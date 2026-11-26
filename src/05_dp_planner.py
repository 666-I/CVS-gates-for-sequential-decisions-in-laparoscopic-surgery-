"""Value table, policy and final evaluation (model-based RL). CPU only.

    python src/05_dp_planner.py

Value table, per video (N_v frames, B_v calls), filled backwards by backward induction:
    V(t, b) = mean over d in bin(t) of max(d - 0.5 + V(t+1, b-1), V(t+1, b))
    V(t, 0) = 0,  V(N, b) = 0
where the d values are the out-of-fold predicted gains on val for that position bin
(outputs/gain_dist.npz). V(t, b) is the extra points expected from frame t to the end with b calls.

Price of a call at frame t with b calls left: V(t+1, b) - V(t+1, b-1).
Policy: call iff dhat_t - 0.5 > price. The budget is a limit, not a target.

Compared with always ResNet, call-on-claim, the no-price rule (call iff dhat_t - 0.5 > 0)
and the hindsight oracle, on val and test, at rho = 0.05, 0.1 and 0.2. The val rows are not a fair
test (the gain distribution comes from val), they are there as a sanity check. Test is the result.

Second planner, "DP planner (prev claim)": the same table with one extra bit of state, x = whether
frame t-1 was a ResNet CVS claim. Claims come in runs (on val a claim is followed by another claim
about 27% of the time, against about 1.2% after a non-claim), and almost all the value of a call sits
on claim frames, so after a claim the next frame is much more likely to be worth a call:
    V(t, b, x) = E[ max(d - 0.5 + V(t+1, b-1, c_t), V(t+1, b, c_t)) ]
where c_t is whether frame t is a claim, which has probability q_x. If it is a claim, d is drawn
from the pool of out-of-fold gains on all val claim frames (pooled over bins, there are only about
38). If not, d is drawn from the non-claim gains of bin(t). Policy at frame t, whose own claim
status c_t is known: call iff dhat_t - 0.5 > V(t+1, b, c_t) - V(t+1, b-1, c_t).
The pools and q values come from the val rows of reward_preds.csv, also when test is evaluated.

Writes outputs/results.csv, outputs/paired.csv, outputs/figures/call_price.png and
outputs/figures/call_price_prev_claim.png.
"""
import argparse

import numpy as np
import pandas as pd

import common as c

RHOS = (0.05, 0.1, 0.2)


def value_table(n, b_max, dist):
    """V as an (n + 1, b_max + 1) array."""
    V = np.zeros((n + 1, b_max + 1))
    for t in range(n - 1, -1, -1):
        d = dist[int(c.time_bin(t, n))][:, None]                     # (draws, 1)
        call = d - c.CALL_COST + V[t + 1, :-1][None, :]              # spend one of b calls
        skip = V[t + 1, 1:][None, :]
        V[t, 1:] = np.maximum(call, skip).mean(0)
    return V


def price(V, t, b):
    return V[t + 1, b] - V[t + 1, b - 1]


# ---------- Planner with one extra bit of state: was the previous frame a claim ----------
# q_0 is estimated per position bin but shrunk towards the pooled rate, by adding SHRINK_FRAMES
# frames at the pooled rate. Claims are rare and sit mostly in the middle of a video, so the
# position matters, but a bin holds only about 3 claims, so a raw per-bin rate would be mostly
# noise. 200 frames is about one bin's worth of val frames after a non-claim, so a bin's own rate
# gets about half the weight. q_1 is pooled over bins (only about 37 frames follow a claim).
SHRINK_FRAMES = 200


def claim_model(val):
    """Pools and probabilities for the prev-claim planner, from val rows only.

    Returns a dict: claim_pool (out-of-fold dhat on all claim frames), nonclaim (one array of dhat
    per bin, non-claim frames), q0 (one value per bin, P(claim | previous frame not a claim),
    shrunk) and q1 (P(claim | previous frame a claim), pooled).
    """
    val = val.sort_values(["video", "t"])
    claim = val["r_cvs"].to_numpy().astype(int)
    prev = val.groupby("video")["r_cvs"].shift(1).fillna(0).to_numpy().astype(int)  # x = 0 before frame 0
    bins, dhat = val["bin"].to_numpy(), val["dhat"].to_numpy()
    q0_pooled = claim[prev == 0].mean()
    q0 = []
    for k in range(c.N_BINS):
        here = (prev == 0) & (bins == k)
        q0.append((claim[here].sum() + SHRINK_FRAMES * q0_pooled) / (here.sum() + SHRINK_FRAMES))
    return {"claim_pool": dhat[claim == 1],
            "nonclaim": [dhat[(claim == 0) & (bins == k)] for k in range(c.N_BINS)],
            "q0": np.array(q0), "q1": claim[prev == 1].mean(), "q0_pooled": q0_pooled}


def value_table_claim(n, b_max, model):
    """V as an (n + 1, b_max + 1, 2) array. The last axis is x, whether frame t-1 was a claim."""
    V = np.zeros((n + 1, b_max + 1, 2))
    for t in range(n - 1, -1, -1):
        k = int(c.time_bin(t, n))
        ev = []                                                      # expected value if c_t = 0, then 1
        for ct, pool in enumerate((model["nonclaim"][k], model["claim_pool"])):
            d = pool[:, None]
            call = d - c.CALL_COST + V[t + 1, :-1, ct][None, :]
            skip = V[t + 1, 1:, ct][None, :]
            ev.append(np.maximum(call, skip).mean(0))
        for x, q in ((0, model["q0"][k]), (1, model["q1"])):
            V[t, 1:, x] = q * ev[1] + (1 - q) * ev[0]
    return V


def price_claim(V, t, b, ct):
    """Price of a call at frame t when frame t itself is (ct = 1) or is not (ct = 0) a claim."""
    return V[t + 1, b, ct] - V[t + 1, b - 1, ct]


def price_summary(videos, dist, model, rho):
    """Mean price at the claim frames of these videos, with all calls left."""
    old, as_claim, as_nonclaim = [], [], []
    for v in videos:
        n, b = len(v), c.budget(len(v), rho)
        V, Vc = value_table(n, b, dist), value_table_claim(n, b, model)
        for t in np.flatnonzero(v["r_cvs"].to_numpy() == 1):
            old.append(price(V, t, b))
            as_claim.append(price_claim(Vc, t, b, 1))
            as_nonclaim.append(price_claim(Vc, t, b, 0))
    return {"claim frames": len(old), "independent frames": np.mean(old),
            "prev claim, c_t = 1 (actual)": np.mean(as_claim), "prev claim, c_t = 0": np.mean(as_nonclaim)}


def policies(v, dist, model, rho):
    """Call vectors for every method on one video."""
    dhat = v["dhat"].to_numpy()
    claim = v["r_cvs"].to_numpy().astype(int)
    n, b_max = len(v), c.budget(len(v), rho)
    V = value_table(n, b_max, dist)
    Vc = value_table_claim(n, b_max, model)
    return {
        "always ResNet": np.zeros(len(v), dtype=int),
        "call-on-claim": c.simulate(v, lambda t, b: v["r_cvs"].iat[t] == 1, rho),
        "no-price": c.simulate(v, lambda t, b: dhat[t] - c.CALL_COST > 0, rho),
        "DP planner": c.simulate(v, lambda t, b: dhat[t] - c.CALL_COST > price(V, t, b), rho),
        "DP planner (prev claim)": c.simulate(
            v, lambda t, b: dhat[t] - c.CALL_COST > price_claim(Vc, t, b, claim[t]), rho),
        "oracle": c.oracle_calls(v, rho),
    }


def paired(per_video, a, b):
    """Mean per-video difference a - b with a 95% interval, resampling whole videos."""
    diff = per_video[a].to_numpy() - per_video[b].to_numpy()
    lo, hi = c.bootstrap_ci(diff)
    return {"comparison": f"{a} - {b}", "mean_diff": diff.mean(), "ci_low": lo, "ci_high": hi,
            "videos_better": int((diff > 0).sum()), "videos_worse": int((diff < 0).sum())}


def plot_price(dist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, n in zip(axes, (30, 100)):
        b_max = c.budget(n)
        V = value_table(n, b_max, dist)
        for b in range(1, b_max + 1):
            ax.plot(range(n), [price(V, t, b) for t in range(n)], label=f"b = {b}")
        ax.set_title(f"N = {n} frames, B = {b_max} calls")
        ax.set_xlabel("frame t")
    axes[0].set_ylabel("price of a call (points)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def plot_price_claim(videos, dist, model, rho, path):
    """Price of a call with and without the prev-claim state, on the video with the longest claim run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def longest_run(v):
        best = run = 0
        for x in v["r_cvs"].to_numpy():
            run = run + 1 if x else 0
            best = max(best, run)
        return best

    v = max(videos, key=lambda v: (longest_run(v), v["r_cvs"].sum()))
    n = len(v)
    b = min(2, c.budget(n, rho))          # with many calls left the price is close to 0, so show 2 left
    claim = v["r_cvs"].to_numpy().astype(int)
    V, Vc = value_table(n, b, dist), value_table_claim(n, b, model)
    t = np.arange(n)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, height_ratios=(3, 2))
    ax.plot(t, [price(V, i, b) for i in t], label="DP planner (independent frames)")
    ax.plot(t, [price_claim(Vc, i, b, claim[i]) for i in t], label="DP planner (prev claim)")
    ax.set_title(f"val video {int(v['video'].iat[0])}, N = {n}, {b} calls left (red = ResNet claims CVS)")
    ax.set_ylabel("price of a call (points)")
    ax.legend(fontsize=7)
    ax2.plot(t, v["dhat"].to_numpy() - c.CALL_COST, color="grey", lw=0.8)
    ax2.set_ylabel("dhat - 0.5")
    ax2.set_xlabel("frame t")
    for a in (ax, ax2):
        for i in t[claim == 1]:
            a.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="val", help="val, test or val,test. Test is for the final run only.")
    splits = c.parse_splits(p.parse_args().splits)
    df = pd.read_csv(c.OUTPUTS_DIR / "reward_preds.csv")
    missing = set(splits) - set(df["split"])
    if missing:
        raise SystemExit(f"reward_preds.csv has no {sorted(missing)} rows. Run 04 with --splits val,test first.")
    npz = np.load(c.OUTPUTS_DIR / "gain_dist.npz")
    dist = [npz[f"bin_{b}"] for b in range(c.N_BINS)]
    val_videos = [v.reset_index(drop=True) for _, v in df[df.split == "val"].groupby("video")]
    model = claim_model(df[df.split == "val"])            # always from val, never from test
    print(f"prev-claim model from val: {len(model['claim_pool'])} claim frames, "
          f"q_1 = {model['q1']:.3f}, q_0 pooled = {model['q0_pooled']:.4f}, "
          f"q_0 per bin = {np.round(model['q0'], 4).tolist()}\n")

    rows, pairs = [], []
    for split in splits:
        videos = [v.reset_index(drop=True) for _, v in df[df.split == split].groupby("video")]
        for rho in RHOS:
            results = {}
            for v in videos:
                for name, calls in policies(v, dist, model, rho).items():
                    results.setdefault(name, []).append(c.video_results(v, calls, rho))
            tables = {name: pd.DataFrame(r) for name, r in results.items()}
            for name, table in tables.items():
                rows.append({"split": split, "rho": rho, "method": name, **c.reward_settings(), **c.summarise(table)})
            returns = pd.DataFrame({name: t["return"] for name, t in tables.items()})
            for a, b in (("DP planner", "no-price"), ("DP planner", "call-on-claim"),
                         ("DP planner (prev claim)", "DP planner"),
                         ("DP planner (prev claim)", "no-price"),
                         ("DP planner (prev claim)", "call-on-claim"),
                         ("no-price", "call-on-claim"), ("oracle", "call-on-claim")):
                pairs.append({"split": split, "rho": rho, **paired(returns, a, b)})

    out = pd.DataFrame(rows)
    gap = out.set_index(["split", "rho", "method"])["mean_return"]
    out["share_of_gap_closed"] = [
        (gap[s, r, m] - gap[s, r, "call-on-claim"]) / (gap[s, r, "oracle"] - gap[s, r, "call-on-claim"])
        if m.startswith("DP planner") else np.nan for s, r, m in zip(out.split, out.rho, out.method)]
    out.to_csv(c.OUTPUTS_DIR / "results.csv", index=False)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(c.OUTPUTS_DIR / "paired.csv", index=False)
    plot_price(dist, c.OUTPUTS_DIR / "figures" / "call_price.png")
    plot_price_claim(val_videos, dist, model, c.RHO, c.OUTPUTS_DIR / "figures" / "call_price_prev_claim.png")

    with pd.option_context("display.width", 220, "display.precision", 2):
        print(out.drop(columns=["videos"]).to_string(index=False))
        print()
        print(pairs.to_string(index=False))
        print("\nMean price of a call at the val claim frames (all calls left):")
        for rho in RHOS:
            print(f"  rho = {rho}:", {k: round(float(x), 3) for k, x in price_summary(val_videos, dist, model, rho).items()})
    print(f"\nwrote results.csv, paired.csv, figures/call_price.png and figures/call_price_prev_claim.png "
          f"to {c.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
