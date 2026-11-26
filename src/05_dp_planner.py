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

Writes outputs/results.csv, outputs/paired.csv and outputs/figures/call_price.png.
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


def policies(v, dist, rho):
    """Call vectors for every method on one video."""
    dhat = v["dhat"].to_numpy()
    V = value_table(len(v), c.budget(len(v), rho), dist)
    return {
        "always ResNet": np.zeros(len(v), dtype=int),
        "call-on-claim": c.simulate(v, lambda t, b: v["r_cvs"].iat[t] == 1, rho),
        "no-price": c.simulate(v, lambda t, b: dhat[t] - c.CALL_COST > 0, rho),
        "DP planner": c.simulate(v, lambda t, b: dhat[t] - c.CALL_COST > price(V, t, b), rho),
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

    rows, pairs = [], []
    for split in splits:
        videos = [v.reset_index(drop=True) for _, v in df[df.split == split].groupby("video")]
        for rho in RHOS:
            results = {}
            for v in videos:
                for name, calls in policies(v, dist, rho).items():
                    results.setdefault(name, []).append(c.video_results(v, calls, rho))
            tables = {name: pd.DataFrame(r) for name, r in results.items()}
            for name, table in tables.items():
                rows.append({"split": split, "rho": rho, "method": name, **c.reward_settings(), **c.summarise(table)})
            returns = pd.DataFrame({name: t["return"] for name, t in tables.items()})
            for a, b in (("DP planner", "no-price"), ("DP planner", "call-on-claim"),
                         ("no-price", "call-on-claim"), ("oracle", "call-on-claim")):
                pairs.append({"split": split, "rho": rho, **paired(returns, a, b)})

    out = pd.DataFrame(rows)
    gap = out.set_index(["split", "rho", "method"])["mean_return"]
    out["share_of_gap_closed"] = [
        (gap[s, r, "DP planner"] - gap[s, r, "call-on-claim"]) / (gap[s, r, "oracle"] - gap[s, r, "call-on-claim"])
        if m == "DP planner" else np.nan for s, r, m in zip(out.split, out.rho, out.method)]
    out.to_csv(c.OUTPUTS_DIR / "results.csv", index=False)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(c.OUTPUTS_DIR / "paired.csv", index=False)
    plot_price(dist, c.OUTPUTS_DIR / "figures" / "call_price.png")

    with pd.option_context("display.width", 220, "display.precision", 2):
        print(out.drop(columns=["videos"]).to_string(index=False))
        print()
        print(pairs.to_string(index=False))
    print(f"\nwrote results.csv, paired.csv and figures/call_price.png to {c.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
