"""Is the val/test shift in ResNet's CVS claims driven by a few videos? CPU only.

    python src/diag_shift.py

ResNet's CVS claims are false about 13 of 38 times on val but 27 of 33 on test, so a reward
model trained on val underestimates the gain from calling on test. This script checks whether
that comes from a handful of videos (a property of the data) or is spread over many videos.

Per split and video: frames, real CVS frames, ResNet claims, false claims, missed CVS, and
AdaptNet claims and false claims (at the cut-offs in use). Per split summary: how concentrated
the false claims are, the false claim rate without the worst video, how many claims fall in
videos with no real CVS, and how claims come in runs of consecutive frames.

A claim means the model predicts all 3 criteria. A false claim is a claim on a frame that is
not a real CVS. Runs are over labelled frames in order inside a video (5 s apart).

Writes docs/results/next_fixes/shift_by_video.csv, shift_summary.csv and
outputs/figures/shift_by_video.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as c

OUT_DIR = c.ROOT / "docs" / "results" / "next_fixes"
FIG = c.OUTPUTS_DIR / "figures" / "shift_by_video.png"


def add_flags(df):
    """Frame level flags: real CVS, ResNet claim, AdaptNet claim, and which claims are false."""
    df = df.sort_values(["split", "video", "frame_idx"]).copy()
    df["cvs"] = c.is_cvs(df[c.Y_COLS].to_numpy())
    df["r_claim"] = df[["ry1", "ry2", "ry3"]].to_numpy().all(axis=1)
    df["a_claim"] = df[["ay1", "ay2", "ay3"]].to_numpy().all(axis=1)
    df["r_false"] = df["r_claim"] & ~df["cvs"]
    df["a_false"] = df["a_claim"] & ~df["cvs"]
    df["r_missed"] = ~df["r_claim"] & df["cvs"]
    return df


def per_video(df):
    g = df.groupby(["split", "video"])
    out = g.agg(frames=("cvs", "size"), real_cvs=("cvs", "sum"),
                r_claims=("r_claim", "sum"), r_false=("r_false", "sum"),
                r_missed=("r_missed", "sum"),
                a_claims=("a_claim", "sum"), a_false=("a_false", "sum")).reset_index()
    return out.astype({k: int for k in out.columns if k not in ("split", "video")})


def run_stats(df):
    """P(claim at t | claim at t-1) and mean run length of consecutive ResNet claims."""
    prev_claim = df.groupby("video")["r_claim"].shift(1, fill_value=False)
    n_prev = int(prev_claim.sum())
    p_cond = (df["r_claim"] & prev_claim).sum() / n_prev if n_prev else np.nan
    runs = []
    for _, claims in df.groupby("video")["r_claim"]:
        length = 0
        for x in claims.to_numpy():
            if x:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
    return p_cond, (np.mean(runs) if runs else np.nan), len(runs)


def summarise(frames, vids, split):
    v = vids[vids["split"] == split]
    total_false = v["r_false"].sum()
    sorted_false = v["r_false"].sort_values(ascending=False).to_numpy()
    worst = v.sort_values("r_false", ascending=False).iloc[0]
    claims_wo, false_wo = v["r_claims"].sum() - worst["r_claims"], total_false - worst["r_false"]
    claiming = v[v["r_claims"] > 0]
    p_cond, run_len, n_runs = run_stats(frames[frames["split"] == split])
    return {
        "split": split,
        "videos": len(v),
        "frames": v["frames"].sum(),
        "resnet_claims": v["r_claims"].sum(),
        "resnet_false_claims": total_false,
        "false_claim_rate": total_false / v["r_claims"].sum(),
        "videos_with_claim": len(claiming),
        "videos_with_false_claim": int((v["r_false"] > 0).sum()),
        "claims_per_claiming_video": claiming["r_claims"].mean(),
        "false_share_top1": sorted_false[:1].sum() / total_false,
        "false_share_top3": sorted_false[:3].sum() / total_false,
        "false_share_top5": sorted_false[:5].sum() / total_false,
        "worst_video": int(worst["video"]),
        "worst_video_false": int(worst["r_false"]),
        "false_claim_rate_without_worst": false_wo / claims_wo,
        "share_videos_with_real_cvs": (v["real_cvs"] > 0).mean(),
        "claims_in_videos_without_cvs": v.loc[v["real_cvs"] == 0, "r_claims"].sum(),
        "claim_rate": v["r_claims"].sum() / v["frames"].sum(),
        "p_claim_given_prev_claim": p_cond,
        "mean_claim_run_length": run_len,
        "claim_runs": n_runs,
        "adaptnet_claims": v["a_claims"].sum(),
        "adaptnet_false_claims": v["a_false"].sum(),
    }


def make_figure(vids):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        v = vids[vids["split"] == split].sort_values("r_claims", ascending=False).reset_index(drop=True)
        x = np.arange(len(v))
        ax.bar(x, v["r_claims"] - v["r_false"], color="tab:green", label="true claims")
        ax.bar(x, v["r_false"], bottom=v["r_claims"] - v["r_false"], color="tab:red", label="false claims")
        ax.set_title(f"{split}: ResNet CVS claims per video")
        ax.set_xlabel("videos, sorted by number of claims")
    axes[0].set_ylabel("claims")
    axes[0].legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)


def main():
    df = add_flags(pd.read_csv(c.OUTPUTS_DIR / "prepped.csv"))
    vids = per_video(df)
    summary = pd.DataFrame([summarise(df, vids, s) for s in ("val", "test")])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vids.to_csv(OUT_DIR / "shift_by_video.csv", index=False)
    summary.to_csv(OUT_DIR / "shift_summary.csv", index=False, float_format="%.4f")
    make_figure(vids)

    pd.set_option("display.width", 200)
    for s in ("val", "test"):
        top = vids[(vids["split"] == s) & (vids["r_claims"] > 0)].sort_values("r_claims", ascending=False)
        print(f"\n{s}: videos with a ResNet claim")
        print(top.to_string(index=False))
    print("\nSummary")
    print(summary.set_index("split").T.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {OUT_DIR / 'shift_by_video.csv'}, {OUT_DIR / 'shift_summary.csv'}, {FIG}")


if __name__ == "__main__":
    main()
