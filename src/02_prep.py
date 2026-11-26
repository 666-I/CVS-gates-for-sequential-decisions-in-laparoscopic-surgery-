"""Add gain labels and state features to outputs/preds.csv. CPU only.

    python src/02_prep.py

Writes outputs/prepped.csv, one row per frame, with:
  ry1..ry3, ay1..ay3  each model's 0/1 prediction (probability >= 0.5)
  score_r, score_a     reward of each model's answer, without the call cost
  gain                 score_a - score_r, the true value of calling (before the 0.5 fee)
  t                    0-based frame index within the video
  state features       common.FEATURE_COLS, built from ResNet output and position only,
                       looking at past frames only, so the router could compute them live

Calls left is part of the planner's state (the b in V(t, b)), not a reward-model feature:
the gain from calling on a frame does not depend on how many calls remain.
"""
import numpy as np
import pandas as pd

import common as c


def main():
    df = pd.read_csv(c.OUTPUTS_DIR / "preds.csv")
    df = df.sort_values(["split", "video", "frame_idx"]).reset_index(drop=True)
    y = df[c.Y_COLS].to_numpy()
    df[["ry1", "ry2", "ry3"]] = (df[c.R_COLS].to_numpy() >= c.THRESHOLD).astype(int)
    df[["ay1", "ay2", "ay3"]] = (df[c.A_COLS].to_numpy() >= c.THRESHOLD).astype(int)
    df["score_r"] = c.score(df[["ry1", "ry2", "ry3"]].to_numpy(), y)
    df["score_a"] = c.score(df[["ay1", "ay2", "ay3"]].to_numpy(), y)
    df["gain"] = df["score_a"] - df["score_r"]

    g = df.groupby(["split", "video"], sort=False)
    df["t"] = g.cumcount()
    df["r_cvs"] = c.is_cvs(df[["ry1", "ry2", "ry3"]]).astype(int)
    df["r_min"] = df[c.R_COLS].min(axis=1)
    df["pos"] = df["t"] / df["N_v"]
    df["frames_left"] = df["N_v"] - df["t"] - 1
    # Share of the previous RECENT frames where ResNet claimed CVS. 0 on the first frame.
    df["recent_claims"] = g["r_cvs"].transform(
        lambda s: s.shift(1).rolling(c.RECENT, min_periods=1).mean()).fillna(0.0)
    df["bin"] = c.time_bin(df["t"], df["N_v"])

    df.to_csv(c.OUTPUTS_DIR / "prepped.csv", index=False)
    print(f"wrote {len(df)} rows to {c.OUTPUTS_DIR / 'prepped.csv'}")

    for split, part in df.groupby("split", sort=False):
        counts = part["gain"].value_counts().sort_index()
        print(f"\n{split}: {part['video'].nunique()} videos, {len(part)} frames, "
              f"ResNet claims CVS on {part['r_cvs'].sum()} frames")
        print("  true gain values (gain: frames): " + ", ".join(f"{k:g}: {v}" for k, v in counts.items()))
        print(f"  frames where calling beats the 0.5 fee: {(part['gain'] > c.CALL_COST).sum()}, "
              f"mean gain {part['gain'].mean():.3f}")
        print(f"  mean gain when ResNet claims CVS: {part.loc[part.r_cvs == 1, 'gain'].mean():.3f}, "
              f"otherwise {part.loc[part.r_cvs == 0, 'gain'].mean():.3f}")
        print("  mean gain by position bin: " +
              " ".join(f"{v:.2f}" for v in part.groupby("bin")["gain"].mean()))


if __name__ == "__main__":
    main()
