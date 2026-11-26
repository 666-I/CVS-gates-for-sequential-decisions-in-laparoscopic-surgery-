"""Add gain labels and state features to outputs/preds.csv. CPU only.

    python src/02_prep.py

Writes outputs/prepped.csv, one row per frame, with:
  ry1..ry3, ay1..ay3  each model's 0/1 prediction (probability >= 0.5)
  score_r, score_a     reward of each model's answer, without the call cost
  gain                 score_a - score_r, the true value of calling (before the 0.5 fee)
  t                    0-based frame index within the video
  a_thr1..a_thr3       the cut-off used for AdaptNet on that frame (see below)
  r_thr1..r_thr3       the cut-off used for ResNet18 (0.5 unless --calibrate-resnet)
  state features       common.FEATURE_COLS, built from ResNet output and position only,
                       looking at past frames only, so the router could compute them live

AdaptNet's positive vs negative prompt scores all sit near 0.5 (roughly 0.43 to 0.56), so 0.5 is
not a meaningful cut. Each criterion gets its own cut-off, chosen on val to maximise that
criterion's accuracy. Val is split into the same 5 video folds as the reward model, and every val
frame uses cut-offs chosen on the other 4 folds, so its gain label never depends on its own video.
Test uses cut-offs chosen on all of val. Test is never used. By default ResNet keeps 0.5, since it
was trained with BCE and its 0.5 is meaningful; --calibrate-resnet chooses its cut-offs the same
way. ResNet's cut-offs also define its CVS claims, so they change the state features and the
call-on-claim baseline too. --no-calibrate uses 0.5 for AdaptNet.
Writes outputs/thresholds.csv.

Calls left is part of the planner's state (the b in V(t, b)), not a reward-model feature:
the gain from calling on a frame does not depend on how many calls remain.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as c


def best_cutoffs(part, cols):
    """Per-criterion cut-off for the probability columns `cols` that maximises accuracy on `part`."""
    out = []
    for a, y in zip(cols, c.Y_COLS):
        x, truth = part[a].to_numpy(), part[y].to_numpy()
        candidates = np.unique(np.quantile(x, np.linspace(0, 1, 801)))
        accuracy = [((x >= g) == truth).mean() for g in candidates]
        out.append(float(candidates[int(np.argmax(accuracy))]))
    return np.array(out)


def cutoffs(df, cols, calibrate):
    """(n, 3) array of cut-offs per frame for `cols`, and a table describing them."""
    thr = np.full((len(df), 3), c.THRESHOLD)
    rows = []
    if calibrate:
        val = np.flatnonzero(df["split"] == "val")
        groups = df["video"].to_numpy()[val]
        for k, (fit, held) in enumerate(GroupKFold(c.N_FOLDS).split(val, groups=groups)):
            cut = best_cutoffs(df.iloc[val[fit]], cols)
            thr[val[held]] = cut
            rows.append({"used_for": f"val fold {k}", **dict(zip(cols, cut))})
        cut = best_cutoffs(df.iloc[val], cols)
        thr[np.flatnonzero(df["split"] == "test")] = cut
        rows.append({"used_for": "test (all val)", **dict(zip(cols, cut))})
    else:
        rows.append({"used_for": "all", **dict(zip(cols, [c.THRESHOLD] * 3))})
    return thr, pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-calibrate", action="store_true", help="use 0.5 for AdaptNet, as in the first runs")
    p.add_argument("--calibrate-resnet", action="store_true",
                   help="also choose ResNet's cut-offs on val, the same way as AdaptNet's")
    args = p.parse_args()

    df = pd.read_csv(c.OUTPUTS_DIR / "preds.csv")
    df = df.sort_values(["split", "video", "frame_idx"]).reset_index(drop=True)
    y = df[c.Y_COLS].to_numpy()
    a_thr, a_table = cutoffs(df, c.A_COLS, calibrate=not args.no_calibrate)
    r_thr, r_table = cutoffs(df, c.R_COLS, calibrate=args.calibrate_resnet)
    table = pd.concat([a_table.assign(model="AdaptNet"), r_table.assign(model="ResNet18")])
    table.to_csv(c.OUTPUTS_DIR / "thresholds.csv", index=False)
    with pd.option_context("display.precision", 4):
        print("AdaptNet cut-offs:\n" + a_table.to_string(index=False))
        print("ResNet18 cut-offs:\n" + r_table.to_string(index=False))
    df[["a_thr1", "a_thr2", "a_thr3"]] = a_thr
    df[["r_thr1", "r_thr2", "r_thr3"]] = r_thr
    df[["ry1", "ry2", "ry3"]] = (df[c.R_COLS].to_numpy() >= r_thr).astype(int)
    df[["ay1", "ay2", "ay3"]] = (df[c.A_COLS].to_numpy() >= a_thr).astype(int)
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
        a_pred, truth = part[["ay1", "ay2", "ay3"]].to_numpy(), part[c.Y_COLS].to_numpy()
        a_claims = c.is_cvs(a_pred)
        print(f"\n{split}: AdaptNet with these cut-offs: accuracy {(a_pred == truth).mean():.3f}, "
              f"CVS claims {a_claims.sum()} ({(a_claims & ~c.is_cvs(truth)).sum()} false), "
              f"mean score {part['score_a'].mean():.3f} (ResNet {part['score_r'].mean():.3f})")
        r_pred = part[["ry1", "ry2", "ry3"]].to_numpy()
        print(f"{split}: ResNet18 with these cut-offs: accuracy {(r_pred == truth).mean():.3f}, "
              f"CVS claims {part['r_cvs'].sum()} ({(part['r_cvs'].astype(bool) & ~c.is_cvs(truth)).sum()} false)")
        print(f"{split}: {part['video'].nunique()} videos, {len(part)} frames, "
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
