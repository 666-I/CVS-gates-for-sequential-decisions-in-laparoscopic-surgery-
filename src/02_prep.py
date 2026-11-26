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

The CVS layer (on by default, --no-cvs-head turns it off) sits on top of AdaptNet: a logistic regression on AdaptNet's image
embeddings (from src/01b_extract_features.py) that predicts CVS directly. It is fitted on the 108
fit videos of train; its yes/no threshold is the one that gives AdaptNet the best mean score on the
12 held-out train videos (under the current reward, lambda included). Val and test are not used.
If it says CVS, AdaptNet answers yes on all three criteria; if it says no but the three cut-offs
all say yes, the criterion closest to its cut-off is set to no; otherwise the cut-offs decide.

The reward's missed-CVS penalty (lambda) comes from the CVS_LAMBDA environment variable, 3 by default.
Run src/01b_extract_features.py first: the CVS layer needs its outputs/features_v1.npz.

Calls left is part of the planner's state (the b in V(t, b)), not a reward-model feature:
the gain from calling on a frame does not depend on how many calls remain.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


HEAD_C = 0.3  # regularisation of the CVS layer, chosen on the held-out train videos (CVS average precision)


def combine(a_probs, cut, p_cvs, t):
    """AdaptNet's 0/1 answers from its cut-offs, overruled on CVS by the CVS layer."""
    pred = (a_probs >= cut).astype(int)
    yes = p_cvs >= t
    pred[yes] = 1
    clash = pred.all(1) & ~yes
    if clash.any():
        rows = np.flatnonzero(clash)
        pred[rows, np.argmin((a_probs - cut)[clash], axis=1)] = 0
    return pred


def cvs_head(df, cut_test, path):
    """Per-frame CVS probability from the CVS layer, and the threshold chosen on held-out train videos."""
    f = np.load(path)
    held = f["train_holdout"].astype(bool)
    head = LogisticRegression(C=HEAD_C, max_iter=3000)
    head.fit(f["train_emb"][~held].astype(np.float32), f["train_y"][~held].all(1))
    p_held = head.predict_proba(f["train_emb"][held].astype(np.float32))[:, 1]
    grid = np.unique(np.quantile(p_held, np.linspace(0.5, 1, 201)))
    y_held = f["train_y"][held]
    t = max(grid, key=lambda g: c.score(combine(f["train_posneg"][held], cut_test, p_held, g), y_held).mean())
    keys = []
    for split in ("val", "test"):
        keys.append(pd.DataFrame({"split": split, "video": f[f"{split}_video"], "frame_idx": f[f"{split}_frame"],
                                  "p_cvs": head.predict_proba(f[f"{split}_emb"].astype(np.float32))[:, 1]}))
    p = df[["split", "video", "frame_idx"]].merge(pd.concat(keys), how="left", on=["split", "video", "frame_idx"])
    if p["p_cvs"].isna().any():
        raise ValueError(f"{path} does not cover every frame in preds.csv; rerun 01b_extract_features.py")
    return p["p_cvs"].to_numpy(), float(t)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-calibrate", action="store_true", help="use 0.5 for AdaptNet, as in the first runs")
    p.add_argument("--no-cvs-head", action="store_true", help="turn off the CVS layer (AdaptNet's cut-offs only)")
    p.add_argument("--features", default=str(c.ROOT / "outputs" / "features_v1.npz"),
                   help="AdaptNet embeddings from 01b_extract_features.py, for the CVS layer")
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
    if not args.no_cvs_head:
        cut_test = a_thr[np.flatnonzero(df["split"] == "test")[0]]
        df["p_cvs"], t = cvs_head(df, cut_test, args.features)
        df[["ay1", "ay2", "ay3"]] = combine(df[c.A_COLS].to_numpy(), a_thr, df["p_cvs"].to_numpy(), t)
        print(f"CVS layer on: threshold {t:.3f}, chosen on held-out train videos")
    else:
        df[["ay1", "ay2", "ay3"]] = (df[c.A_COLS].to_numpy() >= a_thr).astype(int)
    print(f"reward: missed-CVS penalty lambda = {c.MISSED_CVS_PENALTY:g}")
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
              f"misses {(~a_claims & c.is_cvs(truth)).sum()} of {c.is_cvs(truth).sum()} CVS frames, "
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
