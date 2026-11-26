"""Reward model (supervised) and the 10-bin gain distribution. CPU only.

    python src/04_reward_model.py

The reward model predicts the gain from calling AdaptNet on a frame (score_a - score_r, before the
0.5 fee) from the state features in common.FEATURE_COLS. It is a gradient-boosted regressor with
squared error, so it predicts the expected gain, which is what the policy compares with the price.

On val it is fitted five times with GroupKFold by video, so every val frame gets an out-of-fold
prediction from a model that never saw its video. A final model fitted on all of val predicts test.
Test is never used to fit anything.

Writes:
  outputs/reward_preds.csv  prepped.csv plus dhat (out-of-fold on val, final model on test)
  outputs/gain_dist.npz     out-of-fold val predictions sorted into 10 position bins (bin_0..bin_9)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import common as c


def make_model():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=20, random_state=0)


def report(name, part):
    """How well dhat tracks the true gain, overall and on the frames that matter."""
    corr = np.corrcoef(part["dhat"], part["gain"])[0, 1]
    mse = ((part["dhat"] - part["gain"]) ** 2).mean()
    base = ((part["gain"].mean() - part["gain"]) ** 2).mean()
    print(f"{name}: correlation {corr:.3f}, MSE {mse:.3f} (predicting the mean: {base:.3f})")
    top = part.nlargest(max(1, len(part) // 20), "dhat")
    print(f"  top 5% by dhat: mean dhat {top['dhat'].mean():.2f}, mean true gain {top['gain'].mean():.2f}, "
          f"ResNet claims CVS on {top['r_cvs'].mean():.0%} of them")
    for flag, sub in part.groupby("r_cvs"):
        print(f"  ResNet claim={flag}: mean dhat {sub['dhat'].mean():.2f}, mean true gain {sub['gain'].mean():.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="val",
                   help="val, or val,test to also predict test with the final model (final run only)")
    splits = c.parse_splits(p.parse_args().splits)
    df = pd.read_csv(c.OUTPUTS_DIR / "prepped.csv")
    val = df[df.split == "val"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    oof = np.zeros(len(val))
    for k, (fit, held) in enumerate(GroupKFold(c.N_FOLDS).split(val, groups=val["video"])):
        model = make_model().fit(val.loc[fit, c.FEATURE_COLS], val.loc[fit, "gain"])
        oof[held] = model.predict(val.loc[held, c.FEATURE_COLS])
        assert not set(val.loc[fit, "video"]) & set(val.loc[held, "video"])
    val["dhat"] = oof

    final = make_model().fit(val[c.FEATURE_COLS], val["gain"])
    report("val (out-of-fold)", val)
    if "test" in splits:
        test["dhat"] = final.predict(test[c.FEATURE_COLS])
        report("test (final model)", test)
    else:
        test = test.iloc[:0]
    print("feature importance: " + ", ".join(
        f"{n} {w:.2f}" for n, w in sorted(zip(c.FEATURE_COLS, final.feature_importances_), key=lambda x: -x[1])))

    pd.concat([val, test]).to_csv(c.OUTPUTS_DIR / "reward_preds.csv", index=False)
    dist = {f"bin_{b}": val.loc[val["bin"] == b, "dhat"].to_numpy() for b in range(c.N_BINS)}
    np.savez(c.OUTPUTS_DIR / "gain_dist.npz", **dist)
    print("gain distribution (out-of-fold dhat on val) per bin: frames / mean / share above 0.5")
    for b in range(c.N_BINS):
        d = dist[f"bin_{b}"]
        print(f"  bin {b}: {len(d):4d} / {d.mean():6.2f} / {(d > c.CALL_COST).mean():.1%}")


if __name__ == "__main__":
    main()
