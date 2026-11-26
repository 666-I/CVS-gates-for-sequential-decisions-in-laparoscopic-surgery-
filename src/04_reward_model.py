"""Reward model (supervised) and the 10-bin gain distribution. CPU only.

    python src/04_reward_model.py [--model split] [--features compact] [--splits val]

The reward model predicts the gain from calling AdaptNet on a frame (score_a - score_r, before the
0.5 fee) from state features, so it predicts the expected gain, which is what the policy compares
with the price. The models and feature sets are in reward_models.py. The default was chosen on val
(docs/results/next_fixes/reward_models_val.csv and reward_models_lovo_val.csv) by this rule:
the model must use ResNet's output (the router's state is built from it, and step 5 asked for the
claim and miss risk features); a variant is out if its val DP planner has more than 3 extra false CVS
or is clearly worse than the old model; then the best mean claim-frame correlation over 5 fold
seeds wins, with ties (within one SD) going to the simpler model.
  split + compact  On frames where ResNet claims CVS: the chance that the claim is false (logistic
                   regression on ResNet's p_cvs and r_min, position in the video and recent claims)
                   times the mean gain of false claims, plus the rest times the mean gain of true
                   claims. On other frames: the mean gain of that position bin.
  split + slim scores a little higher but its classifier sees only position and recent claims, not
  ResNet's confidence. The original model (200 boosted trees on the 8 raw features) is
  --model gbr200 --features base. All variants lean on val video 161 (13 of the 38 claim frames).

On val it is fitted five times with GroupKFold by video, so every val frame gets an out-of-fold
prediction from a model that never saw its video. A final model fitted on all of val predicts test
(only with --splits val,test). Test is never used to fit anything.

Writes:
  outputs/reward_preds.csv  prepped.csv plus dhat (out-of-fold on val, final model on test)
  outputs/gain_dist.npz     out-of-fold val predictions sorted into 10 position bins (bin_0..bin_9)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as c
import reward_models as rm


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
    p.add_argument("--model", default="split", choices=rm.MODELS, help="reward model (default split)")
    p.add_argument("--features", default="compact", choices=list(rm.FEATURE_SETS),
                   help="feature set from reward_models.py (default compact)")
    p.add_argument("--extra-claims", action="store_true",
                   help="also train the claim classifier on out-of-fold train claim frames (step 6, split only)")
    args = p.parse_args()
    splits = c.parse_splits(args.splits)
    extra = rm.train_oof_claims() if args.extra_claims else None
    if extra is not None:
        print(f"extra train claim frames: {len(extra)} ({extra['false'].mean():.0%} false)")
    df = pd.read_csv(c.OUTPUTS_DIR / "prepped.csv")
    prepped_cols = list(df.columns)
    val = rm.add_risk_features(df[df.split == "val"].reset_index(drop=True))
    test = rm.add_risk_features(df[df.split == "test"].reset_index(drop=True))
    print(f"reward model: {args.model}, features: {args.features}")

    oof = np.zeros(len(val))
    for k, (fit, held) in enumerate(GroupKFold(c.N_FOLDS).split(val, groups=val["video"])):
        assert not set(val.loc[fit, "video"]) & set(val.loc[held, "video"])
        model = rm.make_model(args.model, args.features, extra).fit(val.iloc[fit])
        oof[held] = model.predict(val.iloc[held])
    val["dhat"] = oof

    final = rm.make_model(args.model, args.features, extra).fit(val)
    report("val (out-of-fold)", val)
    if "test" in splits:
        test["dhat"] = final.predict(test)
        report("test (final model)", test)
    else:
        test = test.iloc[:0]
    weights = final.importances()
    if weights:
        print("feature importance or coefficient (final model): " + ", ".join(
            f"{n} {w:.2f}" for n, w in sorted(weights.items(), key=lambda x: -abs(x[1]))))

    pd.concat([val, test])[prepped_cols + ["dhat"]].to_csv(c.OUTPUTS_DIR / "reward_preds.csv", index=False)
    dist = {f"bin_{b}": val.loc[val["bin"] == b, "dhat"].to_numpy() for b in range(c.N_BINS)}
    np.savez(c.OUTPUTS_DIR / "gain_dist.npz", **dist)
    print("gain distribution (out-of-fold dhat on val) per bin: frames / mean / share above 0.5")
    for b in range(c.N_BINS):
        d = dist[f"bin_{b}"]
        print(f"  bin {b}: {len(d):4d} / {d.mean():6.2f} / {(d > c.CALL_COST).mean():.1%}")


if __name__ == "__main__":
    main()
