"""Step 6 check: do out-of-fold train claim frames help the claim classifier? Val only. CPU only.

    python src/eval_extra_claims.py

Compares the default reward model (split + compact) with and without the extra claim frames from
`00_train_resnet.py --fold k` (see reward_models.train_oof_claims), on val out-of-fold:
claim-frame correlation of dhat with the true gain and AUC for "the claim is false", over 5 shuffled
fold seeds. First prints the claim rate and false-claim rate of the extra frames next to val's,
because the fold ResNets use a fixed 8-epoch recipe, not v1's, and may claim differently.

Rule (docs/results/next_fixes/final_run_plan.md, fixed before this was run): use the extra frames
only if their false-claim rate is within 15 points of val's and the mean claim-frame correlation
improves by more than one SD. Writes docs/results/next_fixes/extra_claims_val.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import common as c
import reward_models as rm

SEEDS = range(5)


def oof(val, extra, seed):
    out = np.zeros(len(val))
    folds = GroupKFold(c.N_FOLDS, shuffle=True, random_state=seed)
    for fit, held in folds.split(val, groups=val["video"]):
        out[held] = rm.make_model("split", "compact", extra).fit(val.iloc[fit]).predict(val.iloc[held])
    return out


def main():
    df = pd.read_csv(c.OUTPUTS_DIR / "prepped.csv")
    val = rm.add_risk_features(df[df.split == "val"].reset_index(drop=True))
    extra = rm.train_oof_claims()
    claim = val["r_cvs"].to_numpy() == 1
    false = ~c.is_cvs(val.loc[claim, c.Y_COLS].to_numpy())
    n_train = sum(len(pd.read_csv(f)) for f in sorted((c.ROOT / "outputs" / "train_oof").glob("resnet18_oof_fold*.csv")))
    print(f"val: {claim.sum()} claims in {len(val)} frames ({claim.mean():.1%}), {false.mean():.0%} false")
    print(f"train out-of-fold: {len(extra)} claims in {n_train} frames ({len(extra) / n_train:.1%}), "
          f"{extra['false'].mean():.0%} false, from {extra['video'].nunique()} videos")

    rows = []
    for name, ex in (("val only", None), ("val + train claims", extra)):
        for seed in SEEDS:
            dhat = oof(val, ex, seed)
            rows.append({"variant": name, "seed": seed,
                         "claim_corr": np.corrcoef(dhat[claim], val.loc[claim, "gain"])[0, 1],
                         "claim_auc_false": roc_auc_score(false, dhat[claim])})
    out = pd.DataFrame(rows)
    summary = out.groupby("variant")[["claim_corr", "claim_auc_false"]].agg(["mean", "std"])
    print(summary.round(3).to_string())
    base, new = summary.loc["val only"], summary.loc["val + train claims"]
    rate_ok = abs(extra["false"].mean() - false.mean()) <= 0.15
    corr_ok = new[("claim_corr", "mean")] - base[("claim_corr", "mean")] > base[("claim_corr", "std")]
    print(f"false-claim rates within 15 points: {rate_ok}; correlation up by more than one SD: {corr_ok}; "
          f"use the extra frames: {rate_ok and corr_ok}")
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().assign(train_claims=len(extra), train_false_rate=extra["false"].mean(),
                                 val_claims=int(claim.sum()), val_false_rate=false.mean(),
                                 use_extra=rate_ok and corr_ok).to_csv(
        c.ROOT / "docs" / "results" / "next_fixes" / "extra_claims_val.csv", index=False)


if __name__ == "__main__":
    main()
