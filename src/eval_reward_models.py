"""Compare reward model variants on val only. CPU only.

    python src/eval_reward_models.py

Test rows are dropped as soon as prepped.csv is read, and nothing here asks for them.

For every variant (model x feature set):
  1. Out-of-fold correlation of dhat with the true gain, on the frames where ResNet claims CVS and on
     all frames, over 5 fold seeds (GroupKFold by video, shuffled, random_state = 0..4). Mean and SD.
  2. The default folds (unshuffled, the ones 02_prep and 04 use): the same correlations, plus the val
     result of the planner in 05 at rho 0.05, 0.1 and 0.2: DP planner - no-price, DP planner -
     call-on-claim, and for the DP planner the return, false CVS and missed CVS counts.
     (The val rows of 05 are not an honest test, the gain distribution comes from val. They are
     only used to catch a variant that breaks the planner.)

Selection rule:
  a. The reference is the current model (gbr200 on the base features). A variant is rejected if its
     val DP planner is clearly worse than the reference: at rho 0.05 or 0.1 the per-video return
     difference (variant - reference) has a 95% bootstrap interval (over videos) entirely below 0,
     or the DP planner has more than 3 extra false CVS at any rho.
  b. Among the rest, take the highest mean claim-frame correlation over the 5 seeds.
  c. Every variant within one SD (over seeds) of that best mean counts as tied. Of the tied ones,
     take the simplest: ridge, then split, then hgb_mono, then gbr_small, then gbr200, and with the
     same model the shorter feature list.
  The first version of (a) rejected a variant when its DP return was more than 0.15 per video below
  the reference. That is below the noise (the val DP returns of all variants sit within about one
  point of each other and the intervals span zero), so it rejected 12 of 14 variants for nothing.
  It was replaced by the bootstrap interval above before anything was chosen.

This rule picks split + slim on the tie-break. The default in 04 adds one condition stated up front
(the model must use ResNet's output), which picks split + compact. See 04's docstring.

Writes docs/results/next_fixes/reward_models_val.csv.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common as c
import reward_models as rm

SEEDS = range(5)
RHOS = (0.05, 0.1, 0.2)
COMPLEXITY = {"ridge": 0, "split": 1, "hgb_mono": 2, "gbr_small": 3, "gbr200": 4}
REFERENCE = "gbr200_base"
VARIANTS = [("gbr200", "base"), ("gbr200", "risk"), ("gbr200", "slim"),
            ("gbr_small", "base"), ("gbr_small", "risk"), ("gbr_small", "compact"), ("gbr_small", "slim"),
            ("ridge", "risk"), ("ridge", "slim"), ("ridge", "inter"),
            ("hgb_mono", "risk"), ("hgb_mono", "compact"), ("hgb_mono", "slim"),
            ("split", "compact"), ("split", "slim")]


def oof_predictions(val, model, features, folds):
    """Out-of-fold dhat on val for one variant and one fold splitter."""
    oof = np.zeros(len(val))
    for fit, held in folds.split(val, groups=val["video"]):
        assert not set(val.loc[fit, "video"]) & set(val.loc[held, "video"])
        fitted = rm.make_model(model, features).fit(val.iloc[fit])
        oof[held] = fitted.predict(val.iloc[held])
    return oof


def correlations(val, dhat):
    """Correlation of dhat with gain on ResNet-claim frames and on all frames."""
    def corr(a, b):
        return np.nan if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])
    claim = (val["r_cvs"] == 1).to_numpy()
    return corr(dhat[claim], val["gain"].to_numpy()[claim]), corr(dhat, val["gain"].to_numpy())


def load_planner():
    spec = importlib.util.spec_from_file_location("planner", Path(__file__).with_name("05_dp_planner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def planner_results(planner, val, dhat):
    """Val result of 05 for one set of out-of-fold dhat values. One flat dict."""
    val = val.assign(dhat=dhat)
    dist = [val.loc[val["bin"] == b, "dhat"].to_numpy() for b in range(c.N_BINS)]
    videos = [v.reset_index(drop=True) for _, v in val.groupby("video")]
    out = {}
    for rho in RHOS:
        tables = {}
        for v in videos:
            for name, calls in planner.policies(v, dist, rho).items():
                tables.setdefault(name, []).append((v, calls, c.video_results(v, calls, rho)))
        returns = pd.DataFrame({n: [r["return"] for _, _, r in t] for n, t in tables.items()})
        for a, b, tag in (("DP planner", "no-price", "dp_minus_noprice"),
                          ("DP planner", "call-on-claim", "dp_minus_claim")):
            out[f"{tag}_{rho}"] = planner.paired(returns, a, b)["mean_diff"]
        out[f"dp_return_{rho}"] = returns["DP planner"].mean()
        out[f"video_returns_{rho}"] = returns["DP planner"].to_numpy()
        out[f"dp_false_cvs_{rho}"] = sum(r["false_cvs"] for _, _, r in tables["DP planner"])
        out[f"dp_calls_{rho}"] = sum(r["calls"] for _, _, r in tables["DP planner"])
        missed = 0
        for v, calls, _ in tables["DP planner"]:
            pred = np.where(calls[:, None] == 1, v[["ay1", "ay2", "ay3"]], v[["ry1", "ry2", "ry3"]])
            missed += int((~c.is_cvs(pred) & c.is_cvs(v[c.Y_COLS].to_numpy())).sum())
        out[f"dp_missed_cvs_{rho}"] = missed
    return out


def choose(table):
    """The selection rule from the top of this file."""
    ref = table.set_index("variant").loc[REFERENCE]
    ok = table[table["variant"] != REFERENCE].copy()
    bad = pd.Series(False, index=ok.index)
    for rho in (0.05, 0.1):
        for i in ok.index:
            diff = ok.loc[i, f"video_returns_{rho}"] - ref[f"video_returns_{rho}"]
            bad[i] |= c.bootstrap_ci(diff)[1] < 0
    for rho in RHOS:
        bad |= ok[f"dp_false_cvs_{rho}"] > ref[f"dp_false_cvs_{rho}"] + 3
    ok = pd.concat([ok[~bad], table[table["variant"] == REFERENCE]])
    best = ok.loc[ok["claim_corr_mean"].idxmax()]
    tied = ok[ok["claim_corr_mean"] >= best["claim_corr_mean"] - best["claim_corr_sd"]].copy()
    tied["rank"] = [(COMPLEXITY[m], len(rm.FEATURE_SETS[f])) for m, f in zip(tied["model"], tied["features"])]
    pick = tied.sort_values("rank").iloc[0]
    return pick["variant"], list(ok["variant"]), list(tied["variant"]), list(table.loc[bad.index[bad], "variant"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "results" / "next_fixes"
                                         / "reward_models_val.csv"))
    args = p.parse_args()

    df = pd.read_csv(c.OUTPUTS_DIR / "prepped.csv")
    val = rm.add_risk_features(df[df.split == "val"].reset_index(drop=True))
    del df  # test rows are not kept
    planner = load_planner()

    rows = []
    for model, features in VARIANTS:
        name = f"{model}_{features}"
        seed_corr = np.array([correlations(val, oof_predictions(
            val, model, features, GroupKFold(c.N_FOLDS, shuffle=True, random_state=s))) for s in SEEDS])
        dhat = oof_predictions(val, model, features, GroupKFold(c.N_FOLDS))
        claim_default, all_default = correlations(val, dhat)
        row = {"variant": name, "model": model, "features": features,
               "claim_corr_mean": np.nanmean(seed_corr[:, 0]), "claim_corr_sd": np.nanstd(seed_corr[:, 0]),
               "all_corr_mean": np.nanmean(seed_corr[:, 1]), "all_corr_sd": np.nanstd(seed_corr[:, 1]),
               "claim_corr_default": claim_default, "all_corr_default": all_default,
               **planner_results(planner, val, dhat)}
        rows.append(row)
        print(f"{name:20s} claim corr {row['claim_corr_mean']:.3f} +- {row['claim_corr_sd']:.3f}, "
              f"all {row['all_corr_mean']:.3f} +- {row['all_corr_sd']:.3f}; DP-noprice@0.05 "
              f"{row['dp_minus_noprice_0.05']:+.2f}, DP-claim@0.05 {row['dp_minus_claim_0.05']:+.2f}, "
              f"false CVS {row['dp_false_cvs_0.05']}/{row['dp_false_cvs_0.1']}/{row['dp_false_cvs_0.2']}", flush=True)

    table = pd.DataFrame(rows)
    pick, kept, tied, rejected = choose(table)
    print(f"\nrejected by the DP check: {rejected}\ntied with the best: {tied}\nchosen: {pick}")
    table["chosen"] = table["variant"] == pick
    table = table[[col for col in table.columns if not col.startswith("video_returns_")]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.precision", 3):
        print(table.drop(columns=["model", "features"]).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
