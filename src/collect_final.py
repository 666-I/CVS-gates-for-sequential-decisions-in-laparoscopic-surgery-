"""Gather the final run (outputs/final/<setup>/) into small tables for docs/results/next_fixes/. CPU only.

    python src/collect_final.py

For every setup folder written by src/final_run.sh, takes results.csv and paired.csv and keeps both
splits (val rows are a sanity check, test rows are the result). Writes final_results.csv (one row per
setup, split, rho and method) and final_paired.csv (paired differences with 95% intervals over videos).
"""
import pandas as pd

import common as c

OUT = c.ROOT / "docs" / "results" / "next_fixes"
# Every setup declared in final_run_plan.md (main_v1 or main_v3, whichever the v3 rule chose).
DECLARED = ["adapt_v3", "main_gbr200", "main_split_slim", "main_extra_claims", "main_lambda0", "pc_repro",
            "cheap_v1b_e1", "cheap_v1b_e3", "cheap_r112"]


def main():
    root = c.ROOT / "outputs" / "final"
    mains = [m for m in ("main_v1", "main_v3") if (root / m / "paired.csv").exists()]
    missing = [d for d in DECLARED if not (root / d / "paired.csv").exists()]
    if len(mains) != 1 or missing:
        raise SystemExit(f"final run incomplete: main setups found {mains}, missing {missing}")
    results, paired = [], []
    for folder in sorted(p for p in root.iterdir() if (p / "paired.csv").exists()):
        results.append(pd.read_csv(folder / "results.csv").assign(setup=folder.name))
        paired.append(pd.read_csv(folder / "paired.csv").assign(setup=folder.name))
    results, paired = pd.concat(results), pd.concat(paired)
    results.to_csv(OUT / "final_results.csv", index=False)
    paired.to_csv(OUT / "final_paired.csv", index=False)
    keep = ["DP planner - no-price", "DP planner - call-on-claim", "oracle - call-on-claim"]
    show = paired[(paired.split == "test") & paired.comparison.isin(keep)]
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(show[["setup", "rho", "comparison", "mean_diff", "ci_low", "ci_high"]].to_string(index=False))
        dp = results[(results.split == "test") & (results.method == "DP planner")]
        print(dp[["setup", "rho", "mean_return", "false_cvs", "missed_cvs", "calls", "budget"]].to_string(index=False))
    print(f"wrote {OUT / 'final_results.csv'} and {OUT / 'final_paired.csv'}")


if __name__ == "__main__":
    main()
