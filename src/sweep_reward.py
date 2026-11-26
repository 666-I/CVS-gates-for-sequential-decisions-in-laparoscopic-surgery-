"""Run 02 to 05 once per reward setting, each in its own output folder, on val only. CPU only.

    python src/sweep_reward.py --name lambda --missed 0 3 5 7
    python src/sweep_reward.py --name sensitivity --missed 5 --false 5 10 20 --fee 0.25 0.5 1.0

Every combination of --missed, --false and --fee gets outputs/sweep/<name>/<setting>/, with a copy
of outputs/preds.csv (or --preds), and 02_prep, 03_baselines, 04_reward_model and 05_dp_planner
run there with the matching CVS_* environment variables (see common.py). Test is never touched:
every script runs with --splits val. Extra arguments for 04 and 05 can be passed with
--args04 and --args05, e.g. --args04 "--model ridge".

Writes outputs/sweep/<name>/summary.csv: one row per setting and rho, with the paired differences
(DP planner - no-price, DP planner - call-on-claim, oracle - call-on-claim, with 95% intervals)
and the false CVS, missed CVS and calls of each method. Raw returns change scale with the reward,
so compare settings through the differences, not the returns.
"""
import argparse
import itertools
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import common as c

SRC = Path(__file__).resolve().parent


def run_setting(folder, env, args04, args05, preds):
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(preds, folder / "preds.csv")
    env = {**os.environ, **env, "CVS_OUTPUTS_DIR": str(folder)}
    log = []
    for script, extra in (("02_prep.py", []), ("03_baselines.py", []),
                          ("04_reward_model.py", shlex.split(args04)), ("05_dp_planner.py", shlex.split(args05))):
        cmd = [sys.executable, str(SRC / script), "--splits", "val", *extra]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log.append(f"$ {' '.join(cmd)}\n{out.stdout}{out.stderr}")
        if out.returncode:
            (folder / "log.txt").write_text("\n".join(log))
            raise SystemExit(f"{script} failed in {folder}, see log.txt")
    (folder / "log.txt").write_text("\n".join(log))


def collect(folder, setting):
    results = pd.read_csv(folder / "results.csv")
    paired = pd.read_csv(folder / "paired.csv")
    rows = []
    for rho, part in results.groupby("rho"):
        row = {**setting, "rho": rho}
        for _, p in paired[paired.rho == rho].iterrows():
            key = p["comparison"].replace(" - ", "_minus_").replace(" ", "_").replace("-", "_")
            row[key] = p["mean_diff"]
            row[key + "_lo"] = p["ci_low"]
            row[key + "_hi"] = p["ci_high"]
        for _, m in part.iterrows():
            name = m["method"].replace(" ", "_").replace("-", "_")
            for col in ("false_cvs", "missed_cvs", "calls"):
                row[f"{name}_{col}"] = m[col]
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="sweep folder name in outputs/sweep/")
    p.add_argument("--missed", type=float, nargs="+", default=[c.MISSED_CVS_PENALTY])
    p.add_argument("--false", type=float, nargs="+", default=[c.FALSE_CVS_PENALTY])
    p.add_argument("--fee", type=float, nargs="+", default=[c.CALL_COST])
    p.add_argument("--preds", default=str(c.OUTPUTS_DIR / "preds.csv"))
    p.add_argument("--args04", default="")
    p.add_argument("--args05", default="")
    args = p.parse_args()

    root = c.OUTPUTS_DIR / "sweep" / args.name
    rows = []
    for missed, false, fee in itertools.product(args.missed, args.false, args.fee):
        setting = {"missed_pen": missed, "false_pen": false, "fee": fee}
        folder = root / f"missed{missed:g}_false{false:g}_fee{fee:g}"
        print(f"running {folder.name}", flush=True)
        run_setting(folder, {"CVS_MISSED_CVS_PENALTY": str(missed), "CVS_FALSE_CVS_PENALTY": str(false),
                             "CVS_CALL_COST": str(fee)}, args.args04, args.args05, args.preds)
        rows.extend(collect(folder, setting))
    out = pd.DataFrame(rows)
    out.to_csv(root / "summary.csv", index=False)
    cols = ["missed_pen", "false_pen", "fee", "rho", "DP_planner_minus_no_price", "DP_planner_minus_no_price_lo",
            "DP_planner_minus_no_price_hi", "DP_planner_minus_call_on_claim", "oracle_minus_call_on_claim",
            "DP_planner_false_cvs", "DP_planner_missed_cvs", "DP_planner_calls", "no_price_calls"]
    with pd.option_context("display.width", 250, "display.precision", 2):
        print(out[[col for col in cols if col in out]].to_string(index=False))
    print(f"wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
