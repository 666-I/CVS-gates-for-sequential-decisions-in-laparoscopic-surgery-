#!/bin/zsh
# Step 10: the one final run on test, for the main setup and the cheap-model sweep.
# Run only after every choice has been fixed on val. Each setup has its own folder in outputs/final/,
# holding a copy of its preds.csv, and 02 to 05 run there on val and test.
#
#   src/final_run.sh main outputs/preds.csv
#   src/final_run.sh v1b_e1 outputs/cheap/v1b_e1/preds.csv
#   ARGS04="--model gbr200 --features base" src/final_run.sh main_gbr200 outputs/preds.csv
set -e
setopt pipefail   # a failing script must stop the run, not hide behind tee
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
name=$1; preds=$2
[[ -n "$name" && -f "$preds" ]] || { echo "usage: src/final_run.sh <name> <preds.csv>"; exit 1; }
export CVS_OUTPUTS_DIR=$PWD/outputs/final/$name
mkdir -p $CVS_OUTPUTS_DIR
cp "$preds" $CVS_OUTPUTS_DIR/preds.csv
{
  $PY src/02_prep.py --splits val,test
  $PY src/03_baselines.py --splits val,test
  $PY src/04_reward_model.py --splits val,test ${=ARGS04}
  $PY src/05_dp_planner.py --splits val,test
} 2>&1 | tee $CVS_OUTPUTS_DIR/log.txt
