#!/bin/zsh
# Every setup declared in docs/results/next_fixes/final_run_plan.md, run once on val and test.
# MAIN_PREDS is AdaptNet v1's preds.csv unless the v3 rule switched the main AdaptNet to v3.
#   src/final_all.sh                 # main = v1
#   MAIN=v3 src/final_all.sh         # main = v3 (v1 then runs as a declared secondary)
set -e
cd "$(dirname "$0")/.."
MAIN=${MAIN:-v1}
v1=outputs/preds.csv
v3=outputs/v3/preds.csv

if [[ $MAIN == v3 ]]; then
  src/final_run.sh main_v3 $v3
  src/final_run.sh adapt_v1 $v1
else
  src/final_run.sh main_v1 $v1
  [[ -f $v3 ]] && src/final_run.sh adapt_v3 $v3
fi
main_preds=${(P)MAIN}
ARGS04="--model gbr200 --features base" src/final_run.sh main_gbr200 $main_preds
ARGS04="--model split --features slim" src/final_run.sh main_split_slim $main_preds
ARGS04="--extra-claims" src/final_run.sh main_extra_claims $main_preds
CVS_MISSED_CVS_PENALTY=0 src/final_run.sh main_lambda0 $main_preds
for m in v1b_e1 v1b_e3 r112; do
  src/final_run.sh cheap_$m outputs/cheap/$m/preds.csv
done
.venv/bin/python src/collect_final.py
