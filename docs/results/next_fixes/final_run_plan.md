# Final run plan (written before any test run of the next fixes)

Everything below was fixed on val (or on the train holdout) before test was touched. Test is run once, with `src/final_run.sh`, for exactly the setups listed here.

## Main setup
- Cheap model: ResNet18 v1, cut at 0.5.
- Expensive model: AdaptNet v1 with per-criterion cut-offs from val, unless the v3 rule below switches it.
- Reward: +1 per correct criterion, -10 false CVS, -5 missed CVS (new, lambda = 5, chosen a priori), -0.5 per call.
- Reward model: `--model split --features compact` (claim classifier on p_cvs, r_min, position, recent claims; per-bin mean elsewhere).
- Planner: DP planner (value table V(t, b)). The prev-claim planner is reported next to it (null result on val).
- Budgets: rho 0.05, 0.1, 0.2.

## Declared secondary runs (reported whatever they show)
- `main_gbr200`: same as main but the old reward model, `ARGS04="--model gbr200 --features base"`. On val this is the only setup where the budget binds and the price matters, so it shows the contrast.
- `main_split_slim`: `ARGS04="--model split --features slim"`, the variant the subagent's tie-break picked.
- `main_lambda0`: main setup with `CVS_MISSED_CVS_PENALTY=0` (the old reward), to compare with the PC results.
- Cheap-model sweep (step 9), same AdaptNet answers: `v1b_e1` (retrain of v1, epoch 1), `v1b_e3` (epoch 3; replaces the planned epoch 6, because in this rerun epoch 6 made 0 holdout claims while epoch 3 made 13, 11 false, like the PC's epoch 6), `r112` (112x112 input, best holdout epoch). Chosen on the train holdout only.

## Known risk, written down before the test run
- Step 4 (a diagnostic that looked at test) found that test video 171 produces 18 of test's 27 false ResNet claims and has no real CVS. The split reward model learnt on val that claims inside a run of claims are usually true (recent_claims is its strongest feature). If video 171's false claims come in runs, the main reward model will trust them and not call. This was not used to choose anything; it is stated here so the test result can be read honestly.
- On val, every reward-model variant leans on val video 161 (13 of the 38 claim frames).
- With the split reward model, the DP planner equals no-price on val at every rho and for every cheap model: few frames are worth a call, so the budget never binds and the price is zero. The PC headline (+1.24 over no-price at rho 0.05 on test) came from the old, noisier reward model and is not expected to hold for the main setup.

## Rules for the two pending GPU results
- AdaptNet v3 (all 120 train videos, 12 epochs, no epoch picking): it becomes the main AdaptNet only if it has both a higher val pos/neg mAP and a higher val mean score at its calibrated cut-offs than v1. If it does, the reward-model selection rule in `04_reward_model.py` is applied again on v3's val data (no other re-tuning), and v1 is kept as a declared secondary.
- Step 6 (out-of-fold ResNet on the 120 train videos): its claim frames are added to the claim classifier only if (a) its claim rate and false-claim rate are close to val's (false-claim rate within 15 percentage points), and (b) the val out-of-fold claim-frame correlation, mean over 5 fold seeds, improves by more than one SD. Otherwise it is reported and not used.
