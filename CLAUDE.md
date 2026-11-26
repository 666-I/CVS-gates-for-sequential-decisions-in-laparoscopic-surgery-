# CVS Router: MPHY0043 group project

## How to work with me
- Write the code for me, following the plan below. Work one script at a time, run it, and check the output before moving on. Briefly explain what each script does so I understand it.
- Push back if something I do is wrong or there's a better way.
- Plain, direct language. No em dashes.

## Two computers
- Mac (M5 Pro, 48 GB): now has the dataset and weights too, and runs every script on its GPU through MPS (`common.get_device`: CUDA, then MPS). Python is `.venv/bin/python` (uv, Python 3.12, torch 2.15). MPS runs in full precision, the PC used fp16, so a frame near a cut-off can flip (one ResNet val claim did). The next-fixes results come from the Mac.
- Training PC (RTX 5070, 12 GB, or 8 GB if it is a laptop): has the dataset and weights, runs the GPU scripts. If you are on this one, start at "Current status".
- Code goes through git. `data/`, `weights/` and `outputs/` are not in git: share weights by drive. Copy small training logs to `docs/training_logs/` so they can be pushed.
- Never push images, weights or any part of the dataset (licence).

## Project in one paragraph
Budget-aware Critical View of Safety (CVS) assessment in laparoscopic cholecystectomy. Each labelled frame gets a cheap model's prediction (ResNet18). A router decides per frame whether to also call an expensive model (CVS-AdaptNet) and use its answer instead. Calls are limited by a per-video budget. The router is the RL part.

## Dataset
- Endoscapes2023, the Endoscapes-CVS201 part (all 201 videos). Folder: `data/endoscapes/` (not in git, licence CC BY-NC-SA 4.0, no redistribution).
- Official splits: 120 train / 41 val / 40 test videos. CVS frames every 5 s. Labelled frames: 6,960 train / 2,331 val / 1,799 test (11,090 total). Frames per video run from 7 to 350, median about 30 (the mean is about 55), so a typical video only gets 3 calls at rho = 0.1.
- Use `train/`, `val/`, `test/` folders and `annotation_ds_coco.json` in each. Image names are `{video}_{frame}.jpg`.
- CVS labels are averaged over 3 annotators (0, 0.33, 0.67, 1). Binarise by majority vote: >= 0.5 -> 1.
- Label key is `ds` (confirmed), a list of 3 floats. Video id is `video_id`. Ignore `width`/`height` in the JSON, they are swapped on most frames. Real images are 854x480.
- CVS is true on 5.3% of train, 7.3% of val and 7.6% of test frames (369 / 170 / 136).

## Split rules (important)
- train: train ResNet18 and AdaptNet only. ResNet18 v1, v1b, r112 and AdaptNet v1, v2 use the same 12 held-out train videos (`HOLDOUT_VIDEOS` in `src/common.py`) for picking the best epoch, so val stays untouched. Do not change that list: random draws can hold 0% to 38% of the CVS frames. The holdout is too small to rank models (it rated v2 below v1, and v1b's "weak" epochs above v1 on val), so AdaptNet v3 trains on all 120 train videos for a fixed 12 epochs with no epoch picking.
- Step 6 exception: 5 ResNet18s trained by video fold on all 120 train videos (fixed 8 epochs, `00_train_resnet.py --fold k`) give out-of-fold predictions on train (`outputs/train_oof/`). Each fold's model never saw the videos it predicts. Their claim frames may only feed the router's claim classifier (`04_reward_model.py --extra-claims`, a declared secondary, not the main setup). AdaptNet is never run on train frames for the router.
- val: everything for the router (reward model, gain distribution, value table).
- test: final results only. Never used to build anything.
- Any split inside a dataset is by video, never by frame.

## MDP
- Episode: one video, frames in order. t = 1..N_v (labelled frames).
- Action a_t: 0 = skip (use ResNet), 1 = call AdaptNet (use its answer). Masked to 0 when budget is 0.
- Budget: B_v = max(1, round(rho * N_v)), rho = 0.1, halves round up (45 frames gives 5). Use `common.budget`, because Python's round() gives 4. b_{t+1} = b_t - a_t.
- Criteria: C1 = two structures, C2 = hepatocystic triangle dissection, C3 = cystic plate.
- Final prediction: yhat_t = (1 - a_t) * yhat_t^R + a_t * yhat_t^A (a call replaces all 3 criteria with AdaptNet's answers).
- Reward: R_t = sum_i 1[yhat_ti = y_ti] - 10 * 1[false CVS_t] - 5 * 1[missed CVS_t] - 0.5 * a_t. False CVS = prediction says all 3 met but truth isn't CVS. Missed CVS = truth is CVS but the prediction does not claim all 3 (added in the next fixes; before it, never claiming CVS scored well). The weights come from `CVS_FALSE_CVS_PENALTY`, `CVS_MISSED_CVS_PENALTY` and `CVS_CALL_COST` (defaults 10, 5, 0.5); `CVS_MISSED_CVS_PENALTY=0` gives the old reward.
- Budget size reasoning: P(CVS) <= min_i P(C_i) = P(C2) (about 11.2% train, 12.5% val, 17% test), so rho = 0.1 sits just below that. What really matters is how often ResNet claims CVS, so measure that on val.
- Return: undiscounted sum over the video (gamma = 1).
- State: ResNet confidence per criterion, ResNet CVS-claim flag, weakest criterion prob, position in video t/N_v, recent-claims average (past frames only), calls left, frames left.

## Formulation doc
`docs/formulations.docx` is the group's original write-up. Where it differs from this file, this file wins. Known differences:
- Its "call if Delta > 0.5" rule ignores the price of saving a call. The real rule is in step 7 below.
- Its Example A counts the gain for one criterion only, but a call replaces all three.
- Its q, f and v values are illustrative guesses, not measured.

## Method: DP planner (model-based RL)
1. Prediction table (data prep): run both models once on all val and test frames, save `outputs/preds.csv`.
2. Gain labels (data prep): g_t = score(AdaptNet) - score(ResNet) on val, where score is the reward without call cost.
3. State features (data prep): from ResNet output, position in video, budget. Past frames only.
4. Reward model (SUPERVISED): predicts g_t from state features. 5-fold GroupKFold by video on val -> out-of-fold predictions. Then a final model on all val, used on test. Default since the next fixes: `--model split --features compact` (on claim frames, a logistic regression on p_cvs, r_min, position and recent claims gives P(claim is false), and dhat = p * mean gain of false claims + (1 - p) * mean gain of true claims; on other frames, the mean gain of the position bin). The old 200-tree boosted regressor is `--model gbr200 --features base`. Variants and the selection rule: `reward_models.py`, `eval_reward_models.py`, `docs/results/next_fixes/reward_models_val.csv`.
5. Gain distribution: out-of-fold predicted gains sorted into 10 bins by position in video (t/N_v). Collected, not trained.
6. Value table (RL): V(t, b) filled backwards (backward induction / finite-horizon value iteration):
   V(t,b) = mean over d in bin(t) of max(d - 0.5 + V(t+1,b-1), V(t+1,b)), V(t,0) = 0, V(N+1,b) = 0.
7. Policy (RL): call iff dhat_t - 0.5 > V(t+1,b) - V(t+1,b-1) (the price of a call). Budget is a limit, not a target: calls must always beat the 0.5 fee.
   05 also runs "DP planner (prev claim)", V(t, b, x) with x = whether frame t-1 was a ResNet claim (claims come in runs). It raises the price after a claim but changed almost no decision on val, so the plain DP planner stays the main method.
8. Evaluation on test. Scripts 02 to 05 take `--splits` (default `val`); test only goes through `src/final_run.sh`, once, for the setups in `docs/results/next_fixes/final_run_plan.md`.

## Baselines and evaluation
- Always ResNet; call-on-claim (call whenever ResNet claims CVS while budget lasts); no-price rule (same reward model, call iff dhat - 0.5 > 0); hindsight oracle (per video, take the top B_v frames by true gain where gain > 0.5).
- No-price vs DP planner is the key comparison: it shows what the RL part adds.
- Report: mean return per video, false CVS count, calls used, share of gap closed = (DP - call-on-claim) / (oracle - call-on-claim), CIs by bootstrapping whole videos, rho sweep (0.05, 0.1, 0.2), plot of call price vs frame.
- Do the headroom check (baselines + oracle) right after data prep, before building the reward model.

## Models
- ResNet18: we train it. ImageNet weights, 3-output head, plain BCEWithLogitsLoss (no pos_weight, it would inflate the confidences and the CVS claims), train split only. Sees the whole frame squashed to 224x224. Threshold 0.5.
- CVS-AdaptNet: github.com/CAMMA-public/CVS-AdaptNet. ResNet50 image encoder (224 centre crop upsampled to 299) plus Bio_ClinicalBERT. We fine-tune it from PeskaVLP with `00b_train_adaptnet.py`. Reason: the authors trained on train but picked their epoch on val, which is our router's data, so val gains would look better than test gains. Released checkpoint is the fallback.
- Two changes from the authors' recipe: batches are shuffled, and the epoch is picked by positive vs negative prompt mAP. Do not compare directly with their 57.6 test mAP. v3 does neither: authors' video order (`--no-shuffle`), all 120 train videos, fixed 12 epochs, last epoch kept (`--all-train`).
- For a1..a3 use the positive vs negative prompt score (cut at 0.5). Their "standard" mode has no meaningful 0.5 cut. Save probabilities in preds.csv, threshold later.
- Call cost: ResNet50 at 299 is only about 4x ResNet18's compute (estimate). 01_predict must measure time per frame for both models to justify the 0.5.

## File layout
```
data/endoscapes/        # dataset (not in git)
weights/                # checkpoints (not in git, shared via drive, versioned v1, v2...)
external/CVS-AdaptNet/  # cloned repo
outputs/                # preds.csv, prepped.csv, gain_dist.npz, results.csv, figures/
src/
  common.py             # paths, column names, load_split, binarise, score, budget, holdout
  adaptnet.py           # AdaptNet model, weight loading, prompts, prediction
  00_train_resnet.py    # GPU
  00b_train_adaptnet.py # GPU
  01_predict.py         # GPU -> outputs/preds.csv
  02_prep.py            # gains + state features
  03_baselines.py       # headroom check + oracle
  04_reward_model.py    # 5-fold reward model + gain distribution
  05_dp_planner.py      # value table + policy + evaluation
  reward_models.py      # reward model variants (04 and the evals use them)
  sweep_reward.py       # 02-05 per reward setting (lambda, penalty, fee), val only
  eval_reward_models.py # reward model comparison on val
  eval_extra_claims.py  # step 6: do train out-of-fold claims help the claim classifier
  diag_cvs_metrics.py   # CVS-level AP, precision, recall, F1 per model
  diag_shift.py         # val/test shift by video, claim runs
  final_run.sh          # the one final test run per setup
```
preds.csv columns: split, video, frame_idx, N_v, y1, y2, y3, r1, r2, r3, a1, a2, a3. Sorted by video then frame.

## Known limitations to mention in the report
- Small val set (41 videos); false CVS claims may number in the dozens.
- Gain distribution treats future frames as independent within a time bin (CVS claims really come in runs).
- Budget formula and position feature use N_v, which isn't known in real time.
- The two models see different views: ResNet the whole frame, AdaptNet the authors' centre crop.
- Val and test differ by video: test video 171 alone gives 18 of test's 27 false ResNet claims (it has no real CVS). Val claims are false 34% of the time, test claims 82%, out-of-fold train claims 69%. So val looks unusually kind to ResNet's claims.
- The main reward model learnt on val that claims inside a run are usually true (val leans on video 161, 13 of 38 claim frames). On test this fails on video 171's run of false claims.
- With a sharp reward model the budget never binds, so the planner's price is zero. The value of planning depends on how noisy the gain predictor is.
- AdaptNet at its accuracy-calibrated cut-offs rarely claims CVS (5 true claims on val), so calls can hardly catch missed CVS, even with the missed-CVS penalty.

## Current status
Two rounds of results. The PC round (below, "PC run") used the old reward (no missed-CVS term) and the old boosted reward model. The next-fixes round ran on the Mac (MPS) on branch `next-fixes`: plan, choices and every result in `docs/results/next_fixes/` (start with `final_run_plan.md`, written before test). On the Mac, val ResNet claims are 38 (13 false) instead of 39 (14 false): one frame flipped (fp32 vs fp16).

### Next fixes, final results (test, Mac, one run per declared setup)
Main setup: ResNet18 v1 + AdaptNet v1 (calibrated), reward with missed-CVS penalty 5, reward model `split compact`, DP planner.
DP planner - no-price on test, 95% CI over videos, for the four combinations of reward and reward model:
| rho 0.05 / 0.1 | gbr200 (old reward model) | split compact (new) |
|---|---|---|
| lambda 0 (old reward) | +1.55 [0.34, 3.10] / +0.71 [0.05, 1.59] (`pc_repro`) | 0.00 / 0.00 (`main_lambda0`) |
| lambda 5 (new reward) | +1.15 [0.34, 2.16] / +0.57 [0.04, 1.41] (`main_gbr200`) | 0.00 / 0.00 (`main_v1`, main) |
- The PC headline reproduces on the Mac (PC: +1.24 and +0.35), and survives the new reward. The change of reward model removes it: the split model calls only on claim frames it thinks are false (10 calls on test, budgets 96 to 360), so the budget never binds and the price is zero. Same on val in all 39 sweep settings.
- DP planner - call-on-claim, main setup: -1.18 [-3.92, 0.65], -2.61 [-8.25, 0.65], -4.94 [-15.22, 0.65] at rho 0.05, 0.1, 0.2. Main setup makes 19 false CVS on test against call-on-claim's 13 / 8 / 0: it trusted video 171's run of false claims (the risk written down before the run). With gbr200: -1.09, -2.77, -5.35. Call-on-claim wins on test on the point estimate in every setup, but for the main setup every interval includes 0 (it excludes 0 only for gbr200 at rho 0.2 and `pc_repro` at rho 0.1 and 0.2). On val the planner beat it (+1.09 at rho 0.1).
- The step 5 reward model improved val claim-frame correlation but not test return: main and `main_gbr200` are tied on test (83.28 vs 83.36 / 83.11 / 82.86 per video), with 19 vs 17 to 18 false CVS. `main_split_slim` gives exactly the main numbers on test: its dhat differs, but the same 10 test frames clear the fee. Oracle - call-on-claim on test: 3.44 / 4.22 / 4.72 with lambda 5 (2.06 / 2.73 / 3.23 with lambda 0).
- `main_extra_claims` (step 6 train claims in the classifier, not main because its false-claim rate failed the pre-declared check): 15 false CVS, DP - call-on-claim -0.10 at rho 0.05.
- Prev-claim planner (step 7): 0.00 on test for the main setup, +0.30 [0.00, 0.84] at rho 0.05 in `pc_repro`. Not worth the extra state.
- AdaptNet v3 (step 8, all 120 train videos, 12 epochs, no shuffle, last epoch, about 1.2 h on the Mac): test mAP 0.492 (v1 0.472, v2 0.504, released 0.572). Kept as secondary by the pre-declared val rule (val mAP higher, val mean score lower). In the router: 19 false CVS, oracle headroom over call-on-claim 5.0 / 6.5 / 7.7.
- Cheap-model sweep (step 9, `model_quality.csv`, `final_*.csv`, setups `cheap_*`): DP = no-price for all; oracle headroom does not grow as the cheap model gets worse (v1 4.22, v1b_e1 3.19, v1b_e3 3.62, r112 3.85 at rho 0.1). Call-on-claim captures most of what a claim-happy ResNet adds.
- Step 1 (lambda 0, 3, 5, 7 on val): lambda raises the oracle headroom but barely changes missed CVS (145 to 149 of 170), because AdaptNet rarely claims CVS. Step 2 (penalty 5/10/20, fee 0.25/0.5/1.0): with gbr200 the price helps only when the budget binds (high penalty, low fee); with split, never.
- Step 3 (`cvs_metrics.csv`): CVS-level AP (product score) val ResNet 0.43 vs AdaptNet 0.35, test ResNet 0.19 vs AdaptNet 0.28. Both have recall under 15% at the cut-offs in use.

### CVS layer (tried after the final run, dropped)
Tried from `main` (Uzy's PC work) after the next-fixes final run, then dropped. Nothing here was used to choose the main setup. The code is not in the repo any more; it is in git history (`src/01b_extract_features.py` and `02_prep.py --cvs-head` in commit 370d34c on this branch, and on `main`). Results stay in `docs/results/cvs_layer/` (full precision) and `docs/results/pc_cvs_layer/` (earlier exploratory run).
- What it was: a logistic regression (C = 0.3) on AdaptNet v1's image embeddings that predicts CVS directly, fitted on the 108 fit train videos, threshold (0.388) chosen on the 12 held-out train videos. If it said CVS, AdaptNet answered yes on all three criteria; if it said no but the cut-offs all said yes, the criterion closest to its cut-off was set to no.
- Checks: it used train only (scrambling val and test labels left its outputs and threshold identical; scrambling train labels changed them), its rows and labels lined up with preds.csv, and reruns were identical.
- As a CVS classifier it beat AdaptNet's own scores: CVS AP 0.395 vs 0.346 on val, 0.337 vs 0.284 on test (AUC 0.818 vs 0.809 on test); not sensitive to C (0.1 to 1.0). Its CVS claims were more precise (val 0.62 vs 0.33) but recall stayed low on test (0.05; every model is under 0.15). On val, calls fixed 9 ResNet misses and broke 14 correct CVS calls (6 and 24 without it).
- In the router (lambda 5, full precision): no gain on val, so not adopted by the next-fixes rule. With `split compact` the planner still equals no-price; with `gbr200`, DP - no-price on val was +0.28 vs +0.51 at rho 0.05 and +0.06 vs -0.34 at rho 0.1 (all intervals include 0). On test, for the record only: with `gbr200` it weakened the price effect (+0.39 [-0.38, 1.35] / +0.34 [0.00, 0.95] with it, +1.15 / +0.57 without); with `split compact` it narrowed DP - call-on-claim (-0.57 / -2.01 vs -1.18 / -2.61, 17 false CVS instead of 19).
- Earlier exploratory PC run (`docs/results/pc_cvs_layer/`, lambda 3 + layer + gbr200): DP - no-price +0.82 [0.03, 1.80] / +0.69 [0.06, 1.56] on test, but lambda 3 was picked after seeing test, so it is exploratory only.

### Full precision
- `01_predict.py` now predicts in full precision (float32, TF32 off) by default; `--mixed` gives the old behaviour. With that, the PC reproduces every declared next-fixes test result exactly (`outputs/final_pc/`: `main_v1`, `main_gbr200`, `main_lambda0`, `pc_repro`).
- With the PC's old mixed-precision predictions, one val claim differed (39 vs 38) and `main_gbr200` came out at +0.64 [-0.11, 1.55] / +0.42 [-0.34, 1.38] on test instead of +1.15 [0.34, 2.16] / +0.57 [0.04, 1.41]: same direction, intervals including 0. The effect is real in direction but small enough for one frame to matter; say so in the report.

### PC run: main setup (agreed with the group)
- Cheap model: ResNet18 v1, cut at 0.5 (trained with BCE, so 0.5 is meaningful).
- Expensive model: our AdaptNet v1, with per-criterion cut-offs chosen on val (see 02 below).
- Goal: show that the decider's logic works, so the key comparison is DP planner vs no-price.

### Training (batch 64, peak 7.4 of 11.9 GB, logs in `docs/training_logs/`)
- ResNet18 v1: best holdout mAP 0.400 (epoch 8), overfits after that.
- AdaptNet v1 (shuffled batches): best holdout mAP 0.392 (epoch 14), about 1.3 h.
- AdaptNet v2 (`--no-shuffle`, the authors' order): best holdout mAP 0.285 (epoch 10), but better on test (below). The 12-video holdout is too small to pick epochs or compare models reliably.

### AdaptNet versions
| | test pos/neg mAP | test standard mAP | val mAP |
|---|---|---|---|
| ResNet18 v1 | 0.422 | | 0.396 |
| AdaptNet v1 | 0.472 | | 0.429 |
| AdaptNet v2 | 0.504 | 0.545 | 0.429 |
| Released (authors) | 0.572 | 0.575 | 0.518 |
- The released checkpoint reproduces the paper (56.8 / 57.6) through our `01_predict.py`, so our evaluation is right and the gap is in our fine-tuning (shuffling explains part of it).
- The authors picked their epoch on val, which is our router's data, so the released model is only used as a comparison (`outputs/released/`).
- Time per frame (batch 1, GPU compute): ResNet18 4.2 ms, AdaptNet 10.4 ms, about 2.5x.

### 02 AdaptNet cut-offs
- AdaptNet's pos/neg scores all sit between about 0.43 and 0.56, so 0.5 is not a meaningful cut: at 0.5 the released model claims CVS on 796 test frames (673 false).
- `02_prep.py` picks one cut-off per criterion on val, for that criterion's accuracy, inside the same 5 video folds as the reward model. Test uses cut-offs from all of val. Saved in `thresholds.csv`.
- With them, AdaptNet v1 alone beats ResNet18 on test: mean score 2.32 vs 2.18, 14 CVS claims (6 false) vs 33 (27 false).
- Do not choose cut-offs by the router's reward: the -10 penalty makes "never claim CVS" the best answer.
- `--calibrate-resnet` does the same for ResNet. It sets C1 to 0.976, so ResNet almost never claims CVS and the router has nothing left to do. Not used. Worth discussing: the reward favours a model that never detects CVS.

### PC run: results, main setup (test, old reward and reward model, mean per-video difference, 95% bootstrap CI)
| rho | DP planner - no-price | DP planner - call-on-claim |
|---|---|---|
| 0.05 | +1.24 [0.22, 2.48] | -0.11 [-0.34, 0.09] |
| 0.1 | +0.35 [-0.60, 1.39] | -0.96 [-2.39, 0.01] |
| 0.2 | -0.01 [-0.04, 0.00] | -3.21 [-8.82, -0.07] |
- The price clearly helps when calls are scarce (rho 0.05), where the planner also ties call-on-claim. With more calls, saving them stops mattering.
- v2 and the released AdaptNet give the same picture: the planner beats no-price at rho 0.05 (+2.06 and +1.21, CIs above 0) but loses more to call-on-claim. See `docs/results/adaptnet_comparison.csv`.
- Why call-on-claim is hard to beat: almost all value comes from catching ResNet's false CVS claims, which that rule targets directly; the oracle only beats it by about 2.7 points per video.
- The reward model is weak (test correlation with true gain about 0.35) and underestimates gains on ResNet's claims, because ResNet's claims are false far more often on test (27 of 33) than on val (14 of 39).

### Next
- Decide how to present the main result: the honest headline is that pricing calls helps only when the gain predictor is noisy enough for the budget to bind. The 2x2 above shows it.
- The val/test shift (video 171) dominates every comparison with call-on-claim. A reward model trained on more varied claims (step 6 data, or a new split of train) would be the next lever.
- Delete `weights/cvs_adaptnet_v3_last.pth` (1.6 GB, only for --resume) once v3 is no longer needed.
- The `.bat` helper scripts used on Windows live in `outputs/run/` (not in git). Python on the PC is `%USERPROFILE%\miniconda3\python.exe`. On the Mac, `outputs/run/mps_queue.sh` ran the GPU jobs.
