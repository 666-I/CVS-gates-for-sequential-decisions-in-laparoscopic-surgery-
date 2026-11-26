# CVS Router: MPHY0043 group project

## How to work with me
- Write the code for me, following the plan below. Work one script at a time, run it, and check the output before moving on. Briefly explain what each script does so I understand it.
- Push back if something I do is wrong or there's a better way.
- Plain, direct language. No em dashes.

## Two computers
- Writing computer (Mac, no GPU, no dataset): writes scripts, checks them on fake images.
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
- train: train ResNet18 and AdaptNet only. Both use the same 12 held-out train videos (`HOLDOUT_VIDEOS` in `src/common.py`) for picking the best epoch, so val stays untouched. Do not change that list: random draws can hold 0% to 38% of the CVS frames.
- val: everything for the router (reward model, gain distribution, value table).
- test: final results only. Never used to build anything.
- Any split inside a dataset is by video, never by frame.

## MDP
- Episode: one video, frames in order. t = 1..N_v (labelled frames).
- Action a_t: 0 = skip (use ResNet), 1 = call AdaptNet (use its answer). Masked to 0 when budget is 0.
- Budget: B_v = max(1, round(rho * N_v)), rho = 0.1, halves round up (45 frames gives 5). Use `common.budget`, because Python's round() gives 4. b_{t+1} = b_t - a_t.
- Criteria: C1 = two structures, C2 = hepatocystic triangle dissection, C3 = cystic plate.
- Final prediction: yhat_t = (1 - a_t) * yhat_t^R + a_t * yhat_t^A (a call replaces all 3 criteria with AdaptNet's answers).
- Reward: R_t = sum_i 1[yhat_ti = y_ti] - 10 * 1[false CVS_t] - 0.5 * a_t. False CVS = prediction says all 3 met but truth isn't CVS.
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
4. Reward model (SUPERVISED): gradient-boosted regressor, MSE, predicts g_t from state features. 5-fold GroupKFold by video on val -> out-of-fold predictions. Then a final model on all val, used on test.
5. Gain distribution: out-of-fold predicted gains sorted into 10 bins by position in video (t/N_v). Collected, not trained.
6. Value table (RL): V(t, b) filled backwards (backward induction / finite-horizon value iteration):
   V(t,b) = mean over d in bin(t) of max(d - 0.5 + V(t+1,b-1), V(t+1,b)), V(t,0) = 0, V(N+1,b) = 0.
7. Policy (RL): call iff dhat_t - 0.5 > V(t+1,b) - V(t+1,b-1) (the price of a call). Budget is a limit, not a target: calls must always beat the 0.5 fee.
8. Evaluation on test.

## Baselines and evaluation
- Always ResNet; call-on-claim (call whenever ResNet claims CVS while budget lasts); no-price rule (same reward model, call iff dhat - 0.5 > 0); hindsight oracle (per video, take the top B_v frames by true gain where gain > 0.5).
- No-price vs DP planner is the key comparison: it shows what the RL part adds.
- Report: mean return per video, false CVS count, calls used, share of gap closed = (DP - call-on-claim) / (oracle - call-on-claim), CIs by bootstrapping whole videos, rho sweep (0.05, 0.1, 0.2), plot of call price vs frame.
- Do the headroom check (baselines + oracle) right after data prep, before building the reward model.

## Models
- ResNet18: we train it. ImageNet weights, 3-output head, plain BCEWithLogitsLoss (no pos_weight, it would inflate the confidences and the CVS claims), train split only. Sees the whole frame squashed to 224x224. Threshold 0.5.
- CVS-AdaptNet: github.com/CAMMA-public/CVS-AdaptNet. ResNet50 image encoder (224 centre crop upsampled to 299) plus Bio_ClinicalBERT. We fine-tune it from PeskaVLP with `00b_train_adaptnet.py`. Reason: the authors trained on train but picked their epoch on val, which is our router's data, so val gains would look better than test gains. Released checkpoint is the fallback.
- Two changes from the authors' recipe: batches are shuffled, and the epoch is picked by positive vs negative prompt mAP. Do not compare directly with their 57.6 test mAP.
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
  01b_extract_features.py # GPU -> outputs/features_v1.npz (AdaptNet embeddings for the CVS layer)
  02_prep.py            # AdaptNet cut-offs + CVS layer + gains + state features
  03_baselines.py       # headroom check + oracle
  04_reward_model.py    # 5-fold reward model + gain distribution
  05_dp_planner.py      # value table + policy + evaluation
```
preds.csv columns: split, video, frame_idx, N_v, y1, y2, y3, r1, r2, r3, a1, a2, a3. Sorted by video then frame.

## Known limitations to mention in the report
- Small val set (41 videos); false CVS claims may number in the dozens.
- Gain distribution treats future frames as independent within a time bin (CVS claims really come in runs).
- Budget formula and position feature use N_v, which isn't known in real time.
- The two models see different views: ResNet the whole frame, AdaptNet the authors' centre crop.

## Current status
Everything has run on the training PC (RTX 5070, Miniconda base env, torch 2.15 + CUDA 13.2). Results are in `docs/results/`.

### Main setup (agreed with the group)
- Cheap model: ResNet18 v1, cut at 0.5 (trained with BCE, so 0.5 is meaningful).
- Expensive model: our AdaptNet v1, with per-criterion cut-offs chosen on val and a CVS layer trained on train (see 02 below).
- Reward: missed-CVS penalty lambda = 3 (a false claim still costs 10). lambda = 0 is the original reward.
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

### 02 CVS layer and missed-CVS penalty
- CVS layer: logistic regression (C = 0.3) on AdaptNet v1's image embeddings (`01b_extract_features.py`), predicting CVS directly. Fitted on the 108 fit videos; its threshold (0.388) is the best AdaptNet score on the 12 held-out train videos. Val and test are not used.
- It ranks CVS better than AdaptNet's own scores: CVS average precision 0.39 vs 0.35 on val, 0.34 vs 0.28 on test. On val, calls now fix 9 ResNet misses and break 14 correct CVS calls (was 6 and 24).
- Missed-CVS penalty (`CVS_LAMBDA`, default 3): answers the "never claim CVS" problem. It helps little in practice because every method still misses about 131 to 134 of test's 136 CVS frames; lambda 5 and 7 weaken the planner's advantage. All 8 combinations (lambda 0, 3, 5, 7, layer on/off) are in `docs/results/lambda_comparison.csv`.

### Results, main setup (test, mean per-video difference, 95% bootstrap CI over videos)
| rho | DP planner - no-price | DP planner - call-on-claim | oracle - call-on-claim |
|---|---|---|---|
| 0.05 | +0.82 [0.03, 1.80] | -1.26 [-2.69, -0.16] | +2.61 |
| 0.1 | +0.69 [0.06, 1.56] | -2.61 [-6.90, -0.17] | +3.35 |
| 0.2 | -0.01 [-0.04, 0.00] | -5.16 [-14.13, -0.38] | +3.84 |
- The planner beats no-price at both rho 0.05 and 0.1 with the CIs above 0: pricing calls works. With lambda 0 and no CVS layer this only held at rho 0.05 (+1.24 [0.22, 2.48]).
- Call-on-claim still wins. Test video 171 drives much of that: ResNet falsely claims CVS on 18 of its frames (18 of test's 27 false claims), each worth 11 to 13 points to a call. Without it (sensitivity check only, 171 stays in the main result), planner - call-on-claim at rho 0.1 is -0.63 [-1.29, -0.03] for this setup.
- Not leakage: videos are split cleanly (train 1-120, val 121-161, test 162-201). Other things to state: 28 of ResNet's 39 val claims come from 3 videos, and only 50 of val's 170 CVS frames have all three annotators agreeing on all three criteria.
- Linked frames: a frame worth calling makes the next one 38% likely to be worth calling (12% normally); a ResNet claim is followed by another 26% of the time (1.7% normally). The value table assumes independence.

### Next
- Value table with linked frames: add "was the last frame a ResNet claim" as state, with claim-follows-claim rates from val pooled into 2 or 3 position groups; report with and without.
- Reward model: false-CVS risk features (claim x (1 - r1 r2 r3), its missed-CVS partner), monotonic constraints, ridge as a comparison. Any tuning on val only.
- If AdaptNet is retrained again: train on all 120 train videos for a fixed number of epochs instead of picking the epoch on 12 videos.
- The `.bat` helper scripts used on Windows live in `outputs/run/` (not in git). Python on the PC is `%USERPROFILE%\miniconda3\python.exe`.
