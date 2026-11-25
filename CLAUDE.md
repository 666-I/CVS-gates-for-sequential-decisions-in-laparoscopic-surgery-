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
  02_prep.py            # gains + state features
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
- Written and checked on the Mac: `common.py`, `adaptnet.py`, `00_train_resnet.py`, `00b_train_adaptnet.py`. Labels were checked on the real annotation files, AdaptNet's layers, embeddings and loss against the authors' code. Training has only run on fake images on CPU. Nothing has run on a GPU or on real frames.
- Not written yet: 01 to 05.

### Next, on the training PC
Commands are in README.md (written for bash, adapt them if this is Windows).
1. Set up: install torch 2.7 or newer built for CUDA 12.8 or newer (older builds do not run on a 5070) and run the GPU check in README.md, then install requirements, clone CVS-AdaptNet, get the dataset and PeskaVLP (a zip with PeskaVLP.pth inside).
2. `python src/common.py`. Every split must say [OK]. Stop if not.
3. Run both trainers with `--smoke`. This is the first GPU run, so fix any CUDA problem here. Then delete `weights/*_smoke*`.
4. `python src/00_train_resnet.py --version v1`. Look at mAP, and at CVS claims and false claims. If it never claims CVS, stop and tell the group.
5. `python src/00b_train_adaptnet.py --version v1 --grad-checkpoint` (about 4 hours in the paper, `--resume` if interrupted). The "before training" mAP should be low (paper: 26.6 on test) and then climb. If it has not moved after a few epochs, stop and check. Read the memory and time line printed after 20 batches: Windows can spill into system memory and crawl instead of failing. If that happens or it runs out of memory, use `--batch-size 32` and note it here, since the loss depends on it.
6. Copy both `_log.csv` files to `docs/training_logs/`, put the weights on the drive, update this section with the results, and push.
- Do not load val or test during training, and do not tune anything on them.

### After that
Write `01_predict.py`, run it on the training PC, then 02 and 03 (headroom check) before the reward model.
