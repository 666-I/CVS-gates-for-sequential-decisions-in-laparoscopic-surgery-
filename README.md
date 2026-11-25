# CVS-gates-for-sequential-decisions-in-laparoscopic-surgery-

## Project layout

- `data/endoscapes/` holds the local Endoscapes download.
- `weights/` holds the trained `resnet18_cvs.pt` and the downloaded `cvs_adaptnet.pth`.
- `external/CVS-AdaptNet/` is the location for the CVS-AdaptNet checkout.
- `outputs/` holds generated tables, the gain distribution, and `figures/`.
- `src/` holds the pipeline scripts. `src/common.py` is for shared scoring, column names, and paths.
- `requirements.txt` is for Python dependencies as the pipeline is implemented.

## Pipeline order

The scripts are placeholders. Once implemented, run them in this order from the repository root.

1. `python src/00_train_resnet.py` trains ResNet18 on the training split using a GPU and saves `weights/resnet18_cvs.pt`.
2. `python src/01_predict.py` runs both models on the validation and test splits using a GPU and writes `outputs/preds.csv`.
3. `python src/02_prep.py` adds gain labels and state features and writes `outputs/prepped.csv`.
4. `python src/03_baselines.py` evaluates the baselines and oracle to measure routing headroom.
5. `python src/04_reward_model.py` fits the five-fold reward model and writes the 10-bin gain distribution to `outputs/gain_dist.npz`.
6. `python src/05_dp_planner.py` builds the value table, evaluates the policy, and writes the final method results to `outputs/results.csv`.
