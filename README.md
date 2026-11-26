# CVS-gates-for-sequential-decisions-in-laparoscopic-surgery-

## Project layout

- `data/endoscapes/` holds the local Endoscapes download.
- `weights/` holds the trained `resnet18_cvs_<version>.pt` and `cvs_adaptnet_<version>.pth`, and the downloaded `PeskaVLP.pth`.
- `external/CVS-AdaptNet/` is the location for the CVS-AdaptNet checkout.
- `outputs/` holds generated tables, the gain distribution, and `figures/`.
- `src/` holds the pipeline scripts. `src/common.py` has shared scoring, column names, and paths. `src/adaptnet.py` has the CVS-AdaptNet model.
- `requirements.txt` lists the Python dependencies.

## GPU machine setup

Run everything from the repository root.

We train on an RTX 5070. It needs PyTorch 2.7 or newer built for CUDA 12.8 or newer. Install that first, using the command from pytorch.org for your system, because plain `pip install torch` on Windows gives the CPU build. Then check it. This must print the GPU name and a number, with no "not compatible" warning:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.ones(9, device='cuda').sum().item())"
```

```bash
pip install -r requirements.txt

# Prompts and augmentations come from the authors' repo, at the commit we checked against.
git clone https://github.com/CAMMA-public/CVS-AdaptNet.git external/CVS-AdaptNet
git -C external/CVS-AdaptNet checkout bb21ffbd781b6f943be0daaeeccf1567c21e691d

# Dataset (6.3 GB). Unzips to data/endoscapes/.
curl -L -o endoscapes.zip https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip
unzip -q endoscapes.zip -d data && rm endoscapes.zip

# PeskaVLP starting weights. The download is a zip with PeskaVLP.pth inside.
mkdir -p weights
curl -L -o peskavlp.zip "https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1"
unzip peskavlp.zip -d weights && rm peskavlp.zip

python src/common.py    # every split should say [OK]
```

The trainers also download ImageNet ResNet18 weights (PyTorch) and the Bio_ClinicalBERT config and vocabulary (Hugging Face). If GPU nodes have no internet, fetch both from a login node first:

```bash
python -c "from torchvision.models import resnet18; resnet18(weights='IMAGENET1K_V1')"
python -c "from transformers import AutoConfig, AutoTokenizer as T; n='emilyalsentzer/Bio_ClinicalBERT'; AutoConfig.from_pretrained(n); T.from_pretrained(n)"
export HF_HUB_OFFLINE=1    # on the GPU node
```

To keep data or weights elsewhere, set `CVS_DATA_DIR`, `CVS_WEIGHTS_DIR`, `CVS_OUTPUTS_DIR` or `CVS_ADAPTNET_REPO`.

## Training

Each trainer has `--smoke`, a few batches on random weights to check the machine is set up.

```bash
python src/00_train_resnet.py --smoke
python src/00_train_resnet.py --version v1

python src/00b_train_adaptnet.py --smoke
python src/00b_train_adaptnet.py --version v1 --grad-checkpoint            # the paper reports about 4 hours on an RTX A5500
python src/00b_train_adaptnet.py --version v1 --grad-checkpoint --resume   # if it was interrupted
```

What a healthy AdaptNet run looks like: the "before training" mAP should be low (the paper reports 26.6 for PeskaVLP without fine-tuning, on test) and then climb clearly. The paper's fine-tuned model gets 56.8 test mAP with positive and negative prompts. Our 12-video holdout is small and noisy, so do not expect those exact numbers, but if mAP has not moved after a few epochs, stop and check.

Delete `weights/*_smoke*` before syncing weights to the shared drive. Those files are random.

Both train on 108 of the 120 train videos and pick the best epoch on the other 12 (`HOLDOUT_VIDEOS` in `src/common.py`). Neither loads val or test.

`--grad-checkpoint` is there because the 5070 has 12 GB (8 GB on laptops) and the paper's GPU has 24 GB. It saves memory without changing the maths. After 20 batches the script prints peak memory and an estimated time. On Windows a full card can silently spill into system memory and crawl, so check that line. If memory is at the limit or the time estimate is far too long, stop and use `--batch-size 32`, and note it in the report, since the contrastive loss depends on the batch size.

## Optional: the authors' released AdaptNet

Used only as a comparison. Download `CVSAdaptNet.pth` from the link in `external/CVS-AdaptNet/README.md`, save it as `weights/CVSAdaptNet_released.pth`, then:

```bash
python src/01_predict.py --version v1 --adaptnet-weights weights/CVSAdaptNet_released.pth --out released/preds.csv
CVS_OUTPUTS_DIR=outputs/released python src/02_prep.py   # and the same for 03, 04 and 05
```

## Pipeline order

Run in this order from the repository root. All steps are written and have been run.

1. `python src/00_train_resnet.py` trains ResNet18 and saves `weights/resnet18_cvs_<version>.pt`. Needs a GPU.
2. `python src/00b_train_adaptnet.py` fine-tunes CVS-AdaptNet and saves `weights/cvs_adaptnet_<version>.pth`. Needs a GPU.
3. `python src/01_predict.py` runs both models on the validation and test splits using a GPU and writes `outputs/preds.csv`.
4. `python src/01b_extract_features.py` saves AdaptNet's image embeddings for train, val and test to `outputs/features_v1.npz`, for the CVS layer. Needs a GPU (about 3 minutes).
5. `python src/02_prep.py` picks AdaptNet's cut-offs on val, fits the CVS layer on train, adds gain labels and state features, and writes `outputs/prepped.csv` and `outputs/thresholds.csv`. `--no-cvs-head` turns the CVS layer off.
6. `python src/03_baselines.py` evaluates the baselines and oracle to measure routing headroom.
7. `python src/04_reward_model.py` fits the five-fold reward model and writes the 10-bin gain distribution to `outputs/gain_dist.npz`.
8. `python src/05_dp_planner.py` builds the value table, evaluates the policy, and writes the final method results to `outputs/results.csv`.

The reward's missed-CVS penalty is 3 by default. Set `CVS_LAMBDA` to change it for a run (0 gives the original reward), for example `CVS_LAMBDA=0 python src/02_prep.py`, and the same for steps 6 to 8.
