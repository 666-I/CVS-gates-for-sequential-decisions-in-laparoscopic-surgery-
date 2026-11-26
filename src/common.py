"""Shared paths, column names, data loading and scoring.

Run `python src/common.py` to sanity check the dataset against the known counts.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Paths ----------
# The environment variables let a GPU machine keep data and weights elsewhere.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CVS_DATA_DIR", ROOT / "data" / "endoscapes"))
WEIGHTS_DIR = Path(os.environ.get("CVS_WEIGHTS_DIR", ROOT / "weights"))
OUTPUTS_DIR = Path(os.environ.get("CVS_OUTPUTS_DIR", ROOT / "outputs"))
ADAPTNET_REPO = Path(os.environ.get("CVS_ADAPTNET_REPO", ROOT / "external" / "CVS-AdaptNet"))

# ---------- Column names ----------
CRITERIA = ["C1", "C2", "C3"]
Y_COLS = ["y1", "y2", "y3"]  # ground truth, 0 or 1
R_COLS = ["r1", "r2", "r3"]  # ResNet18 probabilities
A_COLS = ["a1", "a2", "a3"]  # AdaptNet probabilities
PRED_COLS = ["split", "video", "frame_idx", "N_v"] + Y_COLS + R_COLS + A_COLS

# ---------- Constants ----------
THRESHOLD = 0.5
FALSE_CVS_PENALTY = 10.0
CALL_COST = 0.5
RHO = 0.1

# Expected size of each official split: (videos, labelled frames).
EXPECTED = {"train": (120, 6960), "val": (41, 2331), "test": (40, 1799)}

# 12 of the 120 train videos, used only to pick the best epoch of ResNet18 and AdaptNet.
# Video lengths are very uneven (7 to 294 frames), so a plain random draw can land on
# anything from 0% to 38% of the CVS frames. This list is the draw, out of 500 random
# seeds, closest to 10% of frames, of each criterion's positives and of CVS positives
# (seed 434). It was chosen from train labels only.
HOLDOUT_VIDEOS = [10, 13, 22, 31, 42, 47, 60, 63, 67, 74, 87, 104]


# ---------- Labels and scoring ----------
def binarise(values):
    """Majority vote: an annotator average >= 0.5 becomes 1, otherwise 0."""
    return (np.asarray(values, dtype=float) >= 0.5).astype(int)


def is_cvs(labels):
    """True where all three criteria are 1. Takes an (n, 3) array of 0/1."""
    return np.asarray(labels).astype(bool).all(axis=-1)


def score(pred, truth):
    """Per-frame reward without the call cost.

    Number of correct criteria, minus 10 if the prediction claims CVS and that is wrong.
    Both inputs are (n, 3) arrays of 0/1. Probabilities must be thresholded first.
    """
    pred, truth = np.asarray(pred), np.asarray(truth)
    for name, arr in (("pred", pred), ("truth", truth)):
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"score() needs 0/1 values, but {name} has others")
    correct = (pred == truth).sum(axis=-1)
    false_cvs = is_cvs(pred) & ~is_cvs(truth)
    return correct - FALSE_CVS_PENALTY * false_cvs


def budget(n_frames, rho=RHO):
    """B_v = max(1, round(rho * N_v)), with halves rounded up.

    Python's round() sends 4.5 to 4, but the formulation doc's 45-frame example gives 5.
    The 1e-9 guards against products like 0.1 * 35 landing just under a half.
    """
    return max(1, math.floor(rho * n_frames + 0.5 + 1e-9))


# ---------- Data loading ----------
def load_split(split):
    """One row per labelled frame: split, video, frame_idx, N_v, y1..y3, path.

    Sorted by video then frame number, as integers (string order puts 1000 before 200).
    frame_idx is the frame number in the file name. Frames are 5 s apart (125 numbers).
    """
    if split not in EXPECTED:
        raise ValueError(f"split must be one of {list(EXPECTED)}, got {split!r}")
    path = DATA_DIR / split / "annotation_ds_coco.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Set CVS_DATA_DIR or put the dataset in {DATA_DIR}")
    images = json.loads(path.read_text())["images"]

    rows = []
    for image in images:
        video, frame = Path(image["file_name"]).stem.split("_")
        labels = binarise(image["ds"])
        # Image width and height in the JSON are swapped on most frames, so they are not used.
        rows.append((split, int(video), int(frame), *labels,
                     str(DATA_DIR / split / image["file_name"])))
    df = pd.DataFrame(rows, columns=["split", "video", "frame_idx", *Y_COLS, "path"])
    df = df.sort_values(["video", "frame_idx"]).reset_index(drop=True)
    df.insert(3, "N_v", df.groupby("video")["frame_idx"].transform("size"))
    return df


def load_fit_holdout():
    """The train split, divided by video into (fit, holdout)."""
    df = load_split("train")
    held = df["video"].isin(HOLDOUT_VIDEOS)
    return df[~held].reset_index(drop=True), df[held].reset_index(drop=True)


class FrameDataset:
    """Map-style dataset giving (transformed image, float label vector, row number)."""

    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df[Y_COLS].to_numpy(dtype=np.float32, copy=True)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as image:
            x = self.transform(image.convert("RGB"))
        return x, self.labels[i], i


# ---------- Model quality, used while training ----------
def prediction_report(truth, probs):
    """Ranking quality plus how the model behaves at the 0.5 threshold we deploy.

    A criterion with no positives gets AP = nan and is left out of mAP.
    """
    from sklearn.metrics import average_precision_score
    truth, probs = np.asarray(truth).astype(int), np.asarray(probs)
    if not np.isfinite(probs).all():
        raise ValueError("model produced nan or inf probabilities")
    pred = (probs >= THRESHOLD).astype(int)
    out, aps = {}, []
    for k, name in enumerate(CRITERIA):
        ap = average_precision_score(truth[:, k], probs[:, k]) if truth[:, k].any() else float("nan")
        aps.append(ap)
        out[f"AP_{name}"] = float(ap)
        out[f"pos_rate_{name}"] = float(pred[:, k].mean())
    out["mAP"] = float("nan") if np.isnan(aps).all() else float(np.nanmean(aps))
    claims = is_cvs(pred)
    out["accuracy"] = float((pred == truth).mean())
    out["cvs_claims"] = int(claims.sum())
    out["false_cvs"] = int((claims & ~is_cvs(truth)).sum())
    out["true_cvs_frames"] = int(is_cvs(truth).sum())
    out["mean_score"] = float(score(pred, truth).mean())
    return out


# ---------- Router: features, simulation, evaluation ----------
FEATURE_COLS = ["r1", "r2", "r3", "r_cvs", "r_min", "pos", "frames_left", "recent_claims"]
N_BINS = 10
RECENT = 5  # recent_claims looks back this many frames


def time_bin(t, n):
    """Position bin 0..9 of frame index t (0-based) in a video of n frames."""
    return np.minimum((N_BINS * np.asarray(t)) // n, N_BINS - 1)


def simulate(video, want_call, rho=RHO):
    """Run one video in frame order with a budget. Returns a 0/1 array of calls.

    want_call(t, b) is asked on every frame t (0-based row in `video`) while calls are left.
    """
    n = len(video)
    b = budget(n, rho)
    calls = np.zeros(n, dtype=int)
    for t in range(n):
        if b > 0 and want_call(t, b):
            calls[t], b = 1, b - 1
    return calls


def oracle_calls(video, rho=RHO):
    """Hindsight oracle: the top B_v frames by true gain, only where gain beats the call cost."""
    n = len(video)
    gain = video["gain"].to_numpy()
    calls = np.zeros(n, dtype=int)
    for t in np.argsort(-gain, kind="stable")[:budget(n, rho)]:
        if gain[t] > CALL_COST:
            calls[t] = 1
    return calls


def video_results(video, calls, rho=RHO):
    """Return, false CVS count and calls used for one video under a 0/1 call vector."""
    calls = np.asarray(calls)
    pred = np.where(calls[:, None] == 1, video[["ay1", "ay2", "ay3"]], video[["ry1", "ry2", "ry3"]])
    truth = video[Y_COLS].to_numpy()
    return {"return": float(score(pred, truth).sum() - CALL_COST * calls.sum()),
            "false_cvs": int((is_cvs(pred) & ~is_cvs(truth)).sum()),
            "calls": int(calls.sum()), "budget": budget(len(video), rho), "frames": len(video)}


def bootstrap_ci(values, n_boot=10000, seed=0):
    """95% interval for the mean, resampling whole videos."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(per_video):
    """One row from a per-video results table."""
    lo, hi = bootstrap_ci(per_video["return"])
    return {"mean_return": per_video["return"].mean(), "ci_low": lo, "ci_high": hi,
            "false_cvs": int(per_video["false_cvs"].sum()), "calls": int(per_video["calls"].sum()),
            "budget": int(per_video["budget"].sum()), "videos": len(per_video)}


def check_data():
    """Compare the dataset with the published counts and print label rates."""
    ok = True
    for split, (n_videos, n_frames) in EXPECTED.items():
        df = load_split(split)
        videos, frames = df["video"].nunique(), len(df)
        missing = sum(not Path(p).exists() for p in df["path"])
        lengths = df.groupby("video").size()
        good = (videos, frames, missing) == (n_videos, n_frames, 0)
        ok &= good
        rates = ", ".join(f"{c} {df[y].mean():.1%}" for c, y in zip(CRITERIA, Y_COLS))
        print(f"{split}: {videos} videos, {frames} frames, {missing} missing images  "
              f"[{'OK' if good else 'MISMATCH, expected %d videos, %d frames' % (n_videos, n_frames)}]")
        print(f"  positives: {rates}, CVS {is_cvs(df[Y_COLS]).mean():.1%} ({is_cvs(df[Y_COLS]).sum()} frames)")
        print(f"  frames per video: min {lengths.min()}, median {lengths.median():.0f}, "
              f"mean {lengths.mean():.1f}, max {lengths.max()}")
    fit, held = load_fit_holdout()
    print(f"train is divided into fit {fit['video'].nunique()} videos / {len(fit)} frames, "
          f"holdout {held['video'].nunique()} videos / {len(held)} frames "
          f"({is_cvs(held[Y_COLS]).sum()} CVS frames)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check_data() else 1)
