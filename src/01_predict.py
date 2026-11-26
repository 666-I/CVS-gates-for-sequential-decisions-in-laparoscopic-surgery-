"""Run ResNet18 and CVS-AdaptNet once on every val and test frame. Needs a GPU.

    python src/01_predict.py --version v1

Writes outputs/preds.csv: one row per labelled frame with the columns in common.PRED_COLS,
sorted by video then frame. r1..r3 are ResNet18 probabilities, a1..a3 are AdaptNet's
positive vs negative prompt scores. Both are thresholded at 0.5 later, not here.

Also times one frame through each model (batch size 1, GPU compute only, after a warm-up)
and writes outputs/timing.csv, to back up the 0.5 call cost.
"""
import argparse
import importlib
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import adaptnet
import common as c

train_resnet = importlib.import_module("00_train_resnet")  # file name starts with a digit


def time_per_frame(forward, example, device, warmup=20, runs=200):
    """Median milliseconds for one frame through `forward`, GPU compute only."""
    x = example.unsqueeze(0).to(device)
    times = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for i in range(warmup + runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            forward(x)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1", help="weights version for both models")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda")
    resnet_path = c.WEIGHTS_DIR / f"resnet18_cvs_{args.version}.pt"
    adapt_path = c.WEIGHTS_DIR / f"cvs_adaptnet_{args.version}.pth"
    resnet = train_resnet.load_resnet(resnet_path, device)
    adapt = adaptnet.AdaptNet()
    adaptnet.load_weights(adapt, adapt_path, need_logit_scale=True)
    adapt = adapt.to(device).eval()
    text = adaptnet.embed_eval_prompts(adapt, adaptnet.get_tokenizer(), device)
    print(f"loaded {resnet_path.name} and {adapt_path.name}")

    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    tables = []
    for split in ("val", "test"):
        df = c.load_split(split)
        r_probs, r_labels = train_resnet.predict(
            resnet, DataLoader(c.FrameDataset(df, train_resnet.EVAL_TRANSFORM), **opts), device)
        a_probs, _, a_labels = adaptnet.predict(
            adapt, DataLoader(c.FrameDataset(df, adaptnet.EVAL_TRANSFORM), **opts), text, device)
        # Both loaders run in df order, so the labels must line up with the table.
        assert (r_labels == df[c.Y_COLS].to_numpy()).all() and (a_labels == r_labels).all()
        df[c.R_COLS], df[c.A_COLS] = r_probs, a_probs
        tables.append(df[c.PRED_COLS])
        for name, probs in (("ResNet18", r_probs), ("AdaptNet", a_probs)):
            rep = c.prediction_report(df[c.Y_COLS], probs)
            print(f"{split} {name:8s}  mAP {rep['mAP']:.3f}  acc {rep['accuracy']:.3f}  "
                  f"CVS claims {rep['cvs_claims']} ({rep['false_cvs']} false, truth has {rep['true_cvs_frames']})  "
                  f"mean score {rep['mean_score']:.3f}")

    preds = pd.concat(tables).sort_values(["split", "video", "frame_idx"]).reset_index(drop=True)
    c.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(c.OUTPUTS_DIR / "preds.csv", index=False)
    print(f"wrote {len(preds)} rows to {c.OUTPUTS_DIR / 'preds.csv'}")

    example = c.FrameDataset(c.load_split("val").head(1), train_resnet.EVAL_TRANSFORM)[0][0]
    example_a = c.FrameDataset(c.load_split("val").head(1), adaptnet.EVAL_TRANSFORM)[0][0]
    ms_r = time_per_frame(resnet, example, device)
    ms_a = time_per_frame(adapt.encode_image, example_a, device)
    pd.DataFrame([{"model": "ResNet18", "ms_per_frame": ms_r}, {"model": "AdaptNet", "ms_per_frame": ms_a},
                  {"model": "ratio", "ms_per_frame": ms_a / ms_r}]).to_csv(c.OUTPUTS_DIR / "timing.csv", index=False)
    print(f"time per frame: ResNet18 {ms_r:.2f} ms, AdaptNet {ms_a:.2f} ms, ratio {ms_a / ms_r:.1f}x")


if __name__ == "__main__":
    main()
