"""Run ResNet18 and CVS-AdaptNet once on every val and test frame. Needs a GPU (CUDA or MPS).

    python src/01_predict.py --version v1
    python src/01_predict.py --version v1 --adaptnet-weights weights/CVSAdaptNet_released.pth --out released/preds.csv
    python src/01_predict.py --resnet-weights weights/resnet18_cvs_v1b_e1.pt --reuse-adaptnet outputs/preds.csv \
        --out sweep/v1b_e1/preds.csv     # another cheap model, same AdaptNet answers (step 9)

By default both models are ours: resnet18_cvs_<version>.pt and cvs_adaptnet_<version>.pth.
--adaptnet-weights swaps in another AdaptNet checkpoint, e.g. the authors' released one
(weights/CVSAdaptNet_released.pth, link in the CVS-AdaptNet README), used as a comparison. It
reproduces their paper on our test split (0.572 mAP), but its epoch was picked by the authors on
val, so its val predictions are slightly flattering; test is unaffected.

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
    p.add_argument("--adaptnet-weights", default=None,
                   help="AdaptNet checkpoint instead of cvs_adaptnet_<version>.pth, e.g. weights/CVSAdaptNet_released.pth")
    p.add_argument("--resnet-weights", default=None, help="ResNet18 checkpoint instead of resnet18_cvs_<version>.pt")
    p.add_argument("--reuse-adaptnet", default=None,
                   help="copy a1..a3 from this preds.csv instead of running AdaptNet again")
    p.add_argument("--out", default="preds.csv", help="file name in outputs/")
    args = p.parse_args()

    device = c.get_device()
    resnet_path = (c.WEIGHTS_DIR / f"resnet18_cvs_{args.version}.pt" if args.resnet_weights is None
                   else c.Path(args.resnet_weights))
    resnet = train_resnet.load_resnet(resnet_path, device)
    r_transform = train_resnet.eval_transform(train_resnet.checkpoint_size(resnet_path))
    if args.reuse_adaptnet:
        old = pd.read_csv(args.reuse_adaptnet).set_index(["split", "video", "frame_idx"])[c.A_COLS]
        adapt_name = f"AdaptNet answers from {args.reuse_adaptnet}"
    else:
        adapt_path = (c.WEIGHTS_DIR / f"cvs_adaptnet_{args.version}.pth" if args.adaptnet_weights is None
                      else c.Path(args.adaptnet_weights))
        adapt = adaptnet.AdaptNet()
        adaptnet.load_weights(adapt, adapt_path, need_logit_scale=True)
        adapt = adapt.to(device).eval()
        text = adaptnet.embed_eval_prompts(adapt, adaptnet.get_tokenizer(), device)
        adapt_name = adapt_path.name
    print(f"loaded {resnet_path.name} (input {train_resnet.checkpoint_size(resnet_path)}) and {adapt_name}")

    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    tables = []
    for split in ("val", "test"):
        df = c.load_split(split)
        r_probs, r_labels = train_resnet.predict(
            resnet, DataLoader(c.FrameDataset(df, r_transform), **opts), device)
        if args.reuse_adaptnet:
            a_probs = old.loc[list(zip(df["split"], df["video"], df["frame_idx"]))].to_numpy()
            a_standard, a_labels = None, r_labels
        else:
            a_probs, a_standard, a_labels = adaptnet.predict(
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
        if a_standard is not None:
            print(f"{split} AdaptNet standard-mode mAP {c.prediction_report(df[c.Y_COLS], a_standard)['mAP']:.3f}")

    preds = pd.concat(tables).sort_values(["split", "video", "frame_idx"]).reset_index(drop=True)
    c.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (c.OUTPUTS_DIR / args.out).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(c.OUTPUTS_DIR / args.out, index=False)
    print(f"wrote {len(preds)} rows to {c.OUTPUTS_DIR / args.out}")

    if device.type != "cuda" or args.reuse_adaptnet:
        # The 0.5 call cost is backed by timing on the RTX 5070 (docs/results/timing.csv).
        # A Mac GPU gives different numbers, so it does not overwrite that measurement.
        print("per-frame timing skipped (needs CUDA and both models; see docs/results/timing.csv)")
        return
    example = c.FrameDataset(c.load_split("val").head(1), r_transform)[0][0]
    example_a = c.FrameDataset(c.load_split("val").head(1), adaptnet.EVAL_TRANSFORM)[0][0]
    ms_r = time_per_frame(resnet, example, device)
    ms_a = time_per_frame(adapt.encode_image, example_a, device)
    pd.DataFrame([{"model": "ResNet18", "ms_per_frame": ms_r}, {"model": "AdaptNet", "ms_per_frame": ms_a},
                  {"model": "ratio", "ms_per_frame": ms_a / ms_r}]).to_csv(c.OUTPUTS_DIR / "timing.csv", index=False)
    print(f"time per frame: ResNet18 {ms_r:.2f} ms, AdaptNet {ms_a:.2f} ms, ratio {ms_a / ms_r:.1f}x")


if __name__ == "__main__":
    main()
