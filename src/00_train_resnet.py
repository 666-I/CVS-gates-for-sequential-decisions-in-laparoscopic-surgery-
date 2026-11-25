"""Train ResNet18, the cheap model, on the Endoscapes train split. Needs a GPU.

    python src/00_train_resnet.py --version v1

Trains on 108 train videos and keeps the epoch with the best mAP on the other 12
(common.HOLDOUT_VIDEOS). Val and test are never loaded.

Writes to weights/: resnet18_cvs_<version>.pt and resnet18_cvs_<version>_log.csv
`--smoke` runs two batches on CPU with random weights, only to check the code runs.
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T

import common as c

# The whole frame is squashed to 224x224. A centre crop would cut off part of the
# anatomy, and keeping the 854x480 shape would make the "cheap" model nearly twice as slow.
SIZE = (224, 224)
NORMALISE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EVAL_TRANSFORM = T.Compose([T.Resize(SIZE), T.ToTensor(), NORMALISE])
TRAIN_TRANSFORM = T.Compose([
    T.Resize(SIZE),
    T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.ToTensor(),
    NORMALISE,
])


def build_resnet(pretrained):
    """ResNet18 with three outputs (logits for C1, C2, C3)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


def load_resnet(path, device="cpu"):
    """Load a checkpoint saved by this script, ready for inference."""
    model = build_resnet(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    """Probabilities for every frame in the loader, in order, plus the labels."""
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().sigmoid().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.smoke:
        raise SystemExit("No GPU found. Run this on the GPU machine, or use --smoke to check the code.")
    if args.smoke:
        args.version, args.epochs, args.batch_size, args.workers = "smoke", 1, 8, 0

    c.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = c.WEIGHTS_DIR / f"resnet18_cvs_{args.version}.pt"
    log_path = best_path.with_name(best_path.stem + "_log.csv")
    if best_path.exists() and not (args.overwrite or args.smoke):
        raise SystemExit(f"{best_path} already exists. Pick a new --version or pass --overwrite.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fit, held = c.load_fit_holdout()
    if args.smoke:
        fit, held = fit.sample(16, random_state=0), held.sample(16, random_state=0)
    print(f"device {device}, fit {len(fit)} frames, holdout {len(held)} frames")

    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(c.FrameDataset(fit, TRAIN_TRANSFORM), shuffle=True, drop_last=True, **opts)
    held_loader = DataLoader(c.FrameDataset(held, EVAL_TRANSFORM), **opts)

    model = build_resnet(pretrained=not args.smoke).to(device)
    # No pos_weight on purpose. It would push the probabilities up, and the router uses them
    # as confidences and is charged 10 for every false CVS claim.
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} in epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            losses.append(loss.item())

        probs, labels = predict(model, held_loader, device)
        report = c.prediction_report(labels, probs)
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **report})
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"epoch {epoch:2d}  loss {log[-1]['train_loss']:.4f}  mAP {report['mAP']:.4f}  "
              f"acc {report['accuracy']:.4f}  CVS claims {report['cvs_claims']} "
              f"({report['false_cvs']} false, truth has {report['true_cvs_frames']})", flush=True)

        if args.smoke or report["mAP"] > best:
            best = report["mAP"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "holdout_report": report,
                        "holdout_videos": c.HOLDOUT_VIDEOS, "args": vars(args)}, best_path)

    print(f"best holdout mAP {best:.4f}, saved to {best_path}")
    if log[int(np.nanargmax([r['mAP'] for r in log]))]["cvs_claims"] == 0 and not args.smoke:
        print("WARNING: the saved model never claims CVS on the holdout at threshold 0.5. "
              "The router needs claims to act on, so talk to the group before using this version.")


if __name__ == "__main__":
    main()
