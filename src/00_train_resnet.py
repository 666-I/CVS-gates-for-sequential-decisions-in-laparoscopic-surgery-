"""Train ResNet18, the cheap model, on the Endoscapes train split. Needs a GPU (CUDA or MPS).

    python src/00_train_resnet.py --version v1
    python src/00_train_resnet.py --version v1b --save-epochs all        # cheap-model sweep (step 9)
    python src/00_train_resnet.py --version r112 --size 112              # lower input resolution
    python src/00_train_resnet.py --version fold --fold 0 --epochs 8     # out-of-fold train predictions (step 6)

Trains on 108 train videos and keeps the epoch with the best mAP on the other 12
(common.HOLDOUT_VIDEOS). Val and test are never loaded.

Writes to weights/: resnet18_cvs_<version>.pt and resnet18_cvs_<version>_log.csv.
--save-epochs also keeps resnet18_cvs_<version>_e<N>.pt for the listed epochs ("all" for every one).

--fold k is a different mode: the 120 train videos are split into --n-folds groups by video
(GroupKFold), the model trains on all groups but k for a fixed --epochs (no holdout, no epoch
picking), then predicts the frames of group k and writes outputs/train_oof/resnet18_<version>_fold<k>.csv.
Run it for every k to get honest predictions on all train frames from models that never saw them.
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
SIZE = 224
NORMALISE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def eval_transform(size=SIZE):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), NORMALISE])


def train_transform(size=SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        NORMALISE,
    ])


EVAL_TRANSFORM = eval_transform()


def checkpoint_size(path):
    """Input size a checkpoint was trained at (224 for checkpoints saved before --size existed)."""
    args = torch.load(path, map_location="cpu", weights_only=True).get("args", {})
    return int(args.get("size", SIZE))


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
    p.add_argument("--size", type=int, default=SIZE, help="input is squashed to size x size")
    p.add_argument("--save-epochs", default="", help='also save these epochs, e.g. "1,6" or "all"')
    p.add_argument("--fold", type=int, default=None, help="out-of-fold mode, see the top of this file")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    device = c.get_device(allow_cpu=args.smoke)
    if args.smoke:
        device = torch.device("cpu")
        args.version, args.epochs, args.batch_size, args.workers = "smoke", 1, 8, 0
    keep = (set(range(1, args.epochs + 1)) if args.save_epochs == "all"
            else {int(e) for e in args.save_epochs.split(",") if e.strip()})

    c.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.version if args.fold is None else f"{args.version}_fold{args.fold}"
    best_path = c.WEIGHTS_DIR / f"resnet18_cvs_{tag}.pt"
    log_path = best_path.with_name(best_path.stem + "_log.csv")
    if best_path.exists() and not (args.overwrite or args.smoke):
        raise SystemExit(f"{best_path} already exists. Pick a new --version or pass --overwrite.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.fold is None:
        fit, held = c.load_fit_holdout()
    else:
        from sklearn.model_selection import GroupKFold
        train = c.load_split("train")
        folds = list(GroupKFold(args.n_folds).split(train, groups=train["video"]))
        fit_idx, held_idx = folds[args.fold]
        fit, held = train.iloc[fit_idx].reset_index(drop=True), train.iloc[held_idx].reset_index(drop=True)
        assert not set(fit["video"]) & set(held["video"])
    if args.smoke:
        fit, held = fit.sample(16, random_state=0), held.sample(16, random_state=0)
    print(f"device {device}, fit {len(fit)} frames, {'holdout' if args.fold is None else 'out-of-fold'} "
          f"{len(held)} frames ({held['video'].nunique()} videos), input {args.size}x{args.size}")

    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0)
    train_loader = DataLoader(c.FrameDataset(fit, train_transform(args.size)), shuffle=True, drop_last=True, **opts)
    held_loader = DataLoader(c.FrameDataset(held, eval_transform(args.size)), **opts)

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

        state = {"model_state_dict": model.state_dict(), "epoch": epoch, "holdout_report": report,
                 "holdout_videos": sorted(held["video"].unique().tolist()), "args": vars(args)}
        if epoch in keep:
            torch.save(state, best_path.with_name(f"{best_path.stem}_e{epoch}.pt"))
        if args.fold is not None:
            if epoch == args.epochs:  # fixed number of epochs, no picking
                torch.save(state, best_path)
                out = held[["split", "video", "frame_idx", "N_v", *c.Y_COLS]].copy()
                out[c.R_COLS] = probs
                out["fold"] = args.fold
                (c.OUTPUTS_DIR / "train_oof").mkdir(parents=True, exist_ok=True)
                out.to_csv(c.OUTPUTS_DIR / "train_oof" / f"resnet18_{tag}.csv", index=False)
                print(f"wrote out-of-fold predictions for {len(out)} train frames")
            continue
        if args.smoke or report["mAP"] > best:
            best = report["mAP"]
            torch.save(state, best_path)

    if args.fold is not None:
        return
    print(f"best holdout mAP {best:.4f}, saved to {best_path}")
    if log[int(np.nanargmax([r['mAP'] for r in log]))]["cvs_claims"] == 0 and not args.smoke:
        print("WARNING: the saved model never claims CVS on the holdout at threshold 0.5. "
              "The router needs claims to act on, so talk to the group before using this version.")


if __name__ == "__main__":
    main()
