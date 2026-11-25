"""Fine-tune CVS-AdaptNet, the expensive model, from PeskaVLP weights. Needs a GPU.

    python src/00b_train_adaptnet.py --version v1
    python src/00b_train_adaptnet.py --version v1 --resume     # after an interrupted run

Why train it when weights are released: the authors picked their epoch on the val split, which
is our router's data, so AdaptNet would look a little better on val than it does on test.
Here the epoch is picked on the same 12 held-out train videos as ResNet18. Val and test are
never loaded.

Recipe from the paper: Adam, lr 1e-5 (1e-3 for the temperature), weight decay 5e-4, batch 64,
20 epochs, cosine schedule. Two changes, both on purpose:
  - batches are shuffled. The original runs through videos in order, so many batches contain one
    label only, and the contrastive loss has nothing to contrast. `--no-shuffle` restores it.
  - the best epoch is judged with the positive vs negative prompts, the mode the router will use.

Writes to weights/: cvs_adaptnet_<version>.pth, cvs_adaptnet_<version>_log.csv and
cvs_adaptnet_<version>_last.pth (for --resume, safe to delete afterwards).
"""
import argparse
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

import adaptnet
import common as c

# Strengths copied from the authors' configs/config_cvsadaptnet.py.
RAND_AUG = [[dict(type=name, level=level)] for name, level in [
    ("ShearX", 4), ("ShearY", 4), ("Rotate", 15), ("TranslateX", 10), ("TranslateY", 10),
    ("AutoContrast", 1), ("Equalize", 1), ("Contrast", 0.5), ("Color", 0.5), ("Brightness", 0.5),
    ("Sharpness", 0.5)]]


def train_transform():
    path = c.ADAPTNET_REPO / "utilities" / "Augmentations.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Clone CVS-AdaptNet into {c.ADAPTNET_REPO} (see README)")
    sys.path.append(str(c.ADAPTNET_REPO))
    from utilities.Augmentations import RandAugment
    return T.Compose([
        T.Resize((360, 640)),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        RandAugment(RAND_AUG),
        T.ColorJitter(brightness=(0.6, 1.4)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def contrastive_loss(img, txt, labels, scale):
    """Symmetric KL loss from the paper, summed over the three criteria.

    img (B, 768) and txt (3, B, 768) are normalised. For each criterion, frame i should match
    the text of every frame with the same label, not only its own.
    """
    total = 0.0
    for k in range(3):
        same = (labels[:, k, None] == labels[None, :, k]).float()
        target = (same * 10).softmax(1)
        logits = scale * img @ txt[k].T
        for lg in (logits, logits.T):
            total = total + F.kl_div(lg.log_softmax(1), target, reduction="batchmean") / 2
    return total


def sample_texts(prompts, labels):
    """One random matching sentence per frame per criterion, ordered criterion then frame."""
    return [random.choice(prompts[k][int(y)]) for k in range(3) for y in labels[:, k]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1")
    p.add_argument("--init", default=str(c.WEIGHTS_DIR / "PeskaVLP.pth"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="less GPU memory, a bit slower, same result: recomputes the text encoder instead of storing it")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true", help="two CPU batches, random weights, code check only")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cpu" and not args.smoke:
        raise SystemExit("No GPU found. Run this on the GPU machine, or use --smoke to check the code.")
    if args.smoke:
        args.version, args.epochs, args.batch_size, args.workers = "smoke", 1, 4, 0

    c.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = c.WEIGHTS_DIR / f"cvs_adaptnet_{args.version}.pth"
    last_path = best_path.with_name(best_path.stem + "_last.pth")
    log_path = best_path.with_name(best_path.stem + "_log.csv")
    if args.resume and not last_path.exists():
        raise SystemExit(f"Nothing to resume, {last_path} does not exist.")
    if (best_path.exists() or last_path.exists()) and not (args.resume or args.overwrite or args.smoke):
        raise SystemExit(f"{best_path.stem} already exists. Pick a new --version, or pass --resume or --overwrite.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fit, held = c.load_fit_holdout()
    if args.smoke:
        fit, held = fit.sample(8, random_state=0), held.sample(8, random_state=0)
    print(f"device {device}, fit {len(fit)} frames, holdout {len(held)} frames")

    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=use_amp)
    train_loader = DataLoader(c.FrameDataset(fit, train_transform()), shuffle=not args.no_shuffle,
                              drop_last=True, **opts)
    held_loader = DataLoader(c.FrameDataset(held, adaptnet.EVAL_TRANSFORM), **opts)

    tokenizer, prompts = adaptnet.get_tokenizer(), adaptnet.train_prompts()
    if args.smoke:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(adaptnet.BERT_NAME)
        config.num_hidden_layers = 4
        model = adaptnet.AdaptNet(config)
    else:
        model = adaptnet.AdaptNet()
        if not args.resume:
            adaptnet.load_weights(model, args.init, need_logit_scale=False)
    if args.grad_checkpoint:
        model.backbone_text.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)

    optimiser = torch.optim.Adam([
        {"params": [q for n, q in model.named_parameters() if n != "logit_scale"], "lr": 1e-5},
        {"params": [model.logit_scale], "lr": 1e-3},
    ], weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=40)  # 40 as in the original
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start, best, log = 1, -1.0, []
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimiser.load_state_dict(state["optimiser"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        start, best, log = state["epoch"] + 1, state["best"], state["log"]
        # A fresh seed per restart, so the shuffling does not repeat the epochs already seen.
        random.seed(args.seed + start)
        torch.manual_seed(args.seed + start)
        print(f"resuming at epoch {start}, best holdout mAP so far {best:.4f}")
    else:
        text = adaptnet.embed_eval_prompts(model, tokenizer, device)
        posneg, _, labels = adaptnet.predict(model, held_loader, text, device)
        print(f"before training: holdout mAP {c.prediction_report(labels, posneg)['mAP']:.4f}")

    for epoch in range(start, args.epochs + 1):
        model.train()
        losses, began = [], time.time()
        for x, y, _ in train_loader:
            tokens = adaptnet.tokenize(tokenizer, sample_texts(prompts, y), device)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                img = model.encode_image(x)
                txt = model.encode_text(tokens)
            # Loss in full precision. txt rows are ordered criterion then frame.
            loss = contrastive_loss(img, txt.view(3, len(y), -1), y, model.logit_scale.exp())
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} in epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            losses.append(loss.item())
            if use_amp and epoch == start and len(losses) == 20:
                # Windows drivers may spill into slow system memory instead of stopping with an
                # out of memory error, so look at this line before leaving the run alone.
                used = torch.cuda.max_memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(device).total_memory / 1024**3
                per_batch = (time.time() - began) / 20
                print(f"after 20 batches: peak GPU memory {used:.1f} of {total:.1f} GB, {per_batch:.2f} s per batch, "
                      f"about {per_batch * len(train_loader) * args.epochs / 3600:.1f} h for {args.epochs} epochs", flush=True)
        schedule.step()

        text = adaptnet.embed_eval_prompts(model, tokenizer, device)
        posneg, standard, labels = adaptnet.predict(model, held_loader, text, device)
        report = c.prediction_report(labels, posneg)
        standard_map = c.prediction_report(labels, standard)["mAP"]
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                    "logit_scale": model.logit_scale.exp().item(), "standard_mAP": standard_map, **report})
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"epoch {epoch:2d}  loss {log[-1]['train_loss']:.4f}  mAP {report['mAP']:.4f} "
              f"(standard {standard_map:.4f})  acc {report['accuracy']:.4f}  CVS claims {report['cvs_claims']} "
              f"({report['false_cvs']} false, truth has {report['true_cvs_frames']})", flush=True)

        if args.smoke or report["mAP"] > best:
            best = report["mAP"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "holdout_report": report,
                        "holdout_videos": c.HOLDOUT_VIDEOS, "args": vars(args)}, best_path)
        if not args.smoke:
            tmp = last_path.with_suffix(".tmp")  # a kill during saving must not ruin the last good file
            torch.save({"model_state_dict": model.state_dict(), "optimiser": optimiser.state_dict(),
                        "schedule": schedule.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best": best, "log": log}, tmp)
            tmp.replace(last_path)

    print(f"best holdout mAP {best:.4f}, saved to {best_path}")


if __name__ == "__main__":
    main()
