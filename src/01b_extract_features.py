"""Save AdaptNet's image embeddings for every labelled frame, for the CVS head. Needs a GPU.

    python src/01b_extract_features.py --version v1

The CVS head is a small classifier on top of a frozen AdaptNet that predicts CVS (all three
criteria) directly, instead of passing three separate cut-offs. It is fitted on train only, so val
and test stay clean.

Writes outputs/features_<version>.npz with, per split (train, val, test):
  <split>_emb     (n, 768) normalised image embeddings, float16
  <split>_posneg  (n, 3) the usual positive vs negative prompt scores
  <split>_y       (n, 3) binary labels
  <split>_video, <split>_frame
Train rows also carry <split>_holdout (1 for the 12 held-out train videos).
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import adaptnet
import common as c


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1", help="AdaptNet weights version (cvs_adaptnet_<version>.pth)")
    p.add_argument("--adaptnet-weights", default=None, help="AdaptNet checkpoint instead of the versioned one")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda")
    path = (c.WEIGHTS_DIR / f"cvs_adaptnet_{args.version}.pth" if args.adaptnet_weights is None
            else c.Path(args.adaptnet_weights))
    model = adaptnet.AdaptNet()
    adaptnet.load_weights(model, path, need_logit_scale=True)
    model = model.to(device).eval()
    pos, neg = adaptnet.embed_eval_prompts(model, adaptnet.get_tokenizer(), device)
    print(f"loaded {path.name}")

    out = {}
    for split in ("train", "val", "test"):
        df = c.load_split(split)
        loader = DataLoader(c.FrameDataset(df, adaptnet.EVAL_TRANSFORM), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True)
        embs, posneg = [], []
        with torch.no_grad():
            for x, _, _ in loader:
                with torch.autocast(device.type):
                    img = model.encode_image(x.to(device, non_blocking=True))
                s_pos, s_neg = img @ pos.T, img @ neg.T
                embs.append(img.cpu().numpy().astype(np.float16))
                posneg.append(torch.stack([s_pos, s_neg]).softmax(0)[0].cpu().numpy())
        out[f"{split}_emb"] = np.concatenate(embs)
        out[f"{split}_posneg"] = np.concatenate(posneg)
        out[f"{split}_y"] = df[c.Y_COLS].to_numpy(np.int8)
        out[f"{split}_video"] = df["video"].to_numpy()
        out[f"{split}_frame"] = df["frame_idx"].to_numpy()
        if split == "train":
            out[f"{split}_holdout"] = df["video"].isin(c.HOLDOUT_VIDEOS).to_numpy(np.int8)
        print(f"{split}: {len(df)} frames")
    c.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = c.OUTPUTS_DIR / f"features_{args.version}.npz"
    np.savez_compressed(dest, **out)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
