"""CVS-AdaptNet model, weight loading, prompts and prediction.

A compact version of github.com/CAMMA-public/CVS-AdaptNet (CC BY-NC-SA 4.0), commit bb21ffb.
Same layers and weight names, without the hardcoded .cuda() calls and relative weight paths.
The prompt files are read from the cloned repo in external/CVS-AdaptNet.

Image encoder: ResNet50 on a 224 crop, upsampled to 299. Text encoder: Bio_ClinicalBERT.
The text embeddings are computed once, so the per-frame cost is only the ResNet50.
"""
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms as T
from transformers import AutoConfig, AutoModel, AutoTokenizer

import common as c

BERT_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # only config and vocab are downloaded
MAX_TOKENS = 77
INIT_LOGIT_SCALE = float(np.log(1 / 0.07))

# Must match the original exactly, because the released weights expect this view.
EVAL_TRANSFORM = T.Compose([
    T.Resize((360, 640)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.global_embedder = nn.Linear(self.model.fc.in_features, 768)
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)
        return self.global_embedder(self.model(x))


class TextEncoder(nn.Module):
    def __init__(self, bert_config):
        super().__init__()
        self.model = AutoModel.from_config(bert_config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        layers = torch.stack(out.hidden_states[-4:]).sum(0)
        # The original merges word pieces one token at a time, then averages over all 77 slots.
        # That equals the sum over real tokens divided by 77, which is far faster to compute.
        return (layers * attention_mask.unsqueeze(-1)).sum(1) / MAX_TOKENS


class AdaptNet(nn.Module):
    def __init__(self, bert_config=None):
        super().__init__()
        self.backbone_img = ImageEncoder()
        self.backbone_text = TextEncoder(bert_config or AutoConfig.from_pretrained(BERT_NAME))
        self.logit_scale = nn.Parameter(torch.tensor(INIT_LOGIT_SCALE))

    def encode_image(self, x):
        return F.normalize(self.backbone_img(x).float(), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.backbone_text(**tokens).float(), dim=-1)


# ---------- Weights ----------
def clean_state_dict(checkpoint):
    """Weight names as used here, from a PeskaVLP, released AdaptNet, or our own checkpoint."""
    state = checkpoint.get("model_state_dict", checkpoint)
    out = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("model.backbone_"):
            key = key.removeprefix("model.")
        if not key.endswith("position_ids"):  # a buffer newer transformers no longer stores
            out[key] = value
    return out


def load_weights(model, path, need_logit_scale):
    """Load weights and stop on any missing or unexpected name.

    The original test.py uses strict=False, where a bad load would pass silently.
    PeskaVLP has no logit_scale, so that one is allowed to be missing when training starts.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    allowed = set() if need_logit_scale else {"logit_scale"}
    if set(missing) - allowed or unexpected:
        raise RuntimeError(f"{path} does not fit the model. Missing {sorted(set(missing) - allowed)[:5]}, "
                           f"unexpected {list(unexpected)[:5]}")
    return checkpoint


# ---------- Text ----------
def get_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_NAME)


def tokenize(tokenizer, sentences, device="cpu"):
    """Padded only to the longest sentence, which gives the same embedding as padding to 77."""
    enc = tokenizer(list(sentences), return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask", "token_type_ids")}


def _prompt_path(name):
    path = c.ADAPTNET_REPO / "class_prompts" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Clone CVS-AdaptNet into {c.ADAPTNET_REPO} (see README)")
    return path


def eval_prompts():
    """(positive, negative) sentences for C1..C3, the ones the authors' test.sh uses."""
    read = lambda name: [s.strip() for s in _prompt_path(name).read_text(encoding="utf-8").splitlines() if s.strip()]
    pos, neg = read("class_prompt_endoscapes_refined.txt"), read("class_prompt_endoscapes_refined_no_class.txt")
    if len(pos) != 3 or len(neg) != 3:
        raise ValueError(f"expected 3 positive and 3 negative prompts, got {len(pos)} and {len(neg)}")
    return pos, neg


def train_prompts():
    """prompts[criterion][label] is a list of sentences (label 1 = criterion met)."""
    def read(name):
        with open(_prompt_path(name), newline="", encoding="utf-8") as f:
            return [row["sentence"] for row in csv.DictReader(f) if row["sentence"].strip()]
    return [[read(f"class_{k}/augmented_no_class.csv"), read(f"class_{k}/augmented_text.csv")] for k in (1, 2, 3)]


# ---------- Prediction ----------
@torch.no_grad()
def embed_eval_prompts(model, tokenizer, device):
    """Text embeddings for (positive, negative) prompts. Compute once, reuse for every frame."""
    model.eval()
    return tuple(model.encode_text(tokenize(tokenizer, s, device)) for s in eval_prompts())


@torch.no_grad()
def predict(model, loader, text, device):
    """Scores for every frame in the loader, in order.

    Returns (posneg, standard, labels):
      posneg    softmax of positive vs negative prompt. Above 0.5 means the frame is closer to
                the positive prompt, so 0.5 is a natural cut. This is what the router should use.
                The values bunch near 0.5 (no temperature), so treat them as a decision and a
                ranking, not as calibrated probabilities.
      standard  sigmoid of the positive similarity. Fine for mAP, but 0.5 means nothing here.
    """
    model.eval()
    pos, neg = text
    posneg, standard, labels = [], [], []
    for x, y, _ in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            img = model.encode_image(x.to(device, non_blocking=True))
        with torch.autocast(device.type, enabled=False):
            s_pos, s_neg = img @ pos.T, img @ neg.T
        posneg.append(torch.stack([s_pos, s_neg]).softmax(0)[0].cpu().numpy())
        standard.append(s_pos.sigmoid().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(posneg), np.concatenate(standard), np.concatenate(labels)
