"""CVS-level metrics for ResNet18 and AdaptNet, without the router's reward. CPU only.

    python src/diag_cvs_metrics.py

mAP is computed per criterion, but a CVS needs all 3 criteria right. So the only CVS-level
number we had was the false CVS count. This script gives proper CVS numbers, per split
(val and test separately), using only outputs/prepped.csv.

Two kinds of numbers:
  Ranking (AP for CVS): how well a score puts real CVS frames above the others, whatever the
    cut-off. Scores: product of the 3 probabilities, minimum of the 3 probabilities, and for
    AdaptNet also the minimum of (a_i - a_thr_i), which respects its calibrated cut-offs.
    "random" is the CVS prevalence, the AP of a random ranker.
  At the cut-offs in use (ResNet 0.5, AdaptNet the a_thr columns): claims, true claims, false
    claims, real CVS frames, missed CVS, precision, recall and F1.

A claim means the model predicts all 3 criteria. A false claim is a claim on a frame that is
not a real CVS. Writes docs/results/next_fixes/cvs_metrics.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import common as c

OUT = c.ROOT / "docs" / "results" / "next_fixes" / "cvs_metrics.csv"


def safe_div(a, b):
    return a / b if b > 0 else np.nan


def scores_for(df, model):
    """Named CVS scores (higher = more likely CVS) for one model."""
    p = df[c.R_COLS if model == "ResNet18" else c.A_COLS].to_numpy()
    out = {"AP product": p.prod(axis=1), "AP min": p.min(axis=1)}
    if model == "AdaptNet":
        thr = df[["a_thr1", "a_thr2", "a_thr3"]].to_numpy()
        out["AP min margin"] = (p - thr).min(axis=1)  # margin over each criterion's cut-off
    return out


def claims_for(df, model):
    """0/1 CVS claim at the cut-offs in use."""
    cols = ["ry1", "ry2", "ry3"] if model == "ResNet18" else ["ay1", "ay2", "ay3"]
    return df[cols].to_numpy().all(axis=1)


def metrics_row(df, split, model):
    truth = c.is_cvs(df[c.Y_COLS].to_numpy())
    claim = claims_for(df, model)
    true_claims = int((claim & truth).sum())
    n_claims, n_cvs = int(claim.sum()), int(truth.sum())
    precision, recall = safe_div(true_claims, n_claims), safe_div(true_claims, n_cvs)
    f1 = safe_div(2 * precision * recall, precision + recall) if n_claims and true_claims else np.nan
    row = {
        "split": split, "model": model, "frames": len(df), "random": truth.mean(),
        "claims": n_claims, "true_claims": true_claims, "false_claims": n_claims - true_claims,
        "real_cvs": n_cvs, "missed_cvs": n_cvs - true_claims,
        "precision": precision, "recall": recall, "f1": f1,
        "AP product": np.nan, "AP min": np.nan, "AP min margin": np.nan,
    }
    for name, s in scores_for(df, model).items():
        row[name] = average_precision_score(truth, s)
    return row


def main():
    df = pd.read_csv(c.OUTPUTS_DIR / "prepped.csv")
    rows = [metrics_row(df[df["split"] == s], s, m)
            for s in ("val", "test") for m in ("ResNet18", "AdaptNet")]
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {OUT}")
    print("Note: AdaptNet val cut-offs are out-of-fold, test cut-offs use all of val.")


if __name__ == "__main__":
    main()
