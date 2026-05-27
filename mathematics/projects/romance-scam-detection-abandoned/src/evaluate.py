"""Cross-validated evaluation and ablation analysis.

Produces:
    images/roc.png
    images/pr.png
    images/calibration.png
    images/ablation.png  + ablation.csv

Usage:
    python -m src.evaluate --scam data/scam_corpus.jsonl --benign data/benign_corpus.jsonl
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.conversation import load_corpus
from src.pipeline import FEATURE_NAMES, feature_vector

# Feature-family ablation groups
FAMILIES = {
    "LLM":     ["phi_1", "phi_2", "phi_3", "phi_4", "phi_5"],
    "Style":   ["phi_6", "phi_7", "phi_8"],
    "Dyn":     ["phi_9", "phi_10", "phi_11", "phi_12", "phi_13", "phi_14"],
    "HMM":     ["phi_15", "phi_16"],
    "Sem":     ["phi_17", "phi_18", "phi_19"],
}


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc",
                                        max_iter=2000, n_jobs=-1)),
    ])


def _cv_probs(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    probs = np.zeros_like(y, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        pipe = _build_pipeline()
        pipe.fit(X[tr], y[tr])
        probs[te] = pipe.predict_proba(X[te])[:, 1]
    return probs


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                                n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(conf - acc)
    return float(ece)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scam", required=True, type=Path)
    ap.add_argument("--benign", required=True, type=Path)
    ap.add_argument("--out", default=Path("images"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scam = load_corpus(args.scam)
    for c in scam:
        c.label = 1
    benign = load_corpus(args.benign)
    for c in benign:
        c.label = 0
    convs = scam + benign

    X = []
    y = []
    print(f"Extracting features for {len(convs)} conversations...")
    for c in tqdm(convs):
        X.append(feature_vector(c))
        y.append(c.label)
    X = np.vstack(X)
    y = np.array(y)

    # Full-model CV probabilities
    print("Cross-validating full model...")
    probs = _cv_probs(X, y)

    auc_full = roc_auc_score(y, probs)
    ap_full = average_precision_score(y, probs)
    ece_full = _expected_calibration_error(y, probs)
    print(f"Full model:  AUC = {auc_full:.4f}  AP = {ap_full:.4f}  ECE = {ece_full:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Full model (AUC={auc_full:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "roc.png", dpi=140)
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, label=f"AP={ap_full:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "pr.png", dpi=140)
    plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, "o-", label="Empirical")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(f"Calibration (ECE={ece_full:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "calibration.png", dpi=140)
    plt.close()

    # Ablation: drop each family in turn
    print("\nFamily ablation:")
    rows = [("Family removed", "AUC", "delta_AUC")]
    rows.append(("(none)", f"{auc_full:.4f}", "0.0000"))
    family_aucs = {}
    for fam, names in FAMILIES.items():
        idxs = [FEATURE_NAMES.index(n) for n in names if n in FEATURE_NAMES]
        keep = [i for i in range(X.shape[1]) if i not in idxs]
        X_ab = X[:, keep]
        probs_ab = _cv_probs(X_ab, y)
        auc_ab = roc_auc_score(y, probs_ab)
        family_aucs[fam] = auc_ab
        rows.append((fam, f"{auc_ab:.4f}", f"{auc_ab - auc_full:+.4f}"))
        print(f"  -{fam:<5s}  AUC={auc_ab:.4f}  delta={auc_ab - auc_full:+.4f}")

    with open(args.out / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Ablation bar plot
    fam_names = list(family_aucs.keys())
    deltas = [family_aucs[f] - auc_full for f in fam_names]
    plt.figure(figsize=(6, 4))
    plt.bar(fam_names, deltas, color="#5DCAA5")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.ylabel("Delta AUC vs full model")
    plt.title("Feature-family ablation")
    plt.tight_layout()
    plt.savefig(args.out / "ablation.png", dpi=140)
    plt.close()

    print(f"\nWrote: roc.png, pr.png, calibration.png, ablation.png, ablation.csv -> {args.out}")


if __name__ == "__main__":
    main()
