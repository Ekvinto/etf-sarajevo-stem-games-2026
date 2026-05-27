"""Cross-validated evaluation and ablation analysis.

Produces:
    images/roc_ai.png
    images/roc_rb.png
    images/pr_ai.png
    images/pr_rb.png
    images/calibration_ai.png
    images/calibration_rb.png
    images/ablation_ai.png    + ablation_ai.csv
    images/ablation_rb.png    + ablation_rb.csv

Usage:
    python -m src.evaluate --corpus data/corpus.jsonl
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
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.comment import load_corpus
from src.pipeline import (
    AI_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    RB_FEATURE_NAMES,
    extract_features,
)

# Feature-family ablation groups -- AI side
AI_FAMILIES = {
    "LLM-likelihood":   ["psi_1", "psi_2", "psi_3", "psi_4", "psi_5"],
    "Lexical":          ["psi_6"],
    "Punctuation":      ["psi_7"],
    "Hedging":          ["psi_8"],
}

# Feature-family ablation groups -- ragebait side
RB_FAMILIES = {
    "Affect":           ["chi_1", "chi_2"],
    "Moral":            ["chi_3"],
    "Outgroup":         ["chi_4"],
    "Rhetoric":         ["chi_5"],
    "Info-affect":      ["chi_6"],
    "Neutralization":   ["chi_7"],
    "Topic-residual":   ["chi_8"],
    "Semantic":         ["chi_9", "chi_10"],
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


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(p)) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _evaluate(X: np.ndarray, y: np.ndarray, feature_names: list[str],
              families: dict[str, list[str]], name: str,
              out_dir: Path) -> dict:
    """Run cross-validation, ablation, and produce all four plots for a stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} stage ===")

    # Full-model probabilities
    probs_full = _cv_probs(X, y)
    auc_full = roc_auc_score(y, probs_full)
    ap_full = average_precision_score(y, probs_full)
    ece_full = _expected_calibration_error(y, probs_full)
    print(f"Full model: AUC={auc_full:.4f}, AP={ap_full:.4f}, ECE={ece_full:.4f}")

    # ROC
    fpr, tpr, _ = roc_curve(y, probs_full)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Full (AUC={auc_full:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{name} stage ROC")
    plt.legend()
    plt.savefig(out_dir / f"roc_{name.lower()}.png", dpi=130, bbox_inches="tight")
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, probs_full)
    plt.figure()
    plt.plot(rec, prec, label=f"Full (AP={ap_full:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} stage PR")
    plt.legend()
    plt.savefig(out_dir / f"pr_{name.lower()}.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y, probs_full, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.plot(mean_pred, frac_pos, marker="o", label=f"Full (ECE={ece_full:.3f})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(f"{name} stage reliability")
    plt.legend()
    plt.savefig(out_dir / f"calibration_{name.lower()}.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Ablation
    rows = [{"config": "Full", "auc": auc_full, "ap": ap_full, "ece": ece_full}]
    print("Ablation:")
    print(f"  {'-':<20s}  AUC={auc_full:.4f}  AP={ap_full:.4f}")
    for fam, members in families.items():
        keep_idx = [i for i, n in enumerate(feature_names) if n not in members]
        if not keep_idx:
            continue
        X_drop = X[:, keep_idx]
        probs = _cv_probs(X_drop, y)
        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        ece = _expected_calibration_error(y, probs)
        rows.append({"config": f"-{fam}", "auc": auc, "ap": ap, "ece": ece})
        print(f"  -{fam:<19s}  AUC={auc:.4f}  AP={ap:.4f}  (delta {auc - auc_full:+.4f})")

    plt.figure(figsize=(8, 4))
    names = [r["config"] for r in rows]
    aucs = [r["auc"] for r in rows]
    plt.bar(names, aucs)
    plt.axhline(auc_full, color="gray", linestyle="--", alpha=0.5)
    plt.ylabel("ROC AUC")
    plt.title(f"{name} stage feature-family ablation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"ablation_{name.lower()}.png", dpi=130, bbox_inches="tight")
    plt.close()

    with open(out_dir / f"ablation_{name.lower()}.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "auc", "ap", "ece"])
        writer.writeheader()
        writer.writerows(rows)

    return {"auc": auc_full, "ap": ap_full, "ece": ece_full}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--images", default=Path("images"), type=Path)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    print(f"Loading {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} comments")

    X = np.zeros((len(corpus), len(ALL_FEATURE_NAMES)), dtype=float)
    y_ai = np.full(len(corpus), -1, dtype=int)
    y_rb = np.full(len(corpus), -1, dtype=int)
    print("Extracting features...")
    for i, c in enumerate(tqdm(corpus)):
        feats = extract_features(c, skip=set(args.skip))
        for j, n in enumerate(ALL_FEATURE_NAMES):
            X[i, j] = feats[n]
        if c.label_ai is not None:
            y_ai[i] = int(c.label_ai)
        if c.label_ragebait is not None:
            y_rb[i] = int(c.label_ragebait)

    mask_ai = y_ai >= 0
    if mask_ai.sum() < 20:
        print("Too few AI-labeled examples; skipping AI evaluation.")
    else:
        X_ai = X[mask_ai][:, [ALL_FEATURE_NAMES.index(n) for n in AI_FEATURE_NAMES]]
        _evaluate(X_ai, y_ai[mask_ai], AI_FEATURE_NAMES, AI_FAMILIES, "AI", args.images)

    mask_rb = y_rb >= 0
    if mask_rb.sum() < 20:
        print("Too few RB-labeled examples; skipping ragebait evaluation.")
    else:
        X_rb = X[mask_rb][:, [ALL_FEATURE_NAMES.index(n) for n in RB_FEATURE_NAMES]]
        _evaluate(X_rb, y_rb[mask_rb], RB_FEATURE_NAMES, RB_FAMILIES, "RB", args.images)


if __name__ == "__main__":
    main()
