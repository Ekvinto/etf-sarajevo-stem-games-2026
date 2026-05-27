"""Train two logistic-regression classifiers on top of the feature vectors.

Pipeline:
    1. Fit AI-perplexity length-bucketed baseline on the benign subset.
    2. Fit punctuation-baseline on the benign subset.
    3. Fit topic-residual OLS on the benign subset.
    4. Extract features for all comments.
    5. Fit two LRs (AI and ragebait) with 5-fold internal CV.
    6. Calibrate both with Platt scaling (sigmoid).

Usage:
    python -m src.train --corpus data/corpus.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.comment import Comment, load_corpus
from src.features import ai_likelihood, ai_lexical, rb_topic_resid
from src.pipeline import (
    AI_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    RB_FEATURE_NAMES,
    extract_features,
)


def _fit_baselines(benign: list[Comment]) -> None:
    """Fit length-bucketed perplexity baseline, punctuation baseline,
    and topic-residual OLS on benign data."""
    print("Fitting perplexity baseline (length-bucketed)...")
    ai_likelihood.fit_baseline([c.text for c in benign])
    print("Fitting punctuation baseline...")
    ai_lexical.fit_punct_baseline([c.text for c in benign])
    print("Fitting topic-residual OLS...")
    rb_topic_resid.fit_topic_residual(benign)


def _build_dataset(corpus: list[Comment],
                   skip: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for every comment. Returns (X, y_ai, y_rb)."""
    X = np.zeros((len(corpus), len(ALL_FEATURE_NAMES)), dtype=float)
    y_ai = np.full(len(corpus), -1, dtype=int)
    y_rb = np.full(len(corpus), -1, dtype=int)
    print(f"Extracting features for {len(corpus)} comments...")
    for i, c in enumerate(tqdm(corpus)):
        feats = extract_features(c, skip=skip)
        for j, name in enumerate(ALL_FEATURE_NAMES):
            X[i, j] = feats[name]
        if c.label_ai is not None:
            y_ai[i] = int(c.label_ai)
        if c.label_ragebait is not None:
            y_rb[i] = int(c.label_ragebait)
    return X, y_ai, y_rb


def _build_classifier() -> Pipeline:
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegressionCV(
            Cs=10, cv=5, scoring="roc_auc", max_iter=2000, n_jobs=-1,
        )),
    ])
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def _slice_features(X: np.ndarray, names: list[str]) -> np.ndarray:
    indices = [ALL_FEATURE_NAMES.index(n) for n in names]
    return X[:, indices]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path,
                    help="JSONL with both label_ai and label_ragebait set.")
    ap.add_argument("--ai-only", type=Path, default=None,
                    help="Optional additional JSONL with only label_ai (e.g. HC3).")
    ap.add_argument("--rb-only", type=Path, default=None,
                    help="Optional additional JSONL with only label_ragebait (e.g. civil_comments).")
    ap.add_argument("--out", default=Path("models/classifiers.joblib"), type=Path)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Feature modules to skip (e.g. ai_likelihood for fast iteration)")
    ap.add_argument("--no-baselines", action="store_true",
                    help="Skip refitting baselines; assume they already exist.")
    args = ap.parse_args()

    print(f"Loading {args.corpus}...")
    corpus = load_corpus(args.corpus)
    if args.ai_only and args.ai_only.exists():
        print(f"Loading AI-only labels from {args.ai_only}...")
        for c in load_corpus(args.ai_only):
            c.label_ragebait = None  # ensure not used for the RB classifier
            corpus.append(c)
    if args.rb_only and args.rb_only.exists():
        print(f"Loading RB-only labels from {args.rb_only}...")
        for c in load_corpus(args.rb_only):
            c.label_ai = None
            corpus.append(c)
    print(f"Total corpus size: {len(corpus)}")

    if not args.no_baselines:
        benign = [c for c in corpus if c.label_ai == 0 and c.label_ragebait == 0]
        if len(benign) < 30:
            print(f"Warning: only {len(benign)} benign-benign comments; baselines will be weak.")
            benign = [c for c in corpus if c.label_ai == 0 or c.label_ragebait == 0]
        _fit_baselines(benign)

    X, y_ai, y_rb = _build_dataset(corpus, skip=set(args.skip))

    print(f"\nFeature matrix shape: {X.shape}")
    print("NaN fraction per feature:")
    for j, name in enumerate(ALL_FEATURE_NAMES):
        print(f"  {name:>7s} : {np.isnan(X[:, j]).mean():.2%}")

    # ----- AI classifier -----
    mask_ai = y_ai >= 0
    print(f"\nFitting AI classifier on {mask_ai.sum()} labeled examples"
          f" ({(y_ai[mask_ai] == 1).sum()} positive)...")
    X_ai = _slice_features(X[mask_ai], AI_FEATURE_NAMES)
    clf_ai = _build_classifier()
    clf_ai.fit(X_ai, y_ai[mask_ai])

    # ----- Ragebait classifier -----
    mask_rb = y_rb >= 0
    print(f"Fitting Ragebait classifier on {mask_rb.sum()} labeled examples"
          f" ({(y_rb[mask_rb] == 1).sum()} positive)...")
    X_rb = _slice_features(X[mask_rb], RB_FEATURE_NAMES)
    clf_rb = _build_classifier()
    clf_rb.fit(X_rb, y_rb[mask_rb])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "clf_ai": clf_ai,
        "clf_rb": clf_rb,
        "ai_feature_names": AI_FEATURE_NAMES,
        "rb_feature_names": RB_FEATURE_NAMES,
    }, args.out)
    print(f"\nSaved classifiers to {args.out}")


if __name__ == "__main__":
    main()
