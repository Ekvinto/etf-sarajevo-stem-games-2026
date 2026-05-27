"""Train the logistic-regression classifier on top of the 19-D feature vector.

Usage:
    python -m src.train --scam data/scam_corpus.jsonl --benign data/benign_corpus.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.conversation import Conversation, load_corpus
from src.features import hmm_stages
from src.pipeline import FEATURE_NAMES, feature_vector


def _build_dataset(scam_path: Path, benign_path: Path,
                   skip: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[Conversation]]:
    scam = load_corpus(scam_path)
    for c in scam:
        c.label = 1
    benign = load_corpus(benign_path)
    for c in benign:
        c.label = 0
    convs = scam + benign

    X = []
    y = []
    print(f"Extracting features for {len(convs)} conversations...")
    for c in tqdm(convs):
        X.append(feature_vector(c, skip=skip))
        y.append(c.label)
    return np.vstack(X), np.array(y), convs


def _fit_hmms(scam_convs: list[Conversation], benign_convs: list[Conversation]) -> None:
    """Train both HMMs needed by features/hmm_stages.py."""
    print("Fitting scam HMM...")
    hmm_stages.fit_hmm(scam_convs, hmm_stages._SCAM_MODEL_PATH)
    print("Fitting normal HMM...")
    hmm_stages.fit_hmm(benign_convs, hmm_stages._NORMAL_MODEL_PATH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scam", required=True, type=Path)
    ap.add_argument("--benign", required=True, type=Path)
    ap.add_argument("--out", default=Path("models/classifier.joblib"), type=Path)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Feature modules to skip (e.g. detect_gpt for fast iteration)")
    ap.add_argument("--no-hmm", action="store_true", help="Skip HMM training")
    args = ap.parse_args()

    # Train HMMs first so features/hmm_stages.extract returns real numbers
    if not args.no_hmm:
        scam_convs = load_corpus(args.scam)
        benign_convs = load_corpus(args.benign)
        _fit_hmms(scam_convs, benign_convs)

    X, y, _ = _build_dataset(args.scam, args.benign, skip=set(args.skip))

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"NaN fraction per feature:")
    for i, name in enumerate(FEATURE_NAMES):
        nan_frac = np.isnan(X[:, i]).mean()
        print(f"  {name:>7s} : {nan_frac:.2%}")

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegressionCV(
            Cs=10, cv=5, scoring="roc_auc", max_iter=2000, n_jobs=-1,
        )),
    ])
    print("\nFitting classifier (5-fold CV)...")
    clf.fit(X, y)

    train_auc = clf.score(X, y)
    print(f"Train ROC-AUC: {train_auc:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "feature_names": FEATURE_NAMES}, args.out)
    print(f"Saved classifier to {args.out}")


if __name__ == "__main__":
    main()
