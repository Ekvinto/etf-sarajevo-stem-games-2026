"""Account-level Bayesian aggregation.

Given multiple per-comment scores produced by `src.detect` for one username,
compute the posterior probability P(Z_u = 1 | x^(1:n)) that the *account*
is a ragebait bot. Math is in §3.4.2 of the LaTeX report; in short, the
log-odds update is linear in the sum of per-comment scores:

    logit P(Z_u = 1 | x^(1:n))
        = logit pi_0
        + (sum_i p_AR(x^(i))) * log( eta_+(1 - eta_-) / (eta_-(1 - eta_+)) )

with pi_0 = platform base rate of bot accounts (default 0.05),
     eta_+ = bot's per-comment positive rate (default 0.8),
     eta_- = human's per-comment positive rate (default 0.05).

Usage:
    python -m src.aggregate --input data/user_history.jsonl
    # where each line is a comment with the same `username` field
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from src.comment import load_corpus
from src.pipeline import AI_FEATURE_NAMES, RB_FEATURE_NAMES, extract_features

# Defaults; tune from data if/when available.
_PI_0 = 0.05
_ETA_PLUS = 0.80
_ETA_MINUS = 0.05


def _logit(p: float) -> float:
    p = max(min(p, 1 - 1e-9), 1e-9)
    return math.log(p / (1 - p))


def account_posterior(scores: list[float],
                      pi_0: float = _PI_0,
                      eta_plus: float = _ETA_PLUS,
                      eta_minus: float = _ETA_MINUS) -> float:
    """Update P(Z_u = 1) given a list of per-comment p_AR scores."""
    log_ratio = math.log(
        (eta_plus * (1 - eta_minus)) / (eta_minus * (1 - eta_plus))
    )
    log_odds = _logit(pi_0) + sum(scores) * log_ratio
    return 1.0 / (1.0 + math.exp(-log_odds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="JSONL of comments; each line must include `username`.")
    ap.add_argument("--model", default=Path("models/classifiers.joblib"), type=Path)
    ap.add_argument("--pi-0", type=float, default=_PI_0,
                    help="Prior bot rate on the platform.")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"No trained model at {args.model}. Run `python -m src.train` first.")
    bundle = joblib.load(args.model)
    clf_ai, clf_rb = bundle["clf_ai"], bundle["clf_rb"]

    comments = load_corpus(args.input)
    by_user: dict[str, list[float]] = defaultdict(list)
    print(f"Scoring {len(comments)} comments...")
    for c in comments:
        if not c.username:
            continue
        feats = extract_features(c)
        x_ai = np.array([[feats[n] for n in AI_FEATURE_NAMES]], dtype=float)
        x_rb = np.array([[feats[n] for n in RB_FEATURE_NAMES]], dtype=float)
        p_ai = float(clf_ai.predict_proba(x_ai)[0, 1])
        p_rb = float(clf_rb.predict_proba(x_rb)[0, 1])
        by_user[c.username].append(p_ai * p_rb)

    print(f"\n{'Username':<24s}  {'#comments':>9s}  {'P(bot)':>8s}  {'mean p_AR':>10s}")
    print("-" * 60)
    for user, scores in sorted(by_user.items(), key=lambda kv: -account_posterior(kv[1], args.pi_0)):
        post = account_posterior(scores, args.pi_0)
        print(f"{user[:24]:<24s}  {len(scores):>9d}  {post:>8.3f}  {sum(scores)/len(scores):>10.3f}")


if __name__ == "__main__":
    main()
