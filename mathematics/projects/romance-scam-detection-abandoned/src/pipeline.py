"""Feature pipeline: Conversation -> 19-dim phi vector.

Each extractor is called independently. If one fails (NaN or exception),
the corresponding feature(s) are filled with NaN; the trainer / detector
imputes with the training-set median.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np

from src.conversation import Conversation
from src.features import (
    asymmetry,
    detect_gpt,
    hmm_stages,
    perplexity,
    semantic,
    sentiment,
    stylometry,
    token_rank,
    topic_shift,
)

# Order is important: the trained model's feature vector follows this order.
FEATURE_NAMES = [
    "phi_1", "phi_2",                     # perplexity
    "phi_3",                              # detect_gpt
    "phi_4", "phi_5",                     # token_rank
    "phi_6", "phi_7", "phi_8",            # stylometry
    "phi_9", "phi_10",                    # asymmetry
    "phi_11", "phi_12",                   # sentiment
    "phi_13", "phi_14",                   # topic_shift
    "phi_15", "phi_16",                   # hmm_stages
    "phi_17", "phi_18", "phi_19",         # semantic
]

_EXTRACTORS: list[Callable[[Conversation], dict[str, float]]] = [
    perplexity.extract,
    detect_gpt.extract,
    token_rank.extract,
    stylometry.extract,
    asymmetry.extract,
    sentiment.extract,
    topic_shift.extract,
    hmm_stages.extract,
    semantic.extract,
]


def extract_features(conv: Conversation, skip: set[str] | None = None) -> dict[str, float]:
    """Run every extractor; on error fill its features with NaN. Heavy modules
    can be skipped by passing names like {"detect_gpt", "hmm_stages"}."""
    skip = skip or set()
    feats: dict[str, float] = {name: float("nan") for name in FEATURE_NAMES}
    for extractor in _EXTRACTORS:
        mod_name = extractor.__module__.split(".")[-1]
        if mod_name in skip:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = extractor(conv)
            for k, v in out.items():
                feats[k] = float(v) if v is not None else float("nan")
        except Exception as e:  # noqa: BLE001
            print(f"[pipeline] {mod_name} failed: {e}")
    return feats


def feature_vector(conv: Conversation, skip: set[str] | None = None) -> np.ndarray:
    feats = extract_features(conv, skip=skip)
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)


def explain_red_flags(feats: dict[str, float]) -> list[str]:
    """Generate human-readable red flags from feature values.

    Thresholds are conservative; adjust after looking at training distributions.
    """
    flags = []
    if not math.isnan(feats.get("phi_17", float("nan"))) and feats["phi_17"] > 0.85:
        flags.append(f"Semantic match to known scam template (max cos = {feats['phi_17']:.2f})")
    if not math.isnan(feats.get("phi_16", float("nan"))) and feats["phi_16"] > 0.5:
        flags.append("HMM Viterbi path enters Urgency stage")
    if not math.isnan(feats.get("phi_14", float("nan"))) and feats["phi_14"] > 0.10:
        flags.append(f"Late-conversation financial lexicon mass = {feats['phi_14']:.2f}")
    if not math.isnan(feats.get("phi_13", float("nan"))) and feats["phi_13"] > 1.0:
        flags.append(f"Topic-shift KL divergence = {feats['phi_13']:.2f}")
    if not math.isnan(feats.get("phi_3", float("nan"))) and feats["phi_3"] > 2.0:
        flags.append(f"DetectGPT curvature z-score = {feats['phi_3']:.2f}")
    if not math.isnan(feats.get("phi_1", float("nan"))) and feats["phi_1"] < 15:
        flags.append(f"Very low perplexity ({feats['phi_1']:.1f})")
    if not math.isnan(feats.get("phi_11", float("nan"))) and feats["phi_11"] > 2.0:
        flags.append(f"CUSUM sentiment-shift statistic = {feats['phi_11']:.2f}")
    if not math.isnan(feats.get("phi_9", float("nan"))) and feats["phi_9"] > 1.0:
        flags.append("Scammer messages substantially longer than victim's")
    return flags


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    conv = load_conversation(path)
    feats = extract_features(conv)
    for k in FEATURE_NAMES:
        print(f"  {k:>7s} = {feats[k]}")
    print("\nRed flags:")
    for f in explain_red_flags(feats):
        print(f"  - {f}")
