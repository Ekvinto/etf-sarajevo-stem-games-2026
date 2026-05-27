"""Feature pipeline: Comment -> (psi vector, chi vector).

Each extractor is called independently. On failure or NaN, the feature is
filled with NaN and the trainer / detector imputes with the training-set
median.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np

from src.comment import Comment
from src.features import (
    ai_likelihood,
    ai_lexical,
    rb_affect,
    rb_info_affect,
    rb_moral,
    rb_neutralize,
    rb_outgroup,
    rb_rhetoric,
    rb_semantic,
    rb_topic_resid,
)

# Order matters: trained models save these as the feature ordering.
AI_FEATURE_NAMES = [
    "psi_1", "psi_2",          # ai_likelihood (perplexity, burstiness)
    "psi_3",                   # ai_likelihood (DetectGPT)
    "psi_4", "psi_5",          # ai_likelihood (GLTR ranks)
    "psi_6",                   # ai_lexical   (AI phrase fingerprint)
    "psi_7",                   # ai_lexical   (punctuation regularity)
    "psi_8",                   # ai_lexical   (hedging)
]

RB_FEATURE_NAMES = [
    "chi_1", "chi_2",          # rb_affect    (VAD)
    "chi_3",                   # rb_moral     (MFD vice)
    "chi_4",                   # rb_outgroup  (windowed outgroup-NEG)
    "chi_5",                   # rb_rhetoric  (rhetorical patterns)
    "chi_6",                   # rb_info_affect (info-to-affect ratio)
    "chi_7",                   # rb_neutralize (counterfactual gap)
    "chi_8",                   # rb_topic_resid (topic-conditional residual)
    "chi_9", "chi_10",         # rb_semantic  (template similarity)
]

ALL_FEATURE_NAMES = AI_FEATURE_NAMES + RB_FEATURE_NAMES

_AI_EXTRACTORS: list[Callable[[Comment], dict[str, float]]] = [
    ai_likelihood.extract,
    ai_lexical.extract,
]

_RB_EXTRACTORS: list[Callable[[Comment], dict[str, float]]] = [
    rb_affect.extract,
    rb_moral.extract,
    rb_outgroup.extract,
    rb_rhetoric.extract,
    rb_info_affect.extract,
    rb_neutralize.extract,
    rb_topic_resid.extract,
    rb_semantic.extract,
]


def extract_features(comment: Comment,
                     skip: set[str] | None = None) -> dict[str, float]:
    """Run every extractor; on error fill its features with NaN.

    Pass `skip={"ai_likelihood", "rb_neutralize"}` etc. to skip heavy modules
    during fast iteration.
    """
    skip = skip or set()
    feats: dict[str, float] = {name: float("nan") for name in ALL_FEATURE_NAMES}
    for extractor in _AI_EXTRACTORS + _RB_EXTRACTORS:
        mod_name = extractor.__module__.split(".")[-1]
        if mod_name in skip:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = extractor(comment)
            for k, v in out.items():
                feats[k] = float(v) if v is not None else float("nan")
        except Exception as e:  # noqa: BLE001
            print(f"[pipeline] {mod_name} failed: {e}")
    return feats


def feature_vectors(comment: Comment,
                    skip: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return (psi_vec, chi_vec, full_feats_dict)."""
    feats = extract_features(comment, skip=skip)
    psi = np.array([feats[n] for n in AI_FEATURE_NAMES], dtype=float)
    chi = np.array([feats[n] for n in RB_FEATURE_NAMES], dtype=float)
    return psi, chi, feats


def explain_red_flags(feats: dict[str, float]) -> list[str]:
    """Generate human-readable red flags. Conservative thresholds."""
    flags = []

    # ----- AI side -----
    if not math.isnan(feats.get("psi_1", float("nan"))) and feats["psi_1"] < -1.0:
        flags.append(f"Low length-normalized perplexity z-score (psi_1 = {feats['psi_1']:+.2f})")
    if not math.isnan(feats.get("psi_3", float("nan"))) and feats["psi_3"] > 2.0:
        flags.append(f"DetectGPT curvature z-score = {feats['psi_3']:.2f}")
    if not math.isnan(feats.get("psi_4", float("nan"))) and feats["psi_4"] > 0.55:
        flags.append(f"High concentration of top-10 tokens (psi_4 = {feats['psi_4']:.2f})")
    if not math.isnan(feats.get("psi_6", float("nan"))) and feats["psi_6"] > 0.4:
        flags.append(f"LLM lexical fingerprint hit (psi_6 = {feats['psi_6']:.2f})")
    if not math.isnan(feats.get("psi_7", float("nan"))) and feats["psi_7"] > 2.0:
        flags.append(f"Punctuation profile anomalous (psi_7 = {feats['psi_7']:.2f})")
    if not math.isnan(feats.get("psi_8", float("nan"))) and feats["psi_8"] > 1.5:
        flags.append(f"High hedging density (psi_8 = {feats['psi_8']:.2f}/100 tokens)")

    # ----- Ragebait side -----
    if not math.isnan(feats.get("chi_3", float("nan"))) and feats["chi_3"] > 0.10:
        flags.append(f"High moral-vice density (chi_3 = {feats['chi_3']:.3f})")
    if not math.isnan(feats.get("chi_4", float("nan"))) and feats["chi_4"] > 0.20:
        flags.append(f"Strong outgroup-NEG association (chi_4 = {feats['chi_4']:.2f})")
    if not math.isnan(feats.get("chi_5", float("nan"))) and feats["chi_5"] > 1.0:
        flags.append(f"High rhetorical-pattern score (chi_5 = {feats['chi_5']:.2f})")
    if not math.isnan(feats.get("chi_6", float("nan"))) and feats["chi_6"] > 1.5:
        flags.append(f"High affect-to-info ratio (chi_6 = {feats['chi_6']:.2f})")
    if not math.isnan(feats.get("chi_7", float("nan"))) and feats["chi_7"] > 0.4:
        flags.append(f"Neutralization gap large (chi_7 = {feats['chi_7']:.2f}); affect carries the content")
    if not math.isnan(feats.get("chi_8", float("nan"))) and feats["chi_8"] > 0.15:
        flags.append(f"Emotion exceeds topic baseline (chi_8 = {feats['chi_8']:+.2f})")
    if not math.isnan(feats.get("chi_9", float("nan"))) and feats["chi_9"] > 0.75:
        flags.append(f"Semantic match to ragebait template (chi_9 = {feats['chi_9']:.2f})")

    return flags


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    c = load_comment(path)
    feats = extract_features(c)
    print(f"\nFeatures for {path}:")
    print("\nAI features:")
    for k in AI_FEATURE_NAMES:
        print(f"  {k:>7s} = {feats[k]}")
    print("\nRagebait features:")
    for k in RB_FEATURE_NAMES:
        print(f"  {k:>7s} = {feats[k]}")
    print("\nRed flags:")
    for f in explain_red_flags(feats):
        print(f"  - {f}")
