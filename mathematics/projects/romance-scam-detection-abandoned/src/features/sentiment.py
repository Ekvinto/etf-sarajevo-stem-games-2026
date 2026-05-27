"""Sentiment-trajectory features (phi_11, phi_12).

Each scammer message s_t is scored in [-1, +1] by a multilingual transformer.

phi_11 = max_t G_t  with  G_t = max(0, G_{t-1} + (mu_0 - s_t - kappa))
         A one-sided CUSUM against the early-conversation baseline mu_0.
         Detects the negative deflection at the urgency / financial-ask stage.

phi_12 = OLS slope of s_t vs t.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import pipeline

from src.conversation import Conversation

_KAPPA = 0.25


@lru_cache(maxsize=1)
def _load_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=device,
        truncation=True,
        max_length=256,
        use_fast=False,   # SentencePiece slow tokenizer; avoids protobuf conversion path
    )


_LABEL_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def sentiment_scores(texts: list[str]) -> list[float]:
    if not texts:
        return []
    clf = _load_pipeline()
    out = clf(texts, batch_size=8)
    scores = []
    for o in out:
        sign = _LABEL_MAP.get(o["label"].lower(), 0.0)
        scores.append(sign * float(o["score"]))
    return scores


def cusum_max(scores: list[float]) -> float:
    if len(scores) < 4:
        return float("nan")
    half = max(2, len(scores) // 2)
    mu_0 = float(np.median(scores[:half]))
    g, g_max = 0.0, 0.0
    for s in scores:
        g = max(0.0, g + (mu_0 - s - _KAPPA))
        g_max = max(g_max, g)
    return g_max


def ols_slope(scores: list[float]) -> float:
    if len(scores) < 3:
        return float("nan")
    x = np.arange(len(scores), dtype=float)
    y = np.array(scores, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def extract(conv: Conversation) -> dict[str, float]:
    texts = [m.text for m in conv.scammer_messages]
    if len(texts) < 4:
        return {"phi_11": float("nan"), "phi_12": float("nan")}
    scores = sentiment_scores(texts)
    return {
        "phi_11": cusum_max(scores),
        "phi_12": ols_slope(scores),
    }


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
