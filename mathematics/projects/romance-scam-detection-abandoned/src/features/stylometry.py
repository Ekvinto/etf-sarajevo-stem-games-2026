"""Classical stylometry (phi_6, phi_7, phi_8).

phi_6 = MATTR (moving-average TTR), window W = 100.
        MATTR = (1 / (N - W + 1)) * sum_i V_i(W) / W
phi_7 = Yule's K, length-independent vocabulary-richness statistic.
        K = 1e4 * (sum_i i^2 V(i, N) - N) / N^2
        V(i, N) = number of types occurring exactly i times.
phi_8 = |s_hat - 1| where s_hat is the OLS slope of log f vs log r over
        ranks in [10, 1000]. Natural English has s ~ 1.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from src.conversation import Conversation

_W_MATTR = 100
_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def mattr(tokens: list[str], w: int = _W_MATTR) -> float:
    if len(tokens) < w:
        return len(set(tokens)) / max(1, len(tokens))
    ratios = []
    for i in range(len(tokens) - w + 1):
        window = tokens[i:i + w]
        ratios.append(len(set(window)) / w)
    return float(np.mean(ratios))


def yule_k(tokens: list[str]) -> float:
    if not tokens:
        return float("nan")
    n = len(tokens)
    freq_of_freq: Counter = Counter()
    for _, c in Counter(tokens).items():
        freq_of_freq[c] += 1
    s = sum(i * i * v for i, v in freq_of_freq.items())
    return 1e4 * (s - n) / (n * n)


def zipf_deviation(tokens: list[str]) -> float:
    # Zipf is an asymptotic property; below ~150 tokens the OLS slope fit
    # on log-rank vs log-frequency is dominated by noise. Return NaN so the
    # downstream imputer fills with the training-set median.
    if len(tokens) < 150:
        return float("nan")
    counts = sorted(Counter(tokens).values(), reverse=True)
    ranks = list(range(1, len(counts) + 1))
    lo, hi = 10, min(1000, len(counts))
    if hi <= lo:
        lo, hi = 1, len(counts)
    x = np.log(np.array(ranks[lo - 1:hi], dtype=float))
    y = np.log(np.array(counts[lo - 1:hi], dtype=float))
    if len(x) < 2:
        return float("nan")
    slope, _ = np.polyfit(x, y, 1)
    return float(abs(-slope - 1.0))


def extract(conv: Conversation) -> dict[str, float]:
    tokens = _tokenize(conv.scammer_text)
    return {
        "phi_6": mattr(tokens),
        "phi_7": yule_k(tokens),
        "phi_8": zipf_deviation(tokens),
    }


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
