"""Conversational asymmetry (phi_9) and timing bimodality (phi_10).

phi_9  = log( (mean_len_S + 1) / (mean_len_V + 1) )
         Scam bots send longer, more polished messages than emotionally
         invested human victims.

phi_10 = bimodality coefficient of S's inter-message intervals,
         BC = (skew^2 + 1) / (kurt + 3*(n-1)^2 / ((n-2)(n-3)))
         Operators running multiple chats yield bimodal delays.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from src.conversation import Conversation


def _token_count(text: str) -> int:
    return len(text.split())


def length_asymmetry(conv: Conversation) -> float:
    s_lens = [_token_count(m.text) for m in conv.scammer_messages]
    v_lens = [_token_count(m.text) for m in conv.victim_messages]
    if not s_lens or not v_lens:
        return float("nan")
    return math.log((np.mean(s_lens) + 1) / (np.mean(v_lens) + 1))


def timing_bimodality(conv: Conversation) -> float:
    s_msgs = [m for m in conv.scammer_messages if m.timestamp is not None]
    if len(s_msgs) < 4:
        return float("nan")
    s_msgs.sort(key=lambda m: m.timestamp)
    deltas = [
        (s_msgs[i].timestamp - s_msgs[i - 1].timestamp).total_seconds()
        for i in range(1, len(s_msgs))
    ]
    deltas = [d for d in deltas if d > 0]
    n = len(deltas)
    if n < 4:
        return float("nan")
    skew = float(stats.skew(deltas, bias=False))
    kurt = float(stats.kurtosis(deltas, bias=False))
    denom = kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return (skew ** 2 + 1) / denom


def extract(conv: Conversation) -> dict[str, float]:
    return {
        "phi_9": length_asymmetry(conv),
        "phi_10": timing_bimodality(conv),
    }


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
