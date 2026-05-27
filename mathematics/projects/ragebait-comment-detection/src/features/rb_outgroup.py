"""Stage-2 outgroup-negative association (chi_4).

Math (windowed co-occurrence form, an approximation to the PMI definition in
the report when no large reference corpus is available):

    For each outgroup token w_i in x, compute the rate of negative-affect
    tokens in a context window of radius W:
        rate_i = #{ j : |j - i| <= W, valence(w_j) < -0.3 } / (2W + 1)
    chi_4 = mean_i rate_i   (NaN if no outgroup tokens in x)

This captures "the outgroup is mentioned in negatively-charged context,"
which is the operational definition of outgroup targeting for ragebait.

A strict PMI version (uncomment in `chi_4_pmi`) requires a precomputed
P(neg|w), P(neg), P(w) table; for short comments the co-occurrence form
is more sample-efficient and harder to game.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from src.comment import Comment
from src.features.rb_affect import _load_vad

_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)
_OUTGROUP_PATH = Path("data/outgroup_lexicon.txt")

_WINDOW = 5
_NEG_THRESHOLD = -0.3


@lru_cache(maxsize=1)
def _load_outgroup_terms() -> set[str]:
    builtin = {
        # political (kept deliberately mild; avoiding slurs by design)
        "liberals", "conservatives", "leftists", "rightists",
        "democrats", "republicans", "communists", "fascists",
        "progressives", "maga", "woke", "snowflakes",
        # generational
        "boomers", "millennials", "zoomers",
        # institutional
        "media", "elites", "establishment", "globalists",
        "corporations", "billionaires", "politicians",
        # geo
        "westerners", "easterners",
        # ad-hominem mass nouns
        "sheep", "sheeple", "npcs", "shills", "bots", "trolls",
        # plural "they/them" patterns: included via explicit terms only
    }
    if _OUTGROUP_PATH.exists():
        with open(_OUTGROUP_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    builtin.add(w)
    return builtin


def _tokenize_keep_positions(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _is_negative(token: str) -> bool:
    vad = _load_vad()
    entry = vad.get(token)
    return entry is not None and entry[0] < _NEG_THRESHOLD


def chi_4(text: str) -> float:
    outgroup = _load_outgroup_terms()
    tokens = _tokenize_keep_positions(text)
    if not tokens or not outgroup:
        return float("nan")

    og_positions = [i for i, t in enumerate(tokens) if t in outgroup]
    if not og_positions:
        return float("nan")

    rates = []
    for i in og_positions:
        lo = max(0, i - _WINDOW)
        hi = min(len(tokens), i + _WINDOW + 1)
        window = tokens[lo:hi]
        if not window:
            continue
        neg_count = sum(1 for t in window if _is_negative(t))
        rates.append(neg_count / len(window))

    return sum(rates) / len(rates) if rates else float("nan")


def extract(comment: Comment) -> dict[str, float]:
    return {"chi_4": chi_4(comment.text or "")}


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    print(extract(load_comment(path)))
