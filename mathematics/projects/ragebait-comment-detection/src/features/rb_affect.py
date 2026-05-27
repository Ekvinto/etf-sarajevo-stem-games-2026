"""Stage-2 affect features (chi_1, chi_2) from a Valence-Arousal-Dominance lexicon.

Math:
    chi_1 = mean_arousal(x) * |mean_valence(x)|
            High when both intensity AND polarization are high.
    chi_2 = Pr_{w ~ x} ( valence(w) < -0.5 )
            Fraction of strongly-negative words.

The full NRC-VAD lexicon (Mohammad 2018) covers ~20,000 English words.
For this project we ship a curated subset (~300 emotion-laden words with
hand-coded VAD values) in `data/vad_subset.csv`. To use the full lexicon:
  1. Download NRC-VAD-Lexicon.txt from http://saifmohammad.com/WebPages/nrc-vad.html
  2. Place it at `data/NRC-VAD-Lexicon.txt` (tab-separated: word, valence, arousal, dominance).
  3. This module will use it automatically.
"""
from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from pathlib import Path

from src.comment import Comment

_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)

_FULL_VAD_PATH = Path("data/NRC-VAD-Lexicon.txt")
_SUBSET_VAD_PATH = Path("data/vad_subset.csv")


@lru_cache(maxsize=1)
def _load_vad() -> dict[str, tuple[float, float, float]]:
    """Return {word: (valence, arousal, dominance)}.

    Valence is centered: full NRC-VAD is [0, 1]; we map to [-1, 1] via 2v-1
    so that "negative valence" corresponds to v < 0.
    """
    vad: dict[str, tuple[float, float, float]] = {}
    if _FULL_VAD_PATH.exists():
        with open(_FULL_VAD_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                w = parts[0].lower()
                try:
                    v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                vad[w] = (2 * v - 1, a, d)
        return vad

    # Fall back to the curated subset.
    if _SUBSET_VAD_PATH.exists():
        with open(_SUBSET_VAD_PATH, "r", encoding="utf-8-sig") as f:
            # Drop comment lines ('#' as the first non-whitespace char) before
            # feeding to DictReader, otherwise it returns None-valued fields.
            lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            w = (row.get("word") or "").strip().lower()
            if not w or w.startswith("#"):
                continue
            try:
                v = float(row.get("valence"))     # already in [-1, 1]
                a = float(row.get("arousal"))     # in [0, 1]
                d_raw = row.get("dominance")
                d = float(d_raw) if d_raw not in (None, "") else 0.5
            except (ValueError, TypeError):
                continue
            vad[w] = (v, a, d)
    return vad


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def vad_stats(text: str) -> tuple[float, float, float]:
    """Return (mean_arousal, mean_valence, neg_fraction)."""
    vad = _load_vad()
    if not vad:
        return float("nan"), float("nan"), float("nan")
    tokens = _tokenize(text)
    hits = [vad[t] for t in tokens if t in vad]
    if len(hits) < 2:
        return float("nan"), float("nan"), float("nan")
    mean_v = sum(h[0] for h in hits) / len(hits)
    mean_a = sum(h[1] for h in hits) / len(hits)
    neg_frac = sum(1 for h in hits if h[0] < -0.5) / len(tokens)  # over ALL tokens
    return mean_a, mean_v, neg_frac


def chi_1(text: str) -> float:
    a, v, _ = vad_stats(text)
    if math.isnan(a) or math.isnan(v):
        return float("nan")
    return a * abs(v)


def chi_2(text: str) -> float:
    _, _, neg_frac = vad_stats(text)
    return neg_frac


def extract(comment: Comment) -> dict[str, float]:
    text = comment.text or ""
    a, v, neg_frac = vad_stats(text)
    if math.isnan(a):
        return {"chi_1": float("nan"), "chi_2": float("nan")}
    return {
        "chi_1": a * abs(v),
        "chi_2": neg_frac,
    }


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    print(extract(load_comment(path)))
