"""Stage-1 lexical/punctuation features (psi_6, psi_7, psi_8).

psi_6 = log( P(x has AI-phrase) / P(x doesn't) ) using a curated AI lexicon.
        Implemented as a length-normalized hit rate against a phrase set.
        Phrases (and their plural/contracted forms) are matched as substrings
        with word boundaries.
psi_7 = punctuation regularity = sum over patterns of |hat f - mu_H| / sigma_H.
        Baselines come from training-time benign statistics.
psi_8 = hedge density = #hedge-phrase hits / N (N tokens).

Lexicons live in data/. Each file is a plain-text list of phrases, one per
line, "#" introduces a comment. Empty lines ignored.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from src.comment import Comment

_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)

# ----- Files -----
_AI_LEX_PATH = Path("data/ai_lexicon.txt")
_HEDGE_LEX_PATH = Path("data/hedge_lexicon.txt")
_PUNCT_BASELINE_PATH = Path("models/punctuation_baseline.json")


# ============================== psi_6 ==============================
@lru_cache(maxsize=1)
def _load_ai_lexicon() -> list[re.Pattern]:
    """Return compiled regex patterns for the AI-phrase lexicon."""
    builtin = [
        r"\bit'?s worth noting that\b",
        r"\bit'?s important to (note|remember|recognize)\b",
        r"\b(let'?s|let us) (delve|dive) into\b",
        r"\bdelve into\b",
        r"\bnavigate (the|this) (complexities|landscape|nuances)\b",
        r"\ba (tapestry|symphony|kaleidoscope) of\b",
        r"\bin the realm of\b",
        r"\bin today'?s (digital|fast-paced|interconnected) (world|age)\b",
        r"\bmoreover\b",
        r"\bfurthermore\b",
        r"\bin conclusion\b",
        r"\bit goes without saying\b",
        r"\b(plays?|playing) a (crucial|pivotal|vital|key) role\b",
        r"\bunwavering\b",
        r"\b(testament|hallmark) to\b",
        r"\bcomprehensive (understanding|overview|approach)\b",
        r"\bfoster (a sense of|innovation|collaboration|growth)\b",
        r"\bleverage\b",
        r"\bunderscore(s|d)?\b",
        r"\bhighlight(s|ed)?\b",
        r"\bnuance(d|s)?\b",
        r"\bmultifaceted\b",
        r"\bparadigm shift\b",
        r"\bholistic\b",
        # the lone em-dash with no spaces, a strong LLM tell
        r"\w\u2014\w",
    ]
    if _AI_LEX_PATH.exists():
        with open(_AI_LEX_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    builtin.append(line)
    return [re.compile(p, re.IGNORECASE) for p in builtin]


def psi_6(text: str) -> float:
    if not text.strip():
        return float("nan")
    n_tokens = max(1, len(_TOKEN_RE.findall(text)))
    patterns = _load_ai_lexicon()
    hits = sum(1 for p in patterns if p.search(text))
    # Density per ~20 tokens, with log+1 transform so a single hit doesn't dominate
    rate = hits / (n_tokens / 20.0)
    return math.log1p(rate)


# ============================== psi_7 ==============================
def _punct_pattern_counts(text: str) -> dict[str, float]:
    """Return a dict of punctuation-feature rates, normalized by sentence count."""
    n_sent = max(1, len(re.findall(r"[.!?]+", text)) + (1 if text and text[-1] not in ".!?" else 0))
    n_chars = max(1, len(text))
    return {
        "em_dash":          text.count("\u2014") / n_sent,                    # —
        "en_dash":          text.count("\u2013") / n_sent,                    # –
        "semicolon":        text.count(";") / n_sent,
        "ellipsis":         (text.count("...") + text.count("\u2026")) / n_sent,
        "oxford_comma":     len(re.findall(r",\s+and\b", text)) / n_sent,
        "multi_excl":       len(re.findall(r"!{2,}", text)) / n_sent,
        "multi_quest":      len(re.findall(r"\?{2,}", text)) / n_sent,
        "all_caps_run":     len(re.findall(r"\b[A-Z]{3,}\b", text)) / n_sent,
        "double_space":     text.count("  ") / n_chars,
        "smart_quote":      sum(text.count(c) for c in "\u2018\u2019\u201c\u201d") / n_chars,
    }


def _load_punct_baseline() -> dict[str, tuple[float, float]] | None:
    if not _PUNCT_BASELINE_PATH.exists():
        return None
    with open(_PUNCT_BASELINE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: (float(v["mu"]), float(v["sigma"])) for k, v in raw.items()}


def fit_punct_baseline(benign_texts: list[str],
                       out_path: Path = _PUNCT_BASELINE_PATH) -> dict[str, tuple[float, float]]:
    import numpy as np
    sums: dict[str, list[float]] = {}
    for t in benign_texts:
        counts = _punct_pattern_counts(t)
        for k, v in counts.items():
            sums.setdefault(k, []).append(v)
    stats = {k: (float(np.mean(v)), float(np.std(v) + 1e-6)) for k, v in sums.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: {"mu": v[0], "sigma": v[1]} for k, v in stats.items()}, f, indent=2)
    return stats


def psi_7(text: str) -> float:
    if not text.strip():
        return float("nan")
    counts = _punct_pattern_counts(text)
    baseline = _load_punct_baseline()
    if baseline is None:
        # Pre-training: this feature is undefined without a benign baseline.
        # We return NaN so the median-imputer fills sensibly at train time and
        # compare_features.py reports "needs baseline" rather than a misleading
        # raw-density value (raw density runs the wrong direction for AI text,
        # which tends to use cleaner punctuation than casual human writing).
        return float("nan")
    deltas = []
    for k, v in counts.items():
        if k in baseline:
            mu, sigma = baseline[k]
            deltas.append(abs(v - mu) / sigma)
    if not deltas:
        return float("nan")
    return sum(deltas) / len(deltas)


# ============================== psi_8 ==============================
@lru_cache(maxsize=1)
def _load_hedge_lexicon() -> list[re.Pattern]:
    builtin = [
        r"\barguably\b",
        r"\b(it (could|might|may) be argued|one could argue)\b",
        r"\b(many|some) would (say|argue|contend)\b",
        r"\bit'?s important to (remember|note) that\b",
        r"\bin (many|some|certain) (cases|respects|ways)\b",
        r"\bto some extent\b",
        r"\bgenerally speaking\b",
        r"\bby and large\b",
        r"\bperhaps\b",
        r"\bpresumably\b",
        r"\bostensibly\b",
        r"\bseemingly\b",
        r"\bnot necessarily\b",
        r"\bthat (said|being said)\b",
        r"\bon the (one|other) hand\b",
        r"\bit'?s worth (noting|considering) that\b",
    ]
    if _HEDGE_LEX_PATH.exists():
        with open(_HEDGE_LEX_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    builtin.append(line)
    return [re.compile(p, re.IGNORECASE) for p in builtin]


def psi_8(text: str) -> float:
    if not text.strip():
        return float("nan")
    n_tokens = max(1, len(_TOKEN_RE.findall(text)))
    patterns = _load_hedge_lexicon()
    hits = sum(1 for p in patterns if p.search(text))
    return hits / (n_tokens / 100.0)  # per 100 tokens


# ============================== extractor ==============================
def extract(comment: Comment) -> dict[str, float]:
    text = comment.text or ""
    return {
        "psi_6": psi_6(text),
        "psi_7": psi_7(text),
        "psi_8": psi_8(text),
    }


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    print(extract(load_comment(path)))
