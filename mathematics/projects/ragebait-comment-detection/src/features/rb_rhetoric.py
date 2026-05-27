"""Stage-2 rhetorical-pattern features (chi_5).

Combines four sub-features:
    rq    = rhetorical question rate
            (sentences ending in '?' that match a rhetorical-question pattern)
    hyper = hyperbole / absolute-quantifier rate per sentence
    caps  = ALL CAPS runs (>= 3 letters) per N tokens
    excl  = exclamation marks per sentence

    chi_5 = a1*rq + a2*hyper + a3*caps + a4*excl
            with a1..a4 = 1 by default (chi-square reweighting is done at training time).

Each sub-feature is also returned by `subfeatures()` for diagnostic logging.
"""
from __future__ import annotations

import re

from src.comment import Comment

_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)

# Rhetorical-question openers. Conservative; high precision matters more than recall.
_RHETQ_OPENERS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^how (is|can|could|on earth|in the world)\b",
        r"^why (is|on earth|in the world|would anyone)\b",
        r"^who (does|do they think|in their right mind)\b",
        r"^what kind of\b",
        r"^are (we|they) really\b",
        r"^is this (what|really|seriously)\b",
        r"^does anyone (really|honestly|seriously)\b",
        r"^seriously\?",
        r"^really\?",
        r"^make it make sense\b",
        r"^you can'?t make this (up|stuff up)\b",
    ]
]

# Hyperbole / absolute-quantifier vocabulary
_HYPER_RE = re.compile(
    r"\b(literally|every (single|last) (one|time)|always|never|nobody|"
    r"everybody|everyone|no one|all of them|the worst|the best|the most|"
    r"absolutely (everyone|nobody|nothing|everything)|insane|outrageous|"
    r"unbelievable|insanity|madness)\b",
    re.IGNORECASE,
)

_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _is_rhetq(sentence: str) -> bool:
    s = sentence.strip()
    if not s.endswith("?"):
        return False
    for pat in _RHETQ_OPENERS:
        if pat.search(s):
            return True
    return False


def subfeatures(text: str) -> dict[str, float]:
    text = text.strip()
    if not text:
        return {"rq": float("nan"), "hyper": float("nan"),
                "caps": float("nan"), "excl": float("nan")}
    sents = _split_sentences(text)
    n_sent = max(1, len(sents))
    tokens = _TOKEN_RE.findall(text)
    n_tok = max(1, len(tokens))

    rq = sum(1 for s in sents if _is_rhetq(s)) / n_sent
    hyper = len(_HYPER_RE.findall(text)) / n_sent
    caps = len(_CAPS_RE.findall(text)) / n_tok
    excl = text.count("!") / n_sent
    return {"rq": rq, "hyper": hyper, "caps": caps, "excl": excl}


def chi_5(text: str) -> float:
    sub = subfeatures(text)
    vals = [v for v in sub.values()]
    if any(isinstance(v, float) and v != v for v in vals):  # NaN check
        return float("nan")
    # Equal weights at extraction time; the logistic regression will reweight.
    return sum(vals)


def extract(comment: Comment) -> dict[str, float]:
    return {"chi_5": chi_5(comment.text or "")}


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    c = load_comment(path)
    print("subfeatures:", subfeatures(c.text))
    print("chi_5     :", chi_5(c.text))
