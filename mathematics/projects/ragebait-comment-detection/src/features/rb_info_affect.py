"""Stage-2 information-to-affect ratio (chi_6).

Math:
    I(x) = ( #named entities + #numerals + #quoted spans ) / N
    A(x) = mean_arousal(x) * ( |mean_valence(x)| + exclamation_density(x) )
    chi_6 = log( (A(x) + eps) / (I(x) + eps) )

Ragebait carries high A with low I.

We use a *regex* proxy for named entities (capitalized non-initial words +
quoted spans + numerals), rather than a heavy NER model, to keep the feature
extractor CPU-light. A user wanting a more precise NER can swap in a
HuggingFace pipeline; the math is unchanged.
"""
from __future__ import annotations

import math
import re

from src.comment import Comment
from src.features.rb_affect import vad_stats

_EPS = 1e-3
_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)

# A capitalized word that is NOT at the start of a sentence (rough NER proxy).
# We allow up to one preceding lowercase word in the same sentence.
_INITIAL_CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b")
_NUMERAL = re.compile(r"\b\d[\d.,]*\b")
_QUOTED = re.compile(r"[\"\u201c][^\"\u201c\u201d]{3,}[\"\u201d]")
_EXCL_DENS = re.compile(r"!")


def info_content(text: str) -> float:
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return float("nan")
    # Sentence-aware NE proxy: count capitalized words not directly after start-of-sentence
    ne_count = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not sent:
            continue
        # Strip a leading capitalized word (sentence start) so it doesn't inflate NE count.
        ne_count += len(_INITIAL_CAP.findall(sent[1:] if len(sent) > 1 else ""))
        # Plus any capitalized word that follows another in the same sentence
        ne_count += len(re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", sent))
    num_count = len(_NUMERAL.findall(text))
    quote_count = len(_QUOTED.findall(text))
    return (ne_count + num_count + quote_count) / len(tokens)


def affect_content(text: str) -> float:
    mean_a, mean_v, _ = vad_stats(text)
    if math.isnan(mean_a) or math.isnan(mean_v):
        return float("nan")
    n_sent = max(1, len(re.split(r"[.!?]+", text.strip())))
    excl_dens = len(_EXCL_DENS.findall(text)) / n_sent
    return mean_a * (abs(mean_v) + excl_dens)


def chi_6(text: str) -> float:
    if not text.strip():
        return float("nan")
    I = info_content(text)
    A = affect_content(text)
    if math.isnan(I) or math.isnan(A):
        return float("nan")
    return math.log((A + _EPS) / (I + _EPS))


def extract(comment: Comment) -> dict[str, float]:
    return {"chi_6": chi_6(comment.text or "")}


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    c = load_comment(path)
    print(f"  I = {info_content(c.text):.4f}")
    print(f"  A = {affect_content(c.text):.4f}")
    print(f"chi_6 = {chi_6(c.text):.4f}")
