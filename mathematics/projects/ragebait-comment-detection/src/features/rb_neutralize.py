"""Stage-2 counterfactual neutralization gap (chi_7).

Math:
    nu(x)  = N(x)     -- a neutralized rewrite of x
    R(x)   = |nu(x)| / |x|                       (length ratio, capped at 1)
    S(x)   = <phi(x), phi(nu(x))> / (||.|| ||.||) (cosine similarity)
    chi_7  = (1 - R(x)) * (1 - S(x))

Ragebait collapses under neutralization: R is small (rewrite is short),
S is small (rewrite is semantically distant). Substantive comments survive:
both ratios stay near 1, so chi_7 -> 0.

Two implementations of N are provided:
    - 'rule' (default, no extra model): strip affect/hyperbole/ALL CAPS and
      remove rhetorical-question constructions. Cheap, deterministic, robust.
    - 'flan-t5' (optional, better quality): prompt google/flan-t5-base to
      rewrite neutrally. Switch via the NEUTRALIZER env var.
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

import numpy as np

from src.comment import Comment
from src.features.rb_affect import _load_vad
from src.features.rb_rhetoric import _HYPER_RE
from src.features.rb_semantic import _load_embedder

_MODE = os.environ.get("NEUTRALIZER", "rule").lower()

# Words that carry hyperbolic / charged affect; stripped by the rule-based N.
_AFFECT_THRESHOLDS = {"valence_abs": 0.5, "arousal": 0.6}

# Patterns that mark rhetorical-question constructions, replaced by nothing.
_RHETQ_STRIP = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhow (is|can|could|on earth|in the world) [^.!?]+\?",
        r"\bwhy (is|on earth|in the world|would anyone) [^.!?]+\?",
        r"\bmake it make sense\.?",
        r"\byou can'?t make this (up|stuff up)\.?",
        r"\bseriously\?+",
        r"\breally\?+",
    ]
]

# Standalone phrases stripped wholesale
_STRIP_PHRASES = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwake up,? (people|sheeple|everyone)\b",
        r"\bthese (clowns|idiots|morons|fools)\b",
        r"\b(omg|wtf|smh)\b",
    ]
]


def _neutralize_rule(text: str) -> str:
    out = text
    # 1) Strip rhetorical-question and pre-set phrase patterns
    for p in _RHETQ_STRIP + _STRIP_PHRASES:
        out = p.sub("", out)
    # 2) Lowercase ALL CAPS runs of >= 3 letters
    out = re.sub(r"\b[A-Z]{3,}\b", lambda m: m.group(0).lower(), out)
    # 3) Strip hyperbole
    out = _HYPER_RE.sub("", out)
    # 4) Strip strongly-affective words by VAD lookup
    vad = _load_vad()
    if vad:
        def _replace(m: re.Match) -> str:
            w = m.group(0).lower()
            entry = vad.get(w)
            if entry is None:
                return m.group(0)
            v, a, _ = entry
            if abs(v) >= _AFFECT_THRESHOLDS["valence_abs"] and a >= _AFFECT_THRESHOLDS["arousal"]:
                return ""
            return m.group(0)
        out = re.sub(r"\b[\wÀ-ÿ]+\b", _replace, out)
    # 5) Collapse repeated punctuation
    out = re.sub(r"([!?]){2,}", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


@lru_cache(maxsize=1)
def _load_flan_t5():
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch
    name = "google/flan-t5-base"
    tok = T5Tokenizer.from_pretrained(name)
    model = T5ForConditionalGeneration.from_pretrained(name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), device


def _neutralize_flan_t5(text: str) -> str:
    import torch
    tok, model, device = _load_flan_t5()
    prompt = (
        "Rewrite the following text using only neutral, factual language. "
        "Remove all emotionally charged words, hyperbole, and rhetorical "
        "questions; preserve only verifiable claims. If there are no "
        "verifiable claims, output a single dash.\n\n"
        f"Text: {text}\n\nNeutral rewrite:"
    )
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_length=128, num_beams=2)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def neutralize(text: str) -> str:
    if _MODE == "flan-t5":
        try:
            return _neutralize_flan_t5(text)
        except Exception:
            return _neutralize_rule(text)
    return _neutralize_rule(text)


def chi_7(text: str) -> float:
    if not text.strip():
        return float("nan")
    nu = neutralize(text)
    # Length ratio. Use character length: more robust to single-word strip.
    orig_len = max(1, len(text))
    nu_len = len(nu)
    R = min(1.0, nu_len / orig_len)
    # Cosine similarity in embedding space.
    embedder = _load_embedder()
    if not nu.strip() or nu.strip() == "-":
        # Total collapse: similarity undefined; treat as orthogonal.
        S = 0.0
    else:
        embs = embedder.encode([text, nu], normalize_embeddings=True, show_progress_bar=False)
        embs = np.asarray(embs)
        S = float(embs[0] @ embs[1])
        S = max(0.0, S)   # clip negative similarities to 0
    return (1.0 - R) * (1.0 - S)


def extract(comment: Comment) -> dict[str, float]:
    return {"chi_7": chi_7(comment.text or "")}


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    c = load_comment(path)
    nu = neutralize(c.text)
    print(f"original   : {c.text!r}")
    print(f"neutralized: {nu!r}")
    print(f"chi_7      = {chi_7(c.text):.4f}")
