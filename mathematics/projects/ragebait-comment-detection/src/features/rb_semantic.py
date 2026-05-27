"""Stage-2 semantic similarity to a ragebait template corpus (chi_9, chi_10).

Math:
    rho(x) = max_{r in R} cos( phi(x), phi(r) )

    chi_9  = rho(x)              (worst-case template match)
    chi_10 = #{ r in R : cos(phi(x), phi(r)) > 0.75 } / |R|
             (fraction of the ragebait corpus the comment matches)

We use `paraphrase-multilingual-MiniLM-L12-v2` (384-D) so the same embedder
is shared with the neutralization-gap and topic-residual features.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.comment import Comment

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_TEMPLATE_PATH = Path("data/ragebait_templates.jsonl")
# Threshold at which a comment is considered to "match" a template for chi_10.
# 0.5 = same topic + similar framing in MiniLM cosine space. Set deliberately
# below the 0.75 used as the explain_red_flags display threshold for chi_9
# (max similarity); coverage breadth needs a lower bar than worst-case match.
# Raise toward 0.6-0.7 once the template corpus is grown beyond the ~25 seeds.
_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_templates() -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    if _TEMPLATE_PATH.exists():
        with open(_TEMPLATE_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("text", "").strip()
                if t:
                    texts.append(t)
    if not texts:
        return [], np.zeros((0, 384), dtype=np.float32)
    embedder = _load_embedder()
    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return texts, np.asarray(emb)


def chi_9_10(text: str) -> tuple[float, float]:
    if not text.strip():
        return float("nan"), float("nan")
    _, template_emb = _load_templates()
    if template_emb.shape[0] == 0:
        return float("nan"), float("nan")
    embedder = _load_embedder()
    q = embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)
    sims = (np.asarray(q) @ template_emb.T)[0]
    return float(sims.max()), float((sims > _THRESHOLD).mean())


def extract(comment: Comment) -> dict[str, float]:
    text = comment.text or ""
    c9, c10 = chi_9_10(text)
    return {"chi_9": c9, "chi_10": c10}


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    print(extract(load_comment(path)))
