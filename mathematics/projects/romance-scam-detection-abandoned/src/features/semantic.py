"""Semantic similarity to known scam templates (phi_17, phi_18, phi_19).

Embed each scammer message with a multilingual SentenceTransformer.
Compare with cosine similarity to a corpus T of confirmed scam messages.

    rho(m) = max_{t in T}  <phi(m), phi(t)> / (||phi(m)|| ||phi(t)||)

    phi_17 = max_m rho(m)            (worst-case template match)
    phi_18 = mean_m rho(m)
    phi_19 = fraction of m with rho(m) > 0.75
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.conversation import Conversation

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_TEMPLATE_PATH = Path("data/scam_templates.jsonl")
_THRESHOLD = 0.75


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
                if "text" in obj and obj["text"]:
                    texts.append(obj["text"])
    if not texts:
        return [], np.zeros((0, 384), dtype=np.float32)
    embedder = _load_embedder()
    emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return texts, np.asarray(emb)


def extract(conv: Conversation) -> dict[str, float]:
    msgs = [m.text for m in conv.scammer_messages]
    if not msgs:
        return {"phi_17": float("nan"), "phi_18": float("nan"), "phi_19": float("nan")}

    _, template_emb = _load_templates()
    if template_emb.shape[0] == 0:
        return {"phi_17": float("nan"), "phi_18": float("nan"), "phi_19": float("nan")}

    embedder = _load_embedder()
    msg_emb = embedder.encode(msgs, normalize_embeddings=True, show_progress_bar=False)
    msg_emb = np.asarray(msg_emb)

    sims = msg_emb @ template_emb.T                 # [num_msgs, num_templates]
    per_msg_max = sims.max(axis=1)                  # rho(m) for each m

    return {
        "phi_17": float(per_msg_max.max()),
        "phi_18": float(per_msg_max.mean()),
        "phi_19": float((per_msg_max > _THRESHOLD).mean()),
    }


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
