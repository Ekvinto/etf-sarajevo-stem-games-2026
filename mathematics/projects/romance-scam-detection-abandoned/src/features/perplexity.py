"""Perplexity (phi_1) and burstiness (phi_2).

Math:
    perplexity(w) = exp( - 1/N * sum_i log p_theta(w_i | w_<i) )
    burstiness B  = (sigma - mu) / (sigma + mu)    in [-1, 1]
                    Goh & Barabasi (2008), over log-perplexity per sentence.

LLM-generated text typically has:
    - LOWER mean perplexity (model finds it more expected)
    - LOWER burstiness  (more uniform sentence quality)
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.conversation import Conversation

_MODEL_NAME = "gpt2-medium"  # 355M params; swap for "gpt2" on weak CPUs
_MAX_TOKENS = 512


@lru_cache(maxsize=1)
def _load_model():
    """Lazy-load model so importing this module is cheap."""
    tokenizer = GPT2TokenizerFast.from_pretrained(_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(_MODEL_NAME).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def perplexity(text: str) -> float:
    """Per-token perplexity under the reference LM."""
    text = text.strip()
    if not text:
        return float("inf")
    tokenizer, model, device = _load_model()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS).to(device)
    if enc["input_ids"].shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter. Avoids the punkt download."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p) > 3]


def extract(conv: Conversation) -> dict[str, float]:
    """Return {phi_1, phi_2}."""
    scammer_msgs = [m.text for m in conv.scammer_messages]
    if not scammer_msgs:
        return {"phi_1": float("nan"), "phi_2": float("nan")}

    # phi_1: mean per-message perplexity
    msg_pps = [perplexity(m) for m in scammer_msgs]
    msg_pps = [p for p in msg_pps if math.isfinite(p)]
    phi_1 = sum(msg_pps) / max(1, len(msg_pps)) if msg_pps else float("nan")

    # phi_2: burstiness of log-perplexity per sentence
    sentences = [s for m in scammer_msgs for s in _split_sentences(m)]
    if len(sentences) < 2:
        return {"phi_1": phi_1, "phi_2": float("nan")}

    log_pps = []
    for s in sentences:
        p = perplexity(s)
        if math.isfinite(p) and p > 0:
            log_pps.append(math.log(p))
    if len(log_pps) < 2:
        return {"phi_1": phi_1, "phi_2": float("nan")}

    mu = sum(log_pps) / len(log_pps)
    var = sum((x - mu) ** 2 for x in log_pps) / len(log_pps)
    sigma = math.sqrt(var)
    phi_2 = 0.0 if (sigma + mu) == 0 else (sigma - mu) / (sigma + mu)

    return {"phi_1": phi_1, "phi_2": phi_2}


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
