"""Token-rank features in the spirit of GLTR (Gehrmann, Strobelt, Rush; ACL 2019).

For each token w_i:
    r_i = | { v in V : p_theta(v | w_<i) >= p_theta(w_i | w_<i) } |

LLM text uses high-probability (low-rank) tokens disproportionately.
    phi_4 = fraction of tokens with rank <= 10
    phi_5 = fraction of tokens with rank <= 100
"""
from __future__ import annotations

import torch

from src.conversation import Conversation
from src.features.perplexity import _load_model

_MAX_TOKENS = 512


def token_ranks(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    tokenizer, model, device = _load_model()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS).to(device)
    ids = enc["input_ids"][0]
    if len(ids) < 2:
        return []
    with torch.no_grad():
        logits = model(**enc).logits[0]  # [T, V]
    ranks = []
    for t in range(1, len(ids)):
        target = ids[t].item()
        row = logits[t - 1]
        rank = int((row > row[target]).sum().item()) + 1
        ranks.append(rank)
    return ranks


def extract(conv: Conversation) -> dict[str, float]:
    all_ranks: list[int] = []
    for m in conv.scammer_messages:
        all_ranks.extend(token_ranks(m.text))
    if not all_ranks:
        return {"phi_4": float("nan"), "phi_5": float("nan")}
    n = len(all_ranks)
    phi_4 = sum(1 for r in all_ranks if r <= 10) / n
    phi_5 = sum(1 for r in all_ranks if r <= 100) / n
    return {"phi_4": phi_4, "phi_5": phi_5}


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
