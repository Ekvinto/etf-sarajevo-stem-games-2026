"""DetectGPT-style curvature score (phi_3).

Mitchell et al. (ICML 2023). LLM-generated text sits at *local maxima* of
log p_theta. For a perturbation distribution q(. | x),

    d(x) = log p_theta(x) - E_{x~q} [log p_theta(x_tilde)]
         approx  -1/2 tr( Sigma_q * Hess(log p_theta)(x) )

so d(x) > 0 with margin for AI text. We Monte-Carlo this with K perturbations
from a T5 mask-and-fill, and normalize by the empirical std of perturbed log p.

Approximation: log p_theta(x) ~= -N * log PP(x). Documented in the report.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.conversation import Conversation
from src.features.perplexity import _load_model, perplexity

_T5_NAME = "google-t5/t5-small"  # 60M params, CPU-friendly
_K_PERTURB = 10
_MASK_RATIO = 0.15


@lru_cache(maxsize=1)
def _load_t5():
    tok = T5Tokenizer.from_pretrained(_T5_NAME)
    model = T5ForConditionalGeneration.from_pretrained(_T5_NAME).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tok, model, device


def _token_count(text: str) -> int:
    tokenizer, _, _ = _load_model()
    return len(tokenizer.encode(text, truncation=True, max_length=512))


def _log_prob(text: str) -> float:
    pp = perplexity(text)
    if not math.isfinite(pp) or pp <= 0:
        return float("-inf")
    return -_token_count(text) * math.log(pp)


def _perturb(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 5:
        return text
    n_mask = max(1, int(len(words) * _MASK_RATIO))
    idxs = sorted(rng.sample(range(len(words)), min(n_mask, 90)))
    masked = list(words)
    for k, i in enumerate(idxs):
        masked[i] = f"<extra_id_{k}>"
    masked_text = " ".join(masked)

    tok, model, device = _load_t5()
    enc = tok(masked_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_length=128, do_sample=True, top_p=0.95)
    filled = tok.decode(out_ids[0], skip_special_tokens=False)

    fills: dict[int, str] = {}
    for k in range(len(idxs)):
        start_marker = f"<extra_id_{k}>"
        end_marker = f"<extra_id_{k + 1}>"
        if start_marker in filled:
            tail = filled.split(start_marker, 1)[1]
            piece = tail.split(end_marker, 1)[0] if end_marker in tail else tail
            fills[idxs[k]] = piece.strip().replace("</s>", "")
    reconstructed = list(words)
    for i, repl in fills.items():
        reconstructed[i] = repl if repl else words[i]
    return " ".join(reconstructed)


def curvature_score(text: str, k: int = _K_PERTURB, seed: int = 0) -> float:
    rng = random.Random(seed)
    orig = _log_prob(text)
    if not math.isfinite(orig):
        return float("nan")
    perturbed = []
    for _ in range(k):
        p_text = _perturb(text, rng)
        p_logp = _log_prob(p_text)
        if math.isfinite(p_logp):
            perturbed.append(p_logp)
    if len(perturbed) < 2:
        return float("nan")
    mean = sum(perturbed) / len(perturbed)
    var = sum((x - mean) ** 2 for x in perturbed) / (len(perturbed) - 1)
    std = math.sqrt(var) if var > 0 else 1e-8
    return (orig - mean) / std


def extract(conv: Conversation) -> dict[str, float]:
    scores = []
    for m in conv.scammer_messages:
        if len(m.text.split()) >= 20:
            s = curvature_score(m.text)
            if math.isfinite(s):
                scores.append(s)
    phi_3 = sum(scores) / len(scores) if scores else float("nan")
    return {"phi_3": phi_3}


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
