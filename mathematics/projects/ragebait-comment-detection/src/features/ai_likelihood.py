"""Stage-1 likelihood features (psi_1..psi_5).

References:
  - Perplexity / burstiness: classical; Goh & Barabasi (2008) for burstiness.
  - DetectGPT curvature: Mitchell et al. (ICML 2023).
  - GLTR token-rank: Gehrmann, Strobelt, Rush (ACL 2019).

Math (full derivations in the LaTeX report):
  psi_1 = (log PP(x) - mu_H(N)) / sigma_H(N)
            length-normalized log-perplexity z-score against a human baseline
            estimated at training time on benign comments of length ~ N.
  psi_2 = (sigma_pi - mu_pi) / (sigma_pi + mu_pi)
            burstiness coefficient over per-sentence log-perplexity.
            NaN when the comment has < 2 sentences.
  psi_3 = (log p(x) - mean_k log p(perturb_k(x))) / std_k
            DetectGPT curvature, normalized by perturbation-sample std.
            K = 20 perturbations from a T5-small mask-and-fill (15% mask).
  psi_4 = #{tokens with rank <= 10} / N
  psi_5 = #{tokens with rank <= 100} / N
"""
from __future__ import annotations

import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from src.comment import Comment

# ----- Models -----
_GPT2_NAME = "gpt2-medium"          # 355M; swap for "gpt2" on weak CPUs
_T5_NAME = "google-t5/t5-small"     # 60M; perturbation model
_MAX_TOKENS = 512

# ----- DetectGPT settings -----
_K_PERTURB = 20         # more than long-form (was 10) to fight short-text noise
_MASK_RATIO = 0.15

# ----- Baselines for psi_1 -----
# Written by training; loaded at inference.
_BASELINE_PATH = Path("models/perplexity_baseline.json")


@lru_cache(maxsize=1)
def _load_gpt2():
    tokenizer = GPT2TokenizerFast.from_pretrained(_GPT2_NAME)
    # Needed for batched forward passes (padding short sequences in psi_3).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(_GPT2_NAME).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device), device


@lru_cache(maxsize=1)
def _load_t5():
    tok = T5Tokenizer.from_pretrained(_T5_NAME)
    model = T5ForConditionalGeneration.from_pretrained(_T5_NAME).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), device


# ============================== psi_1 ==============================
def perplexity(text: str) -> float:
    text = text.strip()
    if not text:
        return float("nan")
    tokenizer, model, device = _load_gpt2()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS).to(device)
    if enc["input_ids"].shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())


def token_count(text: str) -> int:
    tokenizer, _, _ = _load_gpt2()
    return len(tokenizer.encode(text, truncation=True, max_length=_MAX_TOKENS))


def _load_baseline() -> dict[int, tuple[float, float]] | None:
    """Returns mapping {bucket_center: (mu, sigma)} for log-PP on benign data."""
    if not _BASELINE_PATH.exists():
        return None
    with open(_BASELINE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): (float(v["mu"]), float(v["sigma"])) for k, v in raw.items()}


def fit_baseline(benign_texts: list[str], bucket_size: int = 10,
                 out_path: Path = _BASELINE_PATH) -> dict[int, tuple[float, float]]:
    """Compute length-bucketed mean/std of log-PP on benign comments.

    Buckets are bucket_size tokens wide (default 10): a comment with N tokens
    is matched to the bucket whose center is closest to N.
    """
    import numpy as np
    buckets: dict[int, list[float]] = {}
    for text in benign_texts:
        n = token_count(text)
        if n < 4:
            continue
        pp = perplexity(text)
        if not math.isfinite(pp) or pp <= 1:
            continue
        center = (n // bucket_size) * bucket_size + bucket_size // 2
        buckets.setdefault(center, []).append(math.log(pp))
    stats = {}
    for c, vals in buckets.items():
        if len(vals) < 5:
            continue
        stats[c] = (float(np.mean(vals)), float(np.std(vals) + 1e-6))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): {"mu": v[0], "sigma": v[1]} for k, v in stats.items()}, f, indent=2)
    return stats


def _nearest_bucket(n: int, stats: dict[int, tuple[float, float]]) -> tuple[float, float]:
    nearest = min(stats.keys(), key=lambda k: abs(k - n))
    return stats[nearest]


def psi_1(text: str) -> float:
    pp = perplexity(text)
    if not math.isfinite(pp) or pp <= 1:
        return float("nan")
    stats = _load_baseline()
    if stats is None:
        # No baseline yet (e.g. before training): return raw log-PP, the
        # downstream standardizer will normalize on the training set.
        return math.log(pp)
    n = token_count(text)
    mu, sigma = _nearest_bucket(n, stats)
    return (math.log(pp) - mu) / sigma


# ============================== psi_2 ==============================
def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p) > 3]


def psi_2(text: str) -> float:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return float("nan")
    log_pps = []
    for s in sentences:
        pp = perplexity(s)
        if math.isfinite(pp) and pp > 0:
            log_pps.append(math.log(pp))
    if len(log_pps) < 2:
        return float("nan")
    mu = sum(log_pps) / len(log_pps)
    var = sum((x - mu) ** 2 for x in log_pps) / len(log_pps)
    sigma = math.sqrt(var)
    return 0.0 if (sigma + mu) == 0 else (sigma - mu) / (sigma + mu)


# ============================== psi_3 ==============================
def _batch_log_prob(texts: list[str]) -> list[float]:
    """Batched log p(x) for a list of texts via one GPT-2 forward pass.

    Pads with eos_token; correctly masks padding when summing per-token
    log-probs using the attention mask.
    """
    if not texts:
        return []
    tokenizer, model, device = _load_gpt2()
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=_MAX_TOKENS, padding=True).to(device)
    if enc["input_ids"].shape[1] < 2:
        return [float("-inf")] * len(texts)
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).logits
    # Shift-by-one: predict token t from positions <t.
    shift_logits = logits[:, :-1, :]
    shift_labels = enc["input_ids"][:, 1:]
    attn = enc["attention_mask"][:, 1:].float()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in (token_lp * attn).sum(dim=1).cpu().tolist()]


def _batch_perturb(text: str, k: int, rng: random.Random) -> list[str]:
    """Generate k perturbations of `text` in a single batched T5 generate() call.

    Each perturbation independently masks `_MASK_RATIO` of the words and lets
    T5-small fill them. Same masking distribution as the per-call version,
    just dispatched as one batch instead of k sequential GPU calls -- ~20x
    faster on GPU, no math change.
    """
    words = text.split()
    if len(words) < 5:
        return [text] * k
    n_mask = max(1, int(len(words) * _MASK_RATIO))

    batch_masked, batch_idxs = [], []
    for _ in range(k):
        idxs = sorted(rng.sample(range(len(words)), min(n_mask, 90)))
        masked = list(words)
        for j, i in enumerate(idxs):
            masked[i] = f"<extra_id_{j}>"
        batch_masked.append(" ".join(masked))
        batch_idxs.append(idxs)

    tok, model, device = _load_t5()
    enc = tok(batch_masked, return_tensors="pt", truncation=True,
              max_length=512, padding=True).to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_length=128, do_sample=True, top_p=0.95)

    results = []
    for b in range(len(batch_masked)):
        filled = tok.decode(out_ids[b], skip_special_tokens=False)
        idxs = batch_idxs[b]
        fills: dict[int, str] = {}
        for j in range(len(idxs)):
            start_marker = f"<extra_id_{j}>"
            end_marker = f"<extra_id_{j + 1}>"
            if start_marker in filled:
                tail = filled.split(start_marker, 1)[1]
                piece = tail.split(end_marker, 1)[0] if end_marker in tail else tail
                fills[idxs[j]] = piece.strip().replace("</s>", "")
        reconstructed = list(words)
        for i, repl in fills.items():
            reconstructed[i] = repl if repl else words[i]
        results.append(" ".join(reconstructed))
    return results


def psi_3(text: str, k: int = _K_PERTURB, seed: int = 0) -> float:
    if len(text.split()) < 10:
        return float("nan")
    rng = random.Random(seed)
    perturbed_texts = _batch_perturb(text, k, rng)
    # One batched forward pass: original + all perturbations.
    log_probs = _batch_log_prob([text] + perturbed_texts)
    if not log_probs or not math.isfinite(log_probs[0]):
        return float("nan")
    orig = log_probs[0]
    perturbed = [lp for lp in log_probs[1:] if math.isfinite(lp)]
    if len(perturbed) < 2:
        return float("nan")
    mean = sum(perturbed) / len(perturbed)
    var = sum((x - mean) ** 2 for x in perturbed) / (len(perturbed) - 1)
    std = math.sqrt(var) if var > 0 else 1e-8
    return (orig - mean) / std


# ============================== psi_4, psi_5 ==============================
def token_ranks(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    tokenizer, model, device = _load_gpt2()
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


def psi_4_5(text: str) -> tuple[float, float]:
    ranks = token_ranks(text)
    if not ranks:
        return float("nan"), float("nan")
    n = len(ranks)
    return (sum(1 for r in ranks if r <= 10) / n,
            sum(1 for r in ranks if r <= 100) / n)


# ============================== extractor ==============================
def extract(comment: Comment) -> dict[str, float]:
    text = comment.text or ""
    p4, p5 = psi_4_5(text)
    return {
        "psi_1": psi_1(text),
        "psi_2": psi_2(text),
        "psi_3": psi_3(text),
        "psi_4": p4,
        "psi_5": p5,
    }


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    print(extract(load_comment(path)))
