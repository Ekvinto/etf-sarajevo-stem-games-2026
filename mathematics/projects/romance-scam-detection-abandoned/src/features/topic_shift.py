"""Topic-shift via Kullback-Leibler divergence (phi_13, phi_14).

Partition scammer messages into early window W_E (first half) and late W_L.
Compute Laplace-smoothed word distributions over a content vocabulary plus
a curated financial / scam-indicative lexicon.

    phi_13 = D_KL( p_L || p_E ) = sum_v p_L(v) log( p_L(v) / p_E(v) )
    phi_14 = sum over finance-lexicon terms of p_L(v)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from src.conversation import Conversation

_ALPHA = 0.5
_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)

# Compact built-in lexicon. Extend via data/scam_lexicon.txt for the report.
_FINANCE_LEXICON: set[str] = set()


def _load_lexicon() -> set[str]:
    global _FINANCE_LEXICON
    if _FINANCE_LEXICON:
        return _FINANCE_LEXICON
    builtin = {
        "money", "transfer", "wire", "bank", "account", "swift", "iban",
        "bitcoin", "btc", "crypto", "ethereum", "usdt", "tether", "wallet",
        "investment", "invest", "broker", "trading", "platform", "profit",
        "loan", "fee", "tax", "customs", "duty", "release", "urgent",
        "emergency", "hospital", "surgery", "visa", "passport", "ticket",
        "gold", "deposit", "withdraw", "western", "union", "moneygram",
        "gift", "card", "amazon", "steam", "voucher", "code", "pin",
        "verification", "confirm", "send", "receive", "asap", "immediately",
    }
    extra_path = Path("data/scam_lexicon.txt")
    if extra_path.exists():
        with open(extra_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    builtin.add(w)
    _FINANCE_LEXICON = builtin
    return _FINANCE_LEXICON


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _smoothed_distribution(tokens: list[str], vocab: set[str]) -> dict[str, float]:
    counts = Counter(t for t in tokens if t in vocab)
    n_vocab = len(vocab)
    total = sum(counts.values()) + _ALPHA * n_vocab
    return {v: (counts.get(v, 0) + _ALPHA) / total for v in vocab}


def extract(conv: Conversation) -> dict[str, float]:
    msgs = conv.scammer_messages
    if len(msgs) < 4:
        return {"phi_13": float("nan"), "phi_14": float("nan")}

    half = len(msgs) // 2
    early_tokens = [t for m in msgs[:half] for t in _tokenize(m.text)]
    late_tokens = [t for m in msgs[half:] for t in _tokenize(m.text)]

    if not early_tokens or not late_tokens:
        return {"phi_13": float("nan"), "phi_14": float("nan")}

    finance = _load_lexicon()

    # phi_13: KL divergence over the union of OBSERVED tokens only.
    # Laplace smoothing prevents log(0) but keeps the support realistic.
    vocab = set(early_tokens) | set(late_tokens)
    if not vocab:
        return {"phi_13": float("nan"), "phi_14": float("nan")}

    p_e = _smoothed_distribution(early_tokens, vocab)
    p_l = _smoothed_distribution(late_tokens, vocab)

    kl = 0.0
    for v in vocab:
        if p_l[v] > 0 and p_e[v] > 0:
            kl += p_l[v] * math.log(p_l[v] / p_e[v])

    # phi_14: empirical fraction of late tokens that fall in the finance lexicon.
    # No smoothing here -- smoothing would conflate "lexicon size" with "lexicon usage."
    finance_count = sum(1 for t in late_tokens if t in finance)
    finance_mass = finance_count / len(late_tokens)

    return {"phi_13": float(kl), "phi_14": float(finance_mass)}


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
