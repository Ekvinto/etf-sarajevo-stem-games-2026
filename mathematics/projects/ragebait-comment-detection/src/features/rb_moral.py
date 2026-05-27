"""Stage-2 moral-emotional vice density (chi_3).

The Moral Foundations Dictionary (Graham, Haidt, Nosek 2009) classifies words
into ten categories paired as virtue/vice across five foundations:

    Care / Harm
    Fairness / Cheating
    Loyalty / Betrayal
    Authority / Subversion
    Sanctity / Degradation

Brady et al. (PNAS 2017) showed that moral-emotional language increases the
per-word diffusion of a post on social media by approximately 20%, with the
vice side carrying the lion's share. We use this as the ragebait signal:

    chi_3 = density of vice-side moral words / N

The full MFD2 / eMFD lexicons (~3000 entries) are research-free downloads from
moralfoundations.org. We ship a curated subset (~150 entries) at
`data/mfd_vice.txt`. To use the full MFD2, drop it at `data/MFD2.dic` and this
module will parse it.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from src.comment import Comment

_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)

_MFD2_PATH = Path("data/MFD2.dic")
_VICE_SUBSET_PATH = Path("data/mfd_vice.txt")

# MFD2 .dic category numbers for VICE side
_VICE_CATEGORIES = {
    "Harm.vice", "Cheating.vice", "Betrayal.vice",
    "Subversion.vice", "Degradation.vice",
    # Also map common legacy MFD1 labels:
    "HarmVice", "FairnessVice", "IngroupVice", "AuthorityVice", "PurityVice",
}


@lru_cache(maxsize=1)
def _load_vice() -> set[str]:
    """Return a set of vice-side moral words (and word stems with `*` removed)."""
    vice: set[str] = set()

    # Parse MFD2.dic format if available (LIWC-style):
    #     %
    #     1 Care.virtue
    #     2 Harm.vice
    #     ...
    #     %
    #     abuse 2
    #     compassion 1
    #     ...
    if _MFD2_PATH.exists():
        cat_id_to_name: dict[str, str] = {}
        in_categories = True
        with open(_MFD2_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        sections = content.split("%")
        if len(sections) >= 3:
            for line in sections[1].strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    cat_id_to_name[parts[0]] = parts[1]
            vice_ids = {cid for cid, name in cat_id_to_name.items() if name in _VICE_CATEGORIES}
            for line in sections[2].strip().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                word = parts[0].rstrip("*").lower()
                cat_ids = set(parts[1:])
                if cat_ids & vice_ids:
                    vice.add(word)

    # Always also include the shipped subset
    if _VICE_SUBSET_PATH.exists():
        with open(_VICE_SUBSET_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    vice.add(w.rstrip("*"))

    return vice


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def chi_3(text: str) -> float:
    if not text.strip():
        return float("nan")
    vice = _load_vice()
    if not vice:
        return float("nan")
    tokens = _tokenize(text)
    if not tokens:
        return float("nan")
    # Stem-prefix matching: a token w matches an entry e if w starts with e
    # (this is how the MFD `*` suffix works: "abuse*" matches "abused", "abusing").
    hits = sum(1 for t in tokens if any(t == v or t.startswith(v) for v in vice if len(v) >= 4))
    return hits / len(tokens)


def extract(comment: Comment) -> dict[str, float]:
    return {"chi_3": chi_3(comment.text or "")}


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    print(extract(load_comment(path)))
