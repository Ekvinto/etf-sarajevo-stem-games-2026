"""Stage-2 topic-conditional emotion residual (chi_8).

A comment about a war casualty is *expected* to be more emotional than one
about gardening; absolute arousal can mislead. We compute the residual after
conditioning on topic.

Math:
    Let tau(x) be a sentence-transformer embedding of the parent topic
    (or, if no parent_topic is present, of x itself, projected through the
    same embedder).
    Fit on the BENIGN training corpus an OLS:
        mean_arousal(x) ~= beta_0 + beta^T tau(x) + eps(x)
    Then:
        chi_8(x) = mean_arousal(x) - (beta_0 + beta^T tau(x))

Positive chi_8 = the comment is more aroused than its topic would predict.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.comment import Comment
from src.features.rb_affect import vad_stats
from src.features.rb_semantic import _load_embedder

_MODEL_PATH = Path("models/topic_residual_ols.json")


@lru_cache(maxsize=1)
def _load_ols() -> tuple[np.ndarray, float] | None:
    if not _MODEL_PATH.exists():
        return None
    with open(_MODEL_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return np.asarray(obj["beta"], dtype=np.float32), float(obj["beta_0"])


def fit_topic_residual(benign_comments: list[Comment],
                       out_path: Path = _MODEL_PATH) -> tuple[np.ndarray, float]:
    """Fit OLS mean_arousal ~ tau on the benign subset.

    Returns (beta, beta_0). Saves to disk.
    """
    embedder = _load_embedder()
    texts: list[str] = []
    ys: list[float] = []
    for c in benign_comments:
        a, _, _ = vad_stats(c.text)
        if math.isnan(a):
            continue
        topic = c.parent_topic or c.text
        texts.append(topic)
        ys.append(a)
    if len(texts) < 20:
        # Not enough data to fit meaningfully; save a zero model so the
        # residual returns approximately the raw arousal.
        embedder_dim = 384
        beta = np.zeros(embedder_dim, dtype=np.float32)
        beta_0 = float(np.mean(ys)) if ys else 0.0
    else:
        emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        X = np.asarray(emb, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)
        # Ridge regression with small lambda to avoid singular fits when dim >> n.
        lam = 1.0
        XtX = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        beta_0 = float(y.mean() - X.mean(axis=0) @ beta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"beta": beta.tolist(), "beta_0": float(beta_0)}, f)
    # Bust the cache so subsequent calls see the new model
    _load_ols.cache_clear()
    return beta, beta_0


def chi_8(comment: Comment) -> float:
    mean_a, _, _ = vad_stats(comment.text or "")
    if math.isnan(mean_a):
        return float("nan")
    ols = _load_ols()
    if ols is None:
        # No model yet: just return raw arousal centered at 0.5.
        return mean_a - 0.5
    beta, beta_0 = ols
    embedder = _load_embedder()
    topic = comment.parent_topic or comment.text or ""
    emb = embedder.encode([topic], normalize_embeddings=True, show_progress_bar=False)
    pred = beta_0 + float(np.asarray(emb)[0] @ beta)
    return mean_a - pred


def extract(comment: Comment) -> dict[str, float]:
    return {"chi_8": chi_8(comment)}


if __name__ == "__main__":
    import sys
    from src.comment import load_comment
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/comments/ragebait_ai_01.json"
    print(extract(load_comment(path)))
