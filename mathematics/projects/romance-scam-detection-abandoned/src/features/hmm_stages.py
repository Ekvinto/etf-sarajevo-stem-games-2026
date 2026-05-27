"""HMM over scam playbook stages (phi_15, phi_16).

Hidden states encode the 5-stage playbook:
    {Open, Rapport, Isolate, Hook, Urgency}

Per-message observation features psi(m):
    1. sentiment in {0, 1, 2}        (neg, neu, pos)  -> discretized to bins
    2. length bin    {short, medium, long}
    3. financial lexicon hit count   {0, 1+}
    4. question rate                 {0, low, high}
    5. position bin                  {early, mid, late}

We discretize psi(m) into a single categorical symbol over a small alphabet
(approximate but works for our scale), then fit a CategoricalHMM with
hmmlearn. Two HMMs are trained:
    lambda_S  on labeled scam conversations (weak supervision via lexicon)
    lambda_N  on benign conversations

Features:
    phi_15 = log P(C_S | lambda_S) - log P(C_S | lambda_N)
    phi_16 = 1 if Viterbi path enters Urgency state, else 0

If models/hmm_scam.pkl is missing, both features return NaN and the
pipeline carries on (graceful degradation).
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np

try:
    from hmmlearn import hmm  # type: ignore
    HMMLEARN_AVAILABLE = True
except ImportError:
    hmm = None  # type: ignore
    HMMLEARN_AVAILABLE = False

from src.conversation import Conversation
from src.features.topic_shift import _load_lexicon

_TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)
_STAGES = ["Open", "Rapport", "Isolate", "Hook", "Urgency"]
_N_SYMBOLS = 3 * 3 * 2 * 3 * 3  # = 162; small enough for CategoricalHMM
_MODEL_DIR = Path("models")
_SCAM_MODEL_PATH = _MODEL_DIR / "hmm_scam.pkl"
_NORMAL_MODEL_PATH = _MODEL_DIR / "hmm_normal.pkl"


def _psi_symbol(text: str, position_frac: float, lexicon: set[str]) -> int:
    """Map a message to a single categorical symbol via psi."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    n_tokens = len(tokens)

    # Length bin: short < 8, medium [8, 25), long >= 25
    if n_tokens < 8:
        length_bin = 0
    elif n_tokens < 25:
        length_bin = 1
    else:
        length_bin = 2

    # Sentiment proxy (avoid loading transformer model here for speed during HMM training)
    # We use a simple positive-token / negative-token heuristic.
    pos_words = {"love", "miss", "dear", "happy", "wonderful", "beautiful", "sweetheart"}
    neg_words = {"urgent", "problem", "trouble", "worried", "emergency", "lost", "stuck"}
    pos = sum(1 for t in tokens if t in pos_words)
    neg = sum(1 for t in tokens if t in neg_words)
    if neg > pos:
        sent_bin = 0
    elif pos > neg:
        sent_bin = 2
    else:
        sent_bin = 1

    # Financial-lexicon hits: 0 or 1+
    fin_bin = 1 if any(t in lexicon for t in tokens) else 0

    # Question rate
    q_count = text.count("?")
    if q_count == 0:
        q_bin = 0
    elif q_count <= 1:
        q_bin = 1
    else:
        q_bin = 2

    # Position bin
    if position_frac < 0.33:
        pos_bin = 0
    elif position_frac < 0.66:
        pos_bin = 1
    else:
        pos_bin = 2

    # Pack into a single symbol id in [0, _N_SYMBOLS)
    sym = (
        sent_bin
        + 3 * length_bin
        + 9 * fin_bin
        + 18 * q_bin
        + 54 * pos_bin
    )
    return int(sym)


def conversation_to_symbols(conv: Conversation) -> np.ndarray:
    lexicon = _load_lexicon()
    msgs = conv.scammer_messages
    n = max(1, len(msgs))
    return np.array(
        [[_psi_symbol(m.text, i / n, lexicon)] for i, m in enumerate(msgs)],
        dtype=int,
    )


def _build_blank_hmm() -> "hmm.CategoricalHMM":
    """Upper-triangular transition matrix with self-loops."""
    if not HMMLEARN_AVAILABLE:
        raise RuntimeError("hmmlearn not installed")
    n = len(_STAGES)
    model = hmm.CategoricalHMM(n_components=n, n_iter=30, random_state=0)
    # Initial distribution heavily concentrated on 'Open'
    model.startprob_ = np.array([0.85, 0.10, 0.03, 0.01, 0.01])
    # Upper-triangular transitions (no backsliding)
    A = np.array([
        [0.55, 0.35, 0.05, 0.04, 0.01],
        [0.00, 0.55, 0.30, 0.10, 0.05],
        [0.00, 0.00, 0.55, 0.30, 0.15],
        [0.00, 0.00, 0.00, 0.55, 0.45],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ])
    model.transmat_ = A
    model.n_features = _N_SYMBOLS
    # Uniform emission init, will be re-estimated by Baum-Welch
    model.emissionprob_ = np.full((n, _N_SYMBOLS), 1.0 / _N_SYMBOLS)
    return model


def fit_hmm(convs: list[Conversation], save_path: Path) -> "hmm.CategoricalHMM":
    if not HMMLEARN_AVAILABLE:
        raise RuntimeError(
            "hmmlearn is not installed; cannot fit HMM models. "
            "Install via `pip install hmmlearn` (requires C build tools on Windows), "
            "or use Python 3.11 where prebuilt wheels are available."
        )
    model = _build_blank_hmm()
    all_X = []
    lengths = []
    for c in convs:
        x = conversation_to_symbols(c)
        if len(x) >= 2:
            all_X.append(x)
            lengths.append(len(x))
    if not all_X:
        raise ValueError("No conversations long enough to fit HMM.")
    X = np.concatenate(all_X, axis=0)
    model.fit(X, lengths)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return model


def _load_model_or_none(path: Path):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def extract(conv: Conversation) -> dict[str, float]:
    if not HMMLEARN_AVAILABLE:
        return {"phi_15": float("nan"), "phi_16": float("nan")}
    scam_model = _load_model_or_none(_SCAM_MODEL_PATH)
    normal_model = _load_model_or_none(_NORMAL_MODEL_PATH)
    if scam_model is None or normal_model is None:
        return {"phi_15": float("nan"), "phi_16": float("nan")}

    X = conversation_to_symbols(conv)
    if len(X) < 2:
        return {"phi_15": float("nan"), "phi_16": float("nan")}

    try:
        ll_scam = float(scam_model.score(X))
        ll_normal = float(normal_model.score(X))
        states = scam_model.predict(X)
        urgency_idx = _STAGES.index("Urgency")
        in_urgency = float(int(urgency_idx in states))
    except Exception:
        return {"phi_15": float("nan"), "phi_16": float("nan")}

    return {"phi_15": ll_scam - ll_normal, "phi_16": in_urgency}


if __name__ == "__main__":
    import sys
    from src.conversation import load_conversation
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/conversations/scam_01.json"
    print(extract(load_conversation(path)))
