"""FastAPI server: score a single comment through the two-stage detector.

Run locally (from the project root):
    uvicorn webapp.server:app --host 0.0.0.0 --port 8000

The app is deliberately path-agnostic. It serves the UI at "/" and the API
at "/api/*". When deployed under a subpath (e.g. /mathematics-2026), the
reverse proxy strips that prefix, so this app never needs to know its public
path. The frontend computes the API URL from window.location at runtime.

Endpoints:
    GET  /              -> single-page UI
    GET  /api/health    -> liveness + model-loaded status
    POST /api/score     -> {text, parent_topic?} -> scores + red flags + features
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.comment import Comment
from src.pipeline import (
    AI_FEATURE_NAMES,
    RB_FEATURE_NAMES,
    explain_red_flags,
    extract_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ragebait-web")

# --- Configuration ----------------------------------------------------------
_MODEL_PATH = Path("models/classifiers.joblib")
_STATIC_DIR = Path(__file__).parent / "static"
_MAX_CHARS = 4000           # reject comments longer than this
_MIN_CHARS = 1
_RATE_LIMIT = 20            # max requests ...
_RATE_WINDOW = 60.0         # ... per this many seconds, per client IP

# Human-readable labels for each feature, surfaced in the UI.
_FEATURE_LABELS = {
    "psi_1": "Length-normalized perplexity (z-score)",
    "psi_2": "Burstiness",
    "psi_3": "DetectGPT local curvature",
    "psi_4": "GLTR top-10 token-rank fraction",
    "psi_5": "GLTR top-100 token-rank fraction",
    "psi_6": "LLM lexical fingerprint",
    "psi_7": "Punctuation regularity",
    "psi_8": "Hedging density",
    "chi_1": "Affective intensity (arousal x |valence|)",
    "chi_2": "Strong-negative word fraction",
    "chi_3": "Moral-vice density",
    "chi_4": "Outgroup-negativity association",
    "chi_5": "Rhetorical-pattern score",
    "chi_6": "Information-to-affect ratio",
    "chi_7": "Counterfactual neutralization gap",
    "chi_8": "Topic-conditional emotion residual",
    "chi_9": "Max ragebait-template similarity",
    "chi_10": "Ragebait-template coverage breadth",
}

# --- Shared state -----------------------------------------------------------
# A single lock serializes scoring. Model inference is CPU-heavy; we never
# want two multi-hundred-MB forward passes running at once on a small VPS.
_score_lock = threading.Lock()
_bundle: dict | None = None  # populated at startup by _warm_up()

# Lightweight in-memory rate limiter (sliding window per IP).
_hits: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def _rate_ok(ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        dq = _hits[ip]
        while dq and now - dq[0] > _RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            return False
        dq.append(now)
        return True


def _client_ip(request: Request) -> str:
    """Real client IP, honoring the reverse proxy's X-Forwarded-For header."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _clean(x) -> float | None:
    """JSON-safe: NaN / inf become null (the model imputes internally)."""
    if x is None:
        return None
    xf = float(x)
    return xf if math.isfinite(xf) else None


def _warm_up() -> None:
    """Load the classifier bundle and force every heavy model into memory.

    Running one dummy comment through the pipeline triggers the lazy
    @lru_cache loaders for GPT-2, T5 and the sentence transformer, so the
    first real request is fast instead of taking a minute.
    """
    global _bundle
    t0 = time.time()
    if _MODEL_PATH.exists():
        _bundle = joblib.load(_MODEL_PATH)
        log.info("Loaded classifier bundle from %s", _MODEL_PATH)
    else:
        log.error("No model found at %s -- scoring will return 503.", _MODEL_PATH)

    log.info("Warming up feature extractors (loading GPT-2 / T5 / MiniLM)...")
    try:
        extract_features(Comment(text="This is a short warm-up comment for the detector."))
        log.info("Warm-up complete in %.1f s", time.time() - t0)
    except Exception as e:  # noqa: BLE001
        log.exception("Warm-up failed (server will still start): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_up()
    yield


app = FastAPI(
    title="AI-Ragebait Comment Detector",
    description="Two-stage mathematical detector for AI-generated ragebait comments.",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


# --- API models -------------------------------------------------------------
class ScoreRequest(BaseModel):
    text: str = Field(..., description="Comment text to score.")
    parent_topic: str | None = Field(
        None, description="Optional article / parent-post title for context."
    )


# --- Routes -----------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/api/score")
def score(req: ScoreRequest, request: Request) -> JSONResponse:
    ip = _client_ip(request)
    if not _rate_ok(ip):
        return JSONResponse(
            status_code=429,
            content={"error": f"Rate limit exceeded ({_RATE_LIMIT} requests / "
                               f"{int(_RATE_WINDOW)}s). Please wait a moment."},
        )

    text = (req.text or "").strip()
    if len(text) < _MIN_CHARS:
        return JSONResponse(status_code=400, content={"error": "Comment text is empty."})
    if len(text) > _MAX_CHARS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Comment too long (max {_MAX_CHARS} characters)."},
        )
    if _bundle is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Detection model is not loaded on the server."},
        )

    t0 = time.time()
    comment = Comment(text=text, parent_topic=(req.parent_topic or None))

    # Serialize the heavy work: feature extraction + two model forward passes.
    with _score_lock:
        feats = extract_features(comment)
        flags = explain_red_flags(feats)
        x_ai = np.array([[feats[n] for n in AI_FEATURE_NAMES]], dtype=float)
        x_rb = np.array([[feats[n] for n in RB_FEATURE_NAMES]], dtype=float)
        p_ai = float(_bundle["clf_ai"].predict_proba(x_ai)[0, 1])
        p_rb = float(_bundle["clf_rb"].predict_proba(x_rb)[0, 1])

    p_joint = p_ai * p_rb
    elapsed_ms = int((time.time() - t0) * 1000)

    ai_yes, rb_yes = p_ai >= 0.5, p_rb >= 0.5
    if ai_yes and rb_yes:
        verdict, kind = "Likely AI-generated ragebait", "danger"
    elif ai_yes:
        verdict, kind = "Likely AI-generated, but not ragebait", "warn"
    elif rb_yes:
        verdict, kind = "Likely human-written ragebait", "warn"
    else:
        verdict, kind = "Likely a genuine human comment", "ok"

    def _features(names: list[str]) -> list[dict]:
        return [
            {"name": n, "label": _FEATURE_LABELS.get(n, n), "value": _clean(feats.get(n))}
            for n in names
        ]

    return JSONResponse(content={
        "p_ai": p_ai,
        "p_ragebait": p_rb,
        "p_joint": p_joint,
        "verdict": verdict,
        "verdict_kind": kind,
        "red_flags": flags,
        "features": {
            "ai": _features(AI_FEATURE_NAMES),
            "ragebait": _features(RB_FEATURE_NAMES),
        },
        "elapsed_ms": elapsed_ms,
    })


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
