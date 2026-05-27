# ── AI-Ragebait Detector — web service image ───────────────────────────────
# Builds a self-contained image: CPU-only PyTorch, all three Hugging Face
# models baked in, and the FastAPI server. No downloads happen at runtime.
FROM python:3.11-slim

# Hugging Face / sentence-transformers cache lives inside the image at a
# deterministic path, populated at build time and reused at runtime.
ENV HF_HOME=/app/.hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) CPU-only PyTorch first (avoids pulling ~5 GB of CUDA libraries).
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Remaining web/inference dependencies. torch is already satisfied.
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

# 3) Pre-download the HF models into the image cache (cached build layer).
COPY webapp/prefetch_models.py webapp/prefetch_models.py
RUN python webapp/prefetch_models.py

# 4) Application code, trained model, and runtime data (lexicons, templates).
#    .dockerignore keeps the big training corpora out.
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY webapp/ webapp/

EXPOSE 8000

# Single worker: the model bundle is ~3 GB resident; one process is enough
# for a demo and avoids multiplying memory. Scale with a queue if needed.
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
