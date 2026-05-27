# Deploying the AI-Ragebait Detector

Live target: **https://stem.steleks.ba/mathematics-2026/**

The server already runs the **core-infrastructure** stack (Traefik + Let's
Encrypt). The detector deploys as one extra Docker service that Traefik
discovers automatically through labels — no reverse-proxy config to edit,
no extra ports to open, no certificates to manage by hand.

---

## How it fits together

- `Dockerfile` builds a single image: CPU-only PyTorch, the three Hugging
  Face models baked in (GPT-2 medium, T5 small, MiniLM-L12), and the FastAPI
  server. ~6 GB, and it downloads nothing at runtime.
- `docker-compose.yml` runs that image as the `web` service, attached to the
  existing external `traefik-public` network, with Traefik routing labels.
- Traefik routes `stem.steleks.ba/mathematics-2026/*` to the container,
  strips the `/mathematics-2026` prefix (the app is path-agnostic and serves
  `/` and `/api/*` internally), and issues the TLS certificate.

---

## Prerequisites

1. **Core stack running.** It creates the `traefik-public` network and runs
   Traefik. (Already the case on this server.)
2. **DNS.** `stem.steleks.ba` must resolve to the server's public IP — an A
   record, or a `*.steleks.ba` wildcard. Traefik's Let's Encrypt HTTP
   challenge depends on it. Check from anywhere: `dig +short stem.steleks.ba`.
3. **Model + data in the repo.** Handled in Step 1.

---

## Step 1 — Push everything from your PC

`.gitignore` excludes trained models and data, so three files the app needs
were never pushed. Force-add them (this overrides `.gitignore` for these
specific files; the big training corpora stay ignored):

```bash
git add -f models/classifiers.joblib data/vad_subset.csv data/ragebait_templates.jsonl
git add Dockerfile docker-compose.yml .dockerignore requirements-web.txt webapp/ DEPLOYMENT.md
git commit -m "Web deployment stack (Traefik) + runtime model and data files"
git push
```

Verify all three are tracked before moving on:

```bash
git ls-files | grep -E "classifiers.joblib|vad_subset.csv|ragebait_templates.jsonl"
```

All three must be listed. (`models/*.json` baselines and `data/*.txt`
lexicons are already tracked — only these three were excluded.)

---

## Step 2 — On the server: clone and build

```bash
ssh youruser@your-server
git clone https://github.com/<your-account>/STEM-GAMES-MATHEMATICS-RAGEBAIT.git
cd STEM-GAMES-MATHEMATICS-RAGEBAIT
```

Confirm the model and data arrived with the clone:

```bash
ls -la models/classifiers.joblib data/vad_subset.csv data/ragebait_templates.jsonl
```

If any is missing, redo Step 1 — the build will not work without them.

Build and start:

```bash
docker compose up -d --build
```

The first build downloads PyTorch and ~2.5 GB of model weights and bakes
them into a ~6 GB image — expect **10–20 minutes**. Later code-only rebuilds
take seconds, because the dependency and model-download layers are cached.

---

## Step 3 — Wait for the app to be ready

```bash
docker compose logs -f web
```

Wait for **`Warm-up complete`** in the log — that means the classifier
bundle plus all three models are loaded into memory. The compose healthcheck
allows a 180 s grace window for this.

---

## Step 4 — Open it

**https://stem.steleks.ba/mathematics-2026/**

Traefik obtains the Let's Encrypt certificate on the first HTTPS request
(a few seconds the very first time). Verify from the command line:

```bash
curl https://stem.steleks.ba/mathematics-2026/api/health
# -> {"status":"ok","model_loaded":true}

curl -X POST https://stem.steleks.ba/mathematics-2026/api/score \
     -H "Content-Type: application/json" \
     -d '{"text":"How is this STILL legal? Every single one of these clowns needs to go. Wake up people."}'
```

---

## Updating after a change

```bash
# on your PC:
git add -A && git commit -m "..." && git push

# on the server:
cd STEM-GAMES-MATHEMATICS-RAGEBAIT
git pull
docker compose up -d --build
```

---

## Operating it

```bash
docker compose logs -f web      # live logs
docker compose ps               # status + health state
docker compose restart web      # restart the app
docker compose down             # stop and remove the container
```

**Resources.** The container holds the classifier bundle plus GPT-2 / T5 /
MiniLM in memory — budget ~3–4 GB RAM. CPU scoring is ~2–5 s per comment,
and the server processes one comment at a time (an internal lock prevents
two concurrent forward passes from exhausting memory).

**Tuning.** Constants at the top of `webapp/server.py`: `_MAX_CHARS`
(max comment length), `_RATE_LIMIT` / `_RATE_WINDOW` (per-IP rate limit).
Edit, commit, `git pull` on the server, `docker compose up -d --build`.

---

## Troubleshooting

**Traefik returns 404 for `/mathematics-2026`** — Traefik did not pick up the
labels. Confirm the container is on the shared network:
`docker inspect ragebait-web --format '{{json .NetworkSettings.Networks}}'`
should list `traefik-public`. Confirm the app itself did not crash:
`docker compose logs web`.

**Certificate not issued / browser security warning** — DNS for
`stem.steleks.ba` is not pointing at this server, or ports 80/443 are
blocked upstream. Fix DNS/firewall; Traefik retries automatically.

**`No model found at models/classifiers.joblib` in the logs** — the file was
not in the repo. Redo Step 1, `git pull` on the server, rebuild.

**500 on `/api/score` with a `_fill_dtype` / version error** — scikit-learn
version mismatch. `requirements-web.txt` pins `scikit-learn==1.6.1` (the
version that trained `classifiers.joblib`); make sure the image was rebuilt
after that pin landed: `docker compose up -d --build`.

**502 / "service unavailable" for the first ~3 minutes after `up`** — the app
is still loading models. Wait for `Warm-up complete` in the logs.

**Build runs out of disk** — the image is ~6 GB. Reclaim space from old
layers with `docker system prune`.
