# Bots and how to catch them — Romance-scam chatbot detector

STEM Games 2026, Mathematics arena × Erste.

A mathematical detector for romance-scam chatbot conversations, intended for
deployment at the moment a bank customer initiates an outgoing transfer.
The detector computes a 19-dimensional feature vector spanning LLM-likelihood
statistics, classical stylometry, conversational dynamics, an HMM over scam
playbook stages, and semantic similarity to known scam templates, and fuses
them with L2-regularized logistic regression.

This README explains how to install and run the system. For deeper material:

* [`DATA_COLLECTION.md`](DATA_COLLECTION.md) — where the data comes from,
  how to use the Selenium scraper, ethics, and quotas
* [`MATH_REFERENCE.md`](MATH_REFERENCE.md) — every feature's math definition,
  in one place, for the LaTeX report

---

## 1. Quick start (TL;DR)

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate              # Windows  (use `source .venv/bin/activate` on Unix)
pip install -r requirements.txt

# 2. Copy and fill API keys
copy .env.example .env              # then edit .env in PyCharm

# 3. Collect data (any subset works)
python -m src.data_collection synth   --n 200    # synthetic scams via Anthropic
python -m src.data_collection benign  --n 200    # benign chats via Anthropic
python -m src.data_collection reddit  --limit 200  # real Reddit-pasted chats
python -m src.scrape_selenium --pages 5          # scrape scamletters.com
python -m src.scrape_selenium --postprocess      # convert to conversation schema

# 4. Build the template corpus for the semantic feature
python -m src.data_collection templates \
    --scam-jsonl data/scam_synthetic.jsonl \
    --out data/scam_templates.jsonl

# 5. Train the classifier (fits both HMMs and the logistic regression)
python -m src.train --scam data/scam_synthetic.jsonl --benign data/benign_corpus.jsonl

# 6. Run on a single conversation
python -m src.detect --input tests/conversations/scam_01.json

# 7. Produce ROC/PR/ablation plots for the report
python -m src.evaluate --scam data/scam_synthetic.jsonl --benign data/benign_corpus.jsonl
```

---

## 2. What's in this repo

```
.
├── data/                            # corpora (gitignored)
│   ├── scam_lexicon.txt             # editable financial-urgency wordlist
│   └── scam_templates.jsonl         # seed; extended by data_collection templates
├── images/                          # output plots for the LaTeX report
├── models/                          # trained classifier + HMMs
├── src/
│   ├── conversation.py              # JSON schema, load/save
│   ├── pipeline.py                  # phi(C) -> 19-D feature vector
│   ├── train.py                     # CLI: fit HMMs + logistic regression
│   ├── detect.py                    # CLI: score a single conversation
│   ├── evaluate.py                  # CLI: ROC/PR/calibration/ablation
│   ├── data_collection.py           # CLI: reddit / synth / benign / templates
│   ├── scrape_selenium.py           # CLI: Selenium scraper for scam archives
│   └── features/
│       ├── perplexity.py            # phi_1, phi_2     (GPT-2 medium)
│       ├── detect_gpt.py            # phi_3            (T5-small perturbations)
│       ├── token_rank.py            # phi_4, phi_5     (GLTR)
│       ├── stylometry.py            # phi_6, phi_7, phi_8  (MATTR, Yule K, Zipf)
│       ├── asymmetry.py             # phi_9, phi_10    (length, timing)
│       ├── sentiment.py             # phi_11, phi_12   (CUSUM, OLS)
│       ├── topic_shift.py           # phi_13, phi_14   (KL divergence)
│       ├── hmm_stages.py            # phi_15, phi_16   (forward, Viterbi)
│       └── semantic.py              # phi_17, phi_18, phi_19  (sentence-transformer)
├── tests/conversations/             # 3 scam + 3 benign sample JSON files
├── requirements.txt
├── README.md                        # you are here
├── DATA_COLLECTION.md               # scraping guide
└── MATH_REFERENCE.md                # one-page math summary
```

---

## 3. Installation (detailed)

### 3.1 Python version

Python **3.10 or 3.11** is recommended. PyCharm: `File → Settings → Project → Python Interpreter →
Add Interpreter → Virtualenv Environment → New`.

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

First run downloads transformer weights (~1.5 GB total):
- `gpt2-medium` (355M) — perplexity, token rank, log-prob for DetectGPT
- `google-t5/t5-small` (60M) — DetectGPT mask-and-fill
- `cardiffnlp/twitter-xlm-roberta-base-sentiment` (270M) — sentiment trajectory
- `paraphrase-multilingual-MiniLM-L12-v2` (120M) — sentence embeddings

These cache to `~/.cache/huggingface/` and are not redownloaded.

### 3.3 GPU vs CPU

Detected automatically. On CPU the full pipeline runs in ~5 s per conversation.
On a Colab T4 GPU it drops to ~0.5 s. For training on 1000 conversations:
- CPU: ~90 min (DetectGPT is the bottleneck; consider `--skip detect_gpt`)
- GPU: ~8 min

### 3.4 API keys (`.env`)

Copy `.env.example` to `.env` and fill in **whichever** of these you plan to use:

| Variable | Purpose | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | Synthetic data generation | https://console.anthropic.com/ |
| `OPENAI_API_KEY` | Alternative LLM | https://platform.openai.com/api-keys |
| `REDDIT_CLIENT_ID` | Reddit scraping | https://www.reddit.com/prefs/apps (create "script" app) |
| `REDDIT_CLIENT_SECRET` | Reddit scraping | same |

Selenium scraping does **not** need any keys.

---

## 4. End-to-end workflow

### 4.1 Collect data

The fastest path to a usable dataset is **synthetic only**:

```bash
python -m src.data_collection synth  --n 200
python -m src.data_collection benign --n 200
```

This writes `data/scam_synthetic.jsonl` and `data/benign_corpus.jsonl`,
each ~200 conversations of 30–50 messages. Total time: ~30 minutes, cost
~$5 in API credits.

For higher-quality submissions, add real data:

```bash
python -m src.data_collection reddit --limit 200
python -m src.scrape_selenium --pages 5
python -m src.scrape_selenium --postprocess
```

See [`DATA_COLLECTION.md`](DATA_COLLECTION.md) for source details, ethics,
and how to add more sites to the Selenium scraper.

### 4.2 Build the template corpus

The semantic features (φ₁₇–φ₁₉) compare incoming messages to a corpus of
known scam templates:

```bash
python -m src.data_collection templates \
    --scam-jsonl data/scam_synthetic.jsonl \
    --out data/scam_templates.jsonl
```

A small seed of templates is included in the repo so the feature works
out-of-the-box on the included `tests/conversations/scam_*.json`. **You
should regenerate this after collecting your real scam corpus** so the
distance metric uses your training data, not the seed.

### 4.3 Train

```bash
python -m src.train --scam data/scam_synthetic.jsonl --benign data/benign_corpus.jsonl
```

This:
1. Fits the scam HMM (Baum–Welch on the 5-stage playbook) and saves
   `models/hmm_scam.pkl`.
2. Fits the normal HMM and saves `models/hmm_normal.pkl`.
3. Extracts the 19-D feature vector for every conversation.
4. Fits an L2-regularized logistic regression with 5-fold CV over `C`.
5. Saves `models/classifier.joblib`.

**Speed flags:**

```bash
# Skip the slow DetectGPT feature during iteration
python -m src.train --scam ... --benign ... --skip detect_gpt

# Already-fit HMMs; skip refitting
python -m src.train --scam ... --benign ... --no-hmm
```

### 4.4 Detect

Single conversation:

```bash
python -m src.detect --input tests/conversations/scam_01.json
```

Machine-readable JSON output:

```bash
python -m src.detect --input my_chat.json --json
```

Without a trained model the tool still prints the raw feature vector and
red flags — useful for debugging.

### 4.5 Evaluate (for the report)

```bash
python -m src.evaluate --scam data/scam_synthetic.jsonl --benign data/benign_corpus.jsonl
```

Writes to `images/`:
- `roc.png` — ROC curve, AUC
- `pr.png` — precision–recall, AP
- `calibration.png` — reliability diagram, ECE
- `ablation.png` + `ablation.csv` — per-family ΔAUC

Drop these PNGs straight into the LaTeX report's `images/` folder.

---

## 5. Running individual features (for debugging)

Every feature module is a runnable script:

```bash
python -m src.features.perplexity   tests/conversations/scam_01.json
python -m src.features.detect_gpt   tests/conversations/scam_01.json
python -m src.features.semantic     tests/conversations/scam_01.json
# ... etc.
```

The full pipeline:

```bash
python -m src.pipeline tests/conversations/scam_01.json
```

prints the 19-D vector and triggered red flags. Useful for showing the
examiners on Day 2.

---

## 6. Team task division

| Person | Modules | Day 1 morning deliverable |
|---|---|---|
| **A — Data engineer** | `data_collection.py`, `scrape_selenium.py`, `data/*.jsonl` | 200 synthetic + 200 benign + 200 reddit conversations |
| **B — LLM likelihood** | `features/perplexity.py`, `features/detect_gpt.py`, `features/token_rank.py` | All three modules return finite numbers on tests/scam_01.json |
| **C — Classical stats / dynamics** | `features/stylometry.py`, `features/asymmetry.py`, `features/sentiment.py`, `features/topic_shift.py` | All four modules ditto |
| **D — HMM, semantic, report** | `features/hmm_stages.py`, `features/semantic.py`, LaTeX report | HMM fits on 50 conversations; semantic returns >0.7 on sample scam |

Day 2 is integration, evaluation, and report polish (see chat for the
hour-by-hour plan).

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: gpt2-medium not found` | First call downloads; check internet. Or swap to `"gpt2"` in `src/features/perplexity.py`. |
| `RuntimeError: ANTHROPIC_API_KEY not set` | Edit `.env`, restart PyCharm so it picks up the new env. |
| `selenium.common.exceptions.SessionNotCreatedException` | Make sure Chrome is installed. `webdriver_manager` auto-downloads the matching ChromeDriver. |
| Reddit returns 401 | Make sure the app type is "script" and the user agent string is unique. |
| `phi_3` is NaN for every conversation | DetectGPT needs ≥ 20-word messages. Drop the threshold in `detect_gpt.extract` or skip with `--skip detect_gpt`. |
| `phi_15`, `phi_16` are NaN | HMMs not yet trained. Run `src/train.py` (it fits them) or call `hmm_stages.fit_hmm` manually. |
| `phi_17` etc. NaN | Empty `data/scam_templates.jsonl`. Run the `templates` subcommand. |

---

## 8. License & credits

Built for the STEM Games 2026 Mathematics × Erste challenge.
Methods draw on Mitchell et al. (DetectGPT, 2023), Gehrmann et al. (GLTR, 2019),
Goh & Barabási (burstiness, 2008), Rabiner (HMM tutorial, 1989),
Reimers & Gurevych (Sentence-BERT, 2019). Full citations in the LaTeX report.
