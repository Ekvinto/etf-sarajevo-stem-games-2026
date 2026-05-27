# Bots and how to catch them — AI-ragebait comment detector

STEM Games 2026, Mathematics arena × Erste.

A mathematical detector for AI-generated **ragebait** comments — the kind of
short, inflammatory posts that bot farms increasingly use to seed engagement
and outrage on news sites and social platforms. The detector produces **two
independent calibrated probabilities** for each comment:

- **p̂_AI** — probability the comment was AI-generated
- **p̂_RB** — probability the comment is ragebait

plus a joint risk score **p̂_AR = p̂_AI · p̂_RB**. Per-comment scores can be
aggregated to account-level via a Beta-Bernoulli Bayesian update.

The pipeline uses 18 features in two families (8 AI-likelihood, 10 ragebait),
fused with calibrated logistic regression. For deeper material:

- [`DATA_COLLECTION.md`](DATA_COLLECTION.md) — where the data comes from
  (HC3, civil_comments, wiki_toxic, synthetic), ethics, download instructions
  for the full NRC-VAD and MFD2 lexicons.
- [`MATH_REFERENCE.md`](MATH_REFERENCE.md) — every feature's math definition
  in one place, cross-referenced with the LaTeX report.

---

## 1. Quick start

```bash
# 1. Setup (Python 3.13 supported)
python -m venv .venv
.venv\Scripts\activate              # Windows  (use `source .venv/bin/activate` on Unix)
pip install -r requirements.txt

# 2. Fill API keys
copy .env.example .env              # then edit .env

# 3. Sanity-check the feature directions BEFORE collecting data
python compare_features.py          # should print mostly "OK"

# 4. Collect training data (mix and match)
python -m src.data_collection hc3 --n 500              # human vs ChatGPT
python -m src.data_collection civil --n 1000           # civil_comments (ragebait supervision)
python -m src.data_collection wiki_toxic --n 500       # Wikipedia toxic
python -m src.data_collection synth_ai_rb --n 200      # synthetic AI ragebait (Anthropic API)
python -m src.data_collection synth_benign --n 200     # synthetic substantive (Anthropic API)

# 5. Merge into a single training corpus
python -m src.data_collection merge ^
    --inputs data/hc3.jsonl data/civil_comments.jsonl data/synth_ai_ragebait.jsonl data/synth_benign.jsonl ^
    --out data/corpus.jsonl

# 6. Build the ragebait template corpus (used by chi_9, chi_10)
python -m src.data_collection templates ^
    --input data/corpus.jsonl ^
    --out data/ragebait_templates.jsonl

# 7. Train both classifiers (also fits the perplexity, punctuation, and topic baselines)
python -m src.train --corpus data/corpus.jsonl

# 8. Score a single comment
python -m src.detect --input tests/comments/ragebait_ai_01.json

# 9. Score raw text inline
python -m src.detect --text "Every single one of these clowns lives in a different universe..." ^
                     --parent-topic "Pension reform"

# 10. Produce ROC/PR/calibration/ablation plots for the report
python -m src.evaluate --corpus data/corpus.jsonl

# 11. Score an account's history and produce the Bayesian posterior P(bot)
python -m src.aggregate --input data/user_history.jsonl
```

---

## 2. What's in this repo

```
.
├── data/                            # corpora (gitignored)
│   ├── vad_subset.csv               # curated NRC-VAD subset (works out of the box)
│   ├── ai_lexicon.txt               # editable AI-phrase regex supplement
│   ├── hedge_lexicon.txt            # editable hedging-phrase supplement
│   ├── outgroup_lexicon.txt         # editable outgroup-marker supplement
│   ├── mfd_vice.txt                 # curated MFD vice subset
│   └── ragebait_templates.jsonl     # seed templates; extended by data_collection.py
├── images/                          # output plots for the LaTeX report
├── models/                          # trained classifiers + baselines
├── src/
│   ├── comment.py                   # JSON schema, load/save
│   ├── pipeline.py                  # extract_features(c) -> 18-D dict
│   ├── train.py                     # CLI: fit baselines + two LRs
│   ├── detect.py                    # CLI: score a single comment
│   ├── evaluate.py                  # CLI: ROC/PR/calibration/ablation per stage
│   ├── aggregate.py                 # CLI: account-level Bayesian posterior
│   ├── data_collection.py           # CLI: HF datasets + synthetic via Anthropic
│   └── features/
│       ├── ai_likelihood.py         # psi_1..psi_5   (perplexity, DetectGPT, GLTR)
│       ├── ai_lexical.py            # psi_6, psi_7, psi_8  (AI phrases, punctuation, hedging)
│       ├── rb_affect.py             # chi_1, chi_2   (VAD arousal × |valence|)
│       ├── rb_moral.py              # chi_3          (MFD vice density)
│       ├── rb_outgroup.py           # chi_4          (windowed outgroup-NEG)
│       ├── rb_rhetoric.py           # chi_5          (RhetQ, hyperbole, CAPS, !)
│       ├── rb_info_affect.py        # chi_6          (log A/I)
│       ├── rb_neutralize.py         # chi_7          (counterfactual gap — novel)
│       ├── rb_topic_resid.py        # chi_8          (topic-conditional residual)
│       └── rb_semantic.py           # chi_9, chi_10  (template similarity)
├── tests/comments/                  # 12 fixtures across 4 quadrants
├── compare_features.py              # directional pre-validation
├── requirements.txt
├── README.md                        # you are here
├── DATA_COLLECTION.md
└── MATH_REFERENCE.md
```

---

## 3. Installation (detailed)

### 3.1 Python version

**Python 3.13 is supported**, as are 3.10 / 3.11 / 3.12. All listed
dependencies have 3.13 wheels available on PyPI. In PyCharm:
`File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv
Environment → New`.

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

Models downloaded on first use (cached in `~/.cache/huggingface/`):
- **`gpt2-medium`** (355 MB) — reference LM for ψ₁–ψ₅
- **`google-t5/t5-small`** (240 MB) — perturbation model for ψ₃ (DetectGPT)
- **`paraphrase-multilingual-MiniLM-L12-v2`** (470 MB) — shared embedder for
  χ₇, χ₈, χ₉, χ₁₀
- **`google/flan-t5-base`** (1 GB, *optional*) — instruction-tuned neutralizer
  for χ₇. Only loaded if you set `NEUTRALIZER=flan-t5`. The default
  rule-based neutralizer needs no model.

Total disk footprint with default settings: ~1.1 GB.

### 3.3 CPU vs GPU

Everything runs on CPU. With a CUDA-capable GPU, feature extraction speeds
up roughly 3–5×; the lazy `_load_*` helpers automatically pick `cuda` when
available.

The team's competition data-collection, feature-extraction, and training
runs were done on an **NVIDIA A100 GPU via Google Colab** — see
[`ragebait_train_colab.ipynb`](ragebait_train_colab.ipynb) for the Colab
training notebook.

---

## 4. Data sources

See [`DATA_COLLECTION.md`](DATA_COLLECTION.md) for the full survey. In short:

| Source | What it gives you | Labels |
|---|---|---|
| `Hello-SimpleAI/HC3` | Paired human/ChatGPT short answers | `label_ai ∈ {0,1}`, `label_ragebait = 0` |
| `google/civil_comments` | News-site comments with toxicity scores | `label_ai = 0`, `label_ragebait ∈ {0,1}` (threshold τ = 0.6) |
| `OxAISH-AL-LLM/wiki_toxic` | Wikipedia talk-page toxic labels | `label_ai = 0`, `label_ragebait ∈ {0,1}` |
| Synthetic (Anthropic API) | AI-generated ragebait + substantive | `label_ai = 1`, `label_ragebait ∈ {0,1}` |

The four-quadrant matrix (AI × ragebait) is critical: the two classifiers
need supervision in all four cells to learn that the labels are
**independent**. The `merge` subcommand handles this.

---

## 5. How to interpret the output

A `python -m src.detect --input ...` call prints:

```
AI-generation probability:    0.873
Ragebait probability:         0.912
Joint AI-ragebait risk:       0.796

Top red flags:
  - DetectGPT curvature z-score = 2.31
  - LLM lexical fingerprint hit (psi_6 = 0.52)
  - High moral-vice density (chi_3 = 0.124)
  - Neutralization gap large (chi_7 = 0.61); affect carries the content
  - Semantic match to ragebait template (chi_9 = 0.81)
```

- **p_AI alone high, p_RB low** → AI-generated substantive comment (e.g.
  a customer-support bot answering a policy question). Not ragebait, but
  worth flagging as bot-authored content.
- **p_RB alone high, p_AI low** → human-written outrage post. Not the system's
  primary target but useful for moderation.
- **Both high** → the worst case: an AI ragebait farm. The joint
  `p_AR = p_AI · p_RB` is the primary risk metric.

---

## 6. The novel contribution: counterfactual neutralization (χ₇)

For each comment x we compute a **neutralized rewrite** ν(x) that strips
affective and rhetorical content while preserving any verifiable claims, then
measure:

$$ \chi_7(x) \;=\; (1 - R(x)) \cdot (1 - S(x)) $$

where R = |ν(x)| / |x| is the length ratio and S = cos(φ(x), φ(ν(x))) is the
semantic similarity in embedding space. Ragebait collapses under
neutralization (both ratios → 0, χ₇ → 1); substantive comments survive
(both → 1, χ₇ → 0). This mirrors the curvature argument of DetectGPT but in
the **affective** rather than likelihood dimension.

Two implementations are provided:
- **`rule`** (default, deterministic) — strips affect-lexicon hits, ALL CAPS,
  hyperbole, rhetorical-question patterns. No extra model. Fast.
- **`flan-t5`** (optional, via `NEUTRALIZER=flan-t5`) — prompts Flan-T5-base
  to rewrite the comment in neutral, factual language. Higher quality but
  loads a 1 GB model.

---

## 7. Sanity checks

Before training, run

```bash
python compare_features.py
```

to verify every feature separates the positive class from the negative class
in the expected direction on the 12 fixtures under `tests/comments/`. Any line
that prints `!! WRONG DIRECTION` is a sign/normalization bug worth fixing
before burning compute on training.

To check a single feature:

```bash
python compare_features.py --feature chi_7
```

Heavy modules (perplexity, DetectGPT) can be skipped during fast iteration:

```bash
python compare_features.py --skip ai_likelihood rb_neutralize
```

---

## 8. Results (5-fold cross-validated on a 1,750-comment corpus)

| Stage | ROC AUC | Average Precision | Expected Calibration Error |
|---|---|---|---|
| AI-generation (ψ) | **0.9973** | 0.9881 | 0.0125 |
| Ragebait (χ) | **0.9797** | 0.9697 | 0.0125 |

The AI stage is dominated by the LLM-likelihood family (ψ₁–ψ₅), which
contributes 22.5 AUC points; the lexical-fingerprint feature (ψ₆) adds 4.2
points; punctuation regularity and hedging carry partly-redundant
information that the regularized LR correctly downweights. The ragebait
stage shows a more lopsided ablation in which the semantic family (χ₉, χ₁₀)
contributes 28 AUC points — partly inflated by template-corpus leakage; see
Limitations.

Plots (`images/`) — ROC, precision/recall, reliability, and feature-family
ablation per stage — are generated by `python -m src.evaluate` and are the
ones used in the LaTeX report.

---

## 9. Limitations

- **Template-corpus leakage in the χ ablation.** `data_collection.py
  templates` builds the template set ℛ from the same labeled-ragebait
  corpus used for cross-validation, so each test-fold ragebait comment
  appears in ℛ at cosine ≈ 1 to itself. The 28-point χ_semantic ablation
  delta therefore over-estimates the family's out-of-distribution
  contribution. A holdout-based template split would give a tighter,
  honest estimate (we expect 10–15 points). A `--holdout-frac` flag for
  the `templates` subcommand is the right fix and is left as future work.
- **AI-detection corpus easiness.** The AI-stage 0.997 AUC is achieved on
  a corpus dominated by HC3 (ChatGPT 2022 vs human Q&A) and our Anthropic-
  generated samples. HC3 is a notoriously easy AI-detection benchmark —
  2022-era RLHF outputs have highly distinctive register and discourse
  markers. On harder out-of-distribution AI sources (more recent
  instruction-tuned models, especially with adversarial anti-detection
  prompting), this AUC would drop significantly.
- **English-first.** Lexicons are English. The embedder is multilingual,
  so χ₇/χ₈/χ₉/χ₁₀ generalize across languages, but ψ₆/χ₃/χ₄ would need
  translated lexicons.
- **Adversarial pressure.** A bot that paraphrases its output through a
  second LLM partially defeats ψ₃ and ψ₆. The ragebait features are more
  robust because the rhetorical structure is *the goal* — neutralizing it
  defeats the post.
- **Short text is hard.** Comments under ~8 tokens have so few statistics
  that most likelihood features return NaN; the system imputes the median.
  Account-level aggregation (`src.aggregate`) handles this by pooling many
  short comments.
- **Lexicon-driven features need maintenance.** New AI phrasing tics and
  new outgroup labels appear continuously. Add lines to the `.txt` files
  in `data/` to keep up.

---

## 10. References

Full bibliography in [`MATH_REFERENCE.md`](MATH_REFERENCE.md). Headline
references:

- **DetectGPT** (ψ₃): Mitchell et al., ICML 2023.
- **GLTR** (ψ₄, ψ₅): Gehrmann, Strobelt, Rush, ACL 2019.
- **Burstiness** (ψ₂): Goh & Barabási, EPL 2008.
- **NRC-VAD** (χ₁, χ₂, χ₄): Mohammad, ACL 2018.
- **Moral Foundations Dictionary** (χ₃): Graham, Haidt, Nosek, JPSP 2009;
  eMFD: Hopp et al., 2021.
- **Moral-emotional diffusion** (motivates χ₃, χ₆): Brady et al., PNAS 2017;
  Crockett, Nature Human Behaviour 2017.
- **False-news spread** (motivates the work): Vosoughi, Roy, Aral, Science 2018.
- **HC3 dataset**: Guo et al., 2023.
- **civil_comments dataset**: Borkan et al., WWW 2019.
- **GPT-2** (reference LM): Radford et al., 2019. **T5** (perturbation):
  Raffel et al., JMLR 2020. **MiniLM** (embedder): Wang et al., NeurIPS
  2020 + Reimers & Gurevych, EMNLP 2019.
- **Platt scaling** (calibration): Platt, 1999. **scikit-learn**:
  Pedregosa et al., JMLR 2011.

---

## 11. License

Code: MIT. Data files in `data/`: each lexicon retains its source license.
NRC-VAD and MFD are research-use only; please follow their original terms.
