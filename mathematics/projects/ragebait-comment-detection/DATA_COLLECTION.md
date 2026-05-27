# Data collection guide

This project trains two independent classifiers, so it needs supervision on
two independent labels per comment. The data-collection module
(`src/data_collection.py`) pulls from a mix of public Hugging Face datasets
and Anthropic-generated synthetic data, normalizes everything to the
`Comment` schema, and saves to JSONL files in `data/`.

---

## 1. The four quadrants

| Quadrant | `label_ai` | `label_ragebait` | Example source |
|---|---|---|---|
| AI × ragebait | 1 | 1 | `synth_ai_rb` (Anthropic API) |
| AI × substantive | 1 | 0 | `hc3` (ChatGPT side) or `synth_benign` |
| Human × ragebait | 0 | 1 | `civil` (toxicity ≥ 0.6) or `wiki_toxic` (label=1) |
| Human × substantive | 0 | 0 | `hc3` (human side), `civil` (toxicity < 0.1) |

A balanced training set covers all four cells. Imbalance is OK — the
classifiers handle it via `class_weight='balanced'` in
`LogisticRegressionCV` — but **at least one of each cell must be present**
or one classifier will refuse to fit. The minimum I have run successfully:
~100 examples per cell.

---

## 2. Public Hugging Face datasets

All accessed via `datasets.load_dataset`. None require authentication for
the default configurations used here.

### 2.1 `Hello-SimpleAI/HC3`

> *Guo, Zhang, Wang et al. "How Close is ChatGPT to Human Experts? Comparison
> Corpus, Evaluation, and Detection." 2023.*

Paired human and ChatGPT answers to ~24,000 questions from finance, medicine,
open-domain QA, and Wikipedia. We emit each answer as one `Comment`,
truncated to 8–120 tokens.

```bash
python -m src.data_collection hc3 --n 500
```

Produces `data/hc3.jsonl` with ~500 human and ~500 ChatGPT examples, all
labeled `label_ragebait = 0` (Q&A answers are by construction substantive).

### 2.2 `google/civil_comments`

> *Borkan, Dixon, Sorensen et al. "Nuanced Metrics for Measuring Unintended
> Bias with Real Data for Text Classification." WWW 2019.*

1.8M news-site comments with toxicity scores in [0, 1]. We use:
- `toxicity ≥ 0.6` → `label_ragebait = 1`
- `toxicity < 0.1` → `label_ragebait = 0`

The middle band (0.1–0.6) is ambiguous and dropped. **Note**: toxicity ≠
ragebait perfectly, but high-toxicity short news comments are the closest
publicly available proxy. The dataset is streamed to avoid downloading the
full 1.5 GB.

```bash
python -m src.data_collection civil --n 1000 --threshold 0.6
```

### 2.3 `OxAISH-AL-LLM/wiki_toxic`

A 159 K-comment slice of the Jigsaw competition data, restricted to
Wikipedia talk pages. Simpler binary label.

```bash
python -m src.data_collection wiki_toxic --n 500
```

### 2.4 Other datasets worth knowing about

We did not wire these in, but they are easy drop-ins via a few lines in
`data_collection.py`:

- **`tasksource/jigsaw_toxicity_pred`** — the full Jigsaw competition data
  with severe-toxic, obscene, threat, insult, identity-hate sub-labels.
- **`liamdugan/raid`** — the RAID benchmark (Dugan et al. 2024). Larger
  but heavier than HC3, covers many generator models and domains.
- **`artem9k/ai-text-detection-pile`** — assorted human/AI text from
  multiple domains.
- **`mediabiasgroup/BABE`** — bias annotations on news sentences. Useful
  for the rhetorical-pattern side of ragebait.

---

## 3. Synthetic data via Anthropic API

For the AI × ragebait quadrant, public datasets are thin: ChatGPT in 2023
was actively trained *not* to produce ragebait. We generate synthetic
examples directly via `claude-haiku-4-5-20251001`, prompted with an explicit
ragebait playbook.

```bash
python -m src.data_collection synth_ai_rb --n 200
python -m src.data_collection synth_benign --n 200
```

The prompts are in `src/data_collection.py`:

- `_RAGEBAIT_PROMPT` — instructs the model to follow outgroup framing,
  moral charge, hyperbole, rhetorical confrontation, minimal verifiable
  content. Explicitly forbids slurs and demographic targeting.
- `_BENIGN_PROMPT` — substantive, measured tone, specific claim or relevant
  personal observation.

Topics rotate over a pool of 15 realistic news-comment-section topics
(pension reform, court rulings, climate summit, etc.).

### 3.1 Ethics of synthetic ragebait

Generating ragebait *to detect ragebait* is the same ethical posture as
generating malware samples to train antivirus. We:

- Avoid slurs and demographic targeting in the prompt and review the output
  (the dataset is small enough that spot-checking is feasible).
- Restrict to **mild** outgroup terms (political, generational,
  institutional). Demographic identity terms are out of scope and already
  handled by hate-speech classifiers.
- Do **not** publish the generated ragebait corpus; it is regenerable from
  the prompts in this repo.

---

## 4. Real Reddit / scraping (optional, not wired in)

If you want real news-comment-style data, three options worth knowing:

- **Reddit** (r/news, r/worldnews, r/politics): use `praw` with the
  credentials in `.env`. Top-of-thread comments on hot posts have a usable
  ragebait baseline. Same parsing helpers as the romance-scam project.
  Not wired in here, but `praw` is in `requirements.txt`.
- **Pushshift Reddit dumps**: 1 TB+ of historical Reddit available at
  https://files.pushshift.io. Heavy but the gold standard for large-scale
  comment corpora.
- **News-site scraping**: usually against ToS. We do not recommend.

---

## 5. The template corpus (for χ₉, χ₁₀)

The semantic features compare each comment against a corpus R of confirmed
ragebait templates. To bootstrap R, take confirmed ragebait comments from
your labeled corpus and dedupe:

```bash
python -m src.data_collection templates \
    --input data/corpus.jsonl \
    --out data/ragebait_templates.jsonl
```

This keeps comments with `label_ragebait == 1` and length 8–80 tokens. A
seed of 25 hand-written templates ships with the repo, so χ₉/χ₁₀ produce
useful values even before training data is available.

---

## 6. Lexicons

### 6.1 NRC-VAD (for χ₁, χ₂, χ₄)

> *Mohammad. "Obtaining Reliable Human Ratings of Valence, Arousal, and
> Dominance for 20,000 English Words." ACL 2018.*

A curated ~300-entry subset ships at `data/vad_subset.csv` (valence already
mapped to [-1, 1]). This is enough for the system to produce reasonable
values out of the box.

To use the full lexicon (~20 K entries, much better coverage):

1. Visit http://saifmohammad.com/WebPages/nrc-vad.html
2. Download `NRC-VAD-Lexicon.txt` (research use, free)
3. Place at `data/NRC-VAD-Lexicon.txt`

`src/features/rb_affect.py` auto-detects the full file and uses it
preferentially.

### 6.2 Moral Foundations Dictionary (for χ₃)

> *Graham, Haidt, Nosek. "Liberals and Conservatives Rely on Different
> Sets of Moral Foundations." JPSP 2009.* Plus eMFD: Hopp, Fisher,
> Cornell et al. 2020.

A curated ~140-entry vice subset ships at `data/mfd_vice.txt`. To use the
full MFD2:

1. Visit https://moralfoundations.org/other-materials/
2. Download `MFD2.dic` (LIWC `.dic` format)
3. Place at `data/MFD2.dic`

`src/features/rb_moral.py` parses both the legacy MFD1 (`HarmVice`,
`FairnessVice`, etc.) and MFD2 (`Harm.vice`, etc.) category labels.

### 6.3 In-repo editable lexicons

These are plain-text supplements that **add to** the built-in regex lists
in the feature code (so you cannot break the detector by deleting them):

- `data/ai_lexicon.txt` — additional LLM phrase patterns
- `data/hedge_lexicon.txt` — additional hedging phrases
- `data/outgroup_lexicon.txt` — additional outgroup markers

Each line is a regex; lines starting with `#` are comments.

---

## 7. Privacy & anonymization

The `Comment` schema's `username` field is intended to hold a **salted
hash**, not a plaintext handle. `_hash_user()` in `src/data_collection.py`
takes the raw username, prepends `$USERNAME_SALT` (default
`stem-games-2026`, override in `.env`), SHA-256s, and keeps the first 10
hex characters. This is enough for account-level aggregation while making
the raw handle non-recoverable.

If you scrape Reddit / forums, run handles through `_hash_user()` before
saving to JSONL.

---

## 8. Recommended training corpus composition

For a fairly balanced setup (≈2 K total comments, ~1 hour of feature
extraction on a fast CPU):

```bash
python -m src.data_collection hc3 --n 500           # 500 AI×subst + 500 H×subst
python -m src.data_collection civil --n 800         # 400 H×RB + 400 H×subst
python -m src.data_collection synth_ai_rb --n 200   # 200 AI×RB
python -m src.data_collection synth_benign --n 200  # 200 AI×subst (extra)

python -m src.data_collection merge \
    --inputs data/hc3.jsonl data/civil_comments.jsonl \
             data/synth_ai_ragebait.jsonl data/synth_benign.jsonl \
    --out data/corpus.jsonl
```

Approximate cell distribution:
- AI × ragebait:        200
- AI × substantive:     700
- Human × ragebait:     400
- Human × substantive:  900

The AI × ragebait cell is the smallest because it relies on Anthropic API
calls. Scale `synth_ai_rb --n` higher if you have credits.

---

## 9. Bibliography

Datasets and lexicons used or referenced in this guide:

- **HC3**: Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue,
  J., Wu, Y. *How Close is ChatGPT to Human Experts? Comparison Corpus,
  Evaluation, and Detection.* arXiv:2301.07597, 2023.
  https://huggingface.co/datasets/Hello-SimpleAI/HC3
- **civil_comments**: Borkan, D., Dixon, L., Sorensen, J., Thain, N.,
  Vasserman, L. *Nuanced Metrics for Measuring Unintended Bias with Real
  Data for Text Classification.* WWW 2019.
  https://huggingface.co/datasets/google/civil_comments
- **wiki_toxic**: cjadams, Sorensen, J., Elliott, J., Dixon, L., McDonald,
  M., et al. *Jigsaw Toxic Comment Classification Challenge.* Kaggle, 2017.
  https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic
- **RAID** (mentioned, not wired in): Dugan, L., Hwang, A., Trhlik, F.,
  Ludan, J. M., Zhu, J. M., Xu, H., Ippolito, D., Callison-Burch, C. *RAID:
  A Shared Benchmark for Robust Evaluation of Machine-Generated Text
  Detectors.* ACL 2024. https://huggingface.co/datasets/liamdugan/raid
- **BABE** (mentioned, not wired in): Spinde, T., Plank, M., Krieger, J.-D.,
  Ruas, T., Gipp, B., Aizawa, A. *Neural Media Bias Detection Using
  Distant Supervision With BABE.* EMNLP Findings 2021.
- **NRC-VAD lexicon**: Mohammad, S. *Obtaining Reliable Human Ratings of
  Valence, Arousal, and Dominance for 20,000 English Words.* ACL 2018.
  http://saifmohammad.com/WebPages/nrc-vad.html
- **Moral Foundations Dictionary**: Graham, J., Haidt, J., Nosek, B. A.
  *Liberals and Conservatives Rely on Different Sets of Moral
  Foundations.* JPSP 96(5), 2009. **eMFD**: Hopp, F. R., Fisher, J. T.,
  Cornell, D., Huskey, R., Weber, R., 2021.
  https://moralfoundations.org/other-materials/

Full bibliography of all methodology, model, and theory references is in
[`MATH_REFERENCE.md`](MATH_REFERENCE.md).
