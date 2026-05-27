# Data collection guide

This document explains the four data sources, the Selenium scraper in detail,
and the ethics policy you should be ready to defend in the report.

---

## 1. Data sources at a glance

| Source | Tool | Target | Time | Cost |
|---|---|---|---|---|
| **Synthetic scam conversations** | `data_collection.py synth` | 200 conversations | 30 min | ~$5 (Anthropic) |
| **Synthetic benign conversations** | `data_collection.py benign` | 200 conversations | 30 min | ~$5 (Anthropic) |
| **Reddit pasted chats** | `data_collection.py reddit` | 50–200 conversations | 10 min | free (Reddit API) |
| **Public scam archives** | `scrape_selenium.py` | 100–500 letters | 30 min | free |

**Minimum viable dataset** for Day 1: just `synth` + `benign`. Train on
~400 conversations and you'll see meaningful AUC numbers.

**Strong submission dataset** for Day 2: all four sources combined.

---

## 2. Synthetic conversations (recommended baseline)

`src/data_collection.py synth` calls the Anthropic API with a fixed system
prompt encoding the 5-stage scam playbook. Each call generates one
conversation in JSON. We vary the *flavor* (pig-butchering crypto, military,
widower, oil-rig, doctor-customs) so the model can't collapse to a single
template.

```bash
python -m src.data_collection synth --n 200
```

Output: `data/scam_synthetic.jsonl`.

Benign conversations use the same schema but a friendly-chat prompt:

```bash
python -m src.data_collection benign --n 200
```

Output: `data/benign_corpus.jsonl`.

**Why this works as training data.** Even though our test conversations
will look different in detail, the *structural* signals (stage progression,
length asymmetry, late-conversation finance lexicon, sentiment CUSUM) are
robust across flavors. The risk is that the LLM-likelihood features over-fit
to "what Claude writes" — which is why we deliberately use a *different*
reference model (GPT-2 medium) for perplexity computation.

---

## 3. Reddit (real, but messy)

`r/Scams`, `r/romancescam`, `r/scambait` regularly contain victims pasting
real conversations. The parser at `src/data_collection.py:_parse_pasted_chat`
heuristically recognizes:

```
Him: blah blah
Me: short reply
Him: longer message
```

and similar variations (`scammer:`, `target:`, `victim:`). Lines without
speaker tags are appended to the previous message.

**Setup.**

1. Create a Reddit app at https://www.reddit.com/prefs/apps. Choose
   *"script"*. Copy the client ID (under the app name) and the secret.
2. Put them in `.env`:
   ```
   REDDIT_CLIENT_ID=...
   REDDIT_CLIENT_SECRET=...
   REDDIT_USER_AGENT=stem-games-bot-detector/0.1
   ```
3. Run:
   ```bash
   python -m src.data_collection reddit --limit 200
   ```

**Yield is low.** Of 200 posts, expect maybe 30–60 to actually contain a
parseable chat. The rest will be screenshots (no text), prose descriptions,
or just questions. Run repeatedly across `top`, `hot`, and `new` for
better coverage; you can edit `subreddits=(...)` in `collect_reddit`.

---

## 4. Selenium scraper

`src/scrape_selenium.py` is a configurable Chrome scraper. The default
target is `scamletters.com`, a long-running public archive of romance-scam
letters.

### 4.1 Basic usage

```bash
# Scrape 3 listing pages from scamletters.com (default site)
python -m src.scrape_selenium --pages 3

# Convert the raw output into conversation JSONL
python -m src.scrape_selenium --postprocess
```

This produces:
- `data/scam_letters_raw.jsonl` — one record per scraped page (`url`, `title`, `text`)
- `data/scam_letters_corpus.jsonl` — sentence-grouped pseudo-conversations
  in the same schema as everything else

### 4.2 Adding a new site

Open `src/scrape_selenium.py` and find the `SITES` dictionary:

```python
SITES = {
    "scamletters": {
        "start_url_template": "https://www.scamletters.com/scam-letters/page/{page}/",
        "listing_link_selector": "h2.entry-title a",
        "article_body_selector": "div.entry-content",
        "article_title_selector": "h1.entry-title",
    },
    "generic": {...},
}
```

Add an entry for any site with the same four keys. The selectors are
standard CSS selectors — find them with Chrome DevTools (right-click →
Inspect → copy selector, then trim to what's stable).

Then run:

```bash
python -m src.scrape_selenium --site your_new_site --pages 5
```

For one-off cases without modifying code, use the generic site:

```bash
python -m src.scrape_selenium --site generic --start-url https://some.site/listing --pages 1
```

### 4.3 Other targets worth trying

You'd configure these similarly. **Always check robots.txt and TOS first.**

| Site | Notes |
|---|---|
| `romancescamsnow.com` | Heavy IP archive of romance scam profiles |
| `scam-detector.com` | Aggregated scam reports |
| `bbb.org/scamtracker` | BBB Scam Tracker (US); has structured fields |
| `aarp.org/money/scams-fraud/` | AARP fraud archive |
| Trustpilot reviews of known scam dating sites | Variable yield; many victims paste excerpts |

### 4.4 Debug mode

```bash
python -m src.scrape_selenium --no-headless --pages 1
```

This launches a visible Chrome window so you can see what the selectors
are matching. Useful when a site's HTML differs from what you expected.

### 4.5 Resume / cache

The scraper writes `data/scam_letters_raw.jsonl` in append mode and
remembers every URL it has already scraped. Re-running picks up where
you left off — safe to interrupt with Ctrl-C.

---

## 5. Template corpus for semantic features

After collecting your scam corpus, build the template database that
`features/semantic.py` compares against:

```bash
python -m src.data_collection templates \
    --scam-jsonl data/scam_synthetic.jsonl \
    --out data/scam_templates.jsonl
```

Repeat for each scam JSONL you have, merging into one file:

```bash
python -m src.data_collection templates --scam-jsonl data/scam_synthetic.jsonl --out data/templates_a.jsonl
python -m src.data_collection templates --scam-jsonl data/scam_reddit.jsonl    --out data/templates_b.jsonl
python -m src.data_collection templates --scam-jsonl data/scam_letters_corpus.jsonl --out data/templates_c.jsonl
type data\templates_*.jsonl > data\scam_templates.jsonl
```

On Linux/macOS use `cat` instead of `type`.

The template file is loaded once and embedded at startup of `semantic.py`.
Larger corpus = stronger φ₁₇/φ₁₈/φ₁₉ signal, with diminishing returns
after ~5000 templates.

---

## 6. Ethics & TOS (be ready to discuss this in the report)

Defend the data collection in the report:

1. **Public information only.** Reddit posts, archive sites, and synthetic
   data — no private messages, no logins bypassed.
2. **Respect robots.txt.** The Selenium scraper adds a 2-second delay
   between requests by default. Don't override that for production
   scraping; only adjust during testing on small page counts.
3. **No PII in the published dataset.** Reddit usernames and any phone
   numbers/email addresses in scraped letters should be redacted before
   inclusion in your final corpus. The current pipeline keeps them; a
   redaction pass (regex + manual review) is recommended for any
   redistribution.
4. **Identifiable user agent.** The scraper announces itself as
   `stem-games-bot-detector/0.1 academic-research`. Don't change this to
   impersonate a regular browser — that's a hostility flag.
5. **Synthetic data is your friend.** It avoids most of the above
   concerns. State in the report that the bulk of your training data is
   synthetic and that real-world data is used only for evaluation /
   template extraction.

---

## 7. Provenance — what goes in the report

For the data section of the LaTeX report, the table should look like this
(replace numbers after collection):

| Source | # Conversations | # Messages | Median length | Provenance |
|---|---|---|---|---|
| Anthropic synthetic (scam) | 200 | 7,800 | 38 msgs | LLM generation, 5 flavors |
| Anthropic synthetic (benign) | 200 | 7,500 | 36 msgs | LLM generation, 5 scenarios |
| Reddit r/Scams + r/romancescam | 45 | 720 | 14 msgs | PRAW pull from top-of-year |
| scamletters.com | 180 | 1,600 | 9 pseudo-msgs | Selenium scrape, redacted |
| **Total scam** | 425 | 10,120 | — | — |
| **Total benign** | 200 | 7,500 | — | — |

Combined with a 5-fold CV evaluation, this is enough for AUC numbers
that hold up.
