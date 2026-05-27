"""Data collection orchestrator.

Subcommands:
    hc3        -- Pull paired human/ChatGPT short answers from HuggingFace
                  (Hello-SimpleAI/HC3). Labels: label_ai 0 or 1.
    civil      -- Pull a balanced sample from google/civil_comments.
                  Labels: label_ragebait = 1 if toxicity >= 0.6 else 0.
    wiki_toxic -- Pull from OxAISH-AL-LLM/wiki_toxic.
                  Labels: label_ragebait from the dataset's `label` field.
    synth_ai_rb -- Generate synthetic AI-ragebait via Anthropic API.
                  Labels: label_ai = 1, label_ragebait = 1.
    synth_benign -- Generate synthetic benign comments via Anthropic API.
                  Labels: label_ai = 0/1 (specify), label_ragebait = 0.
    templates  -- Build the ragebait template corpus (data/ragebait_templates.jsonl)
                  by self-extraction from a labeled ragebait corpus.
    merge      -- Combine multiple JSONL files into a single training corpus,
                  shuffling and optionally subsampling.

Outputs JSONL files in data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.comment import Comment, save_corpus

load_dotenv()
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)


def _hash_user(name: str | None) -> str | None:
    if not name:
        return None
    salt = os.environ.get("USERNAME_SALT", "stem-games-2026")
    return "U" + hashlib.sha256((salt + name).encode()).hexdigest()[:10]


# ============================== HC3 ==============================
def collect_hc3(limit: int = 500) -> list[Comment]:
    """Hello-SimpleAI/HC3 — Human ChatGPT Comparison Corpus.

    Each row contains a question, a list of human answers, and a list of
    ChatGPT answers. We emit each individual answer as one Comment,
    truncated to comment-length (8-120 tokens).
    """
    from datasets import load_dataset
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    out: list[Comment] = []
    for row in tqdm(ds, desc="HC3"):
        if len(out) >= limit * 2:
            break
        for text in (row.get("human_answers") or []):
            t = _truncate_comment(text)
            if t:
                out.append(Comment(text=t, label_ai=0, label_ragebait=0,
                                   source=f"HC3:human:{row.get('id', '?')}"))
        for text in (row.get("chatgpt_answers") or []):
            t = _truncate_comment(text)
            if t:
                out.append(Comment(text=t, label_ai=1, label_ragebait=0,
                                   source=f"HC3:chatgpt:{row.get('id', '?')}"))
    random.Random(0).shuffle(out)
    return out[:limit * 2]


def _truncate_comment(text: str, min_tokens: int = 8, max_tokens: int = 120) -> str | None:
    """Take the first sentence(s) up to `max_tokens` tokens; reject if too short."""
    text = (text or "").strip()
    if not text:
        return None
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sents:
        candidate = (out + " " + s).strip() if out else s
        if len(candidate.split()) > max_tokens:
            if not out:
                out = " ".join(candidate.split()[:max_tokens])
            break
        out = candidate
    if len(out.split()) < min_tokens:
        return None
    return out


# ============================== civil_comments ==============================
def collect_civil_comments(limit: int = 1000, toxic_threshold: float = 0.6) -> list[Comment]:
    """google/civil_comments — large news-site comment corpus with toxicity scores.

    Labels:
        label_ai       = 0 (all real human comments)
        label_ragebait = 1 if toxicity >= threshold, else 0
    """
    from datasets import load_dataset
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    pos: list[Comment] = []
    neg: list[Comment] = []
    half = limit // 2
    for row in tqdm(ds, desc="civil_comments"):
        if len(pos) >= half and len(neg) >= half:
            break
        text = (row.get("text") or "").strip()
        if not text:
            continue
        t = _truncate_comment(text)
        if not t:
            continue
        tox = float(row.get("toxicity", 0.0))
        if tox >= toxic_threshold and len(pos) < half:
            pos.append(Comment(text=t, label_ai=0, label_ragebait=1,
                               source=f"civil_comments:tox={tox:.2f}",
                               meta={"toxicity": tox}))
        elif tox < 0.1 and len(neg) < half:
            neg.append(Comment(text=t, label_ai=0, label_ragebait=0,
                               source=f"civil_comments:tox={tox:.2f}",
                               meta={"toxicity": tox}))
    return pos + neg


# ============================== wiki_toxic ==============================
def collect_wiki_toxic(limit: int = 500) -> list[Comment]:
    from datasets import load_dataset
    ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
    pos: list[Comment] = []
    neg: list[Comment] = []
    half = limit // 2
    for row in ds:
        if len(pos) >= half and len(neg) >= half:
            break
        text = (row.get("comment_text") or row.get("text") or "").strip()
        if not text:
            continue
        t = _truncate_comment(text)
        if not t:
            continue
        label = int(row.get("label", 0))
        if label == 1 and len(pos) < half:
            pos.append(Comment(text=t, label_ai=0, label_ragebait=1,
                               source="wiki_toxic"))
        elif label == 0 and len(neg) < half:
            neg.append(Comment(text=t, label_ai=0, label_ragebait=0,
                               source="wiki_toxic"))
    return pos + neg


# ============================== synthetic AI ragebait ==============================
_RAGEBAIT_PROMPT = """You are simulating a comment-section bot farm for academic research \
on detecting AI-generated ragebait. Generate ONE short, inflammatory comment (1-3 sentences, \
under 80 words) on the topic below. The comment should follow the ragebait playbook: \
outgroup framing, moral charge, hyperbole, rhetorical confrontation, and minimal verifiable content. \
Do not use slurs, do not target protected classes by demographic features, and do not include \
real personal information. Topic: {topic}

Output only the comment itself, no preamble."""

_BENIGN_PROMPT = """Write ONE short news-section comment (1-3 sentences, under 80 words) on \
the topic below. The comment should be substantive: it should make a specific claim or share \
a relevant personal observation, with a measured tone. Avoid hyperbole and rhetorical questions. \
Topic: {topic}

Output only the comment itself, no preamble."""

_TOPICS = [
    "Parliament debates pension reform",
    "Local school board rejects new curriculum",
    "Tech company announces major layoffs",
    "City approves bike-lane expansion",
    "Election commission certifies vote count",
    "Court rules on immigration policy",
    "Health agency updates vaccine guidance",
    "Mayor proposes property-tax change",
    "Regulator fines major bank",
    "Power outage hits the capital",
    "University announces new admissions policy",
    "Sports league discusses rule changes",
    "Climate summit concludes with new agreement",
    "Police union responds to oversight bill",
    "Streaming service raises subscription prices",
]


def _anthropic_one(prompt: str) -> str | None:
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")
    client = Anthropic(api_key=key)
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            print(f"  Anthropic attempt {attempt + 1} failed: {e}")
            time.sleep(2 + attempt * 2)
    return None


def synth_ai_ragebait(n: int = 100) -> list[Comment]:
    out: list[Comment] = []
    rng = random.Random(0)
    for _ in tqdm(range(n), desc="synth AI ragebait"):
        topic = rng.choice(_TOPICS)
        prompt = _RAGEBAIT_PROMPT.format(topic=topic)
        text = _anthropic_one(prompt)
        if text:
            out.append(Comment(
                text=text,
                label_ai=1,
                label_ragebait=1,
                parent_topic=topic,
                source="synth:anthropic:ragebait",
            ))
    return out


def synth_benign(n: int = 100, ai: bool = True) -> list[Comment]:
    """Generate substantive comments. If `ai=True`, label as AI-generated (these
    are AI-substantive); else mark as human (use sparingly -- only if you'll
    verify by hand)."""
    out: list[Comment] = []
    rng = random.Random(1)
    for _ in tqdm(range(n), desc="synth benign"):
        topic = rng.choice(_TOPICS)
        prompt = _BENIGN_PROMPT.format(topic=topic)
        text = _anthropic_one(prompt)
        if text:
            out.append(Comment(
                text=text,
                label_ai=1 if ai else 0,
                label_ragebait=0,
                parent_topic=topic,
                source="synth:anthropic:substantive",
            ))
    return out


# ============================== templates ==============================
def build_templates(input_jsonl: Path, out_path: Path, min_tokens: int = 8,
                    max_tokens: int = 80) -> None:
    """Self-extract ragebait templates from a labeled corpus.

    Take every comment with label_ragebait == 1 and length in [min, max] tokens,
    deduplicate, and write to JSONL.
    """
    from src.comment import load_corpus
    corpus = load_corpus(input_jsonl)
    seen: set[str] = set()
    templates = []
    for c in corpus:
        if c.label_ragebait != 1:
            continue
        n = len(c.text.split())
        if not (min_tokens <= n <= max_tokens):
            continue
        key = c.text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        templates.append({"text": c.text.strip(), "source": c.source})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in templates:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {len(templates)} templates -> {out_path}")


# ============================== merge ==============================
def merge_corpora(inputs: list[Path], out_path: Path,
                  shuffle_seed: int = 0,
                  max_total: int | None = None) -> None:
    from src.comment import load_corpus
    all_comments: list[Comment] = []
    for p in inputs:
        if p.exists():
            cs = load_corpus(p)
            print(f"  {p}: {len(cs)} comments")
            all_comments.extend(cs)
        else:
            print(f"  {p}: MISSING")
    random.Random(shuffle_seed).shuffle(all_comments)
    if max_total:
        all_comments = all_comments[:max_total]
    save_corpus(all_comments, out_path)
    print(f"Wrote {len(all_comments)} comments -> {out_path}")


# ============================== CLI ==============================
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_hc3 = sub.add_parser("hc3"); p_hc3.add_argument("--n", type=int, default=500)
    p_civ = sub.add_parser("civil"); p_civ.add_argument("--n", type=int, default=1000)
    p_civ.add_argument("--threshold", type=float, default=0.6)
    p_wt = sub.add_parser("wiki_toxic"); p_wt.add_argument("--n", type=int, default=500)
    p_sar = sub.add_parser("synth_ai_rb"); p_sar.add_argument("--n", type=int, default=100)
    p_sb = sub.add_parser("synth_benign"); p_sb.add_argument("--n", type=int, default=100)
    p_sb.add_argument("--human", action="store_true", help="Mark as label_ai=0 (only with manual review)")

    p_tpl = sub.add_parser("templates")
    p_tpl.add_argument("--input", required=True, type=Path)
    p_tpl.add_argument("--out", type=Path, default=Path("data/ragebait_templates.jsonl"))

    p_mrg = sub.add_parser("merge")
    p_mrg.add_argument("--inputs", nargs="+", required=True, type=Path)
    p_mrg.add_argument("--out", required=True, type=Path)
    p_mrg.add_argument("--max-total", type=int, default=None)

    args = ap.parse_args()

    if args.cmd == "hc3":
        save_corpus(collect_hc3(args.n), OUT_DIR / "hc3.jsonl")
    elif args.cmd == "civil":
        save_corpus(collect_civil_comments(args.n, args.threshold), OUT_DIR / "civil_comments.jsonl")
    elif args.cmd == "wiki_toxic":
        save_corpus(collect_wiki_toxic(args.n), OUT_DIR / "wiki_toxic.jsonl")
    elif args.cmd == "synth_ai_rb":
        save_corpus(synth_ai_ragebait(args.n), OUT_DIR / "synth_ai_ragebait.jsonl")
    elif args.cmd == "synth_benign":
        save_corpus(synth_benign(args.n, ai=not args.human), OUT_DIR / "synth_benign.jsonl")
    elif args.cmd == "templates":
        build_templates(args.input, args.out)
    elif args.cmd == "merge":
        merge_corpora(args.inputs, args.out, max_total=args.max_total)


if __name__ == "__main__":
    main()
