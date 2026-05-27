"""Data collection orchestrator.

Subcommands:
    reddit     -- pull posts from r/Scams + r/romancescam (PRAW API)
    synth      -- generate synthetic scam conversations via Anthropic LLM
    benign     -- download PERSONA-CHAT and synthesize benign dating chats

Outputs JSONL files in data/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.conversation import Conversation, Message, save_corpus

load_dotenv()

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- Reddit
def collect_reddit(subreddits=("Scams", "romancescam", "scambait"),
                   limit: int = 200) -> list[Conversation]:
    import praw  # imported here so missing creds don't break the whole module

    cid = os.environ.get("REDDIT_CLIENT_ID")
    csec = os.environ.get("REDDIT_CLIENT_SECRET")
    ua = os.environ.get("REDDIT_USER_AGENT", "stem-games-bot-detector/0.1")
    if not cid or not csec:
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")

    reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
    convs: list[Conversation] = []
    for sub in subreddits:
        print(f"Pulling r/{sub}...")
        for post in tqdm(reddit.subreddit(sub).top(limit=limit, time_filter="year")):
            text = (post.selftext or "").strip()
            if not text:
                continue
            messages = _parse_pasted_chat(text)
            if len(messages) >= 6:
                convs.append(Conversation(
                    messages=messages,
                    label=1,
                    source=f"reddit.com/r/{sub}/{post.id}",
                ))
    return convs


_LINE_RE = re.compile(
    r"^(?P<speaker>(?:him|her|me|he|she|i|you|them|scammer|victim|target|me:|him:|her:))\b[:\-]?\s*(?P<text>.+)$",
    re.IGNORECASE,
)


def _parse_pasted_chat(text: str) -> list[Message]:
    """Heuristic parser for reddit-pasted chats.

    Heuristic: lines starting with him/her/me/scammer/victim are speaker-tagged.
    We map the first non-me speaker -> S, "me" -> V. Lines without tags get
    attached to the last speaker.
    """
    base_ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    messages: list[Message] = []
    other_label = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if m:
            speaker_raw = m.group("speaker").lower().rstrip(":-").strip()
            content = m.group("text").strip()
            if speaker_raw in {"me", "victim", "target", "i", "you", "me:"}:
                speaker = "V"
            else:
                if other_label is None:
                    other_label = speaker_raw
                speaker = "S"
            ts = base_ts + timedelta(minutes=10 * len(messages))
            messages.append(Message(speaker=speaker, text=content, timestamp=ts))
        elif messages and len(line) > 3:
            messages[-1] = Message(
                speaker=messages[-1].speaker,
                text=messages[-1].text + " " + line,
                timestamp=messages[-1].timestamp,
            )
    return messages


# ---------------------------------------------------------------- Synthetic
_SYSTEM_PROMPT = """You are simulating a romance-scam chatbot conversation for AI-safety research at a university.
Generate a conversation in JSON of 30-50 messages that follows this 5-stage playbook:
  1. Opening / love-bombing  (5-8 messages)
  2. Rapport / daily small talk  (10-15 messages)
  3. Isolation - discourage talking to family  (3-5 messages)
  4. Hook - casual mention of an investment / sick relative / customs fee  (3-5 messages)
  5. Urgency - time-limited money request, specific transfer instructions  (5-10 messages)

Make the scammer's writing slightly over-polished, with mild grammatical perfection.
Make the victim's writing more natural, with occasional typos and short replies.

Output ONLY valid JSON, a list of {"speaker": "S" or "V", "text": "..."} objects.
No commentary, no markdown.
"""


def synth_anthropic(n: int = 200) -> list[Conversation]:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")
    client = anthropic.Anthropic(api_key=api_key)
    variants = [
        "pig-butchering crypto investment scam",
        "military romance scam with deployed soldier persona",
        "wealthy widower with sick relative storyline",
        "oil rig engineer stranded overseas",
        "doctor working overseas needing customs release fee",
    ]
    convs: list[Conversation] = []
    for i in tqdm(range(n)):
        flavor = variants[i % len(variants)]
        try:
            resp = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4000,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Variant: {flavor}. Generate now."}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?", "", text)
                text = re.sub(r"```$", "", text).strip()
            messages_raw = json.loads(text)
            base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
            messages = [
                Message(
                    speaker=m["speaker"],
                    text=m["text"],
                    timestamp=base + timedelta(minutes=15 * j + random.randint(0, 5)),
                )
                for j, m in enumerate(messages_raw)
            ]
            convs.append(Conversation(messages=messages, label=1,
                                      source=f"synthetic:anthropic:{flavor}"))
        except Exception as e:  # noqa: BLE001
            print(f"[synth] skipped: {e}")
            time.sleep(2)
    return convs


# ---------------------------------------------------------------- Benign
def synth_benign_anthropic(n: int = 200) -> list[Conversation]:
    """Generate benign dating-app style chats with the same speaker schema."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env")
    client = anthropic.Anthropic(api_key=api_key)
    flavors = [
        "two people who matched on a dating app planning a first date",
        "long-distance couple chatting about their week",
        "friends catching up after a long time apart",
        "two coworkers becoming friends",
        "a parent and adult child texting about weekend plans",
    ]
    sys = (
        "Generate a realistic, fully benign conversation of 30-50 messages between two people. "
        "Output ONLY a JSON list of {\"speaker\": \"S\" or \"V\", \"text\": \"...\"} objects. "
        "Use natural typos, varied length, no scams, no money requests."
    )
    convs: list[Conversation] = []
    for i in tqdm(range(n)):
        flavor = flavors[i % len(flavors)]
        try:
            resp = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4000,
                system=sys,
                messages=[{"role": "user", "content": f"Scenario: {flavor}. Generate now."}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?", "", text)
                text = re.sub(r"```$", "", text).strip()
            messages_raw = json.loads(text)
            base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
            messages = [
                Message(
                    speaker=m["speaker"],
                    text=m["text"],
                    timestamp=base + timedelta(minutes=15 * j + random.randint(0, 5)),
                )
                for j, m in enumerate(messages_raw)
            ]
            convs.append(Conversation(messages=messages, label=0,
                                      source=f"synthetic:benign:{flavor}"))
        except Exception as e:  # noqa: BLE001
            print(f"[benign] skipped: {e}")
            time.sleep(2)
    return convs


# ---------------------------------------------------------------- Hugging Face datasets

# Speaker tags we recognize. Matches both BothBosu's "Person A:" inline style
# and the multi-agent "Suspect:" / "Innocent:" style. Used with re.split() so
# capturing the tag preserves it in the output list.
_HF_SPEAKER_SPLIT_RE = re.compile(
    r"\b(person\s+[ab]|suspect|innocent|scammer|victim|target|caller|receiver|user|bot)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _hf_map_speaker(raw: str) -> str:
    """Map a raw speaker tag to our 'S' / 'V' schema."""
    raw = raw.strip().lower()
    # Person A by convention is the suspected scammer in BothBosu
    if raw.startswith("person a") or raw in {"suspect", "scammer", "caller", "bot"}:
        return "S"
    return "V"


def _hf_parse_dialogue_string(raw: str) -> list[Message]:
    """Parse a dialogue string into Messages.

    Handles both inline and newline-separated formats:
        "Person A: hello Person B: hi Person A: bye"      (BothBosu inline)
        "Suspect: hello\\nInnocent: hi\\nSuspect: bye"     (multi-agent newline)
        "Scammer: hello\\nVictim: hi"                      (reddit-paste style)
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    parts = _HF_SPEAKER_SPLIT_RE.split(raw)
    # parts = [text_before_first_speaker, speaker_1, content_1, speaker_2, content_2, ...]
    if len(parts) < 3:
        return []

    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    messages: list[Message] = []
    for i in range(1, len(parts) - 1, 2):
        speaker_raw = parts[i]
        content = parts[i + 1].strip()
        if not content:
            continue
        speaker = _hf_map_speaker(speaker_raw)
        ts = base + timedelta(minutes=15 * len(messages))
        messages.append(Message(speaker=speaker, text=content, timestamp=ts))
    return messages


def _hf_row_to_conversation(row: dict, label: int, source: str) -> Conversation | None:
    """Heuristically convert a row from any BothBosu CSV into our Conversation type."""
    # Find the column that holds the dialogue text. Try common names first.
    candidates = ["dialogue", "conversation", "text", "content"]
    raw = None
    for k in candidates:
        if k in row and isinstance(row[k], str) and row[k]:
            raw = row[k]
            break
    if raw is None:
        # Fallback: longest string-valued cell in the row
        str_vals = [(k, v) for k, v in row.items() if isinstance(v, str) and len(v) > 20]
        if not str_vals:
            return None
        raw = max(str_vals, key=lambda kv: len(kv[1]))[1]

    messages = _hf_parse_dialogue_string(raw)
    if len(messages) < 4:
        return None
    return Conversation(messages=messages, label=label, source=source)


def load_hf_bothbosu(dataset_name: str = "BothBosu/Scammer-Conversation",
                    max_per_class: int | None = None) -> tuple[list[Conversation], list[Conversation]]:
    """Load BothBosu/Scammer-Conversation (or sibling) and split into scam vs benign.

    The dataset has two CSV files:
        gen_conver_scamIdentifier_1000.csv  -> scam (label = 1)
        gen_conver_noIdentifier_1000.csv    -> normal (label = 0)
    Hugging Face presents them as separate splits or as one split with a label column;
    we handle both cases.

    Returns (scam_convs, benign_convs).
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name} from Hugging Face...")
    ds = load_dataset(dataset_name)
    print(f"  Available splits: {list(ds.keys())}")
    # Inspect first row of first split so user can see what's there
    first_split = next(iter(ds.values()))
    print(f"  First row keys: {list(first_split[0].keys())}")
    print(f"  First row preview: {str(first_split[0])[:200]}...")

    scam_convs: list[Conversation] = []
    benign_convs: list[Conversation] = []

    for split_name, split in ds.items():
        # Decide the label for this split.
        # Common patterns: split has a 'label' column, or the split name encodes the class.
        infer_label_from_split = None
        sname = split_name.lower()
        if "scam" in sname and "no" not in sname:
            infer_label_from_split = 1
        elif "no" in sname or "benign" in sname or "normal" in sname:
            infer_label_from_split = 0

        for i, row in enumerate(tqdm(split, desc=f"  {split_name}")):
            # Determine row label
            if "label" in row and row["label"] is not None:
                try:
                    label = int(row["label"])
                except (TypeError, ValueError):
                    label = 1 if str(row["label"]).lower().startswith(("scam", "1", "true")) else 0
            elif infer_label_from_split is not None:
                label = infer_label_from_split
            else:
                # Fallback: filename / source hint
                label = 1 if "scam" in dataset_name.lower() else 0

            conv = _hf_row_to_conversation(row, label=label,
                                           source=f"hf:{dataset_name}:{split_name}:{i}")
            if conv is None:
                continue

            if label == 1:
                if max_per_class is None or len(scam_convs) < max_per_class:
                    scam_convs.append(conv)
            else:
                if max_per_class is None or len(benign_convs) < max_per_class:
                    benign_convs.append(conv)

            if max_per_class is not None and \
                    len(scam_convs) >= max_per_class and \
                    len(benign_convs) >= max_per_class:
                break

    print(f"Parsed {len(scam_convs)} scam and {len(benign_convs)} benign conversations.")
    return scam_convs, benign_convs



def extract_templates(scam_jsonl: Path, out_path: Path) -> None:
    """Take all scammer-side messages from the scam corpus and write a JSONL of
    {text: ...} entries. Used by features/semantic.py."""
    from src.conversation import load_corpus
    convs = load_corpus(scam_jsonl)
    seen = set()
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for c in convs:
            for m in c.scammer_messages:
                t = m.text.strip()
                if 8 <= len(t.split()) <= 80 and t not in seen:
                    seen.add(t)
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"Wrote {n} unique templates to {out_path}")


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("reddit"); sp.add_argument("--limit", type=int, default=200)
    sub.add_parser("synth").add_argument("--n", type=int, default=200)
    sub.add_parser("benign").add_argument("--n", type=int, default=200)

    hp = sub.add_parser("huggingface", help="Load BothBosu/Scammer-Conversation (no API key needed)")
    hp.add_argument("--dataset", default="BothBosu/Scammer-Conversation")
    hp.add_argument("--max-per-class", type=int, default=None,
                    help="Optional cap per class for fast iteration (e.g. 200)")

    tp = sub.add_parser("templates")
    tp.add_argument("--scam-jsonl", default=Path("data/scam_corpus.jsonl"), type=Path)
    tp.add_argument("--out", default=Path("data/scam_templates.jsonl"), type=Path)

    args, _ = ap.parse_known_args()
    if args.cmd == "reddit":
        convs = collect_reddit(limit=args.limit)
        save_corpus(convs, OUT_DIR / "scam_reddit.jsonl")
    elif args.cmd == "synth":
        convs = synth_anthropic(n=args.n)
        save_corpus(convs, OUT_DIR / "scam_synthetic.jsonl")
    elif args.cmd == "benign":
        convs = synth_benign_anthropic(n=args.n)
        save_corpus(convs, OUT_DIR / "benign_corpus.jsonl")
    elif args.cmd == "huggingface":
        scam_convs, benign_convs = load_hf_bothbosu(
            dataset_name=args.dataset, max_per_class=args.max_per_class,
        )
        save_corpus(scam_convs, OUT_DIR / "scam_huggingface.jsonl")
        save_corpus(benign_convs, OUT_DIR / "benign_huggingface.jsonl")
        print(f"Wrote {len(scam_convs)} scam and {len(benign_convs)} benign conversations.")
    elif args.cmd == "templates":
        extract_templates(args.scam_jsonl, args.out)


if __name__ == "__main__":
    main()
