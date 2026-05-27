"""Conversation schema and loading utilities.

A conversation is a list of message dicts:
    {"speaker": "S" | "V", "text": str, "timestamp": ISO-8601 str (optional)}

"S" = suspected scammer, "V" = victim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class Message:
    speaker: str
    text: str
    timestamp: datetime | None = None


@dataclass
class Conversation:
    messages: list[Message]
    label: int | None = None
    source: str | None = None

    @property
    def scammer_messages(self) -> list[Message]:
        return [m for m in self.messages if m.speaker == "S"]

    @property
    def victim_messages(self) -> list[Message]:
        return [m for m in self.messages if m.speaker == "V"]

    @property
    def scammer_text(self) -> str:
        return "\n".join(m.text for m in self.scammer_messages)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_conversation(path: str | Path) -> Conversation:
    """Load a single conversation JSON file (list of messages or full object)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        messages_raw = payload
        label = None
        source = str(path)
    else:
        messages_raw = payload["messages"]
        label = payload.get("label")
        source = payload.get("source", str(path))

    messages = [
        Message(
            speaker=m["speaker"],
            text=m["text"],
            timestamp=_parse_ts(m.get("timestamp")),
        )
        for m in messages_raw
    ]
    return Conversation(messages=messages, label=label, source=source)


def load_corpus(jsonl_path: str | Path) -> list[Conversation]:
    """Load a JSONL file where each line is one conversation object."""
    convs = []
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            messages = [
                Message(
                    speaker=m["speaker"],
                    text=m["text"],
                    timestamp=_parse_ts(m.get("timestamp")),
                )
                for m in payload["messages"]
            ]
            convs.append(
                Conversation(
                    messages=messages,
                    label=payload.get("label"),
                    source=payload.get("source"),
                )
            )
    return convs


def save_corpus(convs: Iterable[Conversation], jsonl_path: str | Path) -> None:
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in convs:
            obj = {
                "label": c.label,
                "source": c.source,
                "messages": [
                    {
                        "speaker": m.speaker,
                        "text": m.text,
                        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    }
                    for m in c.messages
                ],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
