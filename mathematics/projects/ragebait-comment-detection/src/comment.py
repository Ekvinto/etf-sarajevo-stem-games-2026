"""Comment schema and loading utilities.

A comment is a dict:
    {
      "text": str,                  # required
      "label_ai": 0 | 1 | None,     # AI-generated?
      "label_ragebait": 0 | 1 | None,
      "username": str | None,       # salted hash; optional
      "timestamp": ISO-8601 str | None,
      "parent_topic": str | None,   # parent post / article title; optional
      "source": str | None,
    }

A corpus is a JSONL file with one such dict per line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class Comment:
    text: str
    label_ai: int | None = None
    label_ragebait: int | None = None
    username: str | None = None
    timestamp: datetime | None = None
    parent_topic: str | None = None
    source: str | None = None
    meta: dict = field(default_factory=dict)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _from_dict(payload: dict) -> Comment:
    return Comment(
        text=payload["text"],
        label_ai=payload.get("label_ai"),
        label_ragebait=payload.get("label_ragebait"),
        username=payload.get("username"),
        timestamp=_parse_ts(payload.get("timestamp")),
        parent_topic=payload.get("parent_topic"),
        source=payload.get("source"),
        meta={k: v for k, v in payload.items()
              if k not in {"text", "label_ai", "label_ragebait",
                           "username", "timestamp", "parent_topic", "source"}},
    )


def load_comment(path: str | Path) -> Comment:
    """Load a single comment from a JSON file."""
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        # Allow a list-of-one for compatibility with corpora dumped as JSON
        payload = payload[0]
    c = _from_dict(payload)
    if c.source is None:
        c.source = str(path)
    return c


def load_corpus(jsonl_path: str | Path) -> list[Comment]:
    """Load a JSONL file where each line is one comment object."""
    comments: list[Comment] = []
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comments.append(_from_dict(json.loads(line)))
    return comments


def save_corpus(comments: Iterable[Comment], jsonl_path: str | Path) -> None:
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in comments:
            obj = {
                "text": c.text,
                "label_ai": c.label_ai,
                "label_ragebait": c.label_ragebait,
                "username": c.username,
                "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                "parent_topic": c.parent_topic,
                "source": c.source,
            }
            obj.update(c.meta)
            # Drop None-valued canonical fields to keep files compact
            obj = {k: v for k, v in obj.items()
                   if not (v is None and k in {"label_ai", "label_ragebait",
                                                "username", "timestamp",
                                                "parent_topic", "source"})}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
