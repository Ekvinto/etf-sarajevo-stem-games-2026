"""CLI entry point: score a single comment.

Examples:
    python -m src.detect --input tests/comments/ragebait_ai_01.json
    python -m src.detect --text "Every single one of these clowns is..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.comment import Comment, load_comment
from src.pipeline import (
    AI_FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    RB_FEATURE_NAMES,
    explain_red_flags,
    extract_features,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect AI-generated ragebait comments.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="Path to comment JSON.")
    g.add_argument("--text", type=str, help="Raw comment text to score.")
    ap.add_argument("--parent-topic", type=str, default=None,
                    help="Optional parent post / article title for --text mode.")
    ap.add_argument("--model", default=Path("models/classifiers.joblib"), type=Path)
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = ap.parse_args()

    if args.input:
        comment = load_comment(args.input)
    else:
        comment = Comment(text=args.text, parent_topic=args.parent_topic)

    feats = extract_features(comment)
    flags = explain_red_flags(feats)

    if not args.model.exists():
        print(f"[WARN] No trained model at {args.model}. Showing features only.")
        if args.json:
            print(json.dumps({"features": feats, "red_flags": flags}, indent=2))
        else:
            _print_features(feats, flags)
        return

    bundle = joblib.load(args.model)
    clf_ai = bundle["clf_ai"]
    clf_rb = bundle["clf_rb"]

    x_ai = np.array([[feats[n] for n in AI_FEATURE_NAMES]], dtype=float)
    x_rb = np.array([[feats[n] for n in RB_FEATURE_NAMES]], dtype=float)
    p_ai = float(clf_ai.predict_proba(x_ai)[0, 1])
    p_rb = float(clf_rb.predict_proba(x_rb)[0, 1])
    p_ar = p_ai * p_rb

    if args.json:
        print(json.dumps({
            "p_ai": p_ai,
            "p_ragebait": p_rb,
            "p_joint": p_ar,
            "features": feats,
            "red_flags": flags,
        }, indent=2))
        return

    print(f"\nAI-generation probability:    {p_ai:.3f}")
    print(f"Ragebait probability:         {p_rb:.3f}")
    print(f"Joint AI-ragebait risk:       {p_ar:.3f}")
    print("\nTop red flags:")
    if flags:
        for f in flags:
            print(f"  - {f}")
    else:
        print("  (none triggered)")


def _print_features(feats: dict[str, float], flags: list[str]) -> None:
    print("\nAI features:")
    for n in AI_FEATURE_NAMES:
        print(f"  {n:>7s} = {feats[n]}")
    print("\nRagebait features:")
    for n in RB_FEATURE_NAMES:
        print(f"  {n:>7s} = {feats[n]}")
    print("\nRed flags:")
    for f in flags:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
