"""CLI entry point.

Examples:
    python -m src.detect --input tests/conversations/scam_01.json
    python -m src.detect --input my_chat.json --model models/classifier.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.conversation import load_conversation
from src.pipeline import FEATURE_NAMES, explain_red_flags, extract_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect romance-scam chatbot conversations.")
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to conversation JSON.")
    ap.add_argument("--model", default=Path("models/classifier.joblib"), type=Path,
                    help="Path to the trained classifier.joblib.")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = ap.parse_args()

    conv = load_conversation(args.input)
    feats = extract_features(conv)
    flags = explain_red_flags(feats)

    if not args.model.exists():
        print(f"[WARN] No trained model at {args.model}. Showing features only.")
        if args.json:
            print(json.dumps({"features": feats, "red_flags": flags}, indent=2))
        else:
            print(f"\nFeatures for {args.input}:")
            for name in FEATURE_NAMES:
                print(f"  {name:>7s} = {feats[name]}")
            print("\nRed flags:")
            for f in flags:
                print(f"  - {f}")
        return

    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    feat_names = bundle.get("feature_names", FEATURE_NAMES)

    x = np.array([[feats[n] for n in feat_names]], dtype=float)
    prob = float(pipe.predict_proba(x)[0, 1])

    if args.json:
        print(json.dumps({"probability": prob, "features": feats, "red_flags": flags}, indent=2))
    else:
        print(f"\nRomance-scam probability: {prob:.3f}")
        print("\nTop red flags:")
        if flags:
            for f in flags:
                print(f"  - {f}")
        else:
            print("  (none triggered)")


if __name__ == "__main__":
    main()
