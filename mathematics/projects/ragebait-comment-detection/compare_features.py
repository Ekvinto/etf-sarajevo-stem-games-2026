"""Directional pre-validation for every feature.

Sanity check: BEFORE training, verify that each feature separates the
positive class from the negative class in the *expected* direction on the
12 fixture comments under tests/comments/.

Usage:
    python compare_features.py
    python compare_features.py --feature psi_3     # just one
    python compare_features.py --skip ai_likelihood rb_neutralize  # fast iteration

If a feature is supposed to go UP for ragebait (e.g. chi_5) but the mean
on ragebait_* fixtures is LOWER than on substantive_* fixtures, the script
prints a red WARNING line. This catches sign/normalization bugs early.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.comment import load_comment
from src.pipeline import (
    AI_FEATURE_NAMES,
    RB_FEATURE_NAMES,
    extract_features,
)

# Direction: +1 means "higher value is more positive class", -1 means
# "lower value is more positive class". For psi_1 (perplexity z-score)
# AI text has LOWER perplexity, so direction is -1.
AI_DIRECTION = {
    "psi_1": -1,   # AI: lower perplexity
    "psi_2": -1,   # AI: lower burstiness
    "psi_3": +1,   # AI: higher curvature
    "psi_4": +1,   # AI: more low-rank tokens
    "psi_5": +1,
    "psi_6": +1,   # AI: more AI-phrases
    "psi_7": +1,   # AI: punctuation profile farther from human baseline
    "psi_8": +1,   # AI: more hedging
}

RB_DIRECTION = {
    "chi_1": +1, "chi_2": +1, "chi_3": +1, "chi_4": +1,
    "chi_5": +1, "chi_6": +1, "chi_7": +1, "chi_8": +1,
    "chi_9": +1, "chi_10": +1,
}


def _mean(values: list[float]) -> float:
    vs = [v for v in values if isinstance(v, float) and not math.isnan(v)]
    return sum(vs) / len(vs) if vs else float("nan")


def _load_fixtures() -> dict[str, list]:
    """Returns dict with keys 'ai_pos', 'ai_neg', 'rb_pos', 'rb_neg' -> list[Comment]."""
    root = Path("tests/comments")
    if not root.exists():
        raise SystemExit("tests/comments/ not found. Run from project root.")
    buckets = {"ai_pos": [], "ai_neg": [], "rb_pos": [], "rb_neg": []}
    for p in sorted(root.glob("*.json")):
        c = load_comment(p)
        if c.label_ai == 1:
            buckets["ai_pos"].append(c)
        elif c.label_ai == 0:
            buckets["ai_neg"].append(c)
        if c.label_ragebait == 1:
            buckets["rb_pos"].append(c)
        elif c.label_ragebait == 0:
            buckets["rb_neg"].append(c)
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", type=str, default=None,
                    help="If given, only check this feature.")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Feature modules to skip (e.g. ai_likelihood for fast iteration)")
    args = ap.parse_args()

    buckets = _load_fixtures()
    print(f"Loaded fixtures: "
          f"AI+ {len(buckets['ai_pos'])}, AI- {len(buckets['ai_neg'])}, "
          f"RB+ {len(buckets['rb_pos'])}, RB- {len(buckets['rb_neg'])}")
    if min(len(v) for v in buckets.values()) < 2:
        print("Need at least 2 fixtures per class. Add more tests/comments/*.json")
        return

    # Extract once per comment, reuse for both stages.
    def feats_for(cs):
        return [extract_features(c, skip=set(args.skip)) for c in cs]

    print("\nExtracting features...")
    ai_pos_f = feats_for(buckets["ai_pos"])
    ai_neg_f = feats_for(buckets["ai_neg"])
    rb_pos_f = feats_for(buckets["rb_pos"])
    rb_neg_f = feats_for(buckets["rb_neg"])

    targets = [args.feature] if args.feature else (AI_FEATURE_NAMES + RB_FEATURE_NAMES)
    print(f"\n{'feature':<8s}  {'class':<7s}  {'pos mean':>10s}  {'neg mean':>10s}  "
          f"{'delta':>10s}  {'expected':>10s}  status")
    print("-" * 78)
    for name in targets:
        if name in AI_FEATURE_NAMES:
            pos_vals = [f.get(name, float("nan")) for f in ai_pos_f]
            neg_vals = [f.get(name, float("nan")) for f in ai_neg_f]
            direction = AI_DIRECTION[name]
            cls = "AI"
        elif name in RB_FEATURE_NAMES:
            pos_vals = [f.get(name, float("nan")) for f in rb_pos_f]
            neg_vals = [f.get(name, float("nan")) for f in rb_neg_f]
            direction = RB_DIRECTION[name]
            cls = "RB"
        else:
            print(f"  {name}: unknown feature")
            continue
        mp, mn = _mean(pos_vals), _mean(neg_vals)
        if math.isnan(mp) or math.isnan(mn):
            mp_s = "NaN" if math.isnan(mp) else f"{mp:.4f}"
            mn_s = "NaN" if math.isnan(mn) else f"{mn:.4f}"
            print(f"{name:<8s}  {cls:<7s}  {mp_s:>10s}  {mn_s:>10s}  "
                  f"{'-':>10s}  {direction:>+10d}  NaN")
            continue
        delta = mp - mn
        agrees = (delta > 0 and direction > 0) or (delta < 0 and direction < 0)
        status = "OK" if agrees else "!! WRONG DIRECTION"
        print(f"{name:<8s}  {cls:<7s}  {mp:>10.4f}  {mn:>10.4f}  "
              f"{delta:>+10.4f}  {direction:>+10d}  {status}")


if __name__ == "__main__":
    main()
