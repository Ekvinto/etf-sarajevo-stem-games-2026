"""Side-by-side comparison of features across all test conversations.

Loads each model only ONCE (thanks to lru_cache in the feature modules) and
prints a table of every feature for every test conversation, plus a directional
sanity check that highlights features whose scam-vs-benign mean is inverted.

Usage:
    python compare_features.py                    # lightweight features only (no model downloads)
    python compare_features.py --all              # include LLM-based features (triggers HF downloads)
    python compare_features.py --skip detect_gpt  # skip specific modules
    python compare_features.py --include perplexity token_rank  # add specific heavy ones
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.conversation import load_conversation
from src.pipeline import FEATURE_NAMES, extract_features

# Modules that need large model downloads (gated behind --all or --include)
HEAVY_MODULES = {
    "perplexity",   # gpt2-medium (~1.4 GB)
    "detect_gpt",   # gpt2-medium + t5-small (~1.6 GB total)
    "token_rank",   # gpt2-medium (~1.4 GB)
    "sentiment",    # xlm-roberta-base-sentiment (~270 MB)
    "semantic",     # MiniLM (~120 MB)
}

# hmm_stages also goes in the skip set because hmmlearn may not be installed
# AND the models aren't trained yet. Its extract() already handles this.

TEST_FILES = ["scam_01", "scam_02", "scam_03",
              "benign_01", "benign_02", "benign_03"]
N_SCAM = 3

# Expected direction for each feature. True = scam should be HIGHER than benign.
EXPECTED_HIGHER = {
    "phi_1": False,   # perplexity:        AI text is more expected -> lower
    "phi_2": False,   # burstiness:        AI text more uniform     -> lower
    "phi_3": True,    # DetectGPT z:       AI sits at local maxima  -> higher
    "phi_4": True,    # GLTR top-10:       AI uses common tokens    -> higher
    "phi_5": True,    # GLTR top-100:      "                         "
    "phi_6": False,   # MATTR:             AI more diverse-ish      -> debatable; small effect
    "phi_7": True,    # Yule K:            scam recycles "dear/love/please" -> higher K
    "phi_8": True,    # Zipf deviation:    AI deviates from natural -> higher
    "phi_9": True,    # length asym:       scammers monologue       -> higher
    "phi_10": True,   # timing bimodality: operator multi-tasking   -> higher
    "phi_11": True,   # CUSUM peak:        sentiment dip at ask     -> higher
    "phi_12": False,  # sentiment slope:   trends negative          -> lower (more negative)
    "phi_13": True,   # topic KL:          scams pivot              -> higher
    "phi_14": True,   # finance mass late: the smoking gun          -> higher
    "phi_15": True,   # HMM log LR
    "phi_16": True,   # urgency indicator
    "phi_17": True,   # max template sim
    "phi_18": True,   # mean template sim
    "phi_19": True,   # frac > threshold
}


def fmt(v: float) -> str:
    if not isinstance(v, (int, float)) or math.isnan(v):
        return "  nan"
    if v == 0:
        return "0.000"
    if abs(v) < 1e-3 or abs(v) > 1e4:
        return f"{v:.1e}"
    return f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Include feature modules that require model downloads")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Feature modules to skip (e.g. detect_gpt sentiment)")
    ap.add_argument("--include", nargs="*", default=[],
                    help="Heavy modules to include even without --all")
    args = ap.parse_args()

    skip = set(args.skip)
    if not args.all:
        skip |= (HEAVY_MODULES - set(args.include))
    skip.add("hmm_stages")  # always skip; needs trained models + hmmlearn

    active_modules = sorted({"perplexity", "detect_gpt", "token_rank",
                             "stylometry", "asymmetry", "sentiment",
                             "topic_shift", "hmm_stages", "semantic"} - skip)
    print(f"Active modules: {active_modules}")
    print(f"Skipped:        {sorted(skip)}\n")

    results: dict[str, dict[str, float]] = {}
    for name in TEST_FILES:
        path = Path("tests/conversations") / f"{name}.json"
        print(f"  processing {name}...")
        conv = load_conversation(path)
        results[name] = extract_features(conv, skip=skip)

    # --- Table -------------------------------------------------------
    col_w = 11
    header = f"{'feature':<8}" + "".join(f"{n:>{col_w}}" for n in TEST_FILES)
    print()
    print(header)
    print("-" * len(header))
    for feat in FEATURE_NAMES:
        row = f"{feat:<8}"
        for name in TEST_FILES:
            row += f"{fmt(results[name].get(feat, float('nan'))):>{col_w}}"
        print(row)

    # --- Directional sanity check ------------------------------------
    print("\n" + "=" * 60)
    print("DIRECTIONAL SANITY CHECK")
    print("=" * 60)
    issues = []
    skipped_count = 0
    ok_count = 0
    for feat in FEATURE_NAMES:
        scam_vals = [results[n][feat] for n in TEST_FILES[:N_SCAM]
                     if not math.isnan(results[n][feat])]
        benign_vals = [results[n][feat] for n in TEST_FILES[N_SCAM:]
                       if not math.isnan(results[n][feat])]
        if len(scam_vals) < 2 or len(benign_vals) < 2:
            skipped_count += 1
            continue
        scam_mean = sum(scam_vals) / len(scam_vals)
        benign_mean = sum(benign_vals) / len(benign_vals)
        exp_higher = EXPECTED_HIGHER.get(feat)
        if exp_higher is None:
            continue
        actually_higher = scam_mean > benign_mean
        match = (actually_higher == exp_higher)
        direction = "scam > benign" if exp_higher else "scam < benign"
        if match:
            ok_count += 1
            print(f"  OK   {feat:<7} expected {direction:<14}  "
                  f"got scam={scam_mean:.3f} benign={benign_mean:.3f}")
        else:
            issues.append(
                f"  WRONG {feat:<7} expected {direction:<14}  "
                f"got scam={scam_mean:.3f} benign={benign_mean:.3f}"
            )

    if issues:
        print("\nPotential bugs (direction inverted from spec):")
        for line in issues:
            print(line)
    print(f"\nSummary: {ok_count} correct, {len(issues)} inverted, "
          f"{skipped_count} skipped (insufficient data)")


if __name__ == "__main__":
    main()
