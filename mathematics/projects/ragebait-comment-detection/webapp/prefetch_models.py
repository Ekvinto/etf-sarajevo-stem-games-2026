"""Pre-download every Hugging Face model the detector needs.

Run during `docker build` so the weights are baked into the image and the
container never downloads anything at runtime. The model names here must
stay in sync with src/features/ai_likelihood.py and src/features/rb_semantic.py.
"""
from __future__ import annotations

import sys


def main() -> None:
    print("Pre-fetching GPT-2 medium ...", flush=True)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    GPT2TokenizerFast.from_pretrained("gpt2-medium")
    GPT2LMHeadModel.from_pretrained("gpt2-medium")

    print("Pre-fetching T5 small ...", flush=True)
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    T5Tokenizer.from_pretrained("google-t5/t5-small")
    T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    print("Pre-fetching sentence transformer (paraphrase-multilingual-MiniLM-L12-v2) ...",
          flush=True)
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("All models cached.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"Pre-fetch failed: {e}", file=sys.stderr)
        sys.exit(1)
