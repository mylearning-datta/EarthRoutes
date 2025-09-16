#!/usr/bin/env python3
"""
Download Mistral-7B-Instruct locally using Hugging Face.

Example:
  python finetuning/scripts/download_mistral.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --out-dir finetuning/models/mistral-7b-instruct
"""

import argparse
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch  # noqa: F401
except Exception as e:
    print("Missing dependencies. Install with: pip install transformers torch")
    raise


def download_model(model_name: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer: {model_name} -> {out}")
    AutoTokenizer.from_pretrained(model_name).save_pretrained(out)

    print(f"Downloading model weights: {model_name} -> {out}")
    AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(out)

    print("Download complete.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download Mistral-7B-Instruct locally")
    p.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1")
    p.add_argument("--out-dir", default="finetuning/models/mistral-7b-instruct")
    args = p.parse_args()

    download_model(args.model, args.out_dir)


if __name__ == "__main__":
    main()
