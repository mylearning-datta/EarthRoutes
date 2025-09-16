#!/usr/bin/env python3
"""
Quick test for base Mistral-7B-Instruct generation without LoRA.

Usage:
  python finetuning/scripts/test_mistral_base.py \
    --model-dir finetuning/models/mistral-7b-instruct \
    --prompt "Suggest sustainable travel from Bangalore to Chennai"
"""

import argparse
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    print("Missing dependencies. Install with: pip install transformers torch")
    raise


def main() -> None:
    p = argparse.ArgumentParser(description="Test base Mistral-7B-Instruct")
    p.add_argument("--model-dir", default="finetuning/models/mistral-7b-instruct")
    p.add_argument("--prompt", default="Suggest sustainable travel options in Mumbai")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}. Run download_mistral.py first.")

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    prompt = f"### INSTRUCTION\n{args.prompt}\n### RESPONSE\n"
    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )

    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
