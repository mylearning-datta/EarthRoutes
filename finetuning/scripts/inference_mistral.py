#!/usr/bin/env python3
"""
Inference script for a Mistral-7B-Instruct model fine-tuned with LoRA.
Loads the base Mistral model and applies a local LoRA adapter for predictions.

Usage examples:
  python finetuning/scripts/inference_mistral.py --model-path finetuning/models/mistral-lora --mode test
  python finetuning/scripts/inference_mistral.py --model-path /absolute/path/to/mistral-lora --mode interactive
"""

import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers or peft not available. Install with:")
    print("pip install transformers peft torch bitsandbytes")


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


class MistralLoraModel:
    def __init__(self, adapter_path: str, base_model_name: str = DEFAULT_BASE_MODEL):
        self.adapter_path = Path(adapter_path)
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self._load()

    def _load(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            return
        if not self.adapter_path.exists():
            print(f"LoRA adapter path does not exist: {self.adapter_path}")
            return

        print(f"Loading tokenizer from adapter: {self.adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model: {self.base_model_name}")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            load_in_8bit=True if torch.cuda.is_available() else False,
        )

        print(f"Applying LoRA adapter from: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()
        print("Mistral + LoRA loaded.")

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def generate(self, instruction: str, max_new_tokens: int = 256) -> Dict:
        if not self.is_ready():
            return {"error": "Model not loaded"}

        prompt = f"### INSTRUCTION:\n{instruction}\n### RESPONSE:\n"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resp = full[len(prompt):].strip()
        return {"instruction": instruction, "response": resp, "full_output": full}


def _test(adapter_path: str) -> None:
    print("Testing Mistral LoRA Inference")
    model = MistralLoraModel(adapter_path)
    if not model.is_ready():
        print("Model not ready. Ensure adapter path is correct and dependencies installed.")
        return

    samples = [
        "I want to travel from Bangalore to Chennai and provide me with suitable sustainable options",
        "I'm visiting Mumbai, suggest sustainable places to visit",
    ]
    for i, q in enumerate(samples, 1):
        print(f"\n{i}. Query: {q}")
        out = model.generate(q)
        if "error" in out:
            print(out["error"])
        else:
            print(f"Response: {out['response']}")


def _interactive(adapter_path: str) -> None:
    model = MistralLoraModel(adapter_path)
    if not model.is_ready():
        print("Model not ready. Ensure adapter path is correct and dependencies installed.")
        return
    print("Interactive mode. Type 'quit' to exit.")
    while True:
        try:
            q = input("\nEnter instruction: ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if not q:
                continue
            out = model.generate(q)
            if "error" in out:
                print(out["error"])
            else:
                print(f"\nResponse: {out['response']}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Mistral LoRA Inference")
    p.add_argument("--model-path", required=True, help="Path to local LoRA adapter directory")
    p.add_argument("--mode", choices=["test", "interactive"], default="test")
    args = p.parse_args()

    if args.mode == "test":
        _test(args.model_path)
    else:
        _interactive(args.model_path)


if __name__ == "__main__":
    main()


