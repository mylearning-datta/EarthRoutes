#!/usr/bin/env python3
"""
MLX-based Fine-Tuning script for travel sustainability model.
Trains Mistral-7B-Instruct-v0.2-4bit with LoRA using Apple's MLX framework.

This script uses the mlx-lm library's built-in training capabilities for proper LoRA fine-tuning.
"""

import json
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import subprocess
import sys
import re
from typing import Iterable

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.tuner import lora
    from mlx_lm.utils import load_config
    from datasets import Dataset, load_dataset, concatenate_datasets
    import numpy as np
    MLX_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    print(f"Warning: MLX not available. Error: {e}")
    print("Install MLX with: pip install mlx mlx-lm")


def load_datasets() -> Dict[str, List[Dict]]:
    """Load the generated datasets."""
    processed_dir = PROJECT_ROOT / "finetuning" / "data" / "processed"
    
    datasets = {}
    for task in ["mode_choice", "sustainable_pois"]:
        file_path = processed_dir / f"{task}.jsonl"
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                datasets[task] = [json.loads(line) for line in f if line.strip()]
        else:
            datasets[task] = []
    
    return datasets


def format_training_example(example: Dict) -> str:
    """Format example for training in MLX format."""
    task = example.get("task", "UNKNOWN")
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    response_json = example.get("response_json", {})
    
    # Create instruction-following format
    text = f"### TASK: {task}\n### INSTRUCTION:\n{instruction}"
    if context:
        text += f"\n### CONTEXT:\n{context}"
    text += f"\n### RESPONSE:\n{response}"
    if response_json:
        text += f"\n### RESPONSE_JSON:\n{json.dumps(response_json, indent=2)}"
    
    return text


def prepare_training_data(datasets: Dict[str, List[Dict]], max_length: int = 1024) -> List[Dict]:
    """Prepare training data for MLX."""
    all_examples = []
    
    for task, examples in datasets.items():
        for example in examples:
            formatted_text = format_training_example(example)
            all_examples.append({
                "text": formatted_text,
                "task": task
            })
    
    return all_examples


def create_mlx_dataset(training_data: List[Dict], tokenizer, max_length: int = 1024):
    """Create MLX-compatible dataset."""
    texts = [example["text"] for example in training_data]
    
    # Tokenize all texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np"
    )
    
    # Convert to MLX arrays
    input_ids = mx.array(tokenized["input_ids"])
    attention_mask = mx.array(tokenized["attention_mask"])
    
    return input_ids, attention_mask


def prepare_mlx_training_data(output_dir: str = "finetuning/data/processed/mlx_data"):
    """Prepare training data in MLX-compatible format."""
    datasets = load_datasets()
    total_examples = sum(len(examples) for examples in datasets.values())
    print(f"Total training examples: {total_examples}")
    
    if total_examples == 0:
        print("No training data found. Run build_dataset.py first.")
        return None
    
    # Prepare training data
    training_data = prepare_training_data(datasets, max_length=1024)
    
    # Create MLX-compatible directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split data into train/validation (80/20)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    valid_data = training_data[split_idx:]
    
    # Save training data
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps({"text": example["text"]}) + "\n")
    
    # Save validation data
    valid_file = output_path / "valid.jsonl"
    with open(valid_file, 'w', encoding='utf-8') as f:
        for example in valid_data:
            f.write(json.dumps({"text": example["text"]}) + "\n")
    
    print(f"Saved {len(train_data)} training examples to {train_file}")
    print(f"Saved {len(valid_data)} validation examples to {valid_file}")
    return str(output_path)


def train_mlx_model_with_cli(
    model_name: str = "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
    output_dir: str = "finetuning/models/mistral-mlx-lora",
    max_length: int = 1024,
    batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
    config_file: Optional[str] = None
):
    """Train the model using MLX CLI commands."""
    
    if not MLX_AVAILABLE:
        print("Cannot train: MLX not available")
        return None
    
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Mistral attention
    
    # Prepare training data
    training_data_file = prepare_mlx_training_data()
    if not training_data_file:
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build MLX training command with correct arguments
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--train",
        "--data", training_data_file,
        "--fine-tune-type", "lora",
        "--num-layers", str(len(target_modules)),
        "--learning-rate", str(learning_rate),
        "--iters", str(num_epochs * 100),  # Convert epochs to iterations (approximate)
        "--batch-size", str(batch_size),
        "--max-seq-length", str(max_length),
        "--save-every", "500",
        "--adapter-path", str(output_path),
        "--steps-per-report", "20"
    ]
    
    print(f"Running MLX training command: {' '.join(cmd)}")

    # Stream output with progress updates
    step_re = re.compile(r"(step|iter)[^0-9]*([0-9]+)\s*/\s*([0-9]+)", re.IGNORECASE)
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            current = 0
            total = None
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip()
                    print(line)
                    m = step_re.search(line)
                    if m:
                        try:
                            current = int(m.group(2))
                            total = int(m.group(3))
                            pct = (current / total * 100.0) if total else 0.0
                            print(f"Progress: {current}/{total} ({pct:.1f}%)", flush=True)
                        except Exception:
                            pass
            returncode = proc.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd)
        print("Training completed successfully!")
        
        # Save training config
        config = {
            "model_name": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "framework": "mlx",
            "training_data_file": training_data_file
        }
        
        with open(output_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None


def train_mlx_model_with_config(config_file: str):
    """Train using a YAML configuration file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    lora_config = config.get('lora', {})
    training_config = config.get('training', {})
    
    return train_mlx_model_with_cli(
        model_name=model_config.get('name', 'mlx-community/Mistral-7B-Instruct-v0.2-4bit'),
        output_dir=training_config.get('output_dir', 'finetuning/models/mistral-mlx-lora'),
        max_length=model_config.get('max_length', 1024),
        batch_size=training_config.get('per_device_train_batch_size', 1),
        num_epochs=training_config.get('num_train_epochs', 3),
        learning_rate=training_config.get('learning_rate', 2e-4),
        lora_r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('alpha', 16),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
    )


def main():
    parser = argparse.ArgumentParser(description="Train Mistral model with MLX")
    parser.add_argument("--model", default="mlx-community/Mistral-7B-Instruct-v0.2-4bit",
                       help="Model name or path")
    parser.add_argument("--output-dir", default="finetuning/models/mistral-mlx-lora",
                       help="Output directory for trained model")
    parser.add_argument("--max-length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--config", type=str, default=None,
                       help="YAML configuration file path")
    
    args = parser.parse_args()
    
    if args.config:
        # Use configuration file
        train_mlx_model_with_config(args.config)
    else:
        # Use command line arguments
        train_mlx_model_with_cli(
            model_name=args.model,
            output_dir=args.output_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )


if __name__ == "__main__":
    main()
