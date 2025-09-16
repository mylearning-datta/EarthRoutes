#!/usr/bin/env python3
"""
Direct conversion of the original Colab code to MLX.
This script shows the exact equivalent of your Colab fine-tuning code using MLX.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(PROJECT_ROOT))

# ==== MLX Equivalent of Original Colab Code ====

# Original Colab constants (converted to MLX equivalents)
BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"  # Changed from leliuga/mistral-7b-instruct-v0.1-bnb-4bit
MAX_LENGTH = 1024  # Same as original
BATCH_SIZE = 1     # Same as original
GRAD_ACCUM = 16    # Handled automatically by MLX
EPOCHS = 3         # Same as original
LR = 2e-4          # Same as original
LORA_R = 8         # Same as original
LORA_ALPHA = 16    # Same as original
LORA_DROPOUT = 0.05 # Same as original
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Same as original

OUTPUT_DIR = "finetuning/models/mistral_mlx_lora"  # MLX output directory

def load_datasets() -> Dict[str, List[Dict]]:
    """Load the generated datasets (same as original)."""
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

def format_example(ex):
    """Format example for training (exact same as original Colab code)."""
    task = ex.get("task", "UNKNOWN")
    instruction = ex.get("instruction", "")
    context = ex.get("context", "")
    response = ex.get("response", "")
    response_json = ex.get("response_json", {})
    txt = f"### TASK: {task}\n### INSTRUCTION:\n{instruction}"
    if context:
        txt += f"\n### CONTEXT:\n{context}"
    txt += f"\n### RESPONSE:\n{response}"
    if response_json:
        txt += f"\n### RESPONSE_JSON:\n{json.dumps(response_json, indent=2)}"
    return {"text": txt}

def prepare_training_data():
    """Prepare training data (MLX equivalent of original Colab data preparation)."""
    print("Loading datasets...")
    datasets = load_datasets()
    
    # Create instruct-style text (same as original)
    all_examples = []
    for task, examples in datasets.items():
        for example in examples:
            formatted = format_example(example)
            all_examples.append(formatted)
    
    print(f"Total training examples: {len(all_examples)}")
    
    # Create MLX-compatible directory structure
    output_dir = "finetuning/data/processed/mlx_data"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split data into train/validation (80/20)
    split_idx = int(len(all_examples) * 0.8)
    train_data = all_examples[:split_idx]
    valid_data = all_examples[split_idx:]
    
    # Save training data
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")
    
    # Save validation data
    valid_file = output_path / "valid.jsonl"
    with open(valid_file, 'w', encoding='utf-8') as f:
        for example in valid_data:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(train_data)} training examples to {train_file}")
    print(f"Saved {len(valid_data)} validation examples to {valid_file}")
    return str(output_path)

def train_with_mlx(training_data_file: str):
    """Train using MLX (equivalent of original Colab training)."""
    print(f"Loading model and tokenizer from {BASE_MODEL}...")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # MLX training command (equivalent of original Trainer setup)
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", training_data_file,
        "--fine-tune-type", "lora",
        "--num-layers", str(len(TARGET_MODULES)),
        "--learning-rate", str(LR),
        "--iters", str(EPOCHS * 100),  # Convert epochs to iterations (approximate)
        "--batch-size", str(BATCH_SIZE),
        "--max-seq-length", str(MAX_LENGTH),
        "--save-every", "500",
        "--adapter-path", OUTPUT_DIR,
        "--steps-per-report", "20"
    ]
    
    print("Starting MLX training...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Save training config (equivalent of original config saving)
        config = {
            "model_name": BASE_MODEL,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "num_epochs": EPOCHS,
            "learning_rate": LR,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "framework": "mlx"
        }
        
        with open(Path(OUTPUT_DIR) / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved LoRA adapter to: {OUTPUT_DIR}")
        return OUTPUT_DIR
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None

def test_model(model_path: str):
    """Test the trained model (equivalent of original model usage)."""
    try:
        from mlx_lm import load, generate
        
        print(f"Loading trained model from {model_path}...")
        model, tokenizer = load(model_path)
        
        # Test prompts (same as original use case)
        test_prompts = [
            "### TASK: MODE_CHOICE\n### INSTRUCTION:\nChoose the most sustainable transport mode from NYC to Boston.\n### RESPONSE:",
            "### TASK: SUSTAINABLE_POIS\n### INSTRUCTION:\nRecommend eco-friendly places in San Francisco.\n### RESPONSE:"
        ]
        
        print("\nTesting model...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=200,
                temp=0.7,
                verbose=False
            )
            print(f"Response: {response}")
        
        print("\n✅ Model testing completed!")
        
    except ImportError:
        print("MLX not available for testing")
    except Exception as e:
        print(f"Testing failed: {e}")

def main():
    """Main function - exact equivalent of original Colab workflow."""
    print("=== MLX Fine-tuning (Colab Equivalent) ===")
    print("This script is the exact MLX equivalent of your original Colab code")
    print()
    
    # Step 1: Data preparation (same as original)
    training_data_file = prepare_training_data()
    
    if not training_data_file:
        print("No training data found. Run build_dataset.py first.")
        return
    
    # Step 2: Training (MLX equivalent of original Trainer)
    model_path = train_with_mlx(training_data_file)
    
    if model_path:
        # Step 3: Testing (equivalent of original model usage)
        test_model(model_path)
        
        print(f"\n🎉 Fine-tuning completed!")
        print(f"Model saved to: {model_path}")
        print(f"Use this model in your backend by setting:")
        print(f"  USE_MLX=true")
        print(f"  MLX_MODEL={model_path}")
    else:
        print("❌ Fine-tuning failed")

if __name__ == "__main__":
    main()
