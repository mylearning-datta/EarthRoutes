#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for travel sustainability model.
Trains Mistral-7B-Instruct with LoRA on MODE_CHOICE and SUSTAINABLE_POIS tasks.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset
    import torch
    from torch.utils.data import DataLoader
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers, peft, or datasets not available. Install with:")
    print("pip install transformers peft datasets torch")


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
    """Format example for training."""
    task = example.get("task", "UNKNOWN")
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    response_json = example.get("response_json", {})
    
    # Create training prompt
    prompt = f"### TASK: {task}\n### INSTRUCTION:\n{instruction}"
    if context:
        prompt += f"\n### CONTEXT:\n{context}"
    prompt += f"\n### RESPONSE:\n{response}"
    
    if response_json:
        prompt += f"\n### RESPONSE_JSON:\n{json.dumps(response_json, indent=2)}"
    
    return prompt


def prepare_training_data(datasets: Dict[str, List[Dict]], max_length: int = 1024) -> List[Dict]:
    """Prepare training data in the format expected by transformers."""
    all_examples = []
    
    for task, examples in datasets.items():
        for example in examples:
            formatted_text = format_training_example(example)
            all_examples.append({
                "text": formatted_text
            })
    
    return all_examples


def train_model(
    model_name: str = "gpt2",
    output_dir: str = "finetuning/models/travel-sustainability-lora",
    max_length: int = 1024,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05
):
    """Train the model with LoRA fine-tuning."""
    
    if not TRANSFORMERS_AVAILABLE:
        print("Cannot train: Required packages not installed")
        return None
    
    print("Loading datasets...")
    datasets = load_datasets()
    total_examples = sum(len(examples) for examples in datasets.values())
    print(f"Total training examples: {total_examples}")
    
    if total_examples == 0:
        print("No training data found. Run build_dataset.py first.")
        return None
    
    # Prepare training data
    training_data = prepare_training_data(datasets, max_length)
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True if torch.cuda.is_available() else False
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - GPT-2 attention layers
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset
    dataset = Dataset.from_list(training_data)
    
    def tokenize_function(examples):
        # Handle both single examples and batches
        texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Remove raw text column to avoid collator trying to tensorize strings
    if "text" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "model_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "total_examples": total_examples
    }
    
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Training completed!")
    return output_dir


def main():
    """Main training function."""
    print("Travel Sustainability Model Training")
    print("=" * 50)
    
    # Check if datasets exist
    datasets = load_datasets()
    if not any(datasets.values()):
        print("No datasets found. Please run:")
        print("1. python finetuning/scripts/extract_facts.py")
        print("2. python finetuning/scripts/build_dataset.py")
        return
    
    # Train model
    model_path = train_model()
    
    if model_path:
        print(f"\nModel saved to: {model_path}")
        print("You can now use this model for inference!")
    else:
        print("Training failed. Check the error messages above.")


if __name__ == "__main__":
    main()
