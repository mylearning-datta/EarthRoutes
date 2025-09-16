# Migration Guide: From PyTorch/Transformers to MLX

This guide shows you exactly how to convert your original Colab fine-tuning code to use MLX instead of PyTorch/Transformers.

## Original Colab Code vs MLX Equivalent

### 1. Model Loading

**Original (PyTorch/Transformers):**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "leliuga/mistral-7b-instruct-v0.1-bnb-4bit",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# Configure LoRA
lora_config = LoraConfig(
    r=8, alpha=16, dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
```

**MLX Equivalent:**
```python
from mlx_lm import load

# Load 4-bit quantized model (native support)
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")

# LoRA is handled automatically by MLX CLI or built-in support
```

### 2. Data Preparation

**Original (PyTorch/Transformers):**
```python
from datasets import load_dataset, Dataset, concatenate_datasets

# Load JSONL as datasets
def load_jsonl_as_dataset(path: str) -> Dataset:
    return load_dataset("json", data_files=path, split="train")

mode_ds = load_jsonl_as_dataset(mode_path)
pois_ds = load_jsonl_as_dataset(pois_path)

# Format examples
def format_example(ex):
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

# Map and concatenate
mode_fmt = mode_ds.map(format_example, remove_columns=mode_ds.column_names)
pois_fmt = pois_ds.map(format_example, remove_columns=pois_ds.column_names)
formatted = concatenate_datasets([mode_fmt, pois_fmt])
```

**MLX Equivalent:**
```python
# Same data preparation, but save in MLX-compatible format
def prepare_mlx_training_data(output_file: str = "mlx_training_data.jsonl"):
    datasets = load_datasets()
    training_data = prepare_training_data(datasets, max_length=1024)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_data:
            f.write(json.dumps({"text": example["text"]}) + "\n")
    
    return output_file
```

### 3. Training

**Original (PyTorch/Transformers):**
```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Tokenize data
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])

# Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    gradient_checkpointing=True,
    report_to="none",
    optim="paged_adamw_8bit",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
)

trainer.train()
```

**MLX Equivalent:**
```python
# Use MLX CLI (recommended)
import subprocess

cmd = [
    "python", "-m", "mlx_lm.lora",
    "--model", "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
    "--train",
    "--data", "mlx_training_data.jsonl",
    "--lora-rank", "8",
    "--lora-alpha", "16",
    "--lora-dropout", "0.05",
    "--learning-rate", "2e-4",
    "--num-epochs", "3",
    "--batch-size", "1",
    "--max-seq-length", "1024",
    "--adapter-path", "./lora_adapters"
]

subprocess.run(cmd, check=True)
```

### 4. Model Saving

**Original (PyTorch/Transformers):**
```python
# Save LoRA adapter + tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

**MLX Equivalent:**
```python
# MLX automatically saves adapters during training
# Or manually:
from mlx_lm import save

save(model, tokenizer, "./lora_adapters")
```

## Step-by-Step Migration

### Step 1: Install MLX
```bash
pip install mlx mlx-lm
```

### Step 2: Update Model Path
Change from:
```python
BASE_MODEL = "leliuga/mistral-7b-instruct-v0.1-bnb-4bit"
```

To:
```python
BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
```

### Step 3: Replace Training Code
Replace your entire training section with:
```python
# Use the provided MLX training script
python finetuning/scripts/train_mlx.py --config finetuning/configs/train_mlx.yaml
```

### Step 4: Update Inference Code
**Original:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**MLX:**
```python
from mlx_lm import load, generate

model, tokenizer = load(OUTPUT_DIR)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
```

## Configuration Migration

### Original Parameters → MLX Equivalent

| Original | MLX | Notes |
|----------|-----|-------|
| `load_in_8bit=True` | Native 4-bit | MLX has better quantization |
| `torch_dtype=torch.float16` | Automatic | MLX handles precision automatically |
| `device_map="auto"` | Automatic | MLX uses all available cores |
| `gradient_checkpointing=True` | Built-in | MLX optimizes memory usage |
| `optim="paged_adamw_8bit"` | `adamw` | MLX has optimized AdamW |

### Training Arguments Mapping

| Transformers | MLX CLI | Description |
|--------------|---------|-------------|
| `per_device_train_batch_size` | `--batch-size` | Batch size |
| `num_train_epochs` | `--num-epochs` | Number of epochs |
| `learning_rate` | `--learning-rate` | Learning rate |
| `max_length` | `--max-seq-length` | Maximum sequence length |
| `output_dir` | `--adapter-path` | Output directory |

## Performance Comparison

### Expected Improvements with MLX:

1. **Memory Usage**: 40-60% reduction
2. **Training Speed**: 2-3x faster on Apple Silicon
3. **Inference Speed**: 3-5x faster
4. **Model Size**: Smaller due to better quantization

### Memory Usage Example:
- **PyTorch**: ~8-12GB for 7B model with LoRA
- **MLX**: ~4-6GB for 7B model with LoRA

## Complete Migration Script

Here's a complete script that converts your Colab code to MLX:

```python
#!/usr/bin/env python3
"""
Complete migration from PyTorch/Transformers to MLX
"""

import json
import subprocess
from pathlib import Path

def migrate_to_mlx():
    """Complete migration process."""
    
    # Step 1: Prepare data (same as original)
    datasets = load_datasets()
    training_data = prepare_training_data(datasets, max_length=1024)
    
    # Step 2: Save in MLX format
    with open("mlx_training_data.jsonl", 'w') as f:
        for example in training_data:
            f.write(json.dumps({"text": example["text"]}) + "\n")
    
    # Step 3: Run MLX training
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        "--train",
        "--data", "mlx_training_data.jsonl",
        "--lora-rank", "8",
        "--lora-alpha", "16",
        "--lora-dropout", "0.05",
        "--learning-rate", "2e-4",
        "--num-epochs", "3",
        "--batch-size", "1",
        "--max-seq-length", "1024",
        "--adapter-path", "./mistral_mlx_lora"
    ]
    
    subprocess.run(cmd, check=True)
    print("Migration completed! Model saved to ./mistral_mlx_lora")

if __name__ == "__main__":
    migrate_to_mlx()
```

## Testing the Migration

After migration, test your model:

```bash
# Test the MLX model
python finetuning/scripts/test_mlx_model.py --model-path ./mistral_mlx_lora

# Compare with original PyTorch model
python finetuning/scripts/compare_models.py
```

## Troubleshooting

### Common Issues:

1. **"MLX not available"**: Install with `pip install mlx mlx-lm`
2. **"Model not found"**: Ensure you're using the correct MLX model path
3. **"Out of memory"**: Reduce batch size or sequence length
4. **"Slow training"**: Ensure you're on Apple Silicon

### Performance Tips:

1. Use 4-bit models for lower memory usage
2. Start with batch size 1 and increase if memory allows
3. Monitor memory usage with Activity Monitor
4. Close other applications during training

This migration will give you the same fine-tuning capabilities with significantly better performance on Apple Silicon!
