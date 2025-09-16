# MLX Fine-tuning Guide

This guide explains how to fine-tune the `mlx-community/Mistral-7B-Instruct-v0.2-4bit` model using Apple's MLX framework for travel sustainability tasks.

## Overview

MLX is Apple's machine learning framework optimized for Apple Silicon (M1/M2/M3 chips). It provides efficient training and inference for large language models with native support for LoRA fine-tuning.

## Prerequisites

### System Requirements
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- At least 16GB RAM (32GB recommended for 7B models)

### Installation

1. **Install MLX and dependencies:**
```bash
pip install -r finetuning/requirements_mlx.txt
```

2. **Verify MLX installation:**
```python
import mlx.core as mx
print(f"MLX version: {mx.__version__}")
print(f"Available devices: {mx.metal.get_device_count()}")
```

## Quick Start

### Option 1: Using Configuration File (Recommended)

1. **Prepare your training data:**
```bash
# Ensure you have processed datasets
python finetuning/scripts/build_dataset.py
```

2. **Run training with configuration:**
```bash
python finetuning/scripts/train_mlx.py --config finetuning/configs/train_mlx.yaml
```

### Option 2: Using Command Line Arguments

```bash
python finetuning/scripts/train_mlx.py \
    --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
    --output-dir finetuning/models/mistral-mlx-lora \
    --epochs 3 \
    --lr 2e-4 \
    --batch-size 1 \
    --lora-r 8 \
    --lora-alpha 16
```

## Configuration

### YAML Configuration (`finetuning/configs/train_mlx.yaml`)

```yaml
# Model Configuration
model:
  name: "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
  max_length: 1024

# LoRA Configuration
lora:
  r: 8                    # LoRA rank
  alpha: 16               # LoRA alpha
  dropout: 0.05           # LoRA dropout
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Configuration
training:
  per_device_train_batch_size: 1
  num_train_epochs: 3
  learning_rate: 2.0e-4
  output_dir: "finetuning/models/mistral-mlx-lora"
```

### Key Parameters

- **LoRA Rank (r)**: Controls the size of the adaptation. Higher values = more parameters but better adaptation
- **LoRA Alpha**: Scaling factor for LoRA weights
- **Target Modules**: Which layers to apply LoRA to (attention layers for Mistral)
- **Learning Rate**: Typically 1e-4 to 5e-4 for LoRA fine-tuning

## Training Process

### Data Format

The script expects training data in JSONL format with the following structure:
```json
{"text": "### TASK: MODE_CHOICE\n### INSTRUCTION:\nChoose the most sustainable transport mode...\n### RESPONSE:\nTrain is the most sustainable option..."}
```

### Training Steps

1. **Data Preparation**: Combines `mode_choice.jsonl` and `sustainable_pois.jsonl`
2. **Model Loading**: Loads the 4-bit quantized Mistral model
3. **LoRA Setup**: Applies LoRA adapters to attention layers
4. **Training**: Fine-tunes using MLX's optimized training loop
5. **Saving**: Saves LoRA adapters and configuration

### Memory Usage

- **Base Model**: ~4GB (4-bit quantized)
- **LoRA Adapters**: ~100-200MB
- **Training**: Additional 2-4GB for gradients and optimizer states

## Monitoring Training

### Logs
Training progress is logged to console with:
- Loss values
- Learning rate schedule
- Memory usage
- Training speed

### Checkpoints
Models are saved every 500 steps and at the end of each epoch to:
```
finetuning/models/mistral-mlx-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── training_config.json
└── tokenizer files
```

## Using the Fine-tuned Model

### Loading for Inference

```python
from mlx_lm import load, generate

# Load the fine-tuned model
model, tokenizer = load("finetuning/models/mistral-mlx-lora")

# Generate responses
response = generate(
    model, 
    tokenizer, 
    prompt="### TASK: MODE_CHOICE\n### INSTRUCTION:\nChoose transport from NYC to Boston",
    max_tokens=200
)
print(response)
```

### Integration with Backend

Update your backend configuration:
```bash
# In backend/.env
USE_MLX=true
MLX_MODEL=finetuning/models/mistral-mlx-lora
FINETUNED_MODEL_VARIANT=mlx_lora
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size to 1
   - Reduce max_length to 512
   - Close other applications

2. **Slow Training**
   - Ensure you're using Apple Silicon
   - Check that MLX is using Metal backend
   - Reduce sequence length if possible

3. **Import Errors**
   - Verify MLX installation: `pip install mlx mlx-lm`
   - Check Python version compatibility

### Performance Tips

1. **Use 4-bit models**: Significantly reduces memory usage
2. **Optimize batch size**: Start with 1, increase if memory allows
3. **Monitor memory**: Use Activity Monitor to track usage
4. **Warm up**: First few batches may be slower

## Comparison with PyTorch

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Memory Usage | Higher | Lower (optimized for Apple Silicon) |
| Training Speed | Good | Excellent on Apple Silicon |
| Model Support | Extensive | Growing (Mistral, Llama, etc.) |
| Quantization | 8-bit/4-bit | Native 4-bit support |
| Platform | Cross-platform | macOS only |

## Advanced Usage

### Custom LoRA Configuration

```python
# Modify target modules for different architectures
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # MLP layers
]
```

### Multi-GPU Training

MLX automatically uses all available GPU cores on Apple Silicon. For multi-device setups, MLX handles distribution internally.

### Hyperparameter Tuning

Recommended ranges for LoRA fine-tuning:
- **Learning Rate**: 1e-5 to 5e-4
- **LoRA Rank**: 4 to 16
- **LoRA Alpha**: 8 to 32
- **Dropout**: 0.05 to 0.1

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/lora)
- [Mistral-7B-Instruct-v0.2-4bit Model](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4bit)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
