#!/bin/bash
# MLX Fine-tuning Script for Travel Sustainability Model
# This script demonstrates how to run MLX fine-tuning

set -e  # Exit on any error

echo "=== MLX Fine-tuning Setup ==="

# Check if we're on macOS with Apple Silicon
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: MLX requires macOS with Apple Silicon"
    exit 1
fi

# Check if MLX is installed
if ! python -c "import mlx" 2>/dev/null; then
    echo "Installing MLX dependencies..."
    pip install -r finetuning/requirements_mlx.txt
fi

# Verify MLX installation
echo "Verifying MLX installation..."
python -c "
import mlx.core as mx
print(f'MLX version: {mx.__version__}')
print(f'Available devices: {mx.metal.get_device_count()}')
"

# Check if training data exists
if [ ! -f "finetuning/data/processed/mode_choice.jsonl" ] || [ ! -f "finetuning/data/processed/sustainable_pois.jsonl" ]; then
    echo "Training data not found. Building datasets..."
    python finetuning/scripts/build_dataset.py
fi

echo "=== Starting MLX Fine-tuning ==="

# Option 1: Use configuration file (recommended)
echo "Running training with configuration file..."
python finetuning/scripts/train_mlx.py --config finetuning/configs/train_mlx.yaml

# Option 2: Use command line arguments (uncomment to use)
# echo "Running training with command line arguments..."
# python finetuning/scripts/train_mlx.py \
#     --model mlx-community/Mistral-7B-Instruct-v0.2-4bit \
#     --output-dir finetuning/models/mistral-mlx-lora \
#     --epochs 3 \
#     --lr 2e-4 \
#     --batch-size 1 \
#     --lora-r 8 \
#     --lora-alpha 16

echo "=== Training Complete ==="
echo "Model saved to: finetuning/models/mistral-mlx-lora"
echo ""
echo "To use the fine-tuned model:"
echo "1. Update backend/.env: USE_MLX=true"
echo "2. Set MLX_MODEL=finetuning/models/mistral-mlx-lora"
echo "3. Restart your backend service"
