#!/usr/bin/env python3
"""
Test script for MLX fine-tuned model.
Demonstrates how to load and use the fine-tuned model for inference.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available. Install with: pip install mlx mlx-lm")


BASE_MLX_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"


def test_model_loading(model_path: str, base_model: str = BASE_MLX_MODEL):
    """Test loading the fine-tuned model.

    If `model_path` is a full MLX model dir/repo, loads it directly.
    If it is an adapters directory (contains adapter_config/adapters.safetensors),
    loads the base model with adapter_path=model_path.
    """
    if not MLX_AVAILABLE:
        print("Cannot test: MLX not available")
        return None
    
    print(f"Loading model from: {model_path}")
    try:
        model_dir = Path(model_path)
        is_adapter_dir = False
        if model_dir.exists() and model_dir.is_dir():
            # Heuristics for adapter directory produced by mlx_lm lora
            adapter_file = model_dir / "adapters.safetensors"
            adapter_cfg = model_dir / "adapter_config.json"
            if adapter_file.exists() or adapter_cfg.exists():
                is_adapter_dir = True

        if is_adapter_dir:
            print(f"Detected adapter directory. Loading base '{base_model}' with adapters from '{model_path}'.")
            model, tokenizer = load(base_model, adapter_path=str(model_path))
        else:
            model, tokenizer = load(model_path)

        print("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def test_inference(model, tokenizer, test_prompts: list):
    """Test inference with sample prompts."""
    if not model or not tokenizer:
        print("Cannot test inference: Model not loaded")
        return
    
    print("\n=== Testing Inference ===")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        print("Response:")
        
        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=200,
                verbose=False
            )
            print(response)
        except Exception as e:
            print(f"❌ Inference failed: {e}")


def create_test_prompts():
    """Create test prompts for the travel sustainability model."""
    return [
        "### TASK: MODE_CHOICE\n### INSTRUCTION:\nChoose the most sustainable transport mode for traveling from New York to Boston.\n### RESPONSE:",
        
        "### TASK: SUSTAINABLE_POIS\n### INSTRUCTION:\nRecommend sustainable places to visit in San Francisco that focus on environmental conservation.\n### RESPONSE:",
        
        "### TASK: MODE_CHOICE\n### INSTRUCTION:\nWhat is the most eco-friendly way to travel from London to Paris?\n### RESPONSE:",
        
        "### TASK: SUSTAINABLE_POIS\n### INSTRUCTION:\nSuggest green tourism activities in Amsterdam that promote sustainability.\n### RESPONSE:"
    ]


def benchmark_performance(model, tokenizer, prompt: str, num_runs: int = 5):
    """Benchmark inference performance."""
    if not model or not tokenizer:
        return
    
    print(f"\n=== Performance Benchmark ({num_runs} runs) ===")
    
    import time
    
    times = []
    for i in range(num_runs):
        start_time = time.time()
        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=100,
                verbose=False
            )
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Run {i+1}: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"Run {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage inference time: {avg_time:.2f}s")
        print(f"Tokens per second: ~{100/avg_time:.1f}")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MLX fine-tuned model")
    parser.add_argument("--model-path", 
                       default="finetuning/models/mistral-mlx-lora",
                       help="Path to the fine-tuned model")
    parser.add_argument("--base-model",
                       default=BASE_MLX_MODEL,
                       help="Base MLX model to load when model-path contains only adapters")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    
    args = parser.parse_args()
    
    print("MLX Fine-tuned Model Test")
    print("=" * 40)
    
    # Test model loading
    loaded = test_model_loading(args.model_path, base_model=args.base_model)
    if loaded is None:
        print("\n❌ Tests failed - could not load model")
        return
    model, tokenizer = loaded
    
    if model and tokenizer:
        # Create test prompts
        test_prompts = create_test_prompts()
        
        # Test inference
        test_inference(model, tokenizer, test_prompts)
        
        # Run benchmark if requested
        if args.benchmark:
            benchmark_performance(model, tokenizer, test_prompts[0])
        
        print("\n✅ All tests completed!")
    else:
        print("\n❌ Tests failed - could not load model")


if __name__ == "__main__":
    main()
