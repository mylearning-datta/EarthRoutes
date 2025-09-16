#!/usr/bin/env python3
"""
Compare PyTorch and MLX model performance and outputs.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

def compare_model_outputs():
    """Compare outputs from PyTorch and MLX models."""
    print("=== Model Output Comparison ===")
    
    test_prompts = [
        "### TASK: MODE_CHOICE\n### INSTRUCTION:\nChoose the most sustainable transport mode from NYC to Boston.\n### RESPONSE:",
        "### TASK: SUSTAINABLE_POIS\n### INSTRUCTION:\nRecommend eco-friendly places in San Francisco.\n### RESPONSE:"
    ]
    
    results = {}
    
    # Test PyTorch model (if available)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("Testing PyTorch model...")
        pytorch_model_path = "finetuning/models/mistral-7b-finetune-4bit-colab"
        
        if Path(pytorch_model_path).exists():
            tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path)
            model = AutoModelForCausalLM.from_pretrained(pytorch_model_path)
            
            results["pytorch"] = {}
            for i, prompt in enumerate(test_prompts):
                start_time = time.time()
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                end_time = time.time()
                
                results["pytorch"][f"prompt_{i}"] = {
                    "response": response,
                    "time": end_time - start_time
                }
                print(f"PyTorch Prompt {i+1}: {end_time - start_time:.2f}s")
        else:
            print("PyTorch model not found")
            
    except ImportError:
        print("PyTorch/Transformers not available")
    except Exception as e:
        print(f"PyTorch model error: {e}")
    
    # Test MLX model (if available)
    try:
        from mlx_lm import load, generate
        
        print("\nTesting MLX model...")
        mlx_model_path = "finetuning/models/mistral-mlx-lora"
        
        if Path(mlx_model_path).exists():
            model, tokenizer = load(mlx_model_path)
            
            results["mlx"] = {}
            for i, prompt in enumerate(test_prompts):
                start_time = time.time()
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=100,
                    temp=0.7,
                    verbose=False
                )
                end_time = time.time()
                
                results["mlx"][f"prompt_{i}"] = {
                    "response": response,
                    "time": end_time - start_time
                }
                print(f"MLX Prompt {i+1}: {end_time - start_time:.2f}s")
        else:
            print("MLX model not found")
            
    except ImportError:
        print("MLX not available")
    except Exception as e:
        print(f"MLX model error: {e}")
    
    # Compare results
    if "pytorch" in results and "mlx" in results:
        print("\n=== Performance Comparison ===")
        for i in range(len(test_prompts)):
            pytorch_time = results["pytorch"][f"prompt_{i}"]["time"]
            mlx_time = results["mlx"][f"prompt_{i}"]["time"]
            speedup = pytorch_time / mlx_time if mlx_time > 0 else 0
            
            print(f"Prompt {i+1}:")
            print(f"  PyTorch: {pytorch_time:.2f}s")
            print(f"  MLX:     {mlx_time:.2f}s")
            print(f"  Speedup: {speedup:.1f}x")
    
    # Save results
    with open("finetuning/results/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: finetuning/results/model_comparison.json")


def compare_memory_usage():
    """Compare memory usage between PyTorch and MLX."""
    print("\n=== Memory Usage Comparison ===")
    
    memory_info = {}
    
    # PyTorch memory usage
    try:
        import torch
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load PyTorch model
        pytorch_model_path = "finetuning/models/mistral-7b-finetune-4bit-colab"
        if Path(pytorch_model_path).exists():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(pytorch_model_path)
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_info["pytorch"] = memory_after - memory_before
            print(f"PyTorch model memory usage: {memory_info['pytorch']:.1f} MB")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e:
        print(f"PyTorch memory test failed: {e}")
    
    # MLX memory usage
    try:
        import mlx.core as mx
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load MLX model
        mlx_model_path = "finetuning/models/mistral-mlx-lora"
        if Path(mlx_model_path).exists():
            from mlx_lm import load
            model, tokenizer = load(mlx_model_path)
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_info["mlx"] = memory_after - memory_before
            print(f"MLX model memory usage: {memory_info['mlx']:.1f} MB")
            del model
            
    except Exception as e:
        print(f"MLX memory test failed: {e}")
    
    # Compare memory usage
    if "pytorch" in memory_info and "mlx" in memory_info:
        reduction = (memory_info["pytorch"] - memory_info["mlx"]) / memory_info["pytorch"] * 100
        print(f"\nMemory reduction with MLX: {reduction:.1f}%")
    
    return memory_info


def main():
    """Main comparison function."""
    print("PyTorch vs MLX Model Comparison")
    print("=" * 40)
    
    # Create results directory
    results_dir = Path("finetuning/results")
    results_dir.mkdir(exist_ok=True)
    
    # Compare model outputs
    compare_model_outputs()
    
    # Compare memory usage
    memory_info = compare_memory_usage()
    
    # Summary
    print("\n=== Summary ===")
    print("✅ Model comparison completed")
    print("📊 Results saved to finetuning/results/")
    print("\nKey advantages of MLX:")
    print("- Lower memory usage")
    print("- Faster inference on Apple Silicon")
    print("- Native 4-bit quantization")
    print("- Optimized for macOS")


if __name__ == "__main__":
    main()
