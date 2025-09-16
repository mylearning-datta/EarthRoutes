#!/usr/bin/env python3
"""
Monitor MLX training progress.
"""

import os
import time
import psutil
from pathlib import Path

def monitor_training():
    """Monitor the MLX training process."""
    print("🔍 Monitoring MLX Training Progress")
    print("=" * 50)
    
    # Check if training process is running
    training_process = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'mlx_lm' in ' '.join(cmdline):
                training_process = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    
    if not training_process:
        print("❌ No MLX training process found")
        return
    
    print(f"✅ Training process found (PID: {training_process.pid})")
    print(f"📊 Process info: {training_process.info['name']}")
    
    # Monitor model directory
    model_dir = Path("finetuning/models/mistral-mlx-lora")
    
    print(f"\n📁 Model directory: {model_dir}")
    
    if model_dir.exists():
        files = list(model_dir.glob("*"))
        print(f"📄 Files in model directory: {len(files)}")
        for file in files:
            size = file.stat().st_size if file.is_file() else 0
            print(f"   - {file.name}: {size:,} bytes")
    else:
        print("📁 Model directory not created yet")
    
    # Monitor system resources
    print(f"\n💻 System Resources:")
    print(f"   - CPU usage: {psutil.cpu_percent()}%")
    print(f"   - Memory usage: {psutil.virtual_memory().percent}%")
    
    # Check process resources
    try:
        proc_info = training_process.as_dict(['cpu_percent', 'memory_info', 'memory_percent'])
        print(f"   - Training process CPU: {proc_info['cpu_percent']}%")
        print(f"   - Training process memory: {proc_info['memory_percent']:.1f}%")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print("   - Process info not available")
    
    # Data summary
    print(f"\n📊 Training Data Summary:")
    train_file = Path("finetuning/data/processed/mlx_data/train.jsonl")
    valid_file = Path("finetuning/data/processed/mlx_data/valid.jsonl")
    
    if train_file.exists():
        train_lines = sum(1 for _ in open(train_file))
        print(f"   - Training examples: {train_lines}")
    
    if valid_file.exists():
        valid_lines = sum(1 for _ in open(valid_file))
        print(f"   - Validation examples: {valid_lines}")
    
    print(f"\n⏱️  Training started at: {time.strftime('%H:%M:%S')}")
    print("🔄 Training is in progress...")

if __name__ == "__main__":
    monitor_training()
