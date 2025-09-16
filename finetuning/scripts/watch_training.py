#!/usr/bin/env python3
"""
Watch MLX training output with step tracking.
"""

import os
import time
import psutil
import subprocess
from pathlib import Path
import re

def find_training_process():
    """Find the MLX training process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'mlx_lm' in ' '.join(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return None

def watch_training():
    """Watch training with step tracking."""
    print("🔍 Watching MLX Training Progress")
    print("=" * 50)
    
    # Find training process
    process = find_training_process()
    if not process:
        print("❌ No MLX training process found")
        return
    
    print(f"✅ Found training process (PID: {process.pid})")
    print("🔄 Watching training progress...")
    print("=" * 50)
    
    # Training parameters (from the command line)
    total_iters = 300
    current_step = 0
    start_time = time.time()
    
    # Model directory
    model_dir = Path("finetuning/models/mistral-mlx-lora")
    
    try:
        while True:
            # Check if process is still running
            if not process.is_running():
                print("\n✅ Training process completed!")
                break
            
            # Get process info
            try:
                proc_info = process.as_dict(['cpu_percent', 'memory_percent', 'status'])
                
                # Estimate progress based on elapsed time
                elapsed = time.time() - start_time
                estimated_progress = min(95, (elapsed / 1800) * 100)  # Assume 30 min total
                
                # Check for new files
                new_files = []
                if model_dir.exists():
                    files = list(model_dir.glob("*"))
                    for file in files:
                        if file.is_file():
                            mtime = file.stat().st_mtime
                            if mtime > start_time:
                                new_files.append(file.name)
                
                # Display progress
                print(f"\r🔄 Step: {current_step}/{total_iters} | "
                      f"Progress: {estimated_progress:.1f}% | "
                      f"CPU: {proc_info['cpu_percent']}% | "
                      f"Memory: {proc_info['memory_percent']:.1f}% | "
                      f"Status: {proc_info['status']}", end='', flush=True)
                
                if new_files:
                    print(f"\n📄 New files: {', '.join(new_files)}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("\n❌ Process no longer available")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user")

def show_training_info():
    """Show training information."""
    process = find_training_process()
    if not process:
        print("❌ No training process found")
        return
    
    print("🔧 Training Information:")
    print("=" * 50)
    
    try:
        cmdline = process.info['cmdline']
        if cmdline:
            # Extract key parameters
            cmd_str = " ".join(cmdline)
            print(f"Command: {cmd_str}")
            print()
            
            # Extract parameters
            iters_match = re.search(r'--iters (\d+)', cmd_str)
            lr_match = re.search(r'--learning-rate ([\d.]+)', cmd_str)
            batch_match = re.search(r'--batch-size (\d+)', cmd_str)
            
            if iters_match:
                print(f"Total iterations: {iters_match.group(1)}")
            if lr_match:
                print(f"Learning rate: {lr_match.group(1)}")
            if batch_match:
                print(f"Batch size: {batch_match.group(1)}")
            
            print()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print("❌ Cannot access process info")

if __name__ == "__main__":
    show_training_info()
    watch_training()
