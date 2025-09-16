#!/usr/bin/env python3
"""
Follow MLX training output in real-time.
"""

import os
import time
import psutil
import subprocess
from pathlib import Path

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

def follow_training():
    """Follow training progress in real-time."""
    print("🔍 Following MLX Training Progress")
    print("=" * 50)
    
    # Find training process
    process = find_training_process()
    if not process:
        print("❌ No MLX training process found")
        return
    
    print(f"✅ Found training process (PID: {process.pid})")
    print("🔄 Following training output...")
    print("=" * 50)
    
    # Monitor model directory for changes
    model_dir = Path("finetuning/models/mistral-mlx-lora")
    last_files = set()
    
    try:
        while True:
            # Check if process is still running
            if not process.is_running():
                print("\n✅ Training process completed!")
                break
            
            # Check for new files in model directory
            if model_dir.exists():
                current_files = set(f.name for f in model_dir.glob("*"))
                new_files = current_files - last_files
                
                if new_files:
                    print(f"\n📄 New files created:")
                    for file in new_files:
                        file_path = model_dir / file
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            print(f"   - {file}: {size:,} bytes")
                    last_files = current_files
            
            # Show process status
            try:
                proc_info = process.as_dict(['cpu_percent', 'memory_percent', 'status'])
                print(f"\r🔄 Status: {proc_info['status']} | CPU: {proc_info['cpu_percent']}% | Memory: {proc_info['memory_percent']:.1f}%", end='', flush=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("\n❌ Process no longer available")
                break
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user")

def show_training_command():
    """Show the actual training command being run."""
    process = find_training_process()
    if not process:
        print("❌ No training process found")
        return
    
    try:
        cmdline = process.info['cmdline']
        if cmdline:
            print("🔧 Training Command:")
            print(" ".join(cmdline))
            print()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        print("❌ Cannot access process info")

if __name__ == "__main__":
    show_training_command()
    follow_training()
