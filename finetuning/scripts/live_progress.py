#!/usr/bin/env python3
"""
Live real-time MLX training progress monitor.
"""

import os
import time
import psutil
import subprocess
from pathlib import Path
import threading
import queue

def get_training_process():
    """Find the MLX training process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'mlx_lm' in ' '.join(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return None

def monitor_model_files(model_dir, file_queue):
    """Monitor model directory for new files."""
    model_path = Path(model_dir)
    if not model_path.exists():
        return
    
    last_files = set()
    while True:
        try:
            current_files = set(f.name for f in model_path.glob("*"))
            new_files = current_files - last_files
            
            if new_files:
                for file in new_files:
                    file_path = model_path / file
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        file_queue.put(f"📄 New file: {file} ({size:,} bytes)")
            
            last_files = current_files
            time.sleep(2)
        except Exception as e:
            file_queue.put(f"❌ File monitoring error: {e}")
            break

def monitor_training_logs(process, log_queue):
    """Monitor training process output."""
    try:
        # Try to get process output if possible
        if hasattr(process, 'stdout') and process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    log_queue.put(f"📝 {line.strip()}")
    except Exception as e:
        log_queue.put(f"❌ Log monitoring error: {e}")

def live_progress():
    """Live progress monitoring."""
    print("🔍 Live MLX Training Progress Monitor")
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    # Find training process
    process = get_training_process()
    if not process:
        print("❌ No MLX training process found")
        return
    
    print(f"✅ Training process found (PID: {process.pid})")
    print(f"📊 Process: {process.info['name']}")
    print()
    
    # Queues for different types of updates
    file_queue = queue.Queue()
    log_queue = queue.Queue()
    
    # Start monitoring threads
    model_dir = "finetuning/models/mistral-mlx-lora"
    file_thread = threading.Thread(target=monitor_model_files, args=(model_dir, file_queue))
    file_thread.daemon = True
    file_thread.start()
    
    # Initial state
    start_time = time.time()
    last_cpu_time = 0
    last_memory = 0
    
    try:
        while True:
            # Clear screen and show header
            os.system('clear' if os.name == 'posix' else 'cls')
            print("🔍 Live MLX Training Progress Monitor")
            print("=" * 60)
            print(f"⏱️  Monitoring since: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
            print(f"🔄 Elapsed time: {int(time.time() - start_time)}s")
            print()
            
            # Process info
            try:
                proc_info = process.as_dict(['cpu_percent', 'memory_info', 'memory_percent', 'status'])
                print(f"📊 Process Status: {proc_info['status']}")
                print(f"💻 CPU Usage: {proc_info['cpu_percent']}%")
                print(f"🧠 Memory Usage: {proc_info['memory_percent']:.1f}%")
                print(f"📈 Memory: {proc_info['memory_info'].rss / 1024 / 1024:.1f} MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("❌ Process no longer available")
                break
            
            # System resources
            print(f"\n💻 System Resources:")
            print(f"   - CPU: {psutil.cpu_percent()}%")
            print(f"   - Memory: {psutil.virtual_memory().percent}%")
            print(f"   - Disk: {psutil.disk_usage('/').percent}%")
            
            # Model directory status
            model_path = Path(model_dir)
            if model_path.exists():
                files = list(model_path.glob("*"))
                print(f"\n📁 Model Directory: {len(files)} files")
                for file in files:
                    if file.is_file():
                        size = file.stat().st_size
                        mtime = time.strftime('%H:%M:%S', time.localtime(file.stat().st_mtime))
                        print(f"   - {file.name}: {size:,} bytes (modified: {mtime})")
            else:
                print(f"\n📁 Model Directory: Not created yet")
            
            # Training data info
            train_file = Path("finetuning/data/processed/mlx_data/train.jsonl")
            valid_file = Path("finetuning/data/processed/mlx_data/valid.jsonl")
            
            if train_file.exists() and valid_file.exists():
                train_lines = sum(1 for _ in open(train_file))
                valid_lines = sum(1 for _ in open(valid_file))
                print(f"\n📊 Training Data:")
                print(f"   - Training examples: {train_lines}")
                print(f"   - Validation examples: {valid_lines}")
                print(f"   - Total: {train_lines + valid_lines}")
            
            # Show recent updates
            print(f"\n🔄 Recent Updates:")
            updates_shown = 0
            while not file_queue.empty() and updates_shown < 5:
                try:
                    update = file_queue.get_nowait()
                    print(f"   {update}")
                    updates_shown += 1
                except queue.Empty:
                    break
            
            while not log_queue.empty() and updates_shown < 5:
                try:
                    update = log_queue.get_nowait()
                    print(f"   {update}")
                    updates_shown += 1
                except queue.Empty:
                    break
            
            if updates_shown == 0:
                print("   - No recent updates")
            
            print(f"\n⏳ Refreshing in 3 seconds... (Ctrl+C to stop)")
            time.sleep(3)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user")
        print(f"⏱️  Total monitoring time: {int(time.time() - start_time)}s")

if __name__ == "__main__":
    live_progress()
