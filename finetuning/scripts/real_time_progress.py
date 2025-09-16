#!/usr/bin/env python3
"""
Real-time MLX training progress monitor with step tracking.
"""

import os
import time
import psutil
import subprocess
from pathlib import Path
import re
import threading
import queue

class MLXTrainingMonitor:
    def __init__(self):
        self.process = None
        self.model_dir = Path("finetuning/models/mistral-mlx-lora")
        self.total_iters = 300  # From the command line
        self.current_step = 0
        self.last_checkpoint = 0
        self.start_time = time.time()
        self.log_queue = queue.Queue()
        
    def find_training_process(self):
        """Find the MLX training process."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'mlx_lm' in ' '.join(cmdline):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        return None
    
    def extract_progress_from_logs(self):
        """Try to extract progress from any available logs."""
        # Check if there are any log files
        log_files = list(self.model_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_log, 'r') as f:
                    content = f.read()
                    # Look for step patterns
                    step_matches = re.findall(r'step (\d+)', content, re.IGNORECASE)
                    if step_matches:
                        self.current_step = int(step_matches[-1])
                        return True
            except Exception:
                pass
        return False
    
    def estimate_progress_from_files(self):
        """Estimate progress based on file creation and modification."""
        if not self.model_dir.exists():
            return 0
        
        files = list(self.model_dir.glob("*"))
        if not files:
            return 0
        
        # Check for adapter files
        adapter_files = [f for f in files if 'adapter' in f.name.lower()]
        if adapter_files:
            # If we have adapter files, we're making progress
            latest_file = max(adapter_files, key=lambda x: x.stat().st_mtime)
            file_age = time.time() - latest_file.stat().st_mtime
            
            # Estimate based on file age and expected training time
            if file_age < 60:  # File created in last minute
                return min(95, (time.time() - self.start_time) / 1800 * 100)  # Assume 30 min total
        
        return 0
    
    def get_training_status(self):
        """Get comprehensive training status."""
        if not self.process:
            self.process = self.find_training_process()
            if not self.process:
                return None
        
        try:
            # Check if process is still running
            if not self.process.is_running():
                return {"status": "completed", "progress": 100}
            
            # Get process info
            proc_info = self.process.as_dict(['cpu_percent', 'memory_percent', 'status'])
            
            # Try to extract progress
            if self.extract_progress_from_logs():
                progress = (self.current_step / self.total_iters) * 100
            else:
                progress = self.estimate_progress_from_files()
            
            return {
                "status": proc_info['status'],
                "progress": progress,
                "current_step": self.current_step,
                "total_iters": self.total_iters,
                "cpu_percent": proc_info['cpu_percent'],
                "memory_percent": proc_info['memory_percent'],
                "elapsed_time": time.time() - self.start_time
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"status": "error", "progress": 0}
    
    def display_progress_bar(self, progress, width=50):
        """Display a progress bar."""
        filled = int(width * progress / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {progress:.1f}%"
    
    def format_time(self, seconds):
        """Format time in human readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {int(seconds%60)}s"
        else:
            return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
    
    def run_monitor(self):
        """Run the real-time monitor."""
        print("🔍 Real-time MLX Training Progress Monitor")
        print("=" * 60)
        print("Press Ctrl+C to stop monitoring")
        print()
        
        # Find training process
        self.process = self.find_training_process()
        if not self.process:
            print("❌ No MLX training process found")
            return
        
        print(f"✅ Training process found (PID: {self.process.pid})")
        print(f"🎯 Target iterations: {self.total_iters}")
        print(f"⏱️  Started monitoring at: {time.strftime('%H:%M:%S')}")
        print()
        
        try:
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Get status
                status = self.get_training_status()
                if not status:
                    print("❌ Cannot get training status")
                    break
                
                # Header
                print("🔍 Real-time MLX Training Progress Monitor")
                print("=" * 60)
                print(f"⏱️  Monitoring since: {time.strftime('%H:%M:%S', time.localtime(self.start_time))}")
                print(f"🔄 Elapsed time: {self.format_time(status['elapsed_time'])}")
                print()
                
                # Progress bar
                progress = status['progress']
                print(f"📊 Training Progress:")
                print(f"   {self.display_progress_bar(progress)}")
                print(f"   Step: {status['current_step']}/{status['total_iters']}")
                print(f"   Status: {status['status']}")
                print()
                
                # Process info
                print(f"💻 Process Information:")
                print(f"   - CPU Usage: {status['cpu_percent']}%")
                print(f"   - Memory Usage: {status['memory_percent']:.1f}%")
                print(f"   - Process Status: {status['status']}")
                print()
                
                # System resources
                print(f"🖥️  System Resources:")
                print(f"   - CPU: {psutil.cpu_percent()}%")
                print(f"   - Memory: {psutil.virtual_memory().percent}%")
                print()
                
                # Model directory status
                if self.model_dir.exists():
                    files = list(self.model_dir.glob("*"))
                    print(f"📁 Model Directory: {len(files)} files")
                    for file in files:
                        if file.is_file():
                            size = file.stat().st_size
                            mtime = time.strftime('%H:%M:%S', time.localtime(file.stat().st_mtime))
                            print(f"   - {file.name}: {size:,} bytes (modified: {mtime})")
                else:
                    print(f"📁 Model Directory: Not created yet")
                
                # Training data info
                train_file = Path("finetuning/data/processed/mlx_data/train.jsonl")
                valid_file = Path("finetuning/data/processed/mlx_data/valid.jsonl")
                
                if train_file.exists() and valid_file.exists():
                    train_lines = sum(1 for _ in open(train_file))
                    valid_lines = sum(1 for _ in open(valid_file))
                    print(f"\n📊 Training Data:")
                    print(f"   - Training examples: {train_lines}")
                    print(f"   - Validation examples: {valid_lines}")
                
                # ETA calculation
                if progress > 0 and status['elapsed_time'] > 0:
                    eta_seconds = (status['elapsed_time'] / progress) * (100 - progress)
                    print(f"\n⏳ Estimated time remaining: {self.format_time(eta_seconds)}")
                
                # Check if completed
                if status['status'] == 'completed' or progress >= 100:
                    print(f"\n🎉 Training completed!")
                    break
                
                print(f"\n⏳ Refreshing in 2 seconds... (Ctrl+C to stop)")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user")
            print(f"⏱️  Total monitoring time: {self.format_time(time.time() - self.start_time)}")

def main():
    monitor = MLXTrainingMonitor()
    monitor.run_monitor()

if __name__ == "__main__":
    main()
