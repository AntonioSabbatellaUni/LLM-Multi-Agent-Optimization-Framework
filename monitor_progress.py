#!/usr/bin/env python3
"""
Monitor BoTorch optimization progress by reading checkpoint files.
"""

import os
import json
import time
import argparse
from datetime import datetime

def find_latest_output_dir(base_dir="outputs"):
    """Find the most recent output directory."""
    if not os.path.exists(base_dir):
        return None
    
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    
    # Sort by modification time
    dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    return os.path.join(base_dir, dirs[0])

def load_checkpoint(checkpoint_path):
    """Load a checkpoint file."""
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except:
        return None

def monitor_progress(output_dir=None, interval=5):
    """Monitor optimization progress."""
    
    if output_dir is None:
        output_dir = find_latest_output_dir()
        if output_dir is None:
            print("❌ No output directory found!")
            return
    
    print(f"🔍 Monitoring progress in: {output_dir}")
    print("Press Ctrl+C to stop monitoring...")
    print("-" * 60)
    
    last_iteration = -1
    
    try:
        while True:
            # Find all checkpoint files
            checkpoint_files = []
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    if f.startswith('checkpoint_iter_') and f.endswith('.json'):
                        iteration = int(f.split('_')[2].split('.')[0])
                        checkpoint_files.append((iteration, f))
            
            # Sort by iteration
            checkpoint_files.sort()
            
            if not checkpoint_files:
                print("⏳ Waiting for first checkpoint...")
                time.sleep(interval)
                continue
            
            # Check if there's a new iteration
            latest_iteration = checkpoint_files[-1][0]
            
            if latest_iteration > last_iteration:
                # Load and display latest checkpoint
                latest_file = checkpoint_files[-1][1]
                checkpoint_path = os.path.join(output_dir, latest_file)
                checkpoint = load_checkpoint(checkpoint_path)
                
                if checkpoint:
                    print(f"\\n📊 Iteration {checkpoint['iteration']} - {datetime.fromisoformat(checkpoint['timestamp']).strftime('%H:%M:%S')}")
                    print(f"   Total evaluations: {checkpoint['n_evaluations']}")
                    print(f"   Pareto solutions: {checkpoint['pareto_info']['n_pareto_solutions']}")
                    print(f"   Best performance: {checkpoint['pareto_info']['best_performance']:.3f}")
                    print(f"   Best cost: {checkpoint['pareto_info']['best_cost']:.3f}")
                    
                    if 'new_candidates_count' in checkpoint:
                        print(f"   New candidates: {checkpoint['new_candidates_count']}")
                
                last_iteration = latest_iteration
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\\n\\n🛑 Monitoring stopped.")
    except Exception as e:
        print(f"\\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor BoTorch optimization progress")
    parser.add_argument("--dir", help="Output directory to monitor")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    monitor_progress(args.dir, args.interval)

if __name__ == "__main__":
    main()
