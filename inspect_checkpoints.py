#!/usr/bin/env python3
"""
Utility to inspect BoTorch checkpoint files.
Shows what data is saved at each iteration.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

def inspect_checkpoint_file(checkpoint_path):
    """Inspect a single checkpoint file."""
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    print(f"📋 Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"   ⏰ Timestamp: {data['timestamp']}")
    print(f"   🔄 Iteration: {data['iteration']}")
    print(f"   📊 Total evaluations: {len(data['X_train'])}")
    if 'X_candidates' in data:
        print(f"   🎯 New candidates: {len(data['X_candidates'])}")
    
    # Show objective statistics
    Y_train = np.array(data['Y_train'])
    performance = Y_train[:, 0]
    cost = -Y_train[:, 1]  # Convert back from negative cost
    
    print(f"   🏆 Best performance: {performance.max():.3f}")
    print(f"   💰 Best cost: {cost.min():.3f}")
    print(f"   📈 Performance range: [{performance.min():.3f}, {performance.max():.3f}]")
    print(f"   💸 Cost range: [{cost.min():.3f}, {cost.max():.3f}]")
    print()

def inspect_experiment_folder(experiment_folder):
    """Inspect all checkpoint files in an experiment folder."""
    checkpoint_files = sorted(Path(experiment_folder).glob("checkpoint_iter_*.json"))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {experiment_folder}")
        return
    
    print(f"📁 Experiment folder: {experiment_folder}")
    print(f"📋 Found {len(checkpoint_files)} checkpoint files")
    print("="*60)
    
    for checkpoint_file in checkpoint_files:
        inspect_checkpoint_file(checkpoint_file)

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoints.py <experiment_folder>")
        print("\nExample:")
        print("  python inspect_checkpoints.py outputs/jun_19_16_26_botorch_multi_objective_llm")
        print("\nOr find the latest experiment:")
        output_dirs = sorted(Path("outputs").glob("*botorch*"), key=os.path.getmtime, reverse=True)
        if output_dirs:
            print(f"\nLatest experiment folder: {output_dirs[0]}")
        sys.exit(1)
    
    experiment_folder = sys.argv[1]
    
    if not os.path.exists(experiment_folder):
        print(f"❌ Folder does not exist: {experiment_folder}")
        sys.exit(1)
    
    inspect_experiment_folder(experiment_folder)

if __name__ == "__main__":
    main()
