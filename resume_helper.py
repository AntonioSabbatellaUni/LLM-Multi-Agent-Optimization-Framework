#!/usr/bin/env python3
"""
Resume Helper for Multi-Agent LLM Optimization

This utility helps you resume optimization runs from checkpoints.
It automatically finds the most recent run and provides easy resume commands.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def find_recent_runs(output_dir: str = "outputs", limit: int = 10):
    """Find the most recent optimization runs."""
    if not os.path.exists(output_dir):
        print(f"❌ Output directory '{output_dir}' not found!")
        return []
    
    runs = []
    for item in os.listdir(output_dir):
        run_path = os.path.join(output_dir, item)
        if os.path.isdir(run_path):
            # Look for checkpoint files
            checkpoints = [f for f in os.listdir(run_path) if f.startswith('checkpoint_iter_')]
            if checkpoints:
                # Get the latest checkpoint
                latest_checkpoint = max(checkpoints)
                checkpoint_path = os.path.join(run_path, latest_checkpoint)
                
                try:
                    with open(checkpoint_path, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    runs.append({
                        'directory': item,
                        'full_path': run_path,
                        'latest_iteration': checkpoint_data.get('iteration', 0),
                        'n_evaluations': checkpoint_data.get('n_evaluations', 0),
                        'timestamp': checkpoint_data.get('timestamp', ''),
                        'checkpoints': len(checkpoints)
                    })
                except Exception as e:
                    print(f"⚠️  Could not read checkpoint {checkpoint_path}: {e}")
    
    # Sort by directory name (which includes timestamp)
    runs.sort(key=lambda x: x['directory'], reverse=True)
    return runs[:limit]


def get_run_info(run_path: str):
    """Get detailed information about a specific run."""
    if not os.path.exists(run_path):
        print(f"❌ Run directory '{run_path}' not found!")
        return None
    
    # Look for config files
    config_files = []
    if os.path.exists(os.path.join(run_path, 'config_used.yaml')):
        config_files.append('config_used.yaml')
    if os.path.exists(os.path.join(run_path, 'architecture_config_used.yaml')):
        config_files.append('architecture_config_used.yaml')
    
    # Get checkpoint info
    checkpoints = [f for f in os.listdir(run_path) if f.startswith('checkpoint_iter_')]
    checkpoints.sort()
    
    # Get the latest checkpoint details
    latest_info = None
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        try:
            with open(os.path.join(run_path, latest_checkpoint), 'r') as f:
                latest_info = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not read latest checkpoint: {e}")
    
    return {
        'path': run_path,
        'config_files': config_files,
        'checkpoints': checkpoints,
        'latest_checkpoint': latest_info
    }


def print_run_summary(runs):
    """Print a summary of recent runs."""
    print("📊 RECENT OPTIMIZATION RUNS")
    print("=" * 80)
    
    for i, run in enumerate(runs, 1):
        print(f"\n{i}. {run['directory']}")
        print(f"   📁 Path: {run['full_path']}")
        print(f"   🏃 Latest iteration: {run['latest_iteration']}")
        print(f"   📈 Evaluations: {run['n_evaluations']}")
        print(f"   💾 Checkpoints: {run['checkpoints']}")
        print(f"   🕐 Timestamp: {run['timestamp']}")


def generate_resume_command(run_path: str, iteration: int = None):
    """Generate the command to resume from a specific run."""
    base_cmd = "python run_multi_agent_optimization.py"
    resume_cmd = f"{base_cmd} --resume-from {run_path}"
    
    if iteration is not None:
        resume_cmd += f" --resume-iteration {iteration}"
    
    return resume_cmd


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Resume Helper for Multi-Agent LLM Optimization')
    parser.add_argument('--list', '-l', action='store_true', help='List recent runs')
    parser.add_argument('--info', '-i', type=str, help='Get detailed info about a specific run')
    parser.add_argument('--resume', '-r', type=str, help='Generate resume command for a specific run')
    parser.add_argument('--iteration', type=int, help='Specific iteration to resume from')
    parser.add_argument('--limit', type=int, default=10, help='Number of recent runs to show (default: 10)')
    
    args = parser.parse_args()
    
    if args.list:
        runs = find_recent_runs(limit=args.limit)
        if not runs:
            print("❌ No recent runs found!")
            return 1
        
        print_run_summary(runs)
        
        print(f"\n🔧 USAGE EXAMPLES:")
        print("-" * 50)
        if runs:
            latest_run = runs[0]
            print(f"📝 Resume latest run:")
            print(f"   {generate_resume_command(latest_run['full_path'])}")
            print(f"📝 Resume from specific iteration:")
            print(f"   {generate_resume_command(latest_run['full_path'], 0)}")
        
        return 0
    
    if args.info:
        run_info = get_run_info(args.info)
        if not run_info:
            return 1
        
        print(f"📊 RUN INFORMATION")
        print("=" * 80)
        print(f"📁 Path: {run_info['path']}")
        print(f"📋 Config files: {', '.join(run_info['config_files'])}")
        print(f"💾 Checkpoints: {len(run_info['checkpoints'])}")
        
        if run_info['checkpoints']:
            print(f"📝 Available checkpoints:")
            for checkpoint in run_info['checkpoints']:
                iteration = checkpoint.replace('checkpoint_iter_', '').replace('.json', '')
                print(f"   • {checkpoint} (iteration {int(iteration)})")
        
        if run_info['latest_checkpoint']:
            latest = run_info['latest_checkpoint']
            print(f"\n🏃 Latest checkpoint:")
            print(f"   • Iteration: {latest.get('iteration', 'unknown')}")
            print(f"   • Evaluations: {latest.get('n_evaluations', 'unknown')}")
            print(f"   • Timestamp: {latest.get('timestamp', 'unknown')}")
            
            if 'pareto_info' in latest:
                pareto = latest['pareto_info']
                print(f"   • Pareto solutions: {pareto.get('n_pareto_solutions', 'unknown')}")
                print(f"   • Best performance: {pareto.get('best_performance', 'unknown'):.3f}")
                print(f"   • Best cost: {pareto.get('best_cost', 'unknown'):.3f}")
        
        print(f"\n🔧 RESUME COMMANDS:")
        print("-" * 50)
        print(f"📝 Resume from latest:")
        print(f"   {generate_resume_command(run_info['path'])}")
        print(f"📝 Resume from specific iteration:")
        print(f"   {generate_resume_command(run_info['path'], 0)}")
        
        return 0
    
    if args.resume:
        command = generate_resume_command(args.resume, args.iteration)
        print(f"🔧 RESUME COMMAND:")
        print("-" * 50)
        print(command)
        
        # Also copy to clipboard if possible
        try:
            import pyperclip
            pyperclip.copy(command)
            print(f"📋 Command copied to clipboard!")
        except ImportError:
            print(f"💡 Install pyperclip to auto-copy commands: pip install pyperclip")
        
        return 0
    
    # Default: show recent runs
    print("🔧 RESUME HELPER")
    print("=" * 80)
    print("Usage examples:")
    print("  python resume_helper.py --list                    # List recent runs")
    print("  python resume_helper.py --info <run_directory>    # Get run details")
    print("  python resume_helper.py --resume <run_directory>  # Generate resume command")
    print("")
    print("Quick start:")
    print("  python resume_helper.py -l                        # Show recent runs")
    
    return 0


if __name__ == "__main__":
    exit(main())
