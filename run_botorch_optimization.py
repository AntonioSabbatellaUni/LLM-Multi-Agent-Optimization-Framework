"""
BoTorch-based optimization pipeline runner with YAML configuration.
Demonstrates the complete multi-objective LLM optimization system using true Bayesian Optimization.
"""

import sys
import os
import torch
import numpy as np
import yaml
import shutil
import argparse
from datetime import datetime
from botorch_optimization import BoTorchOptimizer
from data_processor_v2 import create_evaluator
from basic_optimization import analyze_results, plot_pareto_front, save_results

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file {config_path} not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML configuration: {e}")
        sys.exit(1)

def create_timestamp_folder(base_name: str = "botorch") -> str:
    """Create a timestamp-based output folder."""
    now = datetime.now()
    timestamp = now.strftime("%b_%d_%H_%M").lower()
    output_dir = f"outputs/{timestamp}_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir

def copy_config_to_output(config_path: str, output_dir: str):
    """Copy configuration file to output directory for reproducibility."""
    config_copy_path = os.path.join(output_dir, "config_used.yaml")
    shutil.copy2(config_path, config_copy_path)
    print(f"📋 Configuration copied to: {config_copy_path}")

def print_banner():
    """Print a nice banner."""
    print("="*80)
    print("🚀 BOTORCH MULTI-OBJECTIVE LLM OPTIMIZATION SYSTEM")
    print("="*80)
    print("Finding optimal Performance vs Cost trade-offs using true Bayesian Optimization")
    print("Using Gaussian Process models with qLogExpectedHypervolumeImprovement")
    print("="*80)

def demonstrate_botorch_optimization(config_path: str = "config.yaml", resume_from: str = None, resume_iteration: int = None):
    """Run the complete BoTorch optimization demonstration."""
    
    print_banner()
    
    # Step 0: Load configuration
    config = load_config(config_path)
    
    # Step 1: Create timestamped output folder
    output_dir = create_timestamp_folder(config['experiment']['name'].lower())
    
    # Copy configuration for reproducibility
    copy_config_to_output(config_path, output_dir)
    
    # Step 2: Initialize the system
    print("\n📊 STEP 1: LOADING DATA & INITIALIZING BOTORCH SYSTEM")
    print("-" * 50)
    
    evaluator = create_evaluator()
    
    # Set up BoTorch parameters from config
    dimension = evaluator.n_agents * len(evaluator.feature_names)
    
    # Define bounds from config
    bounds = torch.tensor([
        [config['bounds']['lower']] * dimension, 
        [config['bounds']['upper']] * dimension
    ], dtype=torch.double)
    
    # Define reference point from config
    ref_point = torch.tensor([
        config['reference_point']['performance'], 
        config['reference_point']['cost']
    ], dtype=torch.double)
    
    # Initialize optimizer with output directory for checkpoints
    optimizer = BoTorchOptimizer(evaluator, bounds=bounds, ref_point=ref_point, output_dir=output_dir)
    
    # Show some sample LLMs
    print(f"\nSample LLMs available:")
    for i in [0, 5, 10, 15, 20]:
        if i < len(evaluator.llm_names):
            print(f"  • {evaluator.llm_names[i]}")
    
    print(f"\nOptimization setup:")
    print(f"  • Dimension: {dimension}")
    print(f"  • Device: {optimizer.device}")
    print(f"  • Reference point: {ref_point.tolist()}")
    print(f"  • Initial points: {config['optimization']['n_initial']}")
    print(f"  • Iterations: {config['optimization']['n_iterations']}")
    print(f"  • Batch size: {config['optimization']['batch_size']}")
    
    # Step 3: Run BoTorch optimization
    print(f"\n🎯 STEP 2: RUNNING BOTORCH MULTI-OBJECTIVE OPTIMIZATION")
    print("-" * 50)
    
    # Show resume info if resuming
    if resume_from:
        from utils.checkpoint_manager import CheckpointManager
        print(f"🔁 RESUMING from checkpoint: {resume_from}")
        if resume_iteration is not None:
            print(f"   Starting from iteration: {resume_iteration}")
        else:
            print(f"   Starting from latest iteration")
        
        # Create resume info file and copy checkpoints
        resume_info_path = CheckpointManager.create_resume_info(output_dir, resume_from, resume_iteration)
        print(f"📝 Resume info saved to: {resume_info_path}")
        
        # Load and show copy details
        import json
        with open(resume_info_path, 'r') as f:
            resume_info = json.load(f)
        
        copied_count = resume_info['resume_details']['total_copied']
        source_iter = resume_info['resumed_from']['source_iteration']
        print(f"📁 Copied {copied_count} checkpoint files (iterations 0-{source_iter}) to new folder")
    
    torch_results = optimizer.optimize(
        n_initial=config['optimization']['n_initial'],
        n_iterations=config['optimization']['n_iterations'],
        q=config['optimization']['batch_size'],
        save_interval=1,  # Save checkpoints every iteration
        output_dir=output_dir,  # Pass the output directory
        resume_from=resume_from,  # Resume functionality
        resume_iteration=resume_iteration  # Specific iteration to resume from
    )
    
    # Step 4: Analyze results (convert format for compatibility)
    print(f"\n📈 STEP 3: ANALYZING RESULTS")
    print("-" * 50)
    
    # Convert torch results to format expected by existing analysis functions
    # Note: X and X_all contain the same data (all evaluations)
    # Note: X_pareto and pareto_X contain the same data (Pareto solutions only)
    results = {
        # Primary keys used by analyze_results function
        'X_all': torch_results['X_all'],      # All evaluations made
        'Y_all': torch_results['Y_all'],      # All objectives evaluated
        'X_pareto': torch_results['X_pareto'], # Pareto-optimal solutions only
        'Y_pareto': torch_results['Y_pareto'], # Pareto objectives only
        'pareto_mask': torch_results['pareto_mask'], # Boolean mask for Pareto solutions
        'n_evaluations': torch_results['n_evaluations'],
        
        # Legacy compatibility keys (same data as above)
        'X': torch_results['X_all'],           # Same as X_all (for compatibility)
        'Y': torch_results['Y_all'],           # Same as Y_all (for compatibility)
        'pareto_X': torch_results['X_pareto'], # Same as X_pareto (for compatibility)
        'pareto_Y': torch_results['Y_pareto'], # Same as Y_pareto (for compatibility)
        
        'optimization_history': []  # BoTorch tracks this in checkpoints instead
    }
    
    analysis = analyze_results(optimizer, results)
    
    # Step 5: Generate outputs
    print(f"\n💾 STEP 4: GENERATING OUTPUTS")
    print("-" * 50)
    
    plot_path = plot_pareto_front(results, f'{output_dir}/botorch_pareto_optimization.png')
    results_path = save_results(analysis, f'{output_dir}/botorch_optimization_results.json')
    
    # Also save the torch-specific results
    torch_results_path = f'{output_dir}/botorch_raw_results.json'
    with open(torch_results_path, 'w') as f:
        # Convert torch tensors to lists for JSON serialization
        serializable_results = {
            k: v.tolist() if hasattr(v, 'tolist') else v 
            for k, v in torch_results.items()
        }
        import json
        json.dump(serializable_results, f, indent=2)
    
    # Step 6: Summary
    print(f"\n✨ STEP 5: SUMMARY")
    print("-" * 50)
    
    n_pareto = len(analysis['pareto_solutions'])
    best_perf = analysis['recommendations']['highest_performance']
    best_cost = analysis['recommendations']['lowest_cost']
    best_balance = analysis['recommendations']['best_balance']
    
    print(f"📊 Found {n_pareto} Pareto-optimal solutions from {results['n_evaluations']} evaluations")
    print(f"🏆 Best Performance: {best_perf['performance']:.3f} (Cost: {best_perf['cost']:.3f})")
    print(f"💰 Best Cost: {best_cost['cost']:.3f} (Performance: {best_cost['performance']:.3f})")
    print(f"⚖️  Best Balance: Performance {best_balance['performance']:.3f}, Cost {best_balance['cost']:.3f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Configuration: {output_dir}/config_used.yaml")
    print(f"  • Checkpoints: {output_dir}/checkpoint_iter_*.json")
    print(f"  • Visualization: {plot_path}")
    print(f"  • Analysis Results: {results_path}")
    print(f"  • Raw BoTorch Results: {torch_results_path}")
    
    # Show example configurations
    print(f"\n🔧 EXAMPLE CONFIGURATIONS:")
    print("-" * 50)
    
    configs_to_show = [
        ('🏆 Highest Performance', best_perf),
        ('💰 Lowest Cost', best_cost),
        ('⚖️ Best Balance', best_balance)
    ]
    
    for label, config in configs_to_show:
        print(f"\n{label}:")
        agents = config['agents']
        for agent_config in agents:
            agent_id = agent_config['agent_id']
            role = agent_config['role']
            llm_name = agent_config['assigned_llm']
            print(f"  Agent {agent_id} ({role}): {llm_name}")
    
    # Count checkpoint files
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_iter_')]
    print(f"\n📊 OPTIMIZATION TRACKING:")
    print(f"  • {len(checkpoint_files)} checkpoint files saved")
    print(f"  • Each checkpoint contains X, Y, Pareto info, and new candidates")
    
    print(f"\n" + "="*80)
    print("🎉 BOTORCH OPTIMIZATION COMPLETE!")
    print("="*80)
    
    return analysis

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run BoTorch optimization with optional checkpoint resume')
    parser.add_argument('--config', default='config.yaml', help='Path to config file (default: config.yaml)')
    parser.add_argument('--resume-from', type=str, help='Directory to resume from checkpoint')
    parser.add_argument('--resume-iteration', type=int, help='Specific iteration to resume from (optional)')
    
    args = parser.parse_args()
    config_path = args.config
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"❌ Configuration file '{config_path}' not found!")
        print("Please ensure config.yaml exists in the current directory.")
        return 1
    
    try:
        analysis = demonstrate_botorch_optimization(config_path, args.resume_from, args.resume_iteration)
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
