"""
BoTorch-based optimization pipeline runner.
Demonstrates the complete multi-objective LLM optimization system using true Bayesian Optimization.
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
from botorch_optimization import BoTorchOptimizer
from data_processor_v2 import create_evaluator
from basic_optimization import analyze_results, plot_pareto_front, save_results

def create_timestamp_folder():
    """Create a timestamp-based output folder."""
    now = datetime.now()
    timestamp = now.strftime("%b_%d_%H_%M").lower()
    output_dir = f"outputs/{timestamp}_botorch"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir

def print_banner():
    """Print a nice banner."""
    print("="*80)
    print("🚀 BOTORCH MULTI-OBJECTIVE LLM OPTIMIZATION SYSTEM")
    print("="*80)
    print("Finding optimal Performance vs Cost trade-offs using true Bayesian Optimization")
    print("Using Gaussian Process models with qLogExpectedHypervolumeImprovement")
    print("="*80)

def convert_torch_results_to_basic_format(optimizer, torch_results):
    """Convert BoTorch results to the format expected by existing analysis functions."""
    # Extract results - these are already numpy arrays from the BoTorch optimizer
    X = torch_results['X_all']
    Y = torch_results['Y_all']
    pareto_mask = torch_results['pareto_mask']
    
    # Convert to the format expected by analyze_results
    results = {
        'X': X,
        'Y': Y,
        'pareto_mask': pareto_mask,
        'pareto_X': X[pareto_mask],
        'pareto_Y': Y[pareto_mask],
        'n_evaluations': len(X),
        'optimization_history': []  # BoTorch doesn't track this separately
    }
    
    return results

def demonstrate_botorch_optimization():
    """Run the complete BoTorch optimization demonstration."""
    
    print_banner()
    
    # Step 0: Create timestamped output folder
    output_dir = create_timestamp_folder()
    
    # Step 1: Initialize the system
    print("\n📊 STEP 1: LOADING DATA & INITIALIZING BOTORCH SYSTEM")
    print("-" * 50)
    
    evaluator = create_evaluator()
    
    # Set up BoTorch-specific parameters
    dimension = evaluator.n_agents * len(evaluator.feature_names)
    
    # Define bounds for the continuous optimization space [0, 1]
    bounds = torch.tensor([[0.0] * dimension, [1.0] * dimension], dtype=torch.double)
    
    # Define reference point for hypervolume (slightly below worst possible performance)
    # Performance is maximized (so negative ref point), Cost is minimized (positive ref point)
    ref_point = torch.tensor([-0.1, 1.1], dtype=torch.double)  # [performance, cost]
    
    optimizer = BoTorchOptimizer(evaluator, bounds=bounds, ref_point=ref_point)
    
    # Show some sample LLMs
    print(f"\nSample LLMs available:")
    for i in [0, 5, 10, 15, 20]:
        if i < len(evaluator.llm_names):
            print(f"  • {evaluator.llm_names[i]}")
    
    print(f"\nOptimization setup:")
    print(f"  • Dimension: {dimension}")
    print(f"  • Device: {optimizer.device}")
    print(f"  • Reference point: {ref_point.tolist()}")
    
    # Step 2: Run BoTorch optimization
    print(f"\n🎯 STEP 2: RUNNING BOTORCH MULTI-OBJECTIVE OPTIMIZATION")
    print("-" * 50)
    
    torch_results = optimizer.optimize(
        n_initial=15,     # Initial random samples  
        n_iterations=25   # BO iterations
    )
    
    # Convert torch results to format expected by existing analysis functions
    results = convert_torch_results_to_basic_format(optimizer, torch_results)
    
    # Step 3: Analyze results
    print(f"\n📈 STEP 3: ANALYZING RESULTS")
    print("-" * 50)
    
    analysis = analyze_results(optimizer, results)
    
    # Step 4: Generate outputs
    print(f"\n💾 STEP 4: GENERATING OUTPUTS")
    print("-" * 50)
    
    plot_path = plot_pareto_front(results, f'{output_dir}/botorch_pareto_optimization.png')
    results_path = save_results(analysis, f'{output_dir}/botorch_optimization_results.json')
    
    # Step 5: Summary
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
    print(f"  • Visualization: {plot_path}")
    print(f"  • Detailed Results: {results_path}")
    
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
        for i, agent_config in enumerate(agents):
            llm_name = agent_config['llm']
            print(f"  Agent {i+1}: {llm_name}")
    
    print(f"\n" + "="*80)
    print("🎉 BOTORCH OPTIMIZATION COMPLETE!")
    print("="*80)
    
    return analysis

def main():
    """Main entry point."""
    try:
        analysis = demonstrate_botorch_optimization()
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
