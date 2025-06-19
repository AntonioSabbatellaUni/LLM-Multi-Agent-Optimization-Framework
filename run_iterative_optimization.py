"""
Complete iterative optimization runner with convergence tracking.
Generates all requested visualizations showing BO convergence over iterations.
"""

import sys
import os
from datetime import datetime
from iterative_optimization import (
    IterativeMultiObjectiveOptimizer, 
    analyze_convergence,
    plot_convergence_analysis,
    plot_3d_pareto_evolution,
    save_convergence_results
)
from data_processor_v2 import create_evaluator
from basic_optimization import analyze_results, save_results

def create_timestamp_folder():
    """Create a timestamp-based output folder."""
    now = datetime.now()
    timestamp = now.strftime("%b_%d_%H_%M").lower()
    output_dir = f"outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir

def run_complete_convergence_analysis():
    """Run the complete optimization with convergence tracking."""
    
    print("="*80)
    print("🚀 ITERATIVE MULTI-OBJECTIVE LLM OPTIMIZATION")
    print("="*80)
    print("Tracking Bayesian Optimization convergence over iterations")
    print("Generating performance, cost, and 3D Pareto front evolution plots")
    print("="*80)
    
    # Step 0: Create timestamped output folder
    output_dir = create_timestamp_folder()
    
    # Step 1: Initialize system
    print("\n📊 STEP 1: INITIALIZING SYSTEM")
    print("-" * 50)
    
    evaluator = create_evaluator()
    optimizer = IterativeMultiObjectiveOptimizer(evaluator)
    
    # Step 2: Run iterative optimization with tracking
    print("\n🎯 STEP 2: RUNNING ITERATIVE OPTIMIZATION")
    print("-" * 50)
    
    results = optimizer.optimize_with_tracking(
        n_initial=15,           # Strategic initialization
        n_random_batches=8,     # 8 random search batches
        n_random_per_batch=25,  # 25 points per random batch
        n_local_batches=5,      # 5 local search batches  
        n_local_per_batch=20    # 20 points per local batch
    )
    
    print(f"\n✅ Optimization completed:")
    print(f"   - Total iterations: {results['n_iterations']}")
    print(f"   - Total evaluations: {results['n_evaluations']}")
    print(f"   - Final Pareto solutions: {results['n_pareto_solutions']}")
    
    # Step 3: Convergence analysis
    print("\n📈 STEP 3: CONVERGENCE ANALYSIS")
    print("-" * 50)
    
    convergence = analyze_convergence(results)
    
    print(f"🏆 Performance improvement: {convergence['performance_improvement']:.3f}")
    print(f"💰 Cost improvement: {convergence['cost_improvement']:.3f}")
    print(f"📊 Final Pareto size: {convergence['pareto_size_evolution'][-1]}")
    print(f"📈 Final hypervolume: {convergence['hypervolume_evolution'][-1]:.3f}")
    
    # Step 4: Generate visualizations
    print("\n🎨 STEP 4: GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    # 4.1: Convergence plots (performance, cost, pareto size, hypervolume)
    convergence_path = plot_convergence_analysis(results, f'{output_dir}/convergence')
    
    # 4.2: 3D Pareto front evolution
    pareto_3d_path = plot_3d_pareto_evolution(results, f'{output_dir}/pareto_3d_evolution.png')
    
    # 4.3: Traditional Pareto front (final)
    from basic_optimization import plot_pareto_front
    final_pareto_path = plot_pareto_front(results, f'{output_dir}/final_pareto_front.png')
    
    # Step 5: Save detailed results
    print("\n💾 STEP 5: SAVING RESULTS")
    print("-" * 50)
    
    # 5.1: Convergence data
    convergence_results_path = save_convergence_results(results, convergence, 
                                                       f'{output_dir}/convergence_results.json')
    
    # 5.2: Final optimization analysis  
    analysis = analyze_results(optimizer, results)
    final_results_path = save_results(analysis, f'{output_dir}/final_optimization_results.json')
    
    # Step 6: Summary report
    print("\n📋 STEP 6: SUMMARY REPORT")
    print("-" * 50)
    
    print_convergence_summary(convergence, results)
    
    # Step 7: File outputs
    print("\n📁 OUTPUT FILES GENERATED:")
    print("-" * 50)
    print(f"📊 Convergence Analysis: {convergence_path}")
    print(f"🌌 3D Pareto Evolution: {pareto_3d_path}")
    print(f"📈 Final Pareto Front: {final_pareto_path}")
    print(f"📄 Convergence Data: {convergence_results_path}")
    print(f"📄 Final Results: {final_results_path}")
    
    print(f"\n" + "="*80)
    print("🎉 ITERATIVE OPTIMIZATION ANALYSIS COMPLETE!")
    print("="*80)
    
    return results, convergence, {
        'output_dir': output_dir,
        'convergence_plot': convergence_path,
        'pareto_3d': pareto_3d_path,
        'final_pareto': final_pareto_path,
        'convergence_data': convergence_results_path,
        'final_results': final_results_path
    }

def print_convergence_summary(convergence: dict, results: dict):
    """Print a detailed convergence summary."""
    
    print("🔍 CONVERGENCE SUMMARY:")
    print(f"   Initial best performance: {convergence['best_performance_evolution'][0]:.3f}")
    print(f"   Final best performance: {convergence['best_performance_evolution'][-1]:.3f}")
    print(f"   Performance gain: +{convergence['performance_improvement']:.3f}")
    
    print(f"\n   Initial best cost: {convergence['best_cost_evolution'][0]:.3f}")
    print(f"   Final best cost: {convergence['best_cost_evolution'][-1]:.3f}")
    print(f"   Cost reduction: -{convergence['cost_improvement']:.3f}")
    
    print(f"\n   Initial Pareto size: {convergence['pareto_size_evolution'][0]}")
    print(f"   Final Pareto size: {convergence['pareto_size_evolution'][-1]}")
    
    print(f"\n   Initial hypervolume: {convergence['hypervolume_evolution'][0]:.3f}")
    print(f"   Final hypervolume: {convergence['hypervolume_evolution'][-1]:.3f}")
    print(f"   Hypervolume gain: +{convergence['hypervolume_evolution'][-1] - convergence['hypervolume_evolution'][0]:.3f}")
    
    # Find iteration with biggest improvements
    perf_improvements = [convergence['best_performance_evolution'][i] - convergence['best_performance_evolution'][i-1] 
                        for i in range(1, len(convergence['best_performance_evolution']))]
    
    if perf_improvements:
        best_perf_iter = np.argmax(perf_improvements) + 1
        print(f"\n🚀 Biggest performance jump at iteration {best_perf_iter}")
    
    cost_improvements = [convergence['best_cost_evolution'][i-1] - convergence['best_cost_evolution'][i] 
                        for i in range(1, len(convergence['best_cost_evolution']))]
    
    if cost_improvements:
        best_cost_iter = np.argmax(cost_improvements) + 1
        print(f"💰 Biggest cost reduction at iteration {best_cost_iter}")

def create_summary_visualization(output_dir: str):
    """Create a single summary plot with all key metrics."""
    
    try:
        import json
        with open(f'{output_dir}/convergence_results.json', 'r') as f:
            data = json.load(f)
        
        convergence = data['convergence_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        iterations = convergence['iterations']
        
        # Performance evolution
        ax1.plot(iterations, convergence['best_performance_evolution'], 'b-o', linewidth=3, markersize=8)
        ax1.set_title('🏆 Best Performance Over Iterations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Performance')
        ax1.grid(True, alpha=0.3)
        
        # Cost evolution
        ax2.plot(iterations, convergence['best_cost_evolution'], 'r-s', linewidth=3, markersize=8)
        ax2.set_title('💰 Best Cost Over Iterations', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best Cost (Lower is Better)')
        ax2.grid(True, alpha=0.3)
        
        # Pareto size evolution
        ax3.plot(iterations, convergence['pareto_size_evolution'], 'g-^', linewidth=3, markersize=8)
        ax3.set_title('📊 Pareto Front Size Over Iterations', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Number of Pareto Solutions')
        ax3.grid(True, alpha=0.3)
        
        # Hypervolume evolution
        ax4.plot(iterations, convergence['hypervolume_evolution'], 'm-d', linewidth=3, markersize=8)
        ax4.set_title('📈 Hypervolume Over Iterations', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Hypervolume')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = f'{output_dir}/optimization_summary.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary visualization saved to: {summary_path}")
        return summary_path
        
    except Exception as e:
        print(f"Warning: Could not create summary visualization: {e}")
        return None

if __name__ == "__main__":
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Run complete analysis
        results, convergence, file_paths = run_complete_convergence_analysis()
        output_dir = file_paths['output_dir']
        
        # Create summary visualization
        summary_path = create_summary_visualization(output_dir)
        if summary_path:
            file_paths['summary'] = summary_path
        
        print(f"\n🎯 SUCCESS: Complete iterative optimization analysis finished!")
        print(f"📊 Check the generated plots to see how the Bayesian optimization converged")
        print(f"🌌 The 3D plot shows how the Pareto front evolved over iterations")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)