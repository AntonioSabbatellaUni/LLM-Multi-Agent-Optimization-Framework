"""
3D Pareto Front Evolution Visualization for BoTorch Optimization.
Creates a 3D plot showing how the Pareto front evolves over iterations.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from datetime import datetime


def load_checkpoint_data(experiment_folder):
    """Load all checkpoint data from an experiment folder."""
    checkpoint_files = glob.glob(os.path.join(experiment_folder, "checkpoint_iter_*.json"))
    checkpoint_files.sort()  # Sort by filename to get correct order
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {experiment_folder}")
        return None
    
    print(f"📋 Found {len(checkpoint_files)} checkpoint files")
    
    checkpoints = []
    for checkpoint_file in checkpoint_files:
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            checkpoints.append(data)
        except Exception as e:
            print(f"⚠️  Failed to load {checkpoint_file}: {e}")
            continue
    
    return checkpoints


def compute_pareto_front(Y):
    """Compute Pareto front from objective values."""
    if len(Y) == 0:
        return np.array([])
    
    Y = np.array(Y)
    n_points = Y.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_pareto[i]:
            # Check if this point is dominated by any other point
            dominated = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
            is_pareto[dominated] = False
    
    return is_pareto


def create_3d_pareto_evolution(checkpoints, output_path):
    """Create 3D visualization of Pareto front evolution."""
    print(f"🎨 Creating 3D Pareto evolution plot...")
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different iterations
    colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoints)))
    
    # Track all points for proper axis limits
    all_performance = []
    all_cost = []
    all_iterations = []
    
    pareto_evolution_data = []
    
    for i, checkpoint in enumerate(checkpoints):
        iteration = checkpoint['iteration']
        Y_train = np.array(checkpoint['train_y'])
        
        if len(Y_train) == 0:
            continue
            
        # Extract performance and cost
        performance = Y_train[:, 0]  # First objective
        cost = -Y_train[:, 1]        # Second objective (negated because we minimize cost)
        
        # Find Pareto front for this iteration
        pareto_mask = compute_pareto_front(Y_train)
        pareto_performance = performance[pareto_mask]
        pareto_cost = cost[pareto_mask]
        
        # Store for axis limits
        all_performance.extend(performance)
        all_cost.extend(cost)
        all_iterations.extend([iteration] * len(performance))
        
        # Plot all points (light)
        ax.scatter(performance, cost, [iteration] * len(performance), 
                  c=[colors[i]], alpha=0.3, s=20, label=f'Iter {iteration} (all)')
        
        # Plot Pareto front (bright)
        if len(pareto_performance) > 0:
            ax.scatter(pareto_performance, pareto_cost, [iteration] * len(pareto_performance),
                      c=[colors[i]], alpha=1.0, s=60, marker='D', 
                      edgecolors='black', linewidth=1,
                      label=f'Iter {iteration} (Pareto)')
            
            # Store Pareto evolution data
            pareto_evolution_data.append({
                'iteration': iteration,
                'n_total': len(performance),
                'n_pareto': len(pareto_performance),
                'best_performance': np.max(pareto_performance),
                'best_cost': np.min(pareto_cost),
                'pareto_points': list(zip(pareto_performance, pareto_cost))
            })
    
    # Set labels and title
    ax.set_xlabel('Performance', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_zlabel('Iteration', fontsize=12)
    ax.set_title('BoTorch Pareto Front Evolution\n(3D: Performance vs Cost vs Iteration)', fontsize=14)
    
    # Set axis limits with some padding
    if all_performance and all_cost:
        perf_min, perf_max = min(all_performance), max(all_performance)
        cost_min, cost_max = min(all_cost), max(all_cost)
        perf_range = perf_max - perf_min
        cost_range = cost_max - cost_min
        
        ax.set_xlim(perf_min - 0.1 * perf_range, perf_max + 0.1 * perf_range)
        ax.set_ylim(cost_min - 0.1 * cost_range, cost_max + 0.1 * cost_range)
        ax.set_zlim(-0.5, max(all_iterations) + 0.5)
    
    # Customize the plot
    ax.grid(True, alpha=0.3)
    
    # Create a simplified legend (avoid too many entries)
    handles, labels = ax.get_legend_handles_labels()
    # Keep only Pareto front entries
    pareto_handles = [h for h, l in zip(handles, labels) if 'Pareto' in l]
    pareto_labels = [l for l in labels if 'Pareto' in l]
    
    if len(pareto_handles) <= 8:  # If not too many iterations
        ax.legend(pareto_handles, pareto_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 3D Pareto evolution plot saved to: {output_path}")
    
    return pareto_evolution_data


def create_2d_evolution_plots(checkpoints, output_folder):
    """Create 2D plots showing evolution over iterations."""
    print(f"📈 Creating 2D evolution plots...")
    
    iterations = []
    n_evaluations = []
    n_pareto_solutions = []
    best_performance = []
    best_cost = []
    hypervolume_proxy = []
    
    for checkpoint in checkpoints:
        iteration = checkpoint['iteration']
        Y_train = np.array(checkpoint['train_y'])
        
        if len(Y_train) == 0:
            continue
            
        performance = Y_train[:, 0]
        cost = -Y_train[:, 1]
        
        # Find Pareto front
        pareto_mask = compute_pareto_front(Y_train)
        pareto_performance = performance[pareto_mask]
        pareto_cost = cost[pareto_mask]
        
        iterations.append(iteration)
        n_evaluations.append(len(Y_train))
        n_pareto_solutions.append(len(pareto_performance))
        best_performance.append(np.max(performance) if len(performance) > 0 else 0)
        best_cost.append(np.min(cost) if len(cost) > 0 else 0)
        
        # Simple hypervolume proxy (area under Pareto front)
        if len(pareto_performance) > 1:
            # Sort by performance
            sorted_idx = np.argsort(pareto_performance)
            sorted_perf = pareto_performance[sorted_idx]
            sorted_cost = pareto_cost[sorted_idx]
            # Simple approximation of area
            hv_proxy = np.sum(sorted_perf * (np.max(sorted_cost) - sorted_cost))
        else:
            hv_proxy = pareto_performance[0] if len(pareto_performance) > 0 else 0
        
        hypervolume_proxy.append(hv_proxy)
    
    # Create 2D plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Number of evaluations and Pareto solutions over iterations
    ax1.plot(iterations, n_evaluations, 'b-o', label='Total Evaluations', linewidth=2)
    ax1.plot(iterations, n_pareto_solutions, 'r-s', label='Pareto Solutions', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Count')
    ax1.set_title('Evaluations vs Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best performance over iterations
    ax2.plot(iterations, best_performance, 'g-o', label='Best Performance', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Performance')
    ax2.set_title('Best Performance Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best cost over iterations
    ax3.plot(iterations, best_cost, 'm-o', label='Best Cost', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost')
    ax3.set_title('Best Cost Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hypervolume proxy over iterations
    ax4.plot(iterations, hypervolume_proxy, 'c-o', label='Hypervolume Proxy', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Hypervolume Proxy')
    ax4.set_title('Optimization Progress (Hypervolume)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('BoTorch Optimization Evolution Over Iterations', fontsize=16)
    plt.tight_layout()
    
    evolution_2d_path = os.path.join(output_folder, 'pareto_evolution_2d.png')
    plt.savefig(evolution_2d_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📈 2D evolution plots saved to: {evolution_2d_path}")
    
    return {
        'iterations': iterations,
        'n_evaluations': n_evaluations,
        'n_pareto_solutions': n_pareto_solutions,
        'best_performance': best_performance,
        'best_cost': best_cost,
        'hypervolume_proxy': hypervolume_proxy
    }


def find_latest_botorch_experiment():
    """Find the latest BoTorch experiment folder."""
    output_folders = glob.glob("outputs/*botorch*")
    if not output_folders:
        print("❌ No BoTorch experiment folders found!")
        return None
    
    # Sort by modification time
    output_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_folder = output_folders[0]
    print(f"📁 Using latest experiment: {latest_folder}")
    return latest_folder


def main():
    """Main function."""
    print("="*80)
    print("📊 BOTORCH PARETO FRONT EVOLUTION VISUALIZATION")
    print("="*80)
    
    # Get experiment folder
    if len(sys.argv) > 1:
        experiment_folder = sys.argv[1]
    else:
        experiment_folder = find_latest_botorch_experiment()
        
    if not experiment_folder or not os.path.exists(experiment_folder):
        print(f"❌ Experiment folder not found: {experiment_folder}")
        sys.exit(1)
    
    print(f"📁 Analyzing experiment: {experiment_folder}")
    
    # Load checkpoint data
    checkpoints = load_checkpoint_data(experiment_folder)
    if not checkpoints:
        sys.exit(1)
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    
    # 3D Pareto evolution plot
    evolution_3d_path = os.path.join(experiment_folder, 'pareto_evolution_3d.png')
    pareto_data = create_3d_pareto_evolution(checkpoints, evolution_3d_path)
    
    # 2D evolution plots
    evolution_data = create_2d_evolution_plots(checkpoints, experiment_folder)
    
    # Print summary
    print(f"\n📋 EVOLUTION SUMMARY:")
    print("-" * 50)
    if pareto_data:
        initial = pareto_data[0]
        final = pareto_data[-1]
        print(f"Initial (Iter {initial['iteration']}): {initial['n_pareto']} Pareto solutions from {initial['n_total']} evaluations")
        print(f"Final (Iter {final['iteration']}): {final['n_pareto']} Pareto solutions from {final['n_total']} evaluations")
        print(f"Performance improvement: {initial['best_performance']:.3f} → {final['best_performance']:.3f}")
        print(f"Cost improvement: {initial['best_cost']:.3f} → {final['best_cost']:.3f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • 3D Evolution Plot: {evolution_3d_path}")
    print(f"  • 2D Evolution Plots: {os.path.join(experiment_folder, 'pareto_evolution_2d.png')}")
    
    print(f"\n" + "="*80)
    print("🎉 PARETO EVOLUTION VISUALIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
