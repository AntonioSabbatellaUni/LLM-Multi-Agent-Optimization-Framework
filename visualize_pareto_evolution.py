"""
Pareto Front Evolution Visualization for BoTorch Optimization
Creates 3D visualization showing how the Pareto front evolves over iterations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import argparse

def load_checkpoint_data(checkpoint_dir: str) -> dict:
    """Load all checkpoint files from directory."""
    checkpoint_data = {}
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return checkpoint_data
    
    # Find all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('checkpoint_iter_') and f.endswith('.json')]
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        return checkpoint_data
    
    print(f"📁 Found {len(checkpoint_files)} checkpoint files")
    
    # Load each checkpoint
    for filename in sorted(checkpoint_files):
        try:
            # Extract iteration number
            iter_str = filename.replace('checkpoint_iter_', '').replace('.json', '')
            iteration = int(iter_str)
            
            # Load checkpoint data
            filepath = os.path.join(checkpoint_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            for key in ['train_x', 'train_y', 'new_candidates_x', 'new_candidates_y']:
                if key in data:
                    data[key] = np.array(data[key])
            
            # Convert pareto_mask to boolean
            if 'pareto_info' in data and 'pareto_mask' in data['pareto_info']:
                data['pareto_mask'] = np.array(data['pareto_info']['pareto_mask'], dtype=bool)
            elif 'pareto_mask' in data:
                data['pareto_mask'] = np.array(data['pareto_mask'], dtype=bool)
            
            checkpoint_data[iteration] = data
            
        except Exception as e:
            print(f"⚠️  Error loading {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(checkpoint_data)} checkpoints")
    return checkpoint_data

def create_pareto_surface_evolution(checkpoint_dir: str):
    """Create comprehensive Pareto front evolution visualization."""
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint_data(checkpoint_dir)
    if not checkpoint_data:
        return
    
    print(f"📊 Creating Pareto surface evolution visualization...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Color scheme for iterations
    n_iterations = len(checkpoint_data)
    colors = plt.cm.plasma(np.linspace(0, 1, n_iterations))
    
    # 1. 3D plot with connected Pareto surfaces
    ax1 = fig.add_subplot(231, projection='3d')
    
    pareto_surfaces = []
    iteration_list = sorted(checkpoint_data.keys())
    
    for i, iteration in enumerate(iteration_list):
        data = checkpoint_data[iteration]
        Y = data['train_y']
        pareto_mask = data['pareto_mask']
        pareto_Y = Y[pareto_mask]
        
        if len(pareto_Y) > 1:
            # Sort Pareto points by performance
            sorted_idx = np.argsort(pareto_Y[:, 0])
            sorted_pareto = pareto_Y[sorted_idx]
            
            # Plot Pareto front line in 3D space
            ax1.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                    [iteration] * len(sorted_pareto), 
                    'o-', linewidth=3, markersize=8, alpha=0.9,
                    color=colors[i], label=f'Iter {iteration}')
            
            pareto_surfaces.append((iteration, sorted_pareto))
            
            # Create surface mesh if we have enough points
            if len(sorted_pareto) > 2:
                # Create triangular surface patches
                X_surf = sorted_pareto[:, 0]
                Y_surf = sorted_pareto[:, 1]
                Z_surf = np.full_like(X_surf, iteration)
                
                # Create surface
                ax1.plot_trisurf(X_surf, Y_surf, Z_surf, alpha=0.3, color=colors[i])
    
    # Connect corresponding points between iterations
    if len(pareto_surfaces) > 1:
        for i in range(1, len(pareto_surfaces)):
            curr_iter, curr_pareto = pareto_surfaces[i]
            prev_iter, prev_pareto = pareto_surfaces[i-1]
            
            # Connect similar performance levels
            for curr_point in curr_pareto:
                # Find closest point in previous iteration
                if len(prev_pareto) > 0:
                    distances = np.abs(prev_pareto[:, 0] - curr_point[0])
                    closest_idx = np.argmin(distances)
                    prev_point = prev_pareto[closest_idx]
                    
                    # Draw connection
                    ax1.plot([prev_point[0], curr_point[0]], 
                            [prev_point[1], curr_point[1]], 
                            [prev_iter, curr_iter], 
                            'k--', alpha=0.2, linewidth=1)
    
    ax1.set_xlabel('Performance', fontsize=12)
    ax1.set_ylabel('Cost', fontsize=12)
    ax1.set_zlabel('Iteration', fontsize=12)
    ax1.set_title('3D Pareto Front Evolution\n(Connected Surfaces)', fontsize=14, fontweight='bold')
    
    # 2. 2D Pareto front evolution with filled areas
    ax2 = fig.add_subplot(232)
    
    for i, iteration in enumerate(iteration_list):
        data = checkpoint_data[iteration]
        Y = data['train_y']
        pareto_mask = data['pareto_mask']
        pareto_Y = Y[pareto_mask]
        
        if len(pareto_Y) > 1:
            sorted_idx = np.argsort(pareto_Y[:, 0])
            sorted_pareto = pareto_Y[sorted_idx]
            
            # Plot Pareto front line
            alpha_val = 0.6 + 0.4 * (i / max(1, n_iterations-1))  # Fade in over iterations
            ax2.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                    'o-', alpha=alpha_val, linewidth=3, markersize=8,
                    color=colors[i], label=f'Iteration {iteration}')
            
            # Fill area under Pareto front
            ax2.fill_between(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                           alpha=0.2, color=colors[i])
    
    ax2.set_xlabel('Performance', fontsize=12)
    ax2.set_ylabel('Cost', fontsize=12)
    ax2.set_title('2D Pareto Front Evolution\n(Overlaid with Fill)', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Hypervolume evolution
    ax3 = fig.add_subplot(233)
    
    iterations = []
    hypervolumes = []
    
    for iteration in iteration_list:
        data = checkpoint_data[iteration]
        Y = data['train_y']
        pareto_mask = data['pareto_mask']
        pareto_Y = Y[pareto_mask]
        
        iterations.append(iteration)
        
        if len(pareto_Y) > 0:
            # Calculate approximate hypervolume
            ref_point = np.array([Y[:, 0].min() - 0.1, Y[:, 1].max() + 0.1])
            
            sorted_idx = np.argsort(pareto_Y[:, 0])
            sorted_pareto = pareto_Y[sorted_idx]
            
            hv = 0
            for i in range(len(sorted_pareto)):
                if i == 0:
                    width = sorted_pareto[i, 0] - ref_point[0]
                else:
                    width = sorted_pareto[i, 0] - sorted_pareto[i-1, 0]
                height = ref_point[1] - sorted_pareto[i, 1]
                hv += width * height
            
            hypervolumes.append(max(0, hv))
        else:
            hypervolumes.append(0)
    
    ax3.plot(iterations, hypervolumes, 'bo-', linewidth=3, markersize=10)
    ax3.fill_between(iterations, hypervolumes, alpha=0.3, color='blue')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Hypervolume', fontsize=12)
    ax3.set_title('Hypervolume Evolution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of Pareto points evolution
    ax4 = fig.add_subplot(234)
    
    n_pareto_points = []
    n_total_points = []
    
    for iteration in iteration_list:
        data = checkpoint_data[iteration]
        Y = data['train_y']
        pareto_mask = data['pareto_mask']
        
        n_pareto_points.append(np.sum(pareto_mask))
        n_total_points.append(len(Y))
    
    ax4.plot(iterations, n_pareto_points, 'ro-', linewidth=3, markersize=10, label='Pareto Points')
    ax4.plot(iterations, n_total_points, 'go-', linewidth=3, markersize=10, label='Total Points')
    ax4.fill_between(iterations, n_pareto_points, alpha=0.3, color='red')
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Number of Points', fontsize=12)
    ax4.set_title('Pareto Set Size Evolution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance and Cost bounds evolution
    ax5 = fig.add_subplot(235)
    
    max_performance = []
    min_cost = []
    
    for iteration in iteration_list:
        data = checkpoint_data[iteration]
        Y = data['train_y']
        
        max_performance.append(Y[:, 0].max())
        min_cost.append(Y[:, 1].min())
    
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(iterations, max_performance, 'b-o', linewidth=3, markersize=8, label='Max Performance')
    line2 = ax5_twin.plot(iterations, min_cost, 'r-s', linewidth=3, markersize=8, label='Min Cost')
    
    ax5.set_xlabel('Iteration', fontsize=12)
    ax5.set_ylabel('Max Performance', color='blue', fontsize=12)
    ax5_twin.set_ylabel('Min Cost', color='red', fontsize=12)
    ax5.set_title('Best Performance & Cost Evolution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='center right')
    
    # 6. Acquisition progress (if new candidates data available)
    ax6 = fig.add_subplot(236)
    
    new_candidates_per_iter = []
    for iteration in iteration_list:
        data = checkpoint_data[iteration]
        if 'new_candidates_y' in data and data['new_candidates_y'] is not None:
            new_candidates_per_iter.append(len(data['new_candidates_y']))
        else:
            new_candidates_per_iter.append(0)
    
    ax6.bar(iterations, new_candidates_per_iter, alpha=0.7, color='orange')
    ax6.set_xlabel('Iteration', fontsize=12)
    ax6.set_ylabel('New Candidates', fontsize=12)
    ax6.set_title('Acquisition Function Candidates', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(checkpoint_dir, 'pareto_surface_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Pareto surface evolution saved: {output_path}")
    
    plt.show()
    
    return output_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize Pareto front evolution from BoTorch checkpoints')
    parser.add_argument('checkpoint_dir', nargs='?', 
                       help='Directory containing checkpoint files')
    
    args = parser.parse_args()
    
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        # Find latest BoTorch results directory
        outputs_dir = 'outputs'
        if os.path.exists(outputs_dir):
            botorch_dirs = [d for d in os.listdir(outputs_dir) 
                           if 'botorch' in d.lower() and os.path.isdir(os.path.join(outputs_dir, d))]
            if botorch_dirs:
                # Get most recent
                latest_dir = max(botorch_dirs, key=lambda d: os.path.getmtime(os.path.join(outputs_dir, d)))
                checkpoint_dir = os.path.join(outputs_dir, latest_dir)
                print(f"📁 Using latest BoTorch results: {checkpoint_dir}")
            else:
                print("❌ No BoTorch results directories found!")
                return
        else:
            print("❌ Outputs directory not found!")
            return
    
    create_pareto_surface_evolution(checkpoint_dir)

if __name__ == "__main__":
    main()
