"""
Enhanced multi-objective optimization with iteration tracking for convergence analysis.
Tracks Pareto front evolution over iterations for visualization.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from data_processor_v2 import create_evaluator, NumpyLLMEvaluator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import copy

def is_pareto_optimal(costs: np.ndarray, maximization: List[bool] = None) -> np.ndarray:
    """
    Find Pareto optimal points.
    
    Args:
        costs: Array of shape (n, m) where n is number of points, m is number of objectives
        maximization: List of booleans indicating which objectives to maximize
    
    Returns:
        Boolean array indicating which points are Pareto optimal
    """
    if maximization is None:
        maximization = [True, True]  # Default: maximize both objectives
    
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_efficient[i]:
            # Find points that dominate point i
            for j in range(n_points):
                if i != j:
                    dominates = True
                    strictly_better = False
                    
                    for k, maximize in enumerate(maximization):
                        if maximize:
                            if costs[j, k] < costs[i, k]:  # j is worse than i in objective k
                                dominates = False
                                break
                            elif costs[j, k] > costs[i, k]:  # j is better than i in objective k
                                strictly_better = True
                        else:
                            if costs[j, k] > costs[i, k]:  # j is worse than i in objective k
                                dominates = False
                                break
                            elif costs[j, k] < costs[i, k]:  # j is better than i in objective k
                                strictly_better = True
                    
                    if dominates and strictly_better:
                        is_efficient[i] = False
                        break
    
    return is_efficient

class IterativeMultiObjectiveOptimizer:
    """
    Multi-objective optimizer with iteration tracking for convergence analysis.
    """
    
    def __init__(self, evaluator: NumpyLLMEvaluator):
        self.evaluator = evaluator
        self.n_features = len(evaluator.feature_names)
        self.n_agents = evaluator.n_agents
        self.dimension = self.n_agents * self.n_features
        
        # Store iteration-by-iteration data
        self.iteration_history = []
        self.X_all = []
        self.Y_all = []
        
        print(f"Iterative Optimizer initialized:")
        print(f"  - Search dimension: {self.dimension}")
        print(f"  - Agents: {self.n_agents}")
        print(f"  - Features per agent: {self.n_features}")
    
    def save_iteration_state(self, iteration: int, phase: str, X_current: np.ndarray, Y_current: np.ndarray):
        """Save the current state for convergence analysis."""
        
        # Combine with all previous data
        if len(self.X_all) > 0:
            X_cumulative = np.vstack([np.array(self.X_all), X_current])
            Y_cumulative = np.vstack([np.array(self.Y_all), Y_current])
        else:
            X_cumulative = X_current.copy()
            Y_cumulative = Y_current.copy()
        
        # Find current Pareto front
        pareto_mask = is_pareto_optimal(Y_cumulative, maximization=[True, True])
        X_pareto = X_cumulative[pareto_mask]
        Y_pareto = Y_cumulative[pareto_mask]
        
        # Sort by performance (descending)
        pareto_order = np.argsort(Y_pareto[:, 0])[::-1]
        X_pareto = X_pareto[pareto_order]
        Y_pareto = Y_pareto[pareto_order]
        
        # Calculate metrics
        best_performance = Y_cumulative[:, 0].max()
        best_cost = (-Y_cumulative[:, 1]).min()  # Convert back to positive cost
        
        # Hypervolume approximation (simple)
        ref_point = np.array([0.0, -50.0])  # Reference point for hypervolume
        hypervolume = self.calculate_hypervolume_2d(Y_pareto, ref_point)
        
        iteration_data = {
            'iteration': iteration,
            'phase': phase,
            'n_evaluations': len(Y_cumulative),
            'n_new_points': len(Y_current),
            'best_performance': float(best_performance),
            'best_cost': float(best_cost),
            'n_pareto_solutions': len(Y_pareto),
            'hypervolume': float(hypervolume),
            'X_pareto': X_pareto.tolist(),
            'Y_pareto': Y_pareto.tolist(),
            'pareto_performance': Y_pareto[:, 0].tolist(),
            'pareto_costs': (-Y_pareto[:, 1]).tolist(),  # Convert to positive costs
        }
        
        self.iteration_history.append(iteration_data)
        
        # Update cumulative data
        self.X_all = X_cumulative.tolist()
        self.Y_all = Y_cumulative.tolist()
        
        print(f"  Iteration {iteration} ({phase}): {len(Y_current)} points, "
              f"Best Perf: {best_performance:.3f}, Best Cost: {best_cost:.3f}, "
              f"Pareto: {len(Y_pareto)} solutions")
    
    def calculate_hypervolume_2d(self, pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate 2D hypervolume (simple implementation)."""
        if len(pareto_front) == 0:
            return 0.0
        
        # Sort by first objective (performance)
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for point in sorted_front:
            x, y = point[0], point[1]
            if x > prev_x and y > ref_point[1]:
                hypervolume += (x - prev_x) * (y - ref_point[1])
                prev_x = x
        
        return hypervolume
    
    def initialize_with_good_llms(self, n_points: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize with configurations using good LLMs."""
        
        # Identify some good LLMs based on different criteria
        llm_data = self.evaluator.llm_data
        
        # Find LLMs with high performance in different areas
        high_mmlu = np.argsort(llm_data[:, 0])[-3:]  # Top 3 in MMLU
        high_coding = np.argsort(llm_data[:, 1])[-3:]  # Top 3 in HumanEval  
        high_math = np.argsort(llm_data[:, 3])[-3:]  # Top 3 in MATH
        
        # Find cost-effective LLMs (high normalized cost values = low actual costs)
        high_cost_eff = np.argsort(llm_data[:, 5] + llm_data[:, 6])[-5:]  # Top 5 cost-effective
        
        # Create initial configurations
        X_init = []
        
        # Configuration 1: Best performance overall
        config1 = np.zeros(self.dimension)
        for agent_idx in range(self.n_agents):
            if agent_idx == 0:  # MMLU agent
                best_llm_idx = high_mmlu[-1]
            elif agent_idx == 1:  # Coding agent
                best_llm_idx = high_coding[-1]
            else:  # Math agent
                best_llm_idx = high_math[-1]
            
            # Copy LLM features to agent position
            start_idx = agent_idx * self.n_features
            end_idx = start_idx + self.n_features
            config1[start_idx:end_idx] = llm_data[best_llm_idx, :]
        
        X_init.append(config1)
        
        # Configuration 2: Cost-effective
        config2 = np.zeros(self.dimension)
        for agent_idx in range(self.n_agents):
            best_llm_idx = high_cost_eff[agent_idx % len(high_cost_eff)]
            start_idx = agent_idx * self.n_features
            end_idx = start_idx + self.n_features
            config2[start_idx:end_idx] = llm_data[best_llm_idx, :]
        
        X_init.append(config2)
        
        # Add some mixed configurations
        for _ in range(n_points - 2):
            config = np.zeros(self.dimension)
            for agent_idx in range(self.n_agents):
                # Randomly select from good LLMs
                if np.random.random() < 0.3:  # 30% chance of cost-effective
                    llm_idx = np.random.choice(high_cost_eff)
                else:  # 70% chance of high-performance
                    if agent_idx == 0:
                        llm_idx = np.random.choice(high_mmlu)
                    elif agent_idx == 1:
                        llm_idx = np.random.choice(high_coding)
                    else:
                        llm_idx = np.random.choice(high_math)
                
                start_idx = agent_idx * self.n_features
                end_idx = start_idx + self.n_features
                config[start_idx:end_idx] = llm_data[llm_idx, :]
            
            X_init.append(config)
        
        X_init = np.array(X_init)
        Y_init = self.evaluator.evaluate_agent_system(X_init)
        
        return X_init, Y_init
    
    def random_search_batch(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a batch of random search."""
        X_random = np.random.rand(n_points, self.dimension)
        Y_random = self.evaluator.evaluate_agent_system(X_random)
        return X_random, Y_random
    
    def local_search_batch(self, n_points: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Perform local search around current Pareto points."""
        if len(self.iteration_history) == 0:
            return self.random_search_batch(n_points)
        
        # Get current Pareto solutions
        latest_iteration = self.iteration_history[-1]
        X_pareto = np.array(latest_iteration['X_pareto'])
        
        if len(X_pareto) == 0:
            return self.random_search_batch(n_points)
        
        X_local = []
        points_per_pareto = max(1, n_points // len(X_pareto))
        
        for pareto_point in X_pareto:
            # Generate points around this Pareto point
            for _ in range(points_per_pareto):
                # Add small random perturbations
                noise = np.random.normal(0, 0.1, size=pareto_point.shape)
                new_point = np.clip(pareto_point + noise, 0, 1)
                X_local.append(new_point)
        
        # Fill remaining points with random search
        remaining = n_points - len(X_local)
        if remaining > 0:
            X_random_extra = np.random.rand(remaining, self.dimension)
            X_local.extend(X_random_extra)
        
        X_local = np.array(X_local[:n_points])  # Ensure exact count
        Y_local = self.evaluator.evaluate_agent_system(X_local)
        
        return X_local, Y_local
    
    def optimize_with_tracking(self, 
                             n_initial: int = 15,
                             n_random_batches: int = 8, 
                             n_random_per_batch: int = 25,
                             n_local_batches: int = 5,
                             n_local_per_batch: int = 20) -> Dict[str, Any]:
        """
        Run iterative multi-objective optimization with tracking.
        """
        print("Starting iterative multi-objective optimization...")
        iteration = 0
        
        # Phase 1: Strategic Initialization
        print(f"Phase 1: Strategic initialization with {n_initial} points...")
        X_init, Y_init = self.initialize_with_good_llms(n_initial)
        self.save_iteration_state(iteration, "Initialization", X_init, Y_init)
        iteration += 1
        
        # Phase 2: Random search batches
        print(f"Phase 2: Random search in {n_random_batches} batches...")
        for batch in range(n_random_batches):
            X_random, Y_random = self.random_search_batch(n_random_per_batch)
            self.save_iteration_state(iteration, f"Random_Batch_{batch+1}", X_random, Y_random)
            iteration += 1
        
        # Phase 3: Local search batches
        print(f"Phase 3: Local search in {n_local_batches} batches...")
        for batch in range(n_local_batches):
            X_local, Y_local = self.local_search_batch(n_local_per_batch)
            self.save_iteration_state(iteration, f"Local_Batch_{batch+1}", X_local, Y_local)
            iteration += 1
        
        # Final results
        final_X = np.array(self.X_all)
        final_Y = np.array(self.Y_all)
        
        # Find final Pareto front
        pareto_mask = is_pareto_optimal(final_Y, maximization=[True, True])
        X_pareto = final_X[pareto_mask]
        Y_pareto = final_Y[pareto_mask]
        
        # Sort Pareto solutions by performance (descending)
        pareto_order = np.argsort(Y_pareto[:, 0])[::-1]
        X_pareto = X_pareto[pareto_order]
        Y_pareto = Y_pareto[pareto_order]
        
        print(f"\nOptimization complete!")
        print(f"  - Total iterations: {iteration}")
        print(f"  - Total evaluations: {len(final_Y)}")
        print(f"  - Final Pareto solutions: {len(X_pareto)}")
        
        results = {
            'X_all': final_X,
            'Y_all': final_Y,
            'X_pareto': X_pareto,
            'Y_pareto': Y_pareto,
            'pareto_mask': pareto_mask,
            'n_evaluations': len(final_Y),
            'n_pareto_solutions': len(X_pareto),
            'n_iterations': iteration,
            'iteration_history': self.iteration_history
        }
        
        return results

# Analysis and visualization functions
def analyze_convergence(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the convergence of the optimization process."""
    
    iteration_history = results['iteration_history']
    
    # Extract convergence data
    iterations = [data['iteration'] for data in iteration_history]
    evaluations = [data['n_evaluations'] for data in iteration_history]
    best_performance = [data['best_performance'] for data in iteration_history]
    best_cost = [data['best_cost'] for data in iteration_history]
    n_pareto = [data['n_pareto_solutions'] for data in iteration_history]
    hypervolumes = [data['hypervolume'] for data in iteration_history]
    
    convergence_analysis = {
        'iterations': iterations,
        'evaluations': evaluations,
        'best_performance_evolution': best_performance,
        'best_cost_evolution': best_cost,
        'pareto_size_evolution': n_pareto,
        'hypervolume_evolution': hypervolumes,
        'performance_improvement': best_performance[-1] - best_performance[0] if len(best_performance) > 0 else 0,
        'cost_improvement': best_cost[0] - best_cost[-1] if len(best_cost) > 0 else 0,  # Lower cost is better
        'total_iterations': len(iterations),
        'final_evaluations': evaluations[-1] if evaluations else 0
    }
    
    return convergence_analysis

def plot_convergence_analysis(results: Dict[str, Any], save_prefix: str = 'outputs/convergence'):
    """Create convergence analysis plots."""
    
    convergence = analyze_convergence(results)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    iterations = convergence['iterations']
    
    # Plot 1: Best Performance Evolution
    ax1.plot(iterations, convergence['best_performance_evolution'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Performance')
    ax1.set_title('🏆 Best Performance Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best Cost Evolution
    ax2.plot(iterations, convergence['best_cost_evolution'], 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Cost (Lower is Better)')
    ax2.set_title('💰 Best Cost Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pareto Front Size Evolution
    ax3.plot(iterations, convergence['pareto_size_evolution'], 'g-^', linewidth=2, markersize=6)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Number of Pareto Solutions')
    ax3.set_title('📊 Pareto Front Size Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hypervolume Evolution
    ax4.plot(iterations, convergence['hypervolume_evolution'], 'm-d', linewidth=2, markersize=6)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Hypervolume')
    ax4.set_title('📈 Hypervolume Evolution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    convergence_path = f"{save_prefix}_analysis.png"
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Convergence analysis saved to: {convergence_path}")
    return convergence_path

def plot_3d_pareto_evolution(results: Dict[str, Any], save_path: str = 'outputs/pareto_3d_evolution.png'):
    """Create 3D plot showing Pareto front evolution over iterations."""
    
    iteration_history = results['iteration_history']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for iterations
    colors = plt.cm.viridis(np.linspace(0, 1, len(iteration_history)))
    
    for i, data in enumerate(iteration_history):
        pareto_performance = data['pareto_performance']
        pareto_costs = data['pareto_costs']
        iteration = data['iteration']
        
        if len(pareto_performance) > 0:
            # Create z-coordinates (iteration)
            iterations_z = np.full(len(pareto_performance), iteration)
            
            # Plot Pareto points for this iteration
            ax.scatter(pareto_costs, pareto_performance, iterations_z, 
                      c=[colors[i]], s=50, alpha=0.7, 
                      label=f'Iter {iteration}' if i % 3 == 0 else "")
    
    ax.set_xlabel('Cost')
    ax.set_ylabel('Performance')
    ax.set_zlabel('Iteration')
    ax.set_title('3D Pareto Front Evolution Over Iterations')
    
    # Add legend (show every 3rd iteration to avoid clutter)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(handles[::3], labels[::3], loc='upper left', bbox_to_anchor=(0.05, 0.95))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 3D Pareto evolution saved to: {save_path}")
    return save_path

def save_convergence_results(results: Dict[str, Any], convergence: Dict[str, Any], 
                           save_path: str = 'outputs/convergence_results.json'):
    """Save convergence analysis results."""
    
    # Combine results and convergence analysis
    full_results = {
        'optimization_results': {
            'n_evaluations': results['n_evaluations'],
            'n_pareto_solutions': results['n_pareto_solutions'],
            'n_iterations': results['n_iterations']
        },
        'convergence_analysis': convergence,
        'iteration_history': results['iteration_history']
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_json = convert_numpy(full_results)
    
    with open(save_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"📄 Convergence results saved to: {save_path}")
    return save_path