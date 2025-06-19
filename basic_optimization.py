"""
Basic multi-objective optimization for LLM selection.
Uses simple methods to find Pareto-optimal solutions without heavy dependencies.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from data_processor_v2 import create_evaluator, NumpyLLMEvaluator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json

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

class BasicMultiObjectiveOptimizer:
    """
    Basic multi-objective optimizer using random sampling and grid search.
    """
    
    def __init__(self, evaluator: NumpyLLMEvaluator):
        self.evaluator = evaluator
        self.n_features = len(evaluator.feature_names)
        self.n_agents = evaluator.n_agents
        self.dimension = self.n_agents * self.n_features
        
        # Store all evaluated points
        self.X_all = []
        self.Y_all = []
        
        print(f"Optimizer initialized:")
        print(f"  - Search dimension: {self.dimension}")
        print(f"  - Agents: {self.n_agents}")
        print(f"  - Features per agent: {self.n_features}")
    
    def initialize_with_good_llms(self, n_points: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize with configurations using good LLMs.
        This gives better starting points than pure random.
        """
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
    
    def random_search(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform random search in the continuous space.
        """
        X_random = np.random.rand(n_points, self.dimension)
        Y_random = self.evaluator.evaluate_agent_system(X_random)
        
        return X_random, Y_random
    
    def optimize(self, n_initial: int = 20, n_random: int = 200, n_grid: int = 50) -> Dict[str, Any]:
        """
        Run multi-objective optimization.
        
        Args:
            n_initial: Number of good initial points
            n_random: Number of random search points
            n_grid: Number of grid search points per dimension
        
        Returns:
            Dictionary with optimization results
        """
        print("Starting multi-objective optimization...")
        
        # Phase 1: Initialize with good LLMs
        print(f"Phase 1: Initializing with {n_initial} strategic points...")
        X_init, Y_init = self.initialize_with_good_llms(n_initial)
        
        # Phase 2: Random search
        print(f"Phase 2: Random search with {n_random} points...")
        X_random, Y_random = self.random_search(n_random)
        
        # Phase 3: Grid search around promising regions
        print(f"Phase 3: Grid search with {n_grid} points...")
        X_grid, Y_grid = self.grid_search_around_pareto(X_init, Y_init, n_grid)
        
        # Combine all results
        X_all = np.vstack([X_init, X_random, X_grid])
        Y_all = np.vstack([Y_init, Y_random, Y_grid])
        
        # Find Pareto optimal solutions
        pareto_mask = is_pareto_optimal(Y_all, maximization=[True, True])
        X_pareto = X_all[pareto_mask]
        Y_pareto = Y_all[pareto_mask]
        
        # Sort Pareto solutions by performance (descending)
        pareto_order = np.argsort(Y_pareto[:, 0])[::-1]
        X_pareto = X_pareto[pareto_order]
        Y_pareto = Y_pareto[pareto_order]
        
        self.X_all = X_all
        self.Y_all = Y_all
        
        print(f"\nOptimization complete!")
        print(f"  - Total evaluations: {len(X_all)}")
        print(f"  - Pareto optimal solutions: {len(X_pareto)}")
        
        results = {
            'X_all': X_all,
            'Y_all': Y_all,
            'X_pareto': X_pareto,
            'Y_pareto': Y_pareto,
            'pareto_mask': pareto_mask,
            'n_evaluations': len(X_all),
            'n_pareto_solutions': len(X_pareto)
        }
        
        return results
    
    def grid_search_around_pareto(self, X_init: np.ndarray, Y_init: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grid search around current Pareto optimal solutions.
        """
        pareto_mask = is_pareto_optimal(Y_init, maximization=[True, True])
        X_pareto = X_init[pareto_mask]
        
        if len(X_pareto) == 0:
            # Fallback to random search
            return self.random_search(n_points)
        
        X_grid = []
        points_per_pareto = max(1, n_points // len(X_pareto))
        
        for pareto_point in X_pareto:
            # Generate points around this Pareto point
            for _ in range(points_per_pareto):
                # Add small random perturbations
                noise = np.random.normal(0, 0.1, size=pareto_point.shape)
                new_point = np.clip(pareto_point + noise, 0, 1)
                X_grid.append(new_point)
        
        # Fill remaining points with random search
        remaining = n_points - len(X_grid)
        if remaining > 0:
            X_random_extra = np.random.rand(remaining, self.dimension)
            X_grid.extend(X_random_extra)
        
        X_grid = np.array(X_grid[:n_points])  # Ensure exact count
        Y_grid = self.evaluator.evaluate_agent_system(X_grid)
        
        return X_grid, Y_grid

def analyze_results(optimizer: BasicMultiObjectiveOptimizer, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze optimization results and generate insights.
    """
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("="*60)
    
    X_pareto = results['X_pareto']
    Y_pareto = results['Y_pareto']
    
    # Get configuration details for Pareto solutions
    configs = optimizer.evaluator.get_configuration_details(X_pareto)
    
    analysis = {
        'pareto_solutions': [],
        'statistics': {},
        'recommendations': {}
    }
    
    print(f"\nFound {len(X_pareto)} Pareto-optimal solutions:")
    print("-" * 60)
    
    for i, (config, objectives) in enumerate(zip(configs, Y_pareto)):
        performance, neg_cost = objectives
        cost = -neg_cost
        
        solution = {
            'rank': i + 1,
            'performance': performance,
            'cost': cost,
            'agents': config['agents']
        }
        analysis['pareto_solutions'].append(solution)
        
        if i < 10:  # Show first 10 solutions
            print(f"\n{i+1}. Performance: {performance:.3f}, Cost: {cost:.3f}")
            for agent in config['agents']:
                print(f"   Agent {agent['agent_id']} ({agent['role']}) → {agent['assigned_llm']}")
    
    # Calculate statistics
    all_performance = results['Y_all'][:, 0]
    all_costs = -results['Y_all'][:, 1]
    
    analysis['statistics'] = {
        'performance_range': [float(all_performance.min()), float(all_performance.max())],
        'cost_range': [float(all_costs.min()), float(all_costs.max())],
        'pareto_performance_range': [float(Y_pareto[:, 0].min()), float(Y_pareto[:, 0].max())],
        'pareto_cost_range': [float((-Y_pareto[:, 1]).min()), float((-Y_pareto[:, 1]).max())]
    }
    
    print(f"\n" + "-" * 60)
    print("STATISTICS:")
    print(f"Performance range: {analysis['statistics']['performance_range'][0]:.3f} - {analysis['statistics']['performance_range'][1]:.3f}")
    print(f"Cost range: {analysis['statistics']['cost_range'][0]:.3f} - {analysis['statistics']['cost_range'][1]:.3f}")
    
    # Generate recommendations
    pareto_performance = Y_pareto[:, 0]
    pareto_costs = -Y_pareto[:, 1]
    
    # Best performance
    best_perf_idx = np.argmax(pareto_performance)
    # Best cost (lowest)
    best_cost_idx = np.argmin(pareto_costs)
    # Best balance (simple scoring)
    normalized_perf = (pareto_performance - pareto_performance.min()) / (pareto_performance.max() - pareto_performance.min())
    normalized_cost = 1 - (pareto_costs - pareto_costs.min()) / (pareto_costs.max() - pareto_costs.min())  # Invert cost
    balance_scores = normalized_perf + normalized_cost
    best_balance_idx = np.argmax(balance_scores)
    
    analysis['recommendations'] = {
        'highest_performance': analysis['pareto_solutions'][best_perf_idx],
        'lowest_cost': analysis['pareto_solutions'][best_cost_idx],
        'best_balance': analysis['pareto_solutions'][best_balance_idx]
    }
    
    print(f"\n" + "-" * 60)
    print("RECOMMENDATIONS:")
    print(f"\n🏆 Highest Performance: Solution #{best_perf_idx + 1}")
    print(f"💰 Lowest Cost: Solution #{best_cost_idx + 1}")
    print(f"⚖️  Best Balance: Solution #{best_balance_idx + 1}")
    
    return analysis

def plot_pareto_front(results: Dict[str, Any], save_path: str = 'outputs/pareto_front.png'):
    """
    Plot the Pareto front.
    """
    Y_all = results['Y_all']
    Y_pareto = results['Y_pareto']
    
    # Convert costs to positive values for plotting
    all_costs = -Y_all[:, 1]
    pareto_costs = -Y_pareto[:, 1]
    
    plt.figure(figsize=(10, 6))
    
    # Plot all points
    plt.scatter(all_costs, Y_all[:, 0], alpha=0.5, c='lightblue', s=20, label='All evaluations')
    
    # Plot Pareto front
    plt.scatter(pareto_costs, Y_pareto[:, 0], c='red', s=100, label='Pareto optimal', edgecolor='black', linewidth=1)
    
    # Connect Pareto points
    pareto_order = np.argsort(pareto_costs)
    plt.plot(pareto_costs[pareto_order], Y_pareto[pareto_order, 0], 'r--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Cost')
    plt.ylabel('Performance')
    plt.title('Multi-Objective Optimization: Performance vs Cost Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations for top solutions
    for i, (cost, perf) in enumerate(zip(pareto_costs, Y_pareto[:, 0])):
        if i < 3:  # Annotate top 3
            plt.annotate(f'#{i+1}', (cost, perf), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPareto front plot saved to: {save_path}")
    
    return save_path

def save_results(analysis: Dict[str, Any], save_path: str = 'outputs/optimization_results.json'):
    """
    Save optimization results to JSON file.
    """
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
    
    analysis_json = convert_numpy(analysis)
    
    with open(save_path, 'w') as f:
        json.dump(analysis_json, f, indent=2)
    
    print(f"Results saved to: {save_path}")
    return save_path