"""
Quick visualization script to explore the optimization results.
Run this after installing dependencies: pip install pandas numpy matplotlib
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def find_latest_results():
    """Find the most recent results folder."""
    outputs_dir = 'outputs'
    
    # Get all timestamped folders
    folders = []
    if os.path.exists(outputs_dir):
        for item in os.listdir(outputs_dir):
            item_path = os.path.join(outputs_dir, item)
            if os.path.isdir(item_path) and '_' in item:
                # Check if it contains results
                results_file = os.path.join(item_path, 'llm_optimization_results.json')
                if os.path.exists(results_file):
                    folders.append((item, item_path))
    
    if not folders:
        print("❌ No results found. Please run the optimization first.")
        return None, None
    
    # Sort by folder name (timestamp) and get the latest
    folders.sort(reverse=True)
    latest_folder, latest_path = folders[0]
    
    print(f"📁 Using latest results from: {latest_folder}")
    return latest_folder, latest_path

def show_results_summary():
    """Display a summary of the optimization results."""
    
    # Find latest results
    latest_folder, latest_path = find_latest_results()
    if latest_path is None:
        return None
    
    # Load results
    with open(f'{latest_path}/llm_optimization_results.json', 'r') as f:
        results = json.load(f)
    
    solutions = results['pareto_solutions']
    recommendations = results['recommendations']
    stats = results['statistics']
    
    print("🎯 MULTI-OBJECTIVE LLM OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"📊 Found {len(solutions)} Pareto-optimal solutions")
    print(f"📈 Performance range: {stats['performance_range'][0]:.2f} - {stats['performance_range'][1]:.2f}")
    print(f"💰 Cost range: {stats['cost_range'][0]:.2f} - {stats['cost_range'][1]:.2f}")
    
    print("\n🏆 KEY RECOMMENDATIONS:")
    print("-" * 40)
    
    # Highest Performance
    best_perf = recommendations['highest_performance']
    print(f"\n🚀 MAXIMUM PERFORMANCE:")
    print(f"   Performance: {best_perf['performance']:.3f}")
    print(f"   Cost: {best_perf['cost']:.3f}")
    for agent in best_perf['agents']:
        print(f"   • Agent {agent['agent_id']} ({agent['role']}) → {agent['assigned_llm']}")
    
    # Lowest Cost
    best_cost = recommendations['lowest_cost']
    print(f"\n💰 MINIMUM COST:")
    print(f"   Performance: {best_cost['performance']:.3f}")
    print(f"   Cost: {best_cost['cost']:.3f}")
    for agent in best_cost['agents']:
        print(f"   • Agent {agent['agent_id']} ({agent['role']}) → {agent['assigned_llm']}")
    
    # Best Balance
    best_balance = recommendations['best_balance']
    print(f"\n⚖️  BEST BALANCE:")
    print(f"   Performance: {best_balance['performance']:.3f}")
    print(f"   Cost: {best_balance['cost']:.3f}")
    for agent in best_balance['agents']:
        print(f"   • Agent {agent['agent_id']} ({agent['role']}) → {agent['assigned_llm']}")
    
    return results

def plot_simplified_pareto():
    """Create a simplified Pareto front plot."""
    
    # Find latest results
    latest_folder, latest_path = find_latest_results()
    if latest_path is None:
        return
    
    # Load results
    with open(f'{latest_path}/llm_optimization_results.json', 'r') as f:
        results = json.load(f)
    
    solutions = results['pareto_solutions']
    
    # Extract performance and cost
    performance = [s['performance'] for s in solutions]
    cost = [s['cost'] for s in solutions]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(cost, performance, c='red', s=80, alpha=0.7, edgecolor='black')
    
    # Sort for line connection
    sorted_indices = np.argsort(cost)
    sorted_cost = [cost[i] for i in sorted_indices]
    sorted_perf = [performance[i] for i in sorted_indices]
    plt.plot(sorted_cost, sorted_perf, 'r--', alpha=0.5)
    
    plt.xlabel('Cost', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Pareto Front: LLM Multi-Agent System Optimization', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Highlight key points
    recommendations = results['recommendations']
    
    # Highest performance
    best_perf = recommendations['highest_performance']
    plt.scatter(best_perf['cost'], best_perf['performance'], 
               c='gold', s=150, marker='*', edgecolor='black', linewidth=2,
               label='🏆 Max Performance')
    
    # Lowest cost
    best_cost = recommendations['lowest_cost']
    plt.scatter(best_cost['cost'], best_cost['performance'], 
               c='green', s=150, marker='s', edgecolor='black', linewidth=2,
               label='💰 Min Cost')
    
    # Best balance
    best_balance = recommendations['best_balance']
    plt.scatter(best_balance['cost'], best_balance['performance'], 
               c='blue', s=150, marker='D', edgecolor='black', linewidth=2,
               label='⚖️ Best Balance')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save to the same timestamped folder
    save_path = f'{latest_path}/pareto_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Simplified Pareto plot saved as '{save_path}'")

if __name__ == "__main__":
    try:
        print("Loading optimization results...\n")
        results = show_results_summary()
        
        if results is not None:
            print(f"\n{'='*60}")
            print("📊 CREATING VISUALIZATION...")
            plot_simplified_pareto()
            
            print(f"\n✅ Analysis complete!")
            print("📄 Full results available in the latest timestamped folder")
            print("🖼️  Visualizations saved in the same folder")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've run the optimization first.")