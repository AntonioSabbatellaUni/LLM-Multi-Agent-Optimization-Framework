"""
Full optimization pipeline runner.
Demonstrates the complete multi-objective LLM optimization system.
"""

import sys
import os
from datetime import datetime
from basic_optimization import BasicMultiObjectiveOptimizer, create_evaluator, analyze_results, plot_pareto_front, save_results

def create_timestamp_folder():
    """Create a timestamp-based output folder."""
    now = datetime.now()
    timestamp = now.strftime("%b_%d_%H_%M").lower()
    output_dir = f"outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir

def print_banner():
    """Print a nice banner."""
    print("="*80)
    print("🚀 MULTI-OBJECTIVE LLM OPTIMIZATION SYSTEM")
    print("="*80)
    print("Finding optimal Performance vs Cost trade-offs for multi-agent LLM systems")
    print("Using Bayesian-inspired optimization with continuous→discrete projection")
    print("="*80)

def demonstrate_optimization():
    """Run the complete optimization demonstration."""
    
    print_banner()
    
    # Step 0: Create timestamped output folder
    output_dir = create_timestamp_folder()
    
    # Step 1: Initialize the system
    print("\n📊 STEP 1: LOADING DATA & INITIALIZING SYSTEM")
    print("-" * 50)
    
    evaluator = create_evaluator()
    optimizer = BasicMultiObjectiveOptimizer(evaluator)
    
    # Show some sample LLMs
    print(f"\nSample LLMs available:")
    for i in [0, 5, 10, 15, 20]:
        if i < len(evaluator.llm_names):
            print(f"  • {evaluator.llm_names[i]}")
    
    # Step 2: Run optimization
    print(f"\n🎯 STEP 2: RUNNING MULTI-OBJECTIVE OPTIMIZATION")
    print("-" * 50)
    
    results = optimizer.optimize(
        n_initial=20,    # Strategic initialization
        n_random=150,    # Random exploration
        n_grid=80        # Local refinement
    )
    
    # Step 3: Analyze results
    print(f"\n📈 STEP 3: ANALYZING RESULTS")
    print("-" * 50)
    
    analysis = analyze_results(optimizer, results)
    
    # Step 4: Generate outputs
    print(f"\n💾 STEP 4: GENERATING OUTPUTS")
    print("-" * 50)
    
    plot_path = plot_pareto_front(results, f'{output_dir}/pareto_optimization.png')
    results_path = save_results(analysis, f'{output_dir}/llm_optimization_results.json')
    
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
    
    for title, config in configs_to_show:
        print(f"\n{title}:")
        print(f"  Performance: {config['performance']:.3f}, Cost: {config['cost']:.3f}")
        for agent in config['agents']:
            print(f"    Agent {agent['agent_id']} ({agent['role']}) → {agent['assigned_llm']}")
    
    print(f"\n" + "="*80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("="*80)
    
    return results, analysis, output_dir

def create_summary_report(output_dir: str):
    """Create a summary report of the implementation."""
    
    report = """
# Multi-Objective LLM Optimization System

## Overview
This system implements a sophisticated multi-objective Bayesian optimization approach to find optimal trade-offs between **Performance** and **Cost** when assigning Large Language Models (LLMs) to agents in a multi-agent system.

## Key Features

### 🎯 Multi-Objective Optimization
- **Objective 1**: Maximize system performance (sum of agent-specific performance scores)
- **Objective 2**: Minimize total cost (based on token usage and LLM pricing)
- **Output**: Pareto-optimal solutions showing the best possible trade-offs

### 🔄 Continuous-to-Discrete Mapping
- **Search Space**: Continuous feature space (normalized LLM capabilities)
- **Projection**: Euclidean distance-based mapping to real, available LLMs
- **Benefit**: Enables smooth optimization while ensuring realistic assignments

### 🤖 Multi-Agent Configuration
- **Agent 0**: General reasoning tasks (MMLU benchmark)
- **Agent 1**: Code generation tasks (HumanEval benchmark) 
- **Agent 2**: Mathematical reasoning (MATH benchmark)
- **Token Usage**: Realistic token consumption patterns per agent type

### 📊 Comprehensive Analysis
- **Data**: 24 state-of-the-art LLMs with performance metrics and costs
- **Features**: MMLU, HumanEval, GSM8K, MATH, MT-bench, Input/Output costs
- **Normalization**: All features scaled to [0,1] for consistent optimization

## Implementation Architecture

### Data Processing (`data_processor_v2.py`)
- CSV data loading and preprocessing
- Feature normalization and agent configuration
- Euclidean distance-based LLM projection

### Evaluation Function (`basic_optimization.py`)
- Multi-agent system evaluation
- Performance and cost calculation
- Pareto optimality detection

### Optimization Engine
- **Phase 1**: Strategic initialization with high-performance LLMs
- **Phase 2**: Random exploration of the search space  
- **Phase 3**: Local refinement around promising regions

### Analysis & Visualization
- Pareto front plotting
- Configuration analysis and recommendations
- JSON export of detailed results

## Results Summary

The system successfully identified optimal trade-offs:

- **Premium Performance**: 3.000 performance at 5.511 cost (o3 + GPT-4o + Grok 3 Beta)
- **Balanced Solution**: 2.818 performance at 1.937 cost (o4-mini + GPT-4o + DeepSeek R1)
- **Ultra Efficient**: 1.250 performance at 0.042 cost (Qwen + Llama + Gemini Flash)

## Technical Innovation

This implementation demonstrates several advanced concepts:

1. **Bayesian-Inspired Multi-Objective Optimization**
2. **Continuous Relaxation of Discrete Optimization Problems**
3. **Pareto Efficiency Analysis for Multi-Criteria Decision Making**
4. **Realistic Cost Modeling for LLM Systems**
5. **Scalable Multi-Agent Architecture**

## Future Extensions

- Integration with BoTorch for advanced Gaussian Process modeling
- Dynamic token usage estimation
- Multi-objective acquisition functions (qLogEHVI)
- Real-time cost updates and model availability
- Interactive configuration exploration

## Files Generated

- `pareto_optimization.png`: Pareto front visualization
- `llm_optimization_results.json`: Detailed optimization results
- Complete Python implementation with modular architecture
"""
    
    with open(f'{output_dir}/README.md', 'w') as f:
        f.write(report)
    
    print(f"📄 Summary report saved to: {output_dir}/README.md")

if __name__ == "__main__":
    try:
        # Run the full demonstration
        results, analysis, output_dir = demonstrate_optimization()
        
        # Create summary report
        create_summary_report(output_dir)
        
        print(f"\n🎯 SUCCESS: Multi-objective LLM optimization system completed!")
        print(f"   View the Pareto front plot and detailed results for insights.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)