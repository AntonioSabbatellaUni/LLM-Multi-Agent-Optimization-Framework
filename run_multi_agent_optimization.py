"""
Multi-Agent LLM Optimization Framework Runner

This is the main runner for multi-agent LLM optimization using BoTorch Bayesian Optimization.
Supports multiple evaluation architectures:
- Simulated: Fast CSV-based evaluation for prototyping
- GAIA SmolagentsLibrary: Real benchmark evaluation with actual task execution

Maintains full backward compatibility while enabling extensible architecture support.
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
from basic_optimization import analyze_results, plot_pareto_front, save_results
from env_manager import ensure_environment_ready, validate_environment_for_architecture


def load_config(config_path: str = "config.yaml") -> dict:
    """Load main configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Main configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file {config_path} not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML configuration: {e}")
        sys.exit(1)


def architecture_config_path(architecture_name: str, config: dict) -> str:
    """multi_agent_architectures/<arch>/config.yaml unless the main config sets evaluation.architecture_config."""
    return config.get('evaluation', {}).get('architecture_config') or f"multi_agent_architectures/{architecture_name}/config.yaml"


def load_architecture_config(architecture_name: str, arch_config_path: str = None) -> dict:
    """Load architecture-specific configuration."""
    arch_config_path = arch_config_path or f"multi_agent_architectures/{architecture_name}/config.yaml"
    
    try:
        with open(arch_config_path, 'r') as f:
            arch_config = yaml.safe_load(f)
        print(f"✅ Architecture config loaded: {arch_config_path}")
        return arch_config
    except FileNotFoundError:
        print(f"❌ Architecture config not found: {arch_config_path}")
        print(f"Available architectures: simulated, gaia_smolagents")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing architecture config: {e}")
        sys.exit(1)


def create_evaluator_for_architecture(architecture_name: str, arch_config: dict, config: dict,
                                      arch_config_path: str = None, output_dir: str = None) -> object:
    """Create the appropriate evaluator based on architecture selection."""
    print(f"\n🏗️ CREATING EVALUATOR FOR ARCHITECTURE: {architecture_name}")
    print("-" * 60)
    
    if architecture_name == "simulated":
        print("📊 Using simulated evaluation (fast, CSV-based)")
        
        # Use the legacy approach for backward compatibility
        from multi_agent_architectures.simulated.evaluator import create_evaluator
        evaluator = create_evaluator()
        
        # Override n_agents if specified in architecture config
        if 'n_agents' in arch_config.get('architecture', {}):
            evaluator.n_agents = arch_config['architecture']['n_agents']
            print(f"   Agents overridden to: {evaluator.n_agents}")
        
        return evaluator
        
    elif architecture_name == "gaia_smolagents":
        print("🎯 Using GAIA benchmark evaluation (real, slower)")
        
        # Load LLM data for projection. Features come from the main config (any number); rows with a missing
        # feature are excluded from the pool (logged), not imputed.
        from data_processor_v2 import load_llm_data, load_pool_df
        feature_llm_columns = config.get('features', [])
        csv_path = config.get('data', {}).get('input_file', 'data/llm_data.csv')
        pool_df_full = load_pool_df(csv_path, None, drop_missing=False)
        pool_df = load_pool_df(csv_path, feature_llm_columns or None, drop_missing=True)
        print(f"   Pool: {len(pool_df)}/{len(pool_df_full)} options usable with features {feature_llm_columns}")
        llm_data, llm_names, feature_names = load_llm_data(csv_path, feature_columns=feature_llm_columns, df=pool_df)

        # n_agents = number of optimized SLOTS (agent_roles/slots + enabled optional roles such as visual_qa)
        from multi_agent_architectures.gaia_smolagents.evaluator import create_gaia_evaluator, effective_slots
        slots = effective_slots(arch_config)
        n_agents = len(slots)
        print(f"   GAIA architecture optimizes {n_agents} slots (slot -> roles): {slots}")

        # Optional per-model image capability (pool CSV column `supports_image`, from Go /models modalities.input)
        supports_image = None
        if 'supports_image' in pool_df.columns:
            supports_image = pool_df['supports_image'].fillna(0).astype(int).values.astype(bool)

        evaluator = create_gaia_evaluator(llm_data, llm_names, feature_names, n_agents, supports_image=supports_image,
                                          config_path=arch_config_path, pool_df=pool_df, pool_df_full=pool_df_full,
                                          output_dir=output_dir)
        pool_df.to_csv(os.path.join(output_dir, 'pool_used.csv'), index=False) if output_dir else None
        
        return evaluator
        
    else:
        print(f"❌ Unknown architecture: {architecture_name}")
        print("Available architectures: simulated, gaia_smolagents")
        sys.exit(1)


def create_timestamp_folder(base_name: str, architecture_name: str) -> str:
    """Create a timestamp-based output folder with architecture info."""
    now = datetime.now()
    timestamp = now.strftime("%b_%d_%H_%M").lower()
    output_dir = os.getenv("MALBO_OUTPUT_DIR") or f"outputs/{timestamp}_{base_name}_{architecture_name}"   # fixed dir = resume
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    return output_dir


def copy_configs_to_output(config_path: str, arch_config_path: str, output_dir: str):
    """Copy both main and architecture configs for reproducibility."""
    # Copy main config
    main_config_copy = os.path.join(output_dir, "config_used.yaml")
    shutil.copy2(config_path, main_config_copy)
    
    # Copy architecture config  
    arch_config_copy = os.path.join(output_dir, "architecture_config_used.yaml")
    shutil.copy2(arch_config_path, arch_config_copy)
    
    print(f"📋 Configurations copied to output directory")


def print_banner(architecture_name: str, resume_from: str = None):
    """Print an informative banner."""
    print("="*80)
    print("🚀 MULTI-AGENT LLM OPTIMIZATION FRAMEWORK")
    print("="*80)
    print(f"Architecture: {architecture_name.upper()}")
    if architecture_name == "simulated":
        print("Mode: Fast simulated evaluation using CSV data projection")
    elif architecture_name == "gaia_smolagents":
        print("Mode: Real benchmark evaluation using GAIA dataset")
    print("Finding optimal Performance vs Cost trade-offs using Bayesian Optimization")
    print("Using Gaussian Process models with qLogExpectedHypervolumeImprovement")
    if resume_from:
        print(f"🔁 RESUMING from checkpoint: {resume_from}")
    print("="*80)


def demonstrate_multi_agent_optimization(config_path: str = "config.yaml", resume_from: str = None, resume_iteration: int = None):
    """Run multi-agent LLM optimization with architecture selection."""
    
    # Step 0: Load configurations
    config = load_config(config_path)
    architecture_name = config['evaluation']['architecture']
    # load the config file in  multi_agent_architectures/{architecture_name}/config.yaml
    arch_config_path = architecture_config_path(architecture_name, config)
    arch_config = load_architecture_config(architecture_name, arch_config_path)
    
    print_banner(architecture_name, resume_from)
    
    # Step 1: Create timestamped output folder
    output_dir = create_timestamp_folder(
        config['experiment']['name'].lower(), 
        architecture_name
    )
    
    # Handle resume functionality
    if resume_from:
        print(f"\n🔁 RESUME SETUP")
        print("-" * 50)
        print(f"🔁 RESUMING from checkpoint: {resume_from}")
        if resume_iteration is not None:
            print(f"   Starting from iteration: {resume_iteration}")
        
        # Import CheckpointManager for resume functionality
        from utils.checkpoint_manager import CheckpointManager
        
        # Create resume info and copy previous checkpoints
        resume_info_path = CheckpointManager.create_resume_info(output_dir, resume_from, resume_iteration)
        print(f"📋 Resume info saved: {resume_info_path}")
    
    # Copy configurations for reproducibility
    copy_configs_to_output(config_path, arch_config_path, output_dir)
    
    # Step 2: Initialize the appropriate evaluator
    print("\n📊 STEP 1: INITIALIZING EVALUATION SYSTEM")
    print("-" * 50)
    
    evaluator = create_evaluator_for_architecture(architecture_name, arch_config, config, arch_config_path, output_dir)
    
    # Step 3: Set up BoTorch parameters  
    dimension = evaluator.n_agents * len(evaluator.feature_names)
    
    # array with [[lower_bound (0)] * dimension, [upper_bound (1)] * dimension]
    bounds = torch.tensor([
        [config['bounds']['lower']] * dimension,
        [config['bounds']['upper']] * dimension
    ], dtype=torch.double)
    
    # Usually [-0.1, 1.1] all new points will be better than this reference point
    ref_point = torch.tensor([
        config['reference_point']['performance'],
        config['reference_point']['cost']
    ], dtype=torch.double)
    

    # Initialize optimizer
    opt_cfg = config.get('optimization', {})
    optimizer = BoTorchOptimizer(evaluator, bounds=bounds, ref_point=ref_point, output_dir=output_dir,
                                 seed=int(opt_cfg.get('seed', 42)), initial_design=str(opt_cfg.get('initial_design', 'balanced')),
                                 include_configs=opt_cfg.get('include_configs'), warm_start_from=opt_cfg.get('warm_start_from'))
    
    # n_initial: explicit value in the config wins; else d+1 with d = n_slots x n_features
    n_initial = config['optimization'].get('n_initial')
    if n_initial in (None, 'auto'):
        n_initial = dimension + 1
        print(f"  n_initial not set -> d+1 = {n_initial} (d = {evaluator.n_agents} slots x {len(evaluator.feature_names)} features)")
    config['optimization']['n_initial'] = n_initial

    # Display setup information
    print(f"\nOptimization setup:")
    print(f"  • Architecture: {architecture_name}")
    print(f"  • Agents Number to optimize: {evaluator.n_agents}")
    print(f"  • Features: {len(evaluator.feature_names)}")
    print(f"  • Dimension: {dimension}")
    print(f"  • Device: {optimizer.device}")
    print(f"  • Reference point: {ref_point.tolist()}")
    print(f"  • Initial points: {config['optimization']['n_initial']}")
    print(f"  • Iterations: {config['optimization']['n_iterations']}")
    print(f"  • Batch size: {config['optimization']['batch_size']}")
    
    # Show sample LLMs
    if hasattr(evaluator, 'llm_names') and evaluator.llm_names:
        print(f"\nSample LLMs available:")
        for i in [0, 5, 10, 15, 20]:
            if i < len(evaluator.llm_names):
                print(f"  • {evaluator.llm_names[i]}")
    
    # Step 4: Run optimization
    print(f"\n🎯 STEP 2: RUNNING OPTIMIZATION")
    print("-" * 50)
    
    torch_results = optimizer.optimize(
        n_initial=config['optimization']['n_initial'],
        n_iterations=config['optimization']['n_iterations'], 
        q=config['optimization']['batch_size'],
        save_interval=1,
        output_dir=output_dir,
        resume_from=resume_from,
        resume_iteration=resume_iteration
    )
    
    # Step 5: Analyze results
    print(f"\n📈 STEP 3: ANALYZING RESULTS")
    print("-" * 50)
    
    # Convert results for analysis compatibility
    results = {
        'X_all': torch_results['X_all'],
        'Y_all': torch_results['Y_all'], 
        'X_pareto': torch_results['X_pareto'],
        'Y_pareto': torch_results['Y_pareto'],
        'pareto_mask': torch_results['pareto_mask'],
        'n_evaluations': torch_results['n_evaluations'],
        'X': torch_results['X_all'],  # Legacy compatibility
        'Y': torch_results['Y_all'],  # Legacy compatibility  
        'pareto_X': torch_results['X_pareto'],  # Legacy compatibility
        'pareto_Y': torch_results['Y_pareto'],  # Legacy compatibility
        'optimization_history': []
    }
    
    analysis = analyze_results(optimizer, results)
    
    # Step 6: Generate outputs
    print(f"\n💾 STEP 4: GENERATING OUTPUTS")
    print("-" * 50)
    
    plot_path = plot_pareto_front(results, f'{output_dir}/optimization_pareto_front.png')
    results_path = save_results(analysis, f'{output_dir}/optimization_results.json')
    
    # Save raw torch results
    torch_results_path = f'{output_dir}/raw_torch_results.json'
    with open(torch_results_path, 'w') as f:
        serializable_results = {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in torch_results.items()
        }
        import json
        json.dump(serializable_results, f, indent=2)
    
    # Step 7: Summary
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
    print(f"  • Main Config: {output_dir}/config_used.yaml")
    print(f"  • Architecture Config: {output_dir}/architecture_config_used.yaml")
    print(f"  • Checkpoints: {output_dir}/checkpoint_iter_*.json")
    print(f"  • Visualization: {plot_path}")
    print(f"  • Analysis Results: {results_path}")
    print(f"  • Raw Results: {torch_results_path}")
    
    # Show example configurations
    print(f"\n🔧 EXAMPLE CONFIGURATIONS:")
    print("-" * 50)
    
    configs_to_show = [
        ('🏆 Highest Performance', best_perf),
        ('💰 Lowest Cost', best_cost),
        ('⚖️ Best Balance', best_balance)
    ]
    
    for label, config_result in configs_to_show:
        print(f"\n{label}:")
        agents = config_result['agents']
        for agent_config in agents:
            agent_id = agent_config['agent_id']
            role = agent_config['role']
            llm_name = agent_config['assigned_llm']
            print(f"  Agent {agent_id} ({role}): {llm_name}")
    
    # Count checkpoint files
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_iter_')]
    print(f"\n📊 OPTIMIZATION TRACKING:")
    print(f"  • {len(checkpoint_files)} checkpoint files saved")
    print(f"  • Architecture: {architecture_name}")
    print(f"  • Evaluation method: {'Real GAIA benchmark' if architecture_name == 'gaia_smolagents' else 'Simulated CSV-based'}")
    
    print(f"\n" + "="*80)
    print("🎉 MULTI-AGENT OPTIMIZATION COMPLETE!")
    print(f"Architecture: {architecture_name.upper()}")
    print("="*80)
    
    return analysis


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Multi-Agent LLM Optimization with optional checkpoint resume')
    parser.add_argument('--config', default='config.yaml', help='Path to config file (default: config.yaml)')
    parser.add_argument('--resume-from', type=str, help='Directory to resume from checkpoint')
    parser.add_argument('--resume-iteration', type=int, help='Specific iteration to resume from (optional)')
    
    args = parser.parse_args()
    
    # Set config path
    config_path = args.config
    # if we run in debug the path need to be /teamspace/studios/this_studio/LLM Multi-Agent Optimization Framework/config.yaml
    if not os.path.exists(config_path):
        config_path = "/teamspace/studios/this_studio/LLM Multi-Agent Optimization Framework/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file '{config_path}' not found!")
        print("Please ensure config.yaml exists in the current directory.")
        return 1
    
    try:
        analysis = demonstrate_multi_agent_optimization(config_path, args.resume_from, args.resume_iteration)
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
