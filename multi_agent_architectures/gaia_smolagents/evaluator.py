"""
GAIA SmolagentsLibrary Benchmark Evaluator

This module provides the adapter between BoTorch optimization and real GAIA benchmark execution.
It replaces the simulated evaluation with actual multi-agent task execution.
"""

import os
import sys
import numpy as np
import yaml
from typing import List, Dict, Tuple, Optional
import importlib.util

# Import the base LLM data processing utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_processor_v2 import find_closest_llm_index
from env_manager import env_manager


class GaiaBenchmarkEvaluator:
    """
    Adapter class that connects BoTorch optimization with real GAIA benchmark execution.
    
    This class implements the same interface as NumpyLLMEvaluator but performs actual
    multi-agent system evaluation instead of simulated assessment.
    """
    
    def __init__(self, llm_data: np.ndarray, llm_names: List[str], feature_names: List[str], 
                 n_agents: int, config_path: Optional[str] = None):
        """
        Initialize the GAIA benchmark evaluator.
        
        Args:
            llm_data: Array of LLM feature data for projection
            llm_names: List of LLM names corresponding to data rows
            feature_names: List of feature names (MMLU, HumanEval, etc.)
            n_agents: Number of agents (must be 5 for GAIA architecture)
            config_path: Path to architecture-specific config file
        """
        self.llm_data = llm_data
        self.llm_names = llm_names
        self.feature_names = feature_names
        self.n_agents = n_agents
        
        # Load architecture-specific configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.config = self._load_config(config_path)
        
        # Validate architecture requirements
        self._validate_setup()
        
        # Initialize the executor interface (will be implemented after repo cloning)
        self.executor_interface = None
        self._init_executor_interface()
    
    def _load_config(self, config_path: str) -> dict:
        """Load architecture-specific configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ GAIA architecture config loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Architecture config not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"❌ Error parsing architecture config: {e}")
            raise
    
    def _validate_setup(self):
        """Validate that the architecture setup is correct."""
        expected_agents = self.config['architecture']['n_agents']
        
        if self.n_agents != expected_agents:
            raise ValueError(
                f"Architecture mismatch: Expected {expected_agents} agents for GAIA, "
                f"but got {self.n_agents}. Please check your main config.yaml."
            )
        
        expected_roles = len(self.config['architecture']['agent_roles'])
        if expected_roles != expected_agents:
            raise ValueError(
                f"Configuration error: {expected_agents} agents specified but "
                f"{expected_roles} roles defined in agent_roles."
            )
        
        print(f"✅ GAIA architecture validation passed: {expected_agents} agents")
    
    def _init_executor_interface(self):
        """Initialize the connection to the executor repository."""
        # Check environment before initializing executor
        if not env_manager.is_ready_for_architecture('gaia_smolagents'):
            print("⚠️  Environment not ready for GAIA architecture")
            print("   Some functionality may be limited without proper API keys")
        
        repo_path = os.path.join(os.path.dirname(__file__), 
                                self.config['repository']['path'])
        interface_path = os.path.join(repo_path, 
                                    self.config['repository']['optimization_interface_path'])
        
        if not os.path.exists(repo_path):
            print(f"⚠️  Executor repository not found at: {repo_path}")
            print(f"    Please clone the executor repository to: {repo_path}")
            print(f"    Command: git clone <EXECUTOR_REPO_URL> {repo_path}")
            self.executor_interface = None
            return
        
        if not os.path.exists(interface_path):
            print(f"⚠️  Optimization interface not found at: {interface_path}")
            self.executor_interface = None
            return
        
        try:
            # Add the repository path to sys.path for relative imports
            repo_examples_path = os.path.join(repo_path, "examples", "open_deep_research")
            if repo_examples_path not in sys.path:
                sys.path.insert(0, repo_examples_path)
            
            # Dynamically import the optimization interface
            spec = importlib.util.spec_from_file_location("optimization_interface", interface_path)
            optimization_module = importlib.util.module_from_spec(spec)
            sys.modules["optimization_interface"] = optimization_module
            spec.loader.exec_module(optimization_module)
            
            self.executor_interface = optimization_module
            print(f"✅ Executor interface loaded from: {interface_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to load executor interface: {e}")
            print(f"    This is likely due to missing dependencies in the executor repository.")
            print(f"    Using fallback evaluation instead.")
            self.executor_interface = None
    
    def evaluate_agent_system(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Evaluate agent system configurations using real GAIA benchmark.
        
        Args:
            X_flat: Array of shape (q, n_agents * n_features) containing q configurations
                   in BoTorch's continuous optimization space
        
        Returns:
            Array of shape (q, 2) containing [performance, cost] for each configuration
        """
        q = X_flat.shape[0]  # Number of configurations to evaluate
        n_features = len(self.feature_names)
        results = np.zeros((q, 2), dtype=np.float64)
        
        print(f"\n🚀 GAIA BENCHMARK EVALUATION")
        print(f"   Evaluating {q} configurations using real multi-agent execution")
        print(f"   Architecture: {self.config['architecture']['name']}")
        
        # Check if executor is available
        if self.executor_interface is None:
            print("⚠️  Executor interface not available. Using fallback evaluation.")
            return self._fallback_evaluation(X_flat)
        
        for i in range(q):
            print(f"\n--- Configuration {i+1}/{q} ---")
            
            try:
                # Step 1: Project continuous vector to real LLM configurations
                config_dict = self._project_to_llm_configuration(X_flat[i], n_features)
                print(f"   Mapped to: {config_dict}")
                
                # Step 2: Execute real benchmark
                accuracy, cost = self._execute_real_benchmark(config_dict, run_suffix=f"_{i}")
                
                # Step 3: Normalize results for BoTorch
                normalized_performance = accuracy * self.config['botorch_integration']['performance_normalization']['scale_factor']
                normalized_cost = -cost if self.config['botorch_integration']['cost_normalization']['method'] == 'negative' else cost
                
                results[i, 0] = normalized_performance
                results[i, 1] = normalized_cost
                
                print(f"   ✅ Raw: {accuracy:.1f}% accuracy, ${cost:.4f} cost")
                print(f"   📊 Normalized: {normalized_performance:.3f} performance, {normalized_cost:.3f} cost")
                
            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                print(f"   🔄 Using fallback values")
                
                # Apply fallback values from config
                fallback_acc = self.config['evaluation']['fallback_accuracy']
                fallback_cost = self.config['evaluation']['fallback_cost']
                
                results[i, 0] = fallback_acc * self.config['botorch_integration']['performance_normalization']['scale_factor']
                results[i, 1] = -fallback_cost if self.config['botorch_integration']['cost_normalization']['method'] == 'negative' else fallback_cost
        
        print(f"\n✅ GAIA evaluation completed: {q} configurations processed")
        return results
    
    def get_configuration_details(self, X: np.ndarray) -> List[Dict]:
        """
        Get detailed configuration information for given X points.
        
        This method is required for compatibility with the analysis functions.
        
        Args:
            X: Array of shape (n_points, n_agents * n_features)
            
        Returns:
            List of configuration dictionaries with agent assignments
        """
        n_points = X.shape[0]
        n_features = len(self.feature_names)
        configurations = []
        
        for i in range(n_points):
            # Project to LLM configuration
            config_dict = self._project_to_llm_configuration(X[i], n_features)
            
            # Convert to the format expected by analysis functions
            agents_config = []
            agent_roles = self.config['architecture']['agent_roles']
            
            for agent_idx, role in enumerate(agent_roles):
                agents_config.append({
                    'agent_id': agent_idx,
                    'role': role,
                    'assigned_llm': config_dict[role]['model_id']
                })
            
            configurations.append({
                'agents': agents_config,
                'config_vector': X[i].tolist()
            })
        
        return configurations
    
    def _project_to_llm_configuration(self, X_single: np.ndarray, n_features: int) -> Dict[str, Dict]:
        """
        Project a single continuous vector to a concrete LLM configuration.
        
        Args:
            X_single: Flat vector of length (n_agents * n_features)
            n_features: Number of features per agent
            
        Returns:
            Dictionary mapping agent roles to model configurations
        """
        # Reshape to (n_agents, n_features)
        X_agents = X_single.reshape(self.n_agents, n_features)
        
        # Find closest LLM for each agent using existing projection logic
        llm_indices = find_closest_llm_index(X_agents, self.llm_data)
        
        # Build configuration dictionary for the executor
        agent_roles = self.config['architecture']['agent_roles']
        model_class = self.config['model_config']['default_model_class']
        
        config_dict = {}
        for agent_idx, role in enumerate(agent_roles):
            llm_name = self.llm_names[llm_indices[agent_idx]]
            
            # Use role-specific model class if defined
            role_specific_class = self.config['model_config']['role_specific_classes'].get(role, model_class)
            
            config_dict[role] = {
                'model_class': role_specific_class,
                'model_id': llm_name
            }
        
        return config_dict
    
    def _execute_real_benchmark(self, config_dict: Dict[str, Dict], run_suffix: str = "") -> Tuple[float, float]:
        """
        Execute the real GAIA benchmark with the given configuration.
        
        Args:
            config_dict: Agent model configuration dictionary
            run_suffix: Suffix for the run name
            
        Returns:
            Tuple of (accuracy_percentage, cost_dollars)
        """
        if self.executor_interface is None:
            raise RuntimeError("Executor interface not available")
        
        # Prepare execution parameters
        dataset_limits = self.config['evaluation']['dataset_limits']
        run_name = f"{self.config['evaluation']['run_name_prefix']}{run_suffix}"
        save_detailed = self.config['evaluation']['save_detailed_results']
        
        # Call the executor's evaluation function
        accuracy, cost, _ = self.executor_interface.evaluate_configuration(
            agent_model_configs=config_dict,
            dataset_limits=dataset_limits,
            run_name=run_name,
            save_detailed_results=save_detailed
        )
        
        return accuracy, cost
    
    def _fallback_evaluation(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Fallback evaluation when executor is not available.
        Returns placeholder values for development/testing.
        """
        q = X_flat.shape[0]
        results = np.zeros((q, 2), dtype=np.float64)
        
        print("⚠️  Using fallback evaluation (no real benchmark execution)")
        
        for i in range(q):
            # Generate some reasonable placeholder values
            # This can be replaced with the old simulated evaluation if needed
            fake_accuracy = np.random.uniform(0.1, 0.8)  # 10-80% accuracy
            fake_cost = np.random.uniform(0.01, 1.0)     # $0.01-$1.00 cost
            
            # Apply normalization
            results[i, 0] = fake_accuracy  # Already 0-1 scale
            results[i, 1] = -fake_cost     # Negative for maximization
            
            print(f"   Configuration {i+1}: {fake_accuracy:.1%} accuracy, ${fake_cost:.3f} cost")
        
        return results


def create_gaia_evaluator(llm_data: np.ndarray, llm_names: List[str], 
                         feature_names: List[str], n_agents: int) -> GaiaBenchmarkEvaluator:
    """
    Factory function to create a GAIA benchmark evaluator.
    
    This function provides a clean interface for creating the evaluator
    with the same signature as the simulated version.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    return GaiaBenchmarkEvaluator(
        llm_data=llm_data,
        llm_names=llm_names,
        feature_names=feature_names,
        n_agents=n_agents,
        config_path=config_path
    )


# For backward compatibility and testing
if __name__ == "__main__":
    print("🧪 GAIA Benchmark Evaluator")
    print("This module provides real benchmark evaluation for BoTorch optimization.")
    print("Use create_gaia_evaluator() to create an evaluator instance.")
