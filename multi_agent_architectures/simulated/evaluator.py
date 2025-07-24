"""
Simulated Multi-Agent Architecture Evaluator

This module provides the legacy simulated evaluation functionality as a proper architecture.
It maintains backward compatibility with the existing NumpyLLMEvaluator approach.
"""

import os
import sys
import numpy as np
import yaml
from typing import List, Optional

# Import the original evaluation logic  
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_processor_v2 import NumpyLLMEvaluator, load_llm_data, get_agent_configuration


class SimulatedEvaluator:
    """
    Adapter for the legacy simulated evaluation approach.
    
    This provides the same interface as the GAIA evaluator but uses the existing
    fast simulation based on CSV data projection and distance calculation.
    """
    
    def __init__(self, llm_data: np.ndarray, llm_names: List[str], feature_names: List[str], 
                 n_agents: int, config_path: Optional[str] = None):
        """
        Initialize the simulated evaluator.
        
        Args:
            llm_data: Array of LLM feature data
            llm_names: List of LLM names
            feature_names: List of feature names
            n_agents: Number of agents  
            config_path: Path to architecture config (optional)
        """
        self.llm_data = llm_data
        self.llm_names = llm_names
        self.feature_names = feature_names
        self.n_agents = n_agents
        
        # Load architecture config if provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.config = self._load_config(config_path)
        
        # Create the underlying NumpyLLMEvaluator
        self.numpy_evaluator = NumpyLLMEvaluator(
            llm_data=llm_data,
            llm_names=llm_names,
            feature_names=feature_names,
            n_agents=n_agents
        )
        
        print(f"✅ Simulated evaluator initialized with {n_agents} agents")
    
    def _load_config(self, config_path: str) -> dict:
        """Load architecture-specific configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Simulated architecture config loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Architecture config not found: {config_path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            print(f"❌ Error parsing architecture config: {e}")
            raise
    
    def evaluate_agent_system(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Evaluate agent system using simulated approach.
        
        Args:
            X_flat: Array of shape (q, n_agents * n_features) 
            
        Returns:
            Array of shape (q, 2) containing [performance, cost]
        """
        q = X_flat.shape[0]
        
        print(f"\n🏃‍♂️ SIMULATED EVALUATION")
        print(f"   Evaluating {q} configurations using distance-based projection")
        print(f"   Architecture: {self.config.get('architecture', {}).get('name', 'simulated')}")
        
        # Delegate to the existing NumpyLLMEvaluator
        results = self.numpy_evaluator.evaluate_agent_system(X_flat)
        
        print(f"✅ Simulated evaluation completed: {q} configurations processed")
        return results


def create_simulated_evaluator(llm_data: np.ndarray, llm_names: List[str], 
                              feature_names: List[str], n_agents: int) -> SimulatedEvaluator:
    """
    Factory function to create a simulated evaluator.
    
    This provides the same interface as the GAIA evaluator factory.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    return SimulatedEvaluator(
        llm_data=llm_data,
        llm_names=llm_names,
        feature_names=feature_names,
        n_agents=n_agents,
        config_path=config_path
    )


# Backward compatibility: create_evaluator equivalent
def create_evaluator():
    """
    Create evaluator using the legacy data_processor_v2.create_evaluator approach.
    This maintains full backward compatibility.
    """
    from data_processor_v2 import create_evaluator as legacy_create_evaluator
    return legacy_create_evaluator()


if __name__ == "__main__":
    print("🏃‍♂️ Simulated Architecture Evaluator")
    print("This module provides fast simulated evaluation for BoTorch optimization.")
    print("Use create_simulated_evaluator() to create an evaluator instance.")
