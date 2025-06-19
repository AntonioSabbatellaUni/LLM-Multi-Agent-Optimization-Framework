"""
Data processing module for LLM optimization using pandas and numpy.
This version works with basic scientific Python stack.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def load_llm_data(csv_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load and preprocess LLM data from CSV.
    
    Returns:
        - LLM_DATA: Normalized numpy array of shape (M, D) where M=num_LLMs, D=num_features
        - LLM_NAMES: List of LLM names
        - FEATURE_NAMES: List of feature column names
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} LLMs from {csv_path}")
    
    # Handle missing values (N/A) by forward filling or using median
    df = df.fillna(method='ffill').fillna(df.median(numeric_only=True))
    
    # Define the features we'll use for optimization
    # We'll focus on key performance metrics and costs
    feature_columns = [
        'MMLU', 'HumanEval', 'GSM8K', 'MATH', 'MT_bench',  # Performance features
        'Costo_Input_1M', 'Costo_Output_1M'  # Cost features
    ]
    
    # Extract LLM names
    llm_names = df['Modello'].tolist()
    
    # Extract feature data
    feature_data = df[feature_columns].values
    
    # Convert to numpy array
    llm_data_raw = feature_data.astype(np.float64)
    
    # Normalize features to [0, 1]
    # For performance metrics (higher is better): normalize to [0, 1]
    # For cost metrics (lower is better): we'll invert and normalize
    llm_data_normalized = np.zeros_like(llm_data_raw)
    
    for i, col in enumerate(feature_columns):
        col_data = llm_data_raw[:, i]
        col_min, col_max = col_data.min(), col_data.max()
        
        if col_max == col_min:
            normalized = np.full_like(col_data, 0.5)  # All values are the same
        elif 'Costo' in col:  # Cost features - invert so higher normalized value = lower cost
            # Invert costs: high cost -> low value, low cost -> high value
            normalized = 1.0 - (col_data - col_min) / (col_max - col_min)
        else:  # Performance features - higher is better
            normalized = (col_data - col_min) / (col_max - col_min)
        
        llm_data_normalized[:, i] = normalized
    
    print(f"Features used: {feature_columns}")
    print(f"Data shape: {llm_data_normalized.shape}")
    print(f"Normalization ranges: min={llm_data_normalized.min():.3f}, max={llm_data_normalized.max():.3f}")
    
    return llm_data_normalized, llm_names, feature_columns

def get_agent_configuration() -> Tuple[int, List[str], Dict[str, Dict[str, int]]]:
    """
    Define agent configuration for the multi-agent system.
    
    Returns:
        - N_AGENTS: Number of agents
        - AGENT_ROLES: List of roles (corresponding to feature indices)
        - AGENT_TOKEN_USAGE: Token usage per agent role
    """
    N_AGENTS = 3
    
    # Map agents to specific capabilities
    # Agent 0: General reasoning (MMLU), Agent 1: Coding (HumanEval), Agent 2: Math (MATH)
    AGENT_ROLES = ['MMLU', 'HumanEval', 'MATH']
    
    # Simulated token usage for each agent type
    AGENT_TOKEN_USAGE = {
        'MMLU': {'input': 3000, 'output': 500},      # General knowledge tasks
        'HumanEval': {'input': 4000, 'output': 1200}, # Code generation tasks  
        'MATH': {'input': 2500, 'output': 800}       # Mathematical reasoning
    }
    
    return N_AGENTS, AGENT_ROLES, AGENT_TOKEN_USAGE

def find_closest_llm_index(ideal_features: np.ndarray, llm_data: np.ndarray) -> np.ndarray:
    """
    Find the closest real LLM for each ideal feature vector using Euclidean distance.
    
    Args:
        ideal_features: Array of shape (q, D) - ideal feature vectors
        llm_data: Array of shape (M, D) - real LLM feature matrix
    
    Returns:
        Array of shape (q,) containing indices of closest LLMs
    """
    # Compute pairwise distances: (q, M)
    # Using broadcasting: (q, 1, D) - (1, M, D) -> (q, M, D)
    distances = np.sqrt(np.sum((ideal_features[:, np.newaxis, :] - llm_data[np.newaxis, :, :]) ** 2, axis=2))
    
    # Find indices of minimum distances
    closest_indices = np.argmin(distances, axis=1)
    
    return closest_indices

class NumpyLLMEvaluator:
    """
    Evaluator for multi-agent LLM systems using numpy.
    Maps continuous feature space to discrete LLM assignments and computes objectives.
    """
    
    def __init__(self, 
                 llm_data: np.ndarray, 
                 llm_names: List[str],
                 feature_names: List[str],
                 n_agents: int,
                 agent_roles: List[str],
                 agent_token_usage: Dict[str, Dict[str, int]]):
        """
        Initialize the evaluator.
        """
        self.llm_data = llm_data
        self.llm_names = llm_names
        self.feature_names = feature_names
        self.n_agents = n_agents
        self.agent_roles = agent_roles
        self.agent_token_usage = agent_token_usage
        
        # Create mapping from role names to feature indices
        self.role_to_feature_idx = {}
        for role in agent_roles:
            if role in feature_names:
                self.role_to_feature_idx[role] = feature_names.index(role)
            else:
                # If exact match not found, use a reasonable default
                print(f"Warning: Role '{role}' not found in features. Using first performance feature.")
                self.role_to_feature_idx[role] = 0  # Use first feature as default
        
        # Find cost feature indices
        self.cost_input_idx = feature_names.index('Costo_Input_1M')
        self.cost_output_idx = feature_names.index('Costo_Output_1M')
        
        print(f"Evaluator initialized:")
        print(f"  - {len(llm_names)} LLMs available")
        print(f"  - {n_agents} agents with roles: {agent_roles}")
        print(f"  - Role to feature mapping: {self.role_to_feature_idx}")
    
    def evaluate_agent_system(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Evaluate multi-agent LLM system configurations.
        
        Args:
            X_flat: Array of shape (q, n_agents * n_features) containing q configurations
        
        Returns:
            Y: Array of shape (q, 2) with objectives [Performance, -Cost]
        """
        q = X_flat.shape[0]
        n_features = len(self.feature_names)
        
        # Ensure input has correct dimensions
        expected_dim = self.n_agents * n_features
        if X_flat.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features per configuration, got {X_flat.shape[1]}")
        
        results = np.zeros((q, 2), dtype=np.float64)
        
        for i in range(q):
            # Reshape flat configuration to agent x feature matrix
            X_ideal = X_flat[i].reshape(self.n_agents, n_features)
            
            # Project ideal features to closest real LLMs
            real_llm_indices = find_closest_llm_index(X_ideal, self.llm_data)
            
            # Calculate performance: sum of agent-specific performance scores
            total_performance = 0.0
            for agent_idx in range(self.n_agents):
                llm_idx = real_llm_indices[agent_idx]
                role = self.agent_roles[agent_idx]
                feature_idx = self.role_to_feature_idx[role]
                
                # Get performance score for this agent's role
                performance_score = self.llm_data[llm_idx, feature_idx]
                total_performance += performance_score
            
            # Calculate cost: sum of token costs across all agents
            total_cost = 0.0
            for agent_idx in range(self.n_agents):
                llm_idx = real_llm_indices[agent_idx]
                role = self.agent_roles[agent_idx]
                
                # Get cost per token (note: costs are inverted in normalized data)
                cost_input_normalized = self.llm_data[llm_idx, self.cost_input_idx]
                cost_output_normalized = self.llm_data[llm_idx, self.cost_output_idx]
                
                # Convert normalized costs back to cost factor
                cost_factor = 2.0 - cost_input_normalized - cost_output_normalized
                
                # Get token usage for this agent role
                token_usage = self.agent_token_usage[role]
                input_tokens = token_usage['input']
                output_tokens = token_usage['output']
                
                # Calculate cost (using cost_factor as proxy for actual cost)
                agent_cost = cost_factor * (input_tokens + output_tokens) / 1000.0
                total_cost += agent_cost
            
            # Store results: [Performance, -Cost] (for maximization)
            results[i, 0] = total_performance
            results[i, 1] = -total_cost  # Negative because we want to minimize cost
        
        return results
    
    def get_configuration_details(self, X_flat: np.ndarray) -> List[Dict]:
        """
        Get detailed information about LLM configurations.
        """
        q = X_flat.shape[0]
        n_features = len(self.feature_names)
        configurations = []
        
        for i in range(q):
            X_ideal = X_flat[i].reshape(self.n_agents, n_features)
            real_llm_indices = find_closest_llm_index(X_ideal, self.llm_data)
            
            config = {
                'configuration_id': i,
                'agents': []
            }
            
            for agent_idx in range(self.n_agents):
                llm_idx = real_llm_indices[agent_idx]
                agent_info = {
                    'agent_id': agent_idx,
                    'role': self.agent_roles[agent_idx],
                    'assigned_llm': self.llm_names[llm_idx],
                    'llm_index': llm_idx
                }
                config['agents'].append(agent_info)
            
            configurations.append(config)
        
        return configurations

def create_evaluator(csv_path: str = 'data/llm_data.csv') -> NumpyLLMEvaluator:
    """
    Convenience function to create an evaluator from CSV data.
    """
    llm_data, llm_names, feature_names = load_llm_data(csv_path)
    n_agents, agent_roles, agent_token_usage = get_agent_configuration()
    
    evaluator = NumpyLLMEvaluator(
        llm_data=llm_data,
        llm_names=llm_names,
        feature_names=feature_names,
        n_agents=n_agents,
        agent_roles=agent_roles,
        agent_token_usage=agent_token_usage
    )
    
    return evaluator