"""
Data processing module for LLM optimization.
Handles loading, preprocessing, and normalization of LLM data.
"""

import pandas as pd
import torch
import numpy as np
from typing import Tuple, List, Dict

def load_llm_data(csv_path: str) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Load and preprocess LLM data from CSV.
    
    Returns:
        - LLM_DATA: Normalized tensor of shape (M, D) where M=num_LLMs, D=num_features
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
    
    # Convert to torch tensor
    llm_data_raw = torch.tensor(feature_data, dtype=torch.float64)
    
    # Normalize features to [0, 1]
    # For performance metrics (higher is better): normalize to [0, 1]
    # For cost metrics (lower is better): we'll invert and normalize
    llm_data_normalized = torch.zeros_like(llm_data_raw)
    
    for i, col in enumerate(feature_columns):
        col_data = llm_data_raw[:, i]
        col_min, col_max = col_data.min(), col_data.max()
        
        if 'Costo' in col:  # Cost features - invert so higher normalized value = lower cost
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

def find_closest_llm_index(ideal_features: torch.Tensor, llm_data: torch.Tensor) -> torch.Tensor:
    """
    Find the closest real LLM for each ideal feature vector using Euclidean distance.
    
    Args:
        ideal_features: Tensor of shape (q, D) - ideal feature vectors
        llm_data: Tensor of shape (M, D) - real LLM feature matrix
    
    Returns:
        Tensor of shape (q,) containing indices of closest LLMs
    """
    # Compute pairwise distances: (q, M)
    distances = torch.cdist(ideal_features, llm_data, p=2)
    
    # Find indices of minimum distances
    closest_indices = torch.argmin(distances, dim=1)
    
    return closest_indices

if __name__ == "__main__":
    # Test the data loading
    print("Testing data processor...")
    
    try:
        llm_data, llm_names, feature_names = load_llm_data('data/llm_data.csv')
        n_agents, agent_roles, token_usage = get_agent_configuration()
        
        print(f"\nSuccessfully loaded:")
        print(f"- {len(llm_names)} LLMs")
        print(f"- {len(feature_names)} features: {feature_names}")
        print(f"- {n_agents} agents with roles: {agent_roles}")
        
        print(f"\nFirst few LLMs:")
        for i in range(min(5, len(llm_names))):
            print(f"  {i}: {llm_names[i]}")
        
        print(f"\nNormalized data sample (first 3 LLMs, all features):")
        print(llm_data[:3])
        
        # Test the projection function
        print(f"\nTesting projection function...")
        # Create some random ideal features
        ideal_test = torch.rand(2, len(feature_names), dtype=torch.float64)
        closest_indices = find_closest_llm_index(ideal_test, llm_data)
        
        print(f"Random ideal features projected to LLMs:")
        for i, idx in enumerate(closest_indices):
            print(f"  Ideal {i} -> {llm_names[idx.item()]} (index {idx.item()})")
        
        print("\n✅ Data processor working correctly!")
        
    except Exception as e:
        print(f"❌ Error in data processor: {e}")
        raise