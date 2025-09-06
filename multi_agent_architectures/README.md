# Multi-Agent Architectures

This directory contains different multi-agent system implementations that can be optimized using the BoTorch framework.

## 📁 Structure

```
multi_agent_architectures/
├── README.md                           # This file
├── gaia_smolagents/                    # GAIA benchmark with SmolagentsLibrary
│   ├── README.md                       # Architecture-specific documentation
│   ├── evaluator.py                    # GaiaBenchmarkEvaluator adapter
│   ├── config.yaml                     # Architecture-specific config
│   └── smolagents_repo/                # Cloned executor repository
├── future_architecture_1/              # Future architecture placeholder
│   ├── README.md
│   ├── evaluator.py
│   └── config.yaml
└── future_architecture_2/              # Another future architecture
    ├── README.md
    ├── evaluator.py
    └── config.yaml
```

## 🎯 How to Add New Architectures

1. **Create Architecture Folder**: `mkdir new_architecture_name`
2. **Implement Evaluator**: Create `evaluator.py` with a class that implements the `evaluate_agent_system` method
3. **Add Configuration**: Create `config.yaml` with architecture-specific parameters
4. **Documentation**: Add `README.md` explaining the architecture
5. **Update Main Runner**: Modify the main optimization runners to support the new architecture

## 🔧 Evaluator Interface

Each architecture must implement an evaluator class with this interface:

```python
class ArchitectureEvaluator:
    def __init__(self, llm_data, llm_names, feature_names, n_agents):
        # Initialize with LLM data and configuration
        pass
    
    def evaluate_agent_system(self, X_flat: np.ndarray) -> np.ndarray:
        # Input: X_flat shape (q, n_agents * n_features) - continuous space
        # Output: results shape (q, 2) - [performance, cost] for each configuration
        pass
```

## 🚀 Current Architectures

### **gaia_smolagents** (Real Benchmark)
- **Purpose**: Real evaluation using GAIA benchmark dataset
- **Agents**: 5 specialized agents (manager, search_agent, text_inspector, visual_qa, reformulator)
- **Evaluation**: Actual execution on GAIA tasks with real accuracy/cost metrics
- **Use Case**: Production-ready optimization with real performance data

### **simulated** (Legacy - Built-in)
- **Purpose**: Fast simulated evaluation using CSV data
- **Agents**: Configurable number of agents with role-based features
- **Evaluation**: Distance-based projection to nearest LLM with estimated performance
- **Use Case**: Quick prototyping and testing of optimization algorithms

## 🔄 Switching Between Architectures

The main optimization scripts support architecture selection via config:

```yaml
# In main config.yaml
evaluation:
  architecture: "gaia_smolagents"  # or "simulated"
  # architecture-specific parameters loaded from respective configs
```
