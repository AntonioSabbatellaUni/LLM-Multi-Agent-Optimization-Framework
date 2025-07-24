# LLM Multi-Agent Optimization Framework

## 🎯 Overview
This framework implements sophisticated multi-objective optimization approaches to find optimal trade-offs between **Performance** and **Cost** when assigning Large Language Models (LLMs) to agents in a multi-agent system.

### 🔬 Optimization Methods Available:
- **Heuristic Multi-Objective**: Fast exploration using genetic algorithm-style approach
- **BoTorch Bayesian Optimization**: True Bayesian optimization with Gaussian Process models
- **Iterative Convergence**: Analysis of optimization convergence patterns

## 📁 Project Structure

```
LLM Multi-Agent Optimization Framework/
├── 📁 multi_agent_architectures/        # 🆕 Multi-agent architecture implementations
│   ├── README.md                       # Architecture system documentation
│   ├── simulated/                      # Fast simulated evaluation
│   │   ├── config.yaml                 # Simulated architecture config
│   │   └── evaluator.py                # Simulated evaluation implementation
│   └── gaia_smolagents/                # Real GAIA benchmark evaluation
│       ├── README.md                   # GAIA architecture documentation
│       ├── config.yaml                 # GAIA architecture config
│       ├── evaluator.py                # GAIA benchmark implementation
│       └── smolagents_repo/            # (Cloned executor repository)
├── 📁 data/
│   └── llm_data.csv                    # 24 LLMs with performance & cost data
├── 📁 outputs/                         # All generated results (timestamped folders)
│   ├── jun_19_11_53/                  # Example: Basic optimization run
│   ├── jun_19_12_45_botorch/           # Example: BoTorch optimization run
│   └── [other timestamped runs...]
├── 🐍 Core Python Files:
│   ├── data_processor_v2.py           # Data loading & preprocessing (main)
│   ├── data_processor.py              # Legacy data processor (PyTorch)
│   ├── basic_optimization.py          # Heuristic multi-objective optimization
│   ├── botorch_optimization.py        # Bayesian optimization with BoTorch
│   ├── iterative_optimization.py      # Convergence tracking optimizer
│   ├── run_multi_agent_optimization.py # 🆕 Main runner with multi-architecture support
│   ├── run_full_optimization.py       # Heuristic optimization runner
│   ├── run_botorch_optimization.py    # BoTorch optimization runner with YAML config & checkpoints
│   ├── run_iterative_optimization.py  # Convergence analysis runner
│   ├── monitor_progress.py            # Real-time optimization monitoring tool
│   └── view_results.py                # Results visualization tool
├── 📋 config.yaml                     # Configuration file for BoTorch optimization
├── 📋 requirements.txt                # Python package dependencies
├── 📋 .gitignore                      # Git ignore rules
└── 📋 README.md                       # This documentation
```

## 🚀 Quick Start

### Prerequisites & Installation
```bash
# Install required packages
pip install -r requirements.txt
```

**Required packages:**
- `pandas`, `numpy` - Data processing & numerical computing
- `matplotlib` - Visualization
- `torch` - PyTorch for tensor operations
- `botorch`, `gpytorch` - Bayesian Optimization & Gaussian Processes
- `pyyaml` - YAML configuration file support

### Running Different Optimization Methods

#### 🆕 **Multi-Agent Optimization (Recommended - New Framework)**
The new unified runner supports multiple evaluation architectures with seamless switching.

```bash
# Configure in config.yaml, then run:
python run_multi_agent_optimization.py
```

**Available Architectures:**
- ✅ **Simulated**: Fast evaluation using CSV data (default)
- ✅ **GAIA SmolagentsLibrary**: Real benchmark evaluation on actual tasks

**Configuration Selection:**
```yaml
# In config.yaml
evaluation:
  architecture: "simulated"  # or "gaia_smolagents"
```

#### 1. 🔥 **BoTorch Bayesian Optimization (Legacy)**
Uses true Bayesian optimization with Gaussian Process models and qLogEHVI acquisition function.

```bash
# Configure optimization in config.yaml first, then run:
python run_botorch_optimization.py
```

**Features:**
- ✅ **YAML Configuration**: All parameters in `config.yaml`
- ✅ **Incremental Checkpoints**: Progress saved at each iteration
- ✅ **Reproducible**: Configuration copied to output folder
- ✅ **Real-time Monitoring**: Use `monitor_progress.py` to track progress

**Monitor progress in another terminal:**
```bash
python monitor_progress.py  # Auto-detects latest run
# or specify directory:
python monitor_progress.py --dir outputs/jun_19_12_45_botorch
```

#### 2. 🏃‍♂️ **Basic Multi-Objective Optimization (Fast)**:
```bash
python run_full_optimization.py
```

#### 3. 📈 **Iterative Analysis with Convergence Tracking**:
```bash
python run_iterative_optimization.py
```

## ⚙️ Configuration

### BoTorch Configuration (config.yaml)

The `config.yaml` file controls all BoTorch optimization parameters:

```yaml
# Optimization Parameters
optimization:
  n_initial: 10        # Initial random points
  n_iterations: 15     # BO iterations  
  batch_size: 3        # Candidates per iteration

# Model Parameters  
model:
  num_restarts: 20     # Acquisition optimization restarts
  raw_samples: 1024    # Raw samples for acquisition
  mc_samples: 512      # Monte Carlo samples

# Reference point for hypervolume calculation
reference_point:
  performance: -0.1    # Below worst performance
  cost: 1.1           # Above worst cost
```

**Key Parameters to Adjust:**
- `n_iterations`: More iterations = better results but slower
- `batch_size`: More candidates per iteration = faster convergence
- `n_initial`: More initial points = better initial coverage

## 📊 Generated Outputs

Each run creates a timestamped folder in `outputs/` with:

### BoTorch Optimization Outputs:
```
outputs/jun_19_12_45_botorch/
├── config_used.yaml                    # Configuration used for this run
├── resume_info.json                    # Resume source info (if resumed)
├── checkpoint_iter_000.json            # Initial state (iteration 0)  
├── checkpoint_iter_001.json            # After iteration 1
├── checkpoint_iter_002.json            # After iteration 2
├── ...                                 # One checkpoint per iteration
├── botorch_pareto_optimization.png     # Pareto front visualization
├── botorch_optimization_results.json   # Detailed analysis results
└── botorch_raw_results.json           # Raw BoTorch tensor data
```

### Understanding Checkpoint Files

Each `checkpoint_iter_XXX.json` contains:

```json
{
  "iteration": 5,
  "timestamp": "2025-06-19T12:45:30.123456",
  "train_x": [[...], [...], ...],       // All X points evaluated so far
  "train_y": [[perf, cost], ...],       // All Y values evaluated so far  
  "n_evaluations": 25,
  "new_x": [[...], [...], ...],         // New X candidates from this iteration
  "new_y": [[perf, cost], ...],         // New Y values from this iteration
  "new_candidates_count": 3,
  "pareto_info": {
    "n_pareto_solutions": 8,
    "best_performance": 2.540,
    "best_cost": 0.058,
    "pareto_mask": [true, false, ...]    // Which points are Pareto-optimal
  }
}
```

**Key Checkpoint Data:**
- `train_x`/`train_y`: Complete dataset at this iteration
- `new_x`/`new_y`: What BoTorch suggested and how it performed
- `pareto_info`: Current Pareto front statistics
- `timestamp`: Exact time of this iteration

### Other Optimization Outputs:
```
outputs/jun_19_11_53/
├── pareto_optimization.png             # Pareto front plot
├── optimization_results.json           # Analysis & recommendations  
├── pareto_front.png                    # Alternative visualization
└── optimization_summary.png            # Summary statistics
```

## 🔍 Monitoring & Analysis

### Real-time Progress Monitoring
```bash
# Monitor the latest optimization run
python monitor_progress.py

# Output example:
📊 Iteration 5 - 12:45:30
   Total evaluations: 25
   Pareto solutions: 8
   Best performance: 2.540
   Best cost: 0.058
   New candidates: 3
```

### Post-Run Analysis
```bash
# View detailed results from any run
python view_results.py outputs/jun_19_12_45_botorch/
```

## �️ Multi-Agent Architectures

The framework now supports multiple evaluation architectures, making it extensible for different multi-agent systems and benchmarks.

### 🏃‍♂️ **Simulated Architecture (Default)**
- **Purpose**: Fast prototyping and algorithm development
- **Evaluation**: CSV-based distance projection to real LLMs
- **Speed**: Very fast (~seconds per configuration)
- **Use Case**: Algorithm testing, hyperparameter tuning, initial exploration

### 🎯 **GAIA SmolagentsLibrary Architecture**
- **Purpose**: Real-world evaluation with actual task execution
- **Evaluation**: Live benchmark execution on GAIA dataset tasks
- **Speed**: Slower (~minutes per configuration)
- **Use Case**: Production optimization, final model selection, research validation

### 🔧 **Architecture Configuration**

```yaml
# Main config.yaml
evaluation:
  architecture: "simulated"  # Options: "simulated", "gaia_smolagents"

# Architecture-specific configs are in:
# multi_agent_architectures/<architecture_name>/config.yaml
```

### 🚀 **Adding New Architectures**

1. Create directory: `multi_agent_architectures/your_architecture/`
2. Implement `evaluator.py` with `evaluate_agent_system` method
3. Add `config.yaml` with architecture parameters
4. Update main config to include your architecture option

## �🏆 Key Features

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

## 📈 Example Results

The system successfully identifies optimal trade-offs:

- **🏆 Premium Performance**: 3.000 performance at 5.511 cost (o3 + GPT-4o + Grok 3 Beta)
- **💰 Ultra Efficient**: ~1.400 performance at ~0.030 cost (Qwen + Gemini + cost-effective models)
- **⚖️ Balanced Solution**: ~2.700 performance at ~1.500 cost (Mixed high-performance models)

## 🔧 Technical Innovation

This implementation demonstrates several advanced concepts:

1. **Bayesian Optimization with BoTorch** - True Bayesian optimization using Gaussian processes
2. **Multi-Objective Optimization** - Simultaneous optimization of performance and cost
3. **Continuous-to-Discrete Mapping** - Continuous relaxation of discrete optimization problems
4. **Pareto Efficiency Analysis** - Multi-criteria decision making for optimal trade-offs
5. **Realistic Cost Modeling** - LLM token pricing and usage patterns
6. **Scalable Multi-Agent Architecture** - Task-specialized agent configurations
7. **Timestamped Output Organization** - Systematic experiment tracking

## 🎯 Project Benefits

- **📁 Self-Contained**: All code, data, and outputs in one directory
- **🕐 Timestamped**: Each run creates organized, dated output folders
- **🔄 Portable**: Easy to move or share the entire framework
- **📊 Comprehensive**: Multiple analysis and visualization tools
- **🧹 Clean**: No global paths, everything is relative and organized

## 🚀 Usage

### **Multi-Agent Optimization Framework (Recommended):**
```bash
# Configure optimization parameters and architecture
nano config.yaml

# Run with selected architecture (simulated or gaia_smolagents)
python run_multi_agent_optimization.py

# Resume from checkpoint (works with any architecture)
python run_botorch_optimization.py --resume-from outputs/jul_13_09_52_botorch

# Resume from specific iteration  
python run_botorch_optimization.py --resume-from outputs/jul_13_09_52_botorch --resume-iteration 2
```

### **Legacy BoTorch Bayesian Optimization:**
```bash
# Configure optimization parameters
nano config.yaml

# Run BoTorch optimization with checkpoints
python run_botorch_optimization.py

# Resume from checkpoint (latest iteration)
python run_botorch_optimization.py --resume-from outputs/jul_13_09_52_botorch

# Resume from specific iteration  
python run_botorch_optimization.py --resume-from outputs/jul_13_09_52_botorch --resume-iteration 2

# View results and create visualizations
python view_results.py

# Create 3D Pareto evolution plots from checkpoints
python visualize_pareto_evolution.py
```

### **Alternative Methods:**
```bash
# Run basic heuristic optimization
python run_full_optimization.py

# Run iterative optimization with live tracking
python run_iterative_optimization.py
```

### **Analysis Tools:**
```bash
# Inspect checkpoint files
python inspect_checkpoints.py outputs/latest_folder

# Monitor optimization progress (during runs)
python monitor_progress.py
```

## �📞 Usage Notes

- All file paths are relative to the framework directory
- Run scripts from within the framework directory
- Each optimization run creates a unique timestamped folder
- The system automatically finds the latest results for viewing

---

## 🧪 Testing

To verify checkpoint and optimization functionality:

```bash
# Run all tests
pytest tests/test_checkpoint_system.py -v

# Run only unit tests (faster)
pytest tests/test_checkpoint_system.py::TestCheckpointManager -v
```

Tests cover:
- Checkpoint management (finding, loading, validating)
- Resume functionality (restart optimization from checkpoint)
- Error handling (missing files, directories, or keys)
- Integration (end-to-end BoTorchOptimizer resume)