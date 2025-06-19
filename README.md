# LLM Multi-Agent Optimization Framework

## 🎯 Overview
This framework implements a sophisticated multi-objective Bayesian optimization approach to find optimal trade-offs between **Performance** and **Cost** when assigning Large Language Models (LLMs) to agents in a multi-agent system.

## 📁 Project Structure

```
LLM Multi-Agent Optimization Framework/
├── 📁 data/
│   └── llm_data.csv                    # 24 LLMs with performance & cost data
├── 📁 outputs/                         # All generated results (timestamped folders)
│   ├── jun_19_11_53/                  # Example: Basic optimization run
│   ├── jun_19_11_54/                  # Example: Iterative optimization run
│   └── [other timestamped runs...]
├── 🐍 Core Python Files:
│   ├── data_processor_v2.py           # Data loading & preprocessing (main)
│   ├── data_processor.py              # Legacy data processor (PyTorch)
│   ├── basic_optimization.py          # Multi-objective optimization engine
│   ├── iterative_optimization.py      # Convergence tracking optimizer
│   ├── run_full_optimization.py       # Main optimization runner
│   ├── run_iterative_optimization.py  # Convergence analysis runner
│   └── view_results.py               # Results visualization tool
└── 📋 README.md                       # This documentation
```

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have the required packages
pip install pandas numpy matplotlib
```

### Running the Optimization

1. **Basic Multi-Objective Optimization**:
   ```bash
   cd "LLM Multi-Agent Optimization Framework"
   python run_full_optimization.py
   ```

2. **Iterative Optimization with Convergence Tracking**:
   ```bash
   cd "LLM Multi-Agent Optimization Framework"
   python run_iterative_optimization.py
   ```

3. **View Results**:
   ```bash
   cd "LLM Multi-Agent Optimization Framework"
   python view_results.py
   ```

## 📊 Generated Outputs

Each run creates a timestamped folder in `outputs/` with:

### Basic Optimization (`run_full_optimization.py`):
- `pareto_optimization.png` - Pareto front visualization
- `llm_optimization_results.json` - Detailed optimization results
- `README.md` - Summary report

### Iterative Optimization (`run_iterative_optimization.py`):
- `convergence_analysis.png` - 4-panel convergence analysis
- `pareto_3d_evolution.png` - 3D Pareto front evolution
- `final_pareto_front.png` - Final optimization results
- `optimization_summary.png` - Summary metrics
- `convergence_results.json` - Convergence tracking data
- `final_optimization_results.json` - Final analysis results

### Results Viewer (`view_results.py`):
- `pareto_summary.png` - Simplified Pareto plot
- Automatically finds and uses latest results

## 🏆 Key Features

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

1. **Bayesian-Inspired Multi-Objective Optimization**
2. **Continuous Relaxation of Discrete Optimization Problems**
3. **Pareto Efficiency Analysis for Multi-Criteria Decision Making**
4. **Realistic Cost Modeling for LLM Systems**
5. **Scalable Multi-Agent Architecture**
6. **Timestamped Output Organization**

## 🎯 Project Benefits

- **📁 Self-Contained**: All code, data, and outputs in one directory
- **🕐 Timestamped**: Each run creates organized, dated output folders
- **🔄 Portable**: Easy to move or share the entire framework
- **📊 Comprehensive**: Multiple analysis and visualization tools
- **🧹 Clean**: No global paths, everything is relative and organized

## 📞 Usage Notes

- All file paths are relative to the framework directory
- Run scripts from within the framework directory
- Each optimization run creates a unique timestamped folder
- The system automatically finds the latest results for viewing

---

**Framework Version**: 2.0  
**Date**: June 19, 2025  
**Status**: Production Ready ✅
