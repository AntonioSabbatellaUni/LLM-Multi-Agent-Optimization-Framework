# Cost estimation utilities for different LLM providers
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class CostEstimator:
    """Configurable cost estimation for different LLM models."""
    
    def __init__(self, cost_config_path: Optional[str] = None):
        """Initialize with cost configuration."""
        self.cost_config = self._load_cost_config(cost_config_path)
    
    def _load_cost_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load cost configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default cost configuration
            return self._get_default_cost_config()
    
    def _get_default_cost_config(self) -> dict:
        """Compact model definitions organized by provider."""
        
        # Model data: (input_cost_per_1M, output_cost_per_1M, context_window)
        models = {}
        
        # OpenAI - Updated from CSV
        openai_models = {
            "gpt-4.1-nano": (0.1, 0.4, 1_000_000),  # CSV: 0.1, 0.4
            "gpt-4.1-mini": (0.4, 1.6, 1_000_000),  # CSV: 0.4, 1.6  
            "gpt-4.1": (2.0, 8.0, 1_000_000),       # CSV: 2.0, 8.0
            "gpt-4o-mini": (0.15, 0.6, 128_000),    # CSV: 0.15, 0.6 ✓
            "gpt-4o": (2.5, 10.0, 128_000),         # CSV: 2.5, 10.0 ✓
            "gpt-4-turbo": (10.0, 30.0, 128_000),   # Not in CSV, keeping
            "gpt-3.5-turbo": (0.5, 1.5, 16_000),    # Not in CSV, keeping
            "o3-mini": (1.1, 4.4, 200_000),         # CSV: 1.1, 4.4
            "o4-mini": (1.1, 4.4, 200_000),         # CSV: 1.1, 4.4
        }
        
        # Anthropic - Updated from CSV
        anthropic_models = {
            "claude-3.5-haiku-20241022": (0.8, 4.0, 200_000),    # CSV: 0.80, 4.00 ✓
            "claude-opus-4": (15.0, 75.0, 200_000),              # CSV: 15.0, 75.0 ✓
            "claude-sonnet-4": (3.0, 15.0, 200_000),             # CSV: 3.0, 15.0 ✓
            "claude-3-5-sonnet-20241022": (3.0, 15.0, 200_000),  # Not in CSV, keeping
            "claude-3-haiku-20240307": (0.25, 1.25, 200_000),    # Not in CSV, keeping
        }
        
        # Google - Updated from CSV
        google_models = {
            "gemini-2.5-pro": (1.25, 5.0, 2_000_000),            # CSV: 1.25, 5.0 ✓
            "gemini-2.0-flash-001": (0.15, 0.6, 1_000_000),      # CSV: 0.15, 0.6 (corrected)
            "gemini-2.5-flash": (0.15, 0.6, 1_000_000),          # CSV: 0.15, 0.6 (added from CSV)
            "gemini-2.5-flash-lite": (0.1, 0.4, 1_000_000),      # CSV: 0.1, 0.4 (added from CSV)
            "gemini-1.5-pro": (1.25, 5.0, 2_000_000),            # Not in CSV, keeping
            "gemini-1.5-flash": (0.075, 0.3, 1_000_000),         # Not in CSV, keeping
        }
        
        # Meta/Llama - Updated from CSV 
        meta_models = {
            "llama-3.2-3b-instruct": (0.003, 0.006, 20_000),                    # CSV: 0.003, 0.006
            "meta-llama-llama-3.1-8b-instruct-turbo": (0.015, 0.02, 131_000),   # CSV: 0.015, 0.02
            "meta-llama-llama-3.1-70b-instruct-turbo": (0.88, 0.88, 131_072),   # Not in CSV, keeping
            "llama-3.3-70b-instruct": (0.25, 0.7, 128_000),                     # CSV: 0.25, 0.7
            "llama-4-maverick": (0.27, 0.85, 1_000_000),                        # CSV: 0.27, 0.85 (added)
            "llama-4-scout": (0.18, 0.59, 10_000_000),                          # CSV: 0.18, 0.59 (added)
        }
        
        # Qwen - Updated from CSV
        qwen_models = {
            "qwen3-14b": (0.1, 0.4, 128_000),                      # CSV: 0.1, 0.4
            "qwen-qwen3-235b-a22b-07-25": (0.5, 2.0, 128_000),     # CSV: 0.5, 2.0
            "qwen3-32b": (0.2, 0.8, 128_000),                      # CSV: 0.2, 0.8 (added)
            "qwen3-8b": (0.035, 0.138, 128_000),                   # CSV: 0.035, 0.138 (added)
        }
        
        # xAI - Updated from CSV
        xai_models = {
            "grok-3-beta": (5.0, 20.0, 128_000),                   # CSV: 5.0, 20.0 (corrected)
        }
        
        # DeepSeek - Updated from CSV
        deepseek_models = {
            "deepseek-r1-0528": (2.0, 8.0, 200_000),               # CSV: 2.0, 8.0 (added)
            "deepseek-chat-v3-0324": (1.8, 7.0, 200_000),          # CSV: 1.8, 7.0 (added)
            "deepseek-chat": (0.14, 0.28, 64_000),                 # Legacy, keeping
        }
        
        # Mistral - Updated from CSV
        mistral_models = {
            "codestral-2501": (0.3, 0.9, 256_000),                 # CSV: 0.3, 0.9 (added)
            "mistral-7b-instruct": (0.25, 0.25, 32_000),           # CSV: 0.25, 0.25 (added)
            "magistral-medium-2506": (0.4, 2.0, 32_000),           # CSV: 0.4, 2.0 (added)
            "mistral-large": (2.0, 6.0, 128_000),                  # Legacy, keeping
        }
        
        # Google Gemma - Added from CSV
        gemma_models = {
            "gemma-3-4b-it": (0.02, 0.04, 128_000),                # CSV: 0.02, 0.04
            "gemma-3-12b-it": (0.05, 0.1, 128_000),                # CSV: 0.05, 0.1
            "gemma-3-27b-it": (0.0, 0.0, 128_000),                 # CSV: 0.0, 0.0 (free)
        }
        
        # Merge all models with proper field names
        for provider_models in [openai_models, anthropic_models, google_models, 
                               meta_models, qwen_models, xai_models, deepseek_models, mistral_models, gemma_models]:
            for model, (input_cost, output_cost, context) in provider_models.items():
                models[model] = {
                    "input_cost_per_1m_tokens": input_cost,
                    "output_cost_per_1m_tokens": output_cost,
                    "context_window": context,
                    "provider": "openrouter"  # Since all our models use OpenRouter
                }
        
        return {
            "models": models,
            "default_model": "gpt-4o-mini",
            "default_cost": {"input_cost_per_1m_tokens": 0.15, "output_cost_per_1m_tokens": 0.6}
        }
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Estimate cost for a specific model and token usage."""
        # Normalize model name (remove provider prefixes, etc.)
        normalized_name = self._normalize_model_name(model_name)
        
        # Get model config
        model_config = self.cost_config["models"].get(
            normalized_name, 
            self.cost_config["models"][self.cost_config["default_model"]]
        )

        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in cost configuration, using default model '{self.cost_config['default_model']}' instead.")
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * model_config["input_cost_per_1m_tokens"]
        output_cost = (output_tokens / 1_000_000) * model_config["output_cost_per_1m_tokens"]
        total_cost = input_cost + output_cost
        
        return {
            "model": normalized_name,
            "provider": model_config["provider"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_1m_input": model_config["input_cost_per_1m_tokens"],
            "cost_per_1m_output": model_config["output_cost_per_1m_tokens"],
            "context_window": model_config["context_window"]
        }
    
    def estimate_multi_model_cost(self, model_usage: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Estimate cost for multiple models with different token usage."""
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        model_breakdown = {}
        
        for model_name, tokens in model_usage.items():
            input_tokens = tokens.get("input", 0)
            output_tokens = tokens.get("output", 0)
            
            cost_estimate = self.estimate_cost(model_name, input_tokens, output_tokens)
            model_breakdown[model_name] = cost_estimate
            
            total_cost += cost_estimate["total_cost"]
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        
        return {
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "model_breakdown": model_breakdown
        }
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Simplified normalization with fuzzy matching fallback."""
        from difflib import get_close_matches
        
        # Basic cleanup
        name = model_name.lower()
        if "openrouter/" in name:
            name = name.replace("openrouter/", "").replace("/", "-")
        
        # Exact match check
        available_models = list(self.cost_config["models"].keys())
        if name in available_models:
            return name
        
        # Fuzzy matching fallback
        matches = get_close_matches(name, available_models, n=1, cutoff=0.6)
        if matches:
            print(f"🔍 Fuzzy match: '{model_name}' → '{matches[0]}'")
            return matches[0]
        
        # Default fallback
        print(f"⚠️  Model '{model_name}' not found, using default: {self.cost_config['default_model']}")
        return self.cost_config["default_model"]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        normalized_name = self._normalize_model_name(model_name)
        return self.cost_config["models"].get(
            normalized_name,
            self.cost_config["models"][self.cost_config["default_model"]]
        )
    
    def list_supported_models(self) -> Dict[str, str]:
        """List all supported models and their providers."""
        return {
            model: config["provider"] 
            for model, config in self.cost_config["models"].items()
        }


def create_cost_config_template(output_path: str = "cost_config.yaml"):
    """Create a template cost configuration file."""
    estimator = CostEstimator()
    
    with open(output_path, 'w') as f:
        yaml.dump(estimator.cost_config, f, default_flow_style=False, indent=2)
    
    print(f"Cost configuration template created at: {output_path}")
    print("You can modify the costs and add new models as needed.")


if __name__ == "__main__":
    # Example usage
    estimator = CostEstimator()
    
    # Single model estimation
    cost = estimator.estimate_cost("gpt-4o-mini", 10000, 2000)
    print(f"Cost for gpt-4o-mini: ${cost['total_cost']:.4f}")
    
    # Multi-model estimation
    usage = {
        "gpt-4o-mini": {"input": 50000, "output": 10000},
        "claude-3-haiku-20240307": {"input": 30000, "output": 5000}
    }
    multi_cost = estimator.estimate_multi_model_cost(usage)
    print(f"Total multi-model cost: ${multi_cost['total_cost']:.4f}")
    
    # Create template
    create_cost_config_template("example_cost_config.yaml")
