"""
Programmatic interface for GAIA benchmark evaluation.
Designed to be used by optimization loops (BoTorch, Optuna, etc.).
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

# Import from existing battle-tested modules
from loader import load_agent_models
from utils import EnhancedRunManager
from utils.cost_estimator import CostEstimator
from utils.subset_evaluation import create_deterministic_subset, calculate_final_metrics
from run_gaia_subset_enhanced import load_gaia_dataset, answer_single_question_with_tracking


class GaiaOptimizationInterface:
    """Clean, reusable interface for GAIA benchmark evaluation."""
    
    def __init__(self, base_config_path: Optional[str] = None):
        self.base_config = {}
        if base_config_path:
            import yaml
            with open(base_config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
    
    def evaluate_configuration(
        self, agent_model_configs: Dict[str, Any], dataset_limits: Optional[Dict[str, int]] = None,
        run_name: Optional[str] = None, concurrency: int = 1, random_seed: int = 42,
        save_detailed_results: bool = False
    ) -> Tuple[float, float, Path]:

        """
        Evaluate a specific model configuration on GAIA benchmark.
        
        Args:
            agent_model_configs: Model configuration for each agent role
            dataset_limits: Number of questions per task level
            run_name: Unique identifier for this evaluation
            concurrency: Number of parallel workers
            random_seed: Seed for reproducible dataset subset
            save_detailed_results: Whether to save detailed logs
            
        Returns:
            Tuple of (accuracy_percentage, total_cost_usd, session_directory_path)
        """
        
        # Generate defaults if not provided
        if run_name is None:
            run_name = f"opt_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if dataset_limits is None:
            dataset_limits = self.base_config.get('dataset_limits', {'task_1': 3, 'task_2': 0, 'task_3': 0})
        
        print(f"🚀 Starting evaluation: {run_name}")
        
        try:
            # Load models and setup tracking
            models = load_agent_models(agent_model_configs)
            run_manager = EnhancedRunManager()
            session_dir = run_manager.setup_session("optimization", f"subset_{run_name}")
            run_manager.start_run()
            
            # Load dataset subset
            eval_ds = load_gaia_dataset(use_raw_dataset=False, set_to_run="validation")
            subset_questions = create_deterministic_subset(eval_ds, dataset_limits, random_seed)
            
            if not subset_questions:
                print("⚠️ No questions selected for evaluation")
                return 0.0, 0.0, session_dir
            
            # Process questions
            results = []
            retry_config = self.base_config.get('retry_config', 
                {"max_retries": 2, "base_delay": 1.0, "backoff_factor": 2.0})
            
            if concurrency == 1:
                for i, example in enumerate(tqdm(subset_questions, desc="Processing questions")):
                    result = answer_single_question_with_tracking(example, models, run_manager, retry_config, i)
                    results.append(result)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {executor.submit(answer_single_question_with_tracking, example, models, 
                              run_manager, retry_config, i): i for i, example in enumerate(subset_questions)}
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
                        results.append(future.result())
            
            # Calculate metrics and cost breakdown from results
            # Structure of final_metrics is :
            # {'overall_accuracy': int (e.g. 66.66), 'correct_answers': 2, 'total_questions': 3, 'task_breakdown': {'task_1': {...}}, 'token_usage': {'total_input_tokens': 280029, 'total_output_tokens': 16566, 'total_tokens': 296595}, 'error_analysis': {'parsing_errors': 0, 'iteration_limit_errors': 0, 'agent_errors': 0, 'total_errors': 0}}



            final_metrics = calculate_final_metrics(results) 
            accuracy = final_metrics.get("overall_accuracy", 0.0)
            
            # Aggregate cost data from all results (accurate token counts)
            # NOTE: if we dont pass yaml file, it will use the default cost stored in the python module
            cost_estimator = CostEstimator() 
            total_cost = 0.0
            cost_breakdown = {}
            model_usage = {}  # {model_id: {tokens, agents_set}}
            
            # Aggregate tokens per model from all execution results
            for result in results:
                model_stats = result.get('row_traker_model_stats', {})
                for model_id, stats in model_stats.items():
                    if model_id not in model_usage:
                        model_usage[model_id] = {'input': 0, 'output': 0, 'agents': set()}
                    model_usage[model_id]['input'] += stats.total_input_tokens
                    model_usage[model_id]['output'] += stats.total_output_tokens
                    
                    # Find agents using this model
                    for agent_name, config in agent_model_configs.items():
                        if config.get('model_id') == model_id:
                            model_usage[model_id]['agents'].add(agent_name)
            
            
            # we have directly in results the 'token_traker_model_stats' which contains 
            # the token usage per model, let's check if we can use it, this differ
            # from the previous version where we had to calculate it from the results
            # and is more precise because previuslly we counted onl the total tokens of
            # the messages, but a message contain also the memory tokens - the tokens of the 
            # previous messages in the conversation, so we have to take into account the
            # token usage directly adding the token per model from the model wrapper
            
            # TODO remove try except
            try: 
                c = True # allaw the rebuc console to see model usage
                if c:
                    #reset model_usage to avoid double counting
                    model_usage = {}

                for result in results:
                    for model_id, usage in result.get('token_tracker_model_stats', {}).items():
                        # usage is a ModelUsageStats with model id model_id='..', total_calls: int, total_input_tokens: int, total_output_tokens: int, total_duration=36.52209448814392, first_call=datetime.datetime(2025, 7, 29, 9, 39, 44, 610526), last_call=datetime.datetime(2025, 7, 29, 9, 41, 45, 649176), calls=
                        total_input_tokens = usage.total_input_tokens
                        total_output_tokens = usage.total_output_tokens
                        if model_id not in model_usage:
                            model_usage[model_id] = {'input': 0, 'output': 0, 'agents': set()}
                        model_usage[model_id]['input'] += total_input_tokens
                        model_usage[model_id]['output'] += total_output_tokens

                        # we can find the agents directly in the recorded call with 
                        # results[0]['token_tracker_model_stats']['openrouter/openai/gpt-4.1-mini'].calls[0].agent_name => 'manager'
                        model_usage[model_id]['agents'] = list(set({call.agent_name for call in usage.calls}))

                # for agent_name, config in agent_model_configs.items():
                #             if config.get('model_id') == model_id:
                #                 cost_breakdown[model_id]['agents'].append(agent_name)
            except KeyError:
                print("⚠️ Warning: 'token_traker_model_stats' not found in results, using estimated costs")
            
            # Calculate cost per model and build breakdown
            for model_id, usage in model_usage.items():
                cost_estimate = cost_estimator.estimate_cost(
                    model_name=model_id, 
                    input_tokens=usage['input'], 
                    output_tokens=usage['output']
                )
                model_cost = cost_estimate.get("total_cost", 0.0)
                total_cost += model_cost
                
                cost_breakdown[model_id] = {
                    'model_id': model_id, 
                    'cost': model_cost,
                    'tokens': usage['input'] + usage['output'],
                    'agents': list(usage['agents'])
                }

            # Enhanced result output with cost breakdown
            if save_detailed_results:
                execution_time = time.time() - run_manager.start_time
                run_manager.save_session_summary(session_dir, "optimization", execution_time, final_metrics)
            # TODO ideally I want to save also the cost_breakdown and the model_usage and results
            

            # Print detailed cost breakdown for transparency
            print(f"💰 Cost Breakdown:")
            for agent_name, details in cost_breakdown.items():
                print(f"  {agent_name} ({details['model_id']}): ${details['cost']:.6f} ({details['tokens']:,} tokens)")
            print(f"  Total: ${total_cost:.6f}")
            print(f"✅ Evaluation complete: {accuracy:.2f}% accuracy, ${total_cost:.6f} cost")
            
            return accuracy, total_cost, session_dir
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return 0.0, 0.0, Path("./error")


def evaluate_configuration(agent_model_configs: Dict[str, Any], dataset_limits: Optional[Dict[str, int]] = None,
                          run_name: Optional[str] = None, **kwargs) -> Tuple[float, float, Path]:
    """Simple function interface for quick evaluations."""
    try:
        # this is only used for default values to load from  .. examples/open_deep_research/gaia_subset_config.yaml
        # the configuration actually used in evaluate_configuration is the one passed in the parameters
        interface = GaiaOptimizationInterface("gaia_subset_config.yaml")
    except FileNotFoundError:
        print("⚠️ gaia_subset_config.yaml not found, using interface without base config")
        interface = GaiaOptimizationInterface()
    return interface.evaluate_configuration(agent_model_configs, dataset_limits, run_name, **kwargs)


if __name__ == "__main__":
    import yaml
    
    print("🧪 Testing optimization interface by loading from config file...")
    
    # Test configuration
    # test_models = {
    #     'text_inspector': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4.1-nano'},
    #     'visual_qa': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4o-mini'},
    #     'reformulator': {'model_class': 'LiteLLMModel', 'model_id': 'gpt-4.1-nano'}
    # }
    # test_limits = {'task_1': 1, 'task_2': 0, 'task_3': 0}
    
        # Load configuration from YAML file to ensure consistency with real usage
    try:
        with open("gaia_subset_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract necessary configurations
        test_models = config['agents']
        test_limits = {'task_1': 1, 'task_2': 0, 'task_3': 0}  # Use 1 question for quick test
        
        print(f"✅ Loaded configuration with {len(test_models)} agent models:")
        for agent_name, agent_config in test_models.items():
            print(f"   {agent_name}: {agent_config['model_id']}")
        
    except FileNotFoundError:
        print("❌ Error: gaia_subset_config.yaml not found. Please ensure the config file exists.")
        exit(1)
    except KeyError as e:
        print(f"❌ Error: Your gaia_subset_config.yaml is missing a required key: {e}")
        exit(1)
    
    accuracy, cost, path = evaluate_configuration(
        agent_model_configs=test_models,
        dataset_limits=test_limits,
        run_name="interface_test_from_yaml",
        save_detailed_results=True,
        concurrency=1  # Use sequential processing for easier debugging
    )
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Total Cost: ${cost:.6f}")
    print(f"   Session: {path}")
    print("✅ Interface test completed successfully!")
