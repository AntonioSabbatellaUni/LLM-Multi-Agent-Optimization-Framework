"""
GAIA SmolagentsLibrary Benchmark Evaluator

This module provides the adapter between BoTorch optimization and real GAIA benchmark execution.
It replaces the simulated evaluation with actual multi-agent task execution.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import yaml
import contextlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import the base LLM data processing utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_processor_v2 import find_closest_llm_index
from env_manager import env_manager
from run_progress import update as _progress


class EvaluationPaused(Exception):
    """An evaluation stopped on an infrastructure problem (Go quota / rate limit, provider down, search blocked, crash).
    reason = {kind: 'quota'|'infra', model, error, ...}. Nothing is recorded for it; it is re-run (resumed) later."""

    def __init__(self, reason):
        self.reason = dict(reason or {})
        super().__init__(f"{self.reason.get('kind')} model={self.reason.get('model')}: {str(self.reason.get('error'))[:200]}")



@contextlib.contextmanager
def isolated_executor_env(executor_root_path: Path):
    """
    Context manager per entrare temporaneamente nell'ambiente dell'Esecutore.
    Modifica sys.path e la directory di lavoro, e li ripristina all'uscita.
    
    Args:
        executor_root_path: Path to the executor repository root directory
    """
    original_cwd = os.getcwd()
    original_sys_path = sys.path[:]
    
    try:
        # Aggiungi la radice dell'esecutore al path di Python
        # Questo risolve tutti gli import relativi (es. 'from utils import ...')
        executor_path_str = str(executor_root_path)
        if executor_path_str not in sys.path:
            sys.path.insert(0, executor_path_str)
        
        # Spostati nella directory dell'esecutore
        # Questo risolve i problemi con i percorsi relativi dei file (es. 'data/gaia')
        os.chdir(executor_root_path)
        
        print(f"🔧 Entering isolated executor environment: {executor_root_path}")
        yield # Qui dentro, il codice si comporterà come se fosse eseguito dall'Esecutore
        
    finally:
        # Ripristina tutto allo stato originale, indipendentemente da errori
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path


def effective_slots(config: dict) -> Dict[str, List[str]]:
    """Optimized SLOTS (one BO block of features each) -> the roles they drive (linked roles share one model).
    architecture.slots = {slot: [roles]}; without it every agent_roles entry is its own slot.
    Enabled optional roles (e.g. visual_qa) are appended as extra single-role slots."""
    arch = config['architecture']
    if arch.get('slots'):
        slots = {k: list(v) for k, v in arch['slots'].items()}
    else:
        slots = {r: [r] for r in arch['agent_roles']}
    for name, opt in (arch.get('optional_roles') or {}).items():
        if opt.get('enabled') and name not in slots:
            slots[name] = [name]
    return slots


class GaiaBenchmarkEvaluator:
    """
    Adapter class that connects BoTorch optimization with real GAIA benchmark execution.
    
    This class implements the same interface as NumpyLLMEvaluator but performs actual
    multi-agent system evaluation instead of simulated assessment.
    """
    
    def __init__(self, llm_data: np.ndarray, llm_names: List[str], feature_names: List[str], 
                 n_agents: int, config_path: Optional[str] = None,
                 supports_image: Optional[np.ndarray] = None, pool_df=None, output_dir: Optional[str] = None, pool_df_full=None):
        """
        Initialize the GAIA benchmark evaluator.
        
        Args:
            llm_data: Array of LLM feature data for projection
            llm_names: List of LLM names corresponding to data rows
            feature_names: List of feature names (MMLU, HumanEval, etc.)
            n_agents: Number of agents (should be 5 for GAIA architecture - if less than 5, it will use default LLMs)
            config_path: Path to architecture-specific config file
        """
        self.llm_data = llm_data
        self.llm_names = llm_names
        self.feature_names = feature_names
        self.n_agents = n_agents
        # boolean mask over llm_names: True = model accepts image input (pool CSV column `supports_image`)
        self.supports_image = None if supports_image is None else np.asarray(supports_image, dtype=bool)
        
        # Load architecture-specific configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.config = self._load_config(config_path)
        self.config_dir = Path(config_path).resolve().parent
        self.slots = effective_slots(self.config)          # slot -> [roles]
        self.agent_roles = list(self.slots)                # optimized slot names, in BO block order
        # pool DataFrame aligned with llm_names/llm_data (columns go_model_id, reasoning_effort, prices...); None = legacy pool
        self.pool_df = pool_df
        self.pool_df_full = pool_df_full  # unfiltered pool (prices of fixed roles such as visual_qa)
        self.output_dir = output_dir
        self.last_cost_report = None
        self.eval_counter = 0          # global index of the evaluation in progress (0-based)
        self.last_wall_time = None
        if output_dir:  # lets the executor publish <output_dir>/current_session.json for tools/malbo_status.py
            os.environ['MALBO_OUTPUT_DIR'] = str(Path(output_dir).resolve())
        if 'visual_qa' in self.agent_roles and (self.supports_image is None or not self.supports_image.any()):
            raise ValueError("visual_qa is optimized but the pool has no `supports_image` column / no image-capable model")
        
        # Set up paths for the executor repository
        self.executor_repo_path = Path(__file__).parent / self.config['repository']['path']
        self.executor_root_path = self.executor_repo_path / "examples" / "open_deep_research"
        
        # Validate architecture requirements
        self._validate_setup()
        
        # Check executor availability (but don't initialize yet)
        self.executor_available = self._check_executor_availability()
    
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
        
        expected_agents = len(self.agent_roles)
        if self.n_agents != expected_agents:
            raise ValueError(
                f"Architecture mismatch: {expected_agents} optimized slots {self.agent_roles} but n_agents={self.n_agents}."
            )
        all_roles = [r for rs in self.slots.values() for r in rs]
        if len(set(all_roles)) != len(all_roles):
            raise ValueError(f"A role appears in more than one slot: {self.slots}")
        unknown = set(all_roles) - set(self.config['architecture']['required_roles'])
        if unknown:
            raise ValueError(f"Slot roles not in required_roles: {unknown}")
        if self.pool_df is not None and len(self.pool_df) != len(self.llm_names):
            raise ValueError("pool_df and llm_names are not aligned")
        
        print(f"✅ GAIA architecture validation passed: {expected_agents} agents")
    
    def _check_executor_availability(self) -> bool:
        """Check if the executor repository and interface are available."""
        if not env_manager.is_ready_for_architecture('gaia_smolagents'):
            print("⚠️  Environment not ready for GAIA architecture")
            print("   Some functionality may be limited without proper API keys")
        
        if not self.executor_repo_path.exists():
            print(f"⚠️  Executor repository not found at: {self.executor_repo_path}")
            print(f"    Please clone the executor repository to: {self.executor_repo_path}")
            return False
        
        interface_path = self.executor_root_path / "optimization_interface.py"
        if not interface_path.exists():
            print(f"⚠️  Optimization interface not found at: {interface_path}")
            return False
        
        print(f"✅ Executor repository found at: {self.executor_repo_path}")
        return True
    
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
        if not self.executor_available:
            print("⚠️  Executor interface not available. Using fallback evaluation.")
            return self._fallback_evaluation(X_flat)
        
        if not hasattr(self, 'exhausted_models'):
            self.exhausted_models, self.stop_all, self.pauses = set(), False, []
        for i in range(q):
            print(f"\n--- Configuration {i+1}/{q} ---")
            
            try:
                if self.stop_all:
                    print("   ⏭️  deferred: evaluations paused (global problem)")
                    results[i] = np.nan
                    continue
                # Step 1: Project continuous vector to real LLM configurations
                config_dict = self._project_to_llm_configuration(X_flat[i], n_features)
                print(f"   Mapped to: {self._redact(config_dict)}")
                
                # Step 1.5: Complete configuration with default models for missing roles
                config_dict = self._complete_agent_configuration(config_dict)
                config_summary = ', '.join([f'{role}={cfg.get("pool", {}).get("name") or cfg["model_id"].split("/")[-1]}'
                                            f'(effort={cfg.get("reasoning_effort", "env")})' for role, cfg in config_dict.items()])
                print(f"   Complete config: {config_summary}")

                used = {c['model_id'] for c in config_dict.values()}
                if used & self.exhausted_models:
                    print(f"   ⏭️  deferred: uses quota-exhausted model(s) {sorted(used & self.exhausted_models)}")
                    results[i] = np.nan
                    continue

                # Step 2: Execute real benchmark
                option_per_slot = {sl: config_dict[self.slots[sl][0]].get('pool', {}).get('name') or config_dict[self.slots[sl][0]]['model_id']
                                   for sl in self.agent_roles}
                _t0 = time.time()
                _progress(self.output_dir, eval_index=self.eval_counter, eval_started_at=datetime.now().astimezone().isoformat(),
                          config=option_per_slot, config_index=i, current_session_dir=None)
                accuracy, cost = self._execute_real_benchmark(config_dict, self.eval_counter)
                self.last_wall_time = time.time() - _t0
                self._last_option_per_slot = option_per_slot
                
                # Step 3: Normalize results for BoTorch
                normalized_performance = accuracy * self.config['botorch_integration']['performance_normalization']['scale_factor']
                normalized_cost = -cost if self.config['botorch_integration']['cost_normalization']['method'] == 'negative' else cost

                if normalized_cost == 0:
                    print(f"   ❌ Invalid cost: {normalized_cost}. Skipping this configuration.")
                    normalized_cost = -10.0
                
                results[i, 0] = normalized_performance
                results[i, 1] = normalized_cost
                
                print(f"   ✅ Raw: {accuracy:.1f}% accuracy, ${cost:.4f} cost")
                self._log_evaluation(i, config_dict, accuracy, cost)
                self.eval_counter += 1
                _progress(self.output_dir, evals_done=self.eval_counter)
                print(f"   📊 Normalized: {normalized_performance:.3f} performance, {normalized_cost:.3f} cost")
                
            except EvaluationPaused as e:
                results[i] = np.nan
                r = e.reason
                self.pauses.append(dict(r, config_index=i, ts=datetime.now().astimezone().isoformat()))
                m = r.get('model')
                if r.get('kind') == 'quota' and m:
                    self.exhausted_models.add(m if '/' in m else f"openai/{m}")
                    self.exhausted_models.add(m)
                    print(f"   ⏸️  paused: quota/rate limit on {m} -> deferring every point that uses it, continuing with the others")
                else:
                    self.stop_all = True
                    print(f"   ⏸️  paused (global): {e} -> deferring all remaining points")
                if len(self.pauses) >= int(os.getenv('MALBO_MAX_PAUSES_PER_RUN', '4')):
                    self.stop_all = True   # many models failing in one run: probably not a per-model cap -> wait and resume
                _progress(self.output_dir, last_pause=self.pauses[-1], exhausted_models=sorted(self.exhausted_models))
                continue
            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                print(f"   🔄 Using fallback values")
                self.eval_counter += 1
                
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
            agent_roles = self.agent_roles
            
            for agent_idx, slot in enumerate(agent_roles):
                first = self.slots[slot][0]
                agents_config.append({
                    'agent_id': agent_idx,
                    'role': slot,
                    'roles': self.slots[slot],
                    'assigned_llm': config_dict[first]['model_id'],
                    'pool_option': config_dict[first].get('pool', {}).get('name')
                })
            
            configurations.append({
                'agents': agents_config,
                'config_vector': X[i].tolist()
            })
        
        return configurations
    
    def _project_to_llm_configuration(self, X_single: np.ndarray, n_features: int) -> Dict[str, Dict]:
        """
        Project a single continuous vector to a concrete LLM configuration.
        One block of n_features per SLOT; every role linked to the slot gets the same pool option.

        Returns: role -> model kwargs (+ `reasoning_effort` and `pool` price metadata when the pool is a Go pool)
        """
        X_agents = X_single.reshape(self.n_agents, n_features)

        # closest LLM per slot (euclidean); the visual_qa slot only among image-capable models
        llm_indices = find_closest_llm_index(X_agents, self.llm_data)
        for idx, slot in enumerate(self.agent_roles):
            if slot == 'visual_qa' and self.supports_image is not None:
                cand = np.flatnonzero(self.supports_image)
                llm_indices[idx] = cand[find_closest_llm_index(X_agents[idx:idx + 1], self.llm_data[cand])[0]]

        model_class = self.config['model_config']['default_model_class']
        config_dict = {}
        for slot_idx, slot in enumerate(self.agent_roles):
            row = int(llm_indices[slot_idx])
            for role in self.slots[slot]:
                role_class = self.config['model_config']['role_specific_classes'].get(role, model_class)
                config_dict[role] = {
                    'model_class': role_class,
                    **self._pool_model_fields(row),
                    **self._extra_model_kwargs(role)
                }
        return config_dict

    def _pool_model_fields(self, row: int) -> Dict:
        """model_id (+ reasoning_effort, pool prices) for pool row `row`."""
        name = self.llm_names[row]
        if self.pool_df is None or 'go_model_id' not in self.pool_df.columns:
            return {'model_id': name}
        r = self.pool_df.iloc[row]
        return {'model_id': f"openai/{r['go_model_id']}",
                # value to send for this role; '' = send nothing (loader gives it precedence over env)
                'reasoning_effort': '' if pd.isna(r.get('reasoning_effort')) else str(r['reasoning_effort']),
                'pool': self._price_dict(r)}

    @staticmethod
    def _price_dict(r) -> Dict:
        f = lambda k, d=0.0: d if k not in r or pd.isna(r[k]) else float(r[k])
        return {'name': str(r['Modello']), 'aa_in': f('Costo_Input_1M'), 'aa_cache': f('AA_CacheHit_1M'),
                'aa_out': f('Costo_Output_1M'), 'go_in': f('Go_Input_1M'), 'go_cache': f('Go_CacheRead_1M'),
                'go_out': f('Go_Output_1M'), 'go_cache_write': f('Go_CacheWrite_1M'), 'go_peak': f('Go_peak_multiplier', 1.0)}

    @staticmethod
    def _redact(cfg: Dict) -> Dict:
        """Copy of a role->model-kwargs dict that is safe to print (api_key hidden)."""
        return {r: {k: ('***' if k == 'api_key' else v) for k, v in c.items()} for r, c in cfg.items()}

    def _extra_model_kwargs(self, role: Optional[str] = None) -> Dict:
        """Per-model kwargs (api_base, api_key, max_tokens...) for self-hosted endpoints.
        From model_config.extra_model_kwargs and/or env MALBO_API_BASE (revival)."""
        kw = dict(self.config.get('model_config', {}).get('extra_model_kwargs') or {})
        if os.getenv('MALBO_API_BASE'):
            kw.update(api_base=os.environ['MALBO_API_BASE'], api_key=os.getenv('MALBO_API_KEY', 'dummy'),
                      max_tokens=int(os.getenv('MALBO_MAX_TOKENS', 32768)))
            if os.getenv('OPENCODE_API_KEY') and not os.getenv('MALBO_API_KEY'):
                kw['api_key'] = os.environ['OPENCODE_API_KEY']
        # some free Go models 422 on list-of-parts messages; visual_qa needs real content parts so it is never flattened
        if os.getenv('MALBO_FLATTEN') == '1' and role != 'visual_qa':
            kw['flatten_messages_as_text'] = True
        return kw

    def _complete_agent_configuration(self, config_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """Complete configuration with default models for missing roles (visual_qa fixed to MALBO_VISUAL_QA_MODEL_ID,
        priced with the matching pool row of the full pool CSV)."""
        required_roles = self.config['architecture']['required_roles']
        default_model_class = self.config['model_config']['default_model_class']
        default_model_id = self.config['model_config']['default_model_id']
        
        for role in required_roles:
            if role not in config_dict:
                model_id = os.getenv('MALBO_DEFAULT_MODEL_ID', default_model_id)
                if role == 'visual_qa':  # fixed vision model when not optimized
                    model_id = os.getenv('MALBO_VISUAL_QA_MODEL_ID', 'openai/deepseek-v4.1-flash')
                config_dict[role] = {
                    'model_class': default_model_class,
                    'model_id': model_id,
                    **self._extra_model_kwargs(role)
                }
                pool = self._fixed_role_pool(model_id)
                if pool:
                    config_dict[role]['pool'] = pool
        
        return config_dict

    def _fixed_role_pool(self, model_id: str) -> Optional[Dict]:
        """Prices for a role that is not optimized: first row of the pool whose go_model_id matches model_id."""
        df = self.pool_df
        if df is None or 'go_model_id' not in df.columns:
            return None
        gid = model_id.split('/')[-1]
        full = self.pool_df_full
        for d in (df, full):
            if d is not None:
                hit = d[d['go_model_id'] == gid]
                if len(hit):
                    return self._price_dict(hit.iloc[0])
        print(f"   ⚠️ no pool row for fixed model {model_id}: its calls will be unpriced")
        return None
    
    def _execute_real_benchmark(self, config_dict: Dict[str, Dict], i: int) -> Tuple[float, float]:
        """
        Execute the real GAIA benchmark with the given configuration using context manager.
        
        Args:
            config_dict: Agent model configuration dictionary
            i: Configuration index for logging
            
        Returns:
            Tuple of (accuracy_percentage, cost_dollars)
        """
        print(f"🔧 Executing REAL benchmark for configuration {i+1}...  ")
        try:
            # We need to ensure the executor environment is isolated
            # to avoid conflicts with other modules or imports.
            if self.output_dir:
                os.environ['MALBO_QCACHE_DIR'] = str((Path(self.output_dir) / 'qcache').resolve())   # per-question resume cache
            with isolated_executor_env(self.executor_root_path):
                from optimization_interface import evaluate_configuration

                # Prepare execution parameters
                dataset_limits = dict(self.config['evaluation']['dataset_limits'])
                tf = self.config['evaluation'].get('task_ids_file')
                if tf and not os.getenv('MALBO_TASK_IDS_FILE'):
                    os.environ['MALBO_TASK_IDS_FILE'] = str((self.config_dir / tf).resolve())  # fixed GAIA task list (env wins)
                if os.getenv('MALBO_TASK1'):
                    dataset_limits['task_1'] = int(os.environ['MALBO_TASK1'])
                run_name = f"{self.config['evaluation']['run_name_prefix']}_{i}"
                par = os.getenv('MALBO_PARALLEL_TASKS') or self.config['evaluation'].get('parallel_tasks') or 1
                par = max(1, int(par))
                save_detailed = self.config['evaluation']['save_detailed_results']

                print(f"Start Evaluation: Dataset limits {dataset_limits}, Run name {run_name}, Save detailed: {save_detailed}")
                #
                accuracy, cost, session_dir = evaluate_configuration(
                    agent_model_configs=config_dict,
                    dataset_limits=dataset_limits,
                    run_name=run_name,
                    save_detailed_results=save_detailed,
                    concurrency=par
                )
                session_dir = Path(session_dir).resolve()   # relative to the executor dir: resolve before the cwd is restored
            
            self.last_cost_report = None
            rep_path = Path(session_dir) / "cost_report.json"
            if rep_path.exists():
                self.last_cost_report = json.loads(rep_path.read_text())
                self.last_cost_report['session_dir'] = str(session_dir)
            print(f"  ✅ Results: Accuracy={accuracy:.2f}%, Cost=${cost:.4f}")
            return accuracy, cost

        except Exception as e:
            import traceback
            if type(e).__name__ == 'EvaluationPaused':   # from optimization_interface (utils.abort)
                raise EvaluationPaused(getattr(e, 'reason', {})) from e
            print(f"  ❌ Error during benchmark execution: {e}")
            traceback.print_exc() # Print full traceback for debugging
            # no penalty point (it would be a fake observation for the GP): pause, the point is re-run on resume
            raise EvaluationPaused({'kind': 'infra', 'model': None, 'error': f"{type(e).__name__}: {e}"}) from e
    
    def _log_evaluation(self, i: int, config_dict: Dict[str, Dict], accuracy: float, cost: float):
        """Append one record per evaluation to evaluations.jsonl."""
        rep = self.last_cost_report or {}
        rec = {'config_index': i, 'eval_index': self.eval_counter, 'wall_time_s': self.last_wall_time,
               'ts_end': datetime.now().astimezone().isoformat(),
               'options': getattr(self, '_last_option_per_slot', None), 'accuracy': accuracy, 'cost_objective': cost,
               'cost_source': 'pool_aa_prices' if rep else 'fallback_cost_estimator',
               'config': {r: {'model_id': c['model_id'], 'pool_option': c.get('pool', {}).get('name'),
                              'reasoning_effort': c.get('reasoning_effort')} for r, c in config_dict.items()},
               'cost_report': rep}
        if rep:
            print(f"   💰 aa_cost=${rep['aa_cost']:.4f} go_expected=${rep['go_expected']:.4f} go_list_cost=${rep['go_list_cost']:.4f} "
                  f"go_allowance=${rep['go_allowance']:.4f} {'⚠️ MISMATCH >2%' if rep['mismatch'] else 'consistent'}")
        self._append_record(rec)

    def _append_record(self, rec: Dict):
        d = Path(self.output_dir or os.getenv('MALBO_EVAL_LOG_DIR') or '.')
        try:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / 'evaluations.jsonl', 'a') as f:
                f.write(json.dumps(rec, default=str) + '\n')
        except Exception as e:
            print(f"   ⚠️ could not write evaluations.jsonl: {e}")

    def reuse_evaluation(self, rec: Dict) -> np.ndarray:
        """Y row of an earlier record, computed as in evaluate_agent_system; copied to evaluations.jsonl with reused=True."""
        bi = self.config['botorch_integration']
        perf = float(rec['accuracy']) * bi['performance_normalization']['scale_factor']
        cost = float(rec['cost_objective'])
        cost = -cost if bi['cost_normalization']['method'] == 'negative' else cost
        if cost == 0:
            cost = -10.0
        out = {k: v for k, v in rec.items() if k != '_source'}
        out.update(reused=True, reused_from=rec.get('_source'), reused_at=datetime.now().astimezone().isoformat())
        own = Path(self.output_dir or os.getenv('MALBO_EVAL_LOG_DIR') or '.') / 'evaluations.jsonl'
        copies = [json.loads(l) for l in own.read_text().splitlines() if l.strip()] if own.exists() else []
        if not any(c.get('options') == rec['options'] for c in copies):   # resume: already in this run's log
            self._append_record(out)
        print(f"   ♻️ reused {rec['options']}: acc={rec['accuracy']:.1f}% aa_cost=${float(rec['cost_objective']):.4f} (from {rec.get('_source')})")
        return np.array([perf, cost], dtype=np.float64)

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
                         feature_names: List[str], n_agents: int,
                         supports_image: Optional[np.ndarray] = None, config_path: Optional[str] = None,
                         pool_df=None, pool_df_full=None, output_dir: Optional[str] = None) -> GaiaBenchmarkEvaluator:
    """
    Factory function to create a GAIA benchmark evaluator.
    
    This function provides a clean interface for creating the evaluator
    with the same signature as the simulated version.
    """
    config_path = config_path or os.path.join(os.path.dirname(__file__), 'config.yaml')
    return GaiaBenchmarkEvaluator(
        llm_data=llm_data,
        llm_names=llm_names,
        feature_names=feature_names,
        n_agents=n_agents,
        config_path=config_path,
        supports_image=supports_image,
        pool_df=pool_df, pool_df_full=pool_df_full, output_dir=output_dir
    )


# For backward compatibility and testing
if __name__ == "__main__":
    print("🧪 GAIA Benchmark Evaluator")
    print("This module provides real benchmark evaluation for BoTorch optimization.")
    print("Use create_gaia_evaluator() to create an evaluator instance.")
