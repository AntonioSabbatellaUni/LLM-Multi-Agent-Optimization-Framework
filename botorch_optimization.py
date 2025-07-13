"""
BoTorch-based multi-objective optimization for LLM selection.
Uses true Bayesian Optimization with Gaussian Process models and qLogEHVI acquisition function.
"""

import torch
import numpy as np
import json
import os
import time
from typing import Dict, Any, Tuple
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.hypervolume import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from datetime import datetime

# Import from our existing files
from data_processor_v2 import NumpyLLMEvaluator
from logging_utils import (
    setup_optimization_logger, log_optimization_start, log_system_info, 
    log_gpu_info, log_iteration_timing, log_optimization_complete
)

class BoTorchOptimizer:
    """
    BoTorch-based multi-objective optimizer using Gaussian Process models
    and qLogExpectedHypervolumeImprovement acquisition function.
    """
    
    def __init__(self, evaluator: NumpyLLMEvaluator, bounds: torch.Tensor = None, ref_point: torch.Tensor = None, output_dir: str = None):
        """
        Initialize the BoTorch optimizer.
        
        Args:
            evaluator: The LLM evaluation function
            bounds: Search space bounds (shape: 2 x dimension)
            ref_point: Reference point for hypervolume calculation
            output_dir: Directory to save checkpoints (optional)
        """
        self.evaluator = evaluator
        self.dimension = evaluator.n_agents * len(evaluator.feature_names)
        self.output_dir = output_dir
        
        # Setup logging
        self.logger = setup_optimization_logger('BoTorchOptimizer', output_dir)
        
        # Device and dtype setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        
        # Log GPU info
        log_gpu_info(self.logger, self.device)
        
        # Set default bounds [0, 1] for normalized feature space
        if bounds is None:
            bounds = torch.tensor([[0.0] * self.dimension, [1.0] * self.dimension], dtype=self.dtype)
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        
        # Set default reference point (slightly below worst possible values)
        if ref_point is None:
            ref_point = torch.tensor([0.0, -50.0], dtype=self.dtype)  # [min_performance, min_negative_cost]
        self.ref_point = ref_point.to(device=self.device, dtype=self.dtype)
        
        # Log system configuration
        log_system_info(self.logger, self.device, self.dimension, self.bounds.shape, self.ref_point)
        if self.output_dir:
            self.logger.info(f"  - Checkpoints will be saved to: {self.output_dir}")
    def _initialize_model(self, train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
        """
        Initialize and fit a ModelListGP with standardized outcomes.
        
        Args:
            train_x: Training inputs (n_points x dimension)
            train_y: Training outputs (n_points x n_objectives)
            
        Returns:
            Fitted ModelListGP
        """
        # Ensure tensors are on correct device and dtype
        train_x = train_x.to(device=self.device, dtype=self.dtype)
        train_y = train_y.to(device=self.device, dtype=self.dtype)
        
        models = []
        for i in range(train_y.shape[-1]):  # For each objective
            train_objective = train_y[..., i:i+1]
            
            # Create SingleTaskGP with outcome standardization
            model = SingleTaskGP(
                train_x, 
                train_objective, 
                outcome_transform=Standardize(m=1)
            )
            
            # Move model to GPU
            model = model.to(device=self.device, dtype=self.dtype)
            
            # Fit the model
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            models.append(model)
        
        # Create ModelListGP and move to GPU
        model = ModelListGP(*models)
        model = model.to(device=self.device, dtype=self.dtype)
        return model
    
    def _generate_initial_data(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate initial training data by sampling real LLM configurations.
        
        Args:
            n_points: Number of initial points to generate
            
        Returns:
            Initial X and Y tensors
        """
        print(f"Generating {n_points} initial points...")
        
        # Get available LLM data
        llm_data = self.evaluator.llm_data  # Shape: (n_llms, n_features)
        n_llms, n_features = llm_data.shape
        n_agents = self.evaluator.n_agents
        
        X_init = []
        
        for _ in range(n_points):
            # Sample a configuration by randomly selecting LLMs for each agent
            config = np.zeros(self.dimension)
            
            for agent_idx in range(n_agents):
                # Randomly select an LLM for this agent
                llm_idx = np.random.randint(0, n_llms)
                
                # Copy LLM features to agent position in configuration
                start_idx = agent_idx * n_features
                end_idx = start_idx + n_features
                config[start_idx:end_idx] = llm_data[llm_idx, :]
            
            X_init.append(config)
        
        X_init = np.array(X_init)
        
        # Evaluate initial points
        Y_init = self.evaluator.evaluate_agent_system(X_init)
        
        # Convert to torch tensors
        X_init_torch = torch.tensor(X_init, dtype=self.dtype, device=self.device)
        Y_init_torch = torch.tensor(Y_init, dtype=self.dtype, device=self.device)
        
        self.logger.info(f"Initial data generated: X shape {X_init_torch.shape}, Y shape {Y_init_torch.shape}")
        return X_init_torch, Y_init_torch
    
    def optimize(self, n_initial: int = 20, n_iterations: int = 50, q: int = 5, save_interval: int = 5, output_dir: str = None) -> Dict[str, Any]:
        """
        Run Bayesian optimization loop.
        
        Args:
            n_initial: Number of initial random points
            n_iterations: Number of BO iterations
            q: Batch size for acquisition optimization
            save_interval: Interval for saving checkpoints
            output_dir: Directory for saving output files
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("="*80)
        self.logger.info("🚀 BOTORCH BAYESIAN OPTIMIZATION")
        self.logger.info("="*80)
        self.logger.info(f"Running {n_iterations} iterations with batch size {q}")
        self.logger.info("="*80)
        
        # Step 1: Generate initial data
        train_x, train_y = self._generate_initial_data(n_initial)
        
        # Track all evaluations
        all_x = [train_x]
        all_y = [train_y]
        
        # Track iteration times for ETA calculation
        iteration_times = []
        
        # Save initial state
        if output_dir:
            self._save_checkpoint(train_x, train_y, 0, output_dir)
        
        # Step 2: Bayesian Optimization loop
        for iteration in range(1, n_iterations + 1):
            iter_start_time = time.time()
            self.logger.info(f"\n🔄 Iteration {iteration}/{n_iterations}")
            
            # Fit GP model
            gp_start_time = time.time()
            self.logger.info("  📊 Fitting GP model...")
            model = self._initialize_model(train_x, train_y)
            gp_time = time.time() - gp_start_time
            
            # Get current Pareto front for partitioning
            self.logger.info("  🎯 Computing Pareto front...")
            pareto_mask = is_non_dominated(train_y)
            pareto_y = train_y[pareto_mask]
            
            # Create partitioning for hypervolume
            try:
                partitioning = NondominatedPartitioning(
                    ref_point=self.ref_point,
                    Y=pareto_y
                )
                
                # Create acquisition function
                acq_start_time = time.time()
                self.logger.info("  🎲 Optimizing acquisition function...")
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
                
                acq_func = qLogExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=self.ref_point,
                    partitioning=partitioning,
                    sampler=sampler
                )
                
                # Optimize acquisition function
                candidates, acq_values = optimize_acqf(
                    acq_function=acq_func,
                    bounds=self.bounds,
                    q=q,
                    num_restarts=20,
                    raw_samples=1024,
                )
                acq_time = time.time() - acq_start_time
                
                self.logger.info(f"  ✅ Found {q} new candidates")
                
            except Exception as e:
                self.logger.warning(f"  ⚠️  Acquisition optimization failed: {e}")
                self.logger.info("  🎲 Falling back to random sampling...")
                # Fallback: random sampling
                acq_time = 0.0  # No acquisition time for fallback
                candidates = torch.rand(q, self.dimension, dtype=self.dtype, device=self.device)
                candidates = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidates
            
            # Evaluate new candidates
            eval_start_time = time.time()
            self.logger.info("  📈 Evaluating new candidates...")
            candidates_np = candidates.detach().cpu().numpy()
            new_y_np = self.evaluator.evaluate_agent_system(candidates_np)
            new_y = torch.tensor(new_y_np, dtype=self.dtype, device=self.device)
            eval_time = time.time() - eval_start_time
            
            # Update training data
            train_x = torch.cat([train_x, candidates], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)
            
            # Track all data
            all_x.append(candidates)
            all_y.append(new_y)
            
            # Clear GPU memory cache periodically
            if self.device.type == "cuda" and iteration % 5 == 0:
                torch.cuda.empty_cache()
            
            # Incremental saving
            if output_dir and (iteration % save_interval == 0):
                self._save_checkpoint(train_x, train_y, iteration, output_dir, 
                                    new_x=candidates, new_y=new_y)
                self.logger.info(f"💾 Checkpoint saved at iteration {iteration}")
            
            # Progress update
            current_pareto_mask = is_non_dominated(train_y)
            n_pareto = current_pareto_mask.sum().item()
            best_performance = train_y[:, 0].max().item()
            best_cost = (-train_y[:, 1]).min().item()
            
            # Calculate iteration timing and ETA
            iter_total_time = time.time() - iter_start_time
            iteration_times.append(iter_total_time)
            
            # Calculate ETA
            if len(iteration_times) > 0:
                avg_time = sum(iteration_times) / len(iteration_times)
                remaining_iterations = n_iterations - iteration
                eta_seconds = avg_time * remaining_iterations
                eta_str = f"{eta_seconds:.1f}s" if eta_seconds < 60 else f"{eta_seconds/60:.1f}m"
            else:
                eta_str = "calculating..."
            
            self.logger.info(f"  ⏱️  Timing: GP={gp_time:.1f}s | Acq={acq_time:.1f}s | Eval={eval_time:.1f}s | Total={iter_total_time:.1f}s | ETA={eta_str} remaining")
            self.logger.info(f"  📊 Current: {len(train_y)} evaluations, {n_pareto} Pareto solutions")
            self.logger.info(f"  🏆 Best performance: {best_performance:.3f}, Best cost: {best_cost:.3f}")
        
        # Collect final results
        self.logger.info(f"\n✅ Optimization complete!")
        
        # Convert back to numpy for compatibility with existing analysis functions
        X_all = torch.cat(all_x, dim=0).detach().cpu().numpy()
        Y_all = torch.cat(all_y, dim=0).detach().cpu().numpy()
        
        # Find final Pareto front
        pareto_mask = is_non_dominated(torch.tensor(Y_all)).numpy()
        X_pareto = X_all[pareto_mask]
        Y_pareto = Y_all[pareto_mask]
        
        # Sort by performance (descending)
        pareto_order = np.argsort(Y_pareto[:, 0])[::-1]
        X_pareto = X_pareto[pareto_order]
        Y_pareto = Y_pareto[pareto_order]
        
        self.logger.info(f"  📊 Final: {len(Y_all)} total evaluations")
        self.logger.info(f"  🎯 Final Pareto front: {len(Y_pareto)} solutions")
        self.logger.info(f"  🏆 Best performance: {Y_pareto[0, 0]:.3f}")
        self.logger.info(f"  💰 Best cost: {(-Y_pareto[:, 1]).min():.3f}")
        
        results = {
            'X_all': X_all,
            'Y_all': Y_all,
            'X_pareto': X_pareto,
            'Y_pareto': Y_pareto,
            'pareto_mask': pareto_mask,
            'n_evaluations': len(Y_all),
            'n_pareto_solutions': len(Y_pareto),
            'n_iterations': n_iterations,
            'method': 'BoTorch'
        }
        
        return results
    
    def _save_checkpoint(self, train_x: torch.Tensor, train_y: torch.Tensor, iteration: int, output_dir: str,
                        new_x: torch.Tensor = None, new_y: torch.Tensor = None):
        """
        Save checkpoint data at each iteration.
        
        Args:
            train_x: All training inputs so far
            train_y: All training outputs so far
            iteration: Current iteration number
            output_dir: Directory to save checkpoints
            new_x: New candidates from this iteration (optional)
            new_y: New evaluations from this iteration (optional)
        """
        if not output_dir:
            return
            
        # Create checkpoint data
        checkpoint = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'train_x': train_x.detach().cpu().numpy().tolist(),
            'train_y': train_y.detach().cpu().numpy().tolist(),
            'n_evaluations': len(train_y),
        }
        
        # Add new candidates if provided
        if new_x is not None and new_y is not None:
            checkpoint['new_x'] = new_x.detach().cpu().numpy().tolist()
            checkpoint['new_y'] = new_y.detach().cpu().numpy().tolist()
            checkpoint['new_candidates_count'] = len(new_x)
        
        # Compute Pareto front info
        pareto_mask = is_non_dominated(train_y)
        pareto_y = train_y[pareto_mask]
        
        checkpoint['pareto_info'] = {
            'n_pareto_solutions': pareto_mask.sum().item(),
            'best_performance': train_y[:, 0].max().item(),
            'best_cost': (-train_y[:, 1]).min().item(),
            'pareto_mask': pareto_mask.cpu().numpy().tolist()
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_iter_{iteration:03d}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"  💾 Checkpoint saved: {checkpoint_path}")
