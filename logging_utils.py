"""
Logging utilities for BoTorch optimization system.
Provides centralized logging configuration for console and file output.
"""

import logging
import os
from datetime import datetime


def setup_optimization_logger(name: str = 'BoTorchOptimizer', output_dir: str = None) -> logging.Logger:
    """
    Setup logging to both console and file for optimization runs.
    
    Args:
        name: Logger name
        output_dir: Directory to save log file (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')  # Simple format for console
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Detailed format for file
    
    # Console handler - always enabled
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler - only if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'botorch_optimization.log')
        
        file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite existing log
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Notify that file logging is enabled
        print(f"📝 Optimization logs will be saved to: {log_file}")
    
    return logger


def log_optimization_start(logger: logging.Logger, n_iterations: int, batch_size: int):
    """Log optimization start banner."""
    logger.info("="*80)
    logger.info("🚀 BOTORCH BAYESIAN OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Running {n_iterations} iterations with batch size {batch_size}")
    logger.info("="*80)


def log_system_info(logger: logging.Logger, device, dimension: int, bounds_shape, ref_point):
    """Log system configuration information."""
    logger.info(f"BoTorch Optimizer initialized:")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Search dimension: {dimension}")
    logger.info(f"  - Bounds: {bounds_shape}")
    logger.info(f"  - Reference point: {ref_point}")


def log_gpu_info(logger: logging.Logger, device):
    """Log GPU availability and information."""
    if device.type == "cuda":
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logger.info(f"🚀 GPU acceleration enabled: {gpu_name}")
        logger.info(f"   GPU Memory: {gpu_memory:.1f} GB")
    else:
        logger.warning("⚠️  Running on CPU - GPU not available")


def log_iteration_timing(logger: logging.Logger, iteration: int, n_iterations: int, 
                        gp_time: float, acq_time: float, eval_time: float, 
                        total_time: float, eta_str: str, n_evaluations: int, 
                        n_pareto: int, best_performance: float, best_cost: float):
    """Log comprehensive iteration timing and progress information."""
    logger.info(f"  ⏱️  Timing: GP={gp_time:.1f}s | Acq={acq_time:.1f}s | Eval={eval_time:.1f}s | Total={total_time:.1f}s | ETA={eta_str} remaining")
    logger.info(f"  📊 Current: {n_evaluations} evaluations, {n_pareto} Pareto solutions")
    logger.info(f"  🏆 Best performance: {best_performance:.3f}, Best cost: {best_cost:.3f}")


def log_optimization_complete(logger: logging.Logger, n_evaluations: int, n_pareto: int, 
                             best_performance: float, best_cost: float):
    """Log optimization completion summary."""
    logger.info("✅ Optimization complete!")
    logger.info(f"  📊 Final: {n_evaluations} total evaluations")
    logger.info(f"  🎯 Final Pareto front: {n_pareto} solutions")
    logger.info(f"  🏆 Best performance: {best_performance:.3f}")
    logger.info(f"  💰 Best cost: {best_cost:.3f}")
