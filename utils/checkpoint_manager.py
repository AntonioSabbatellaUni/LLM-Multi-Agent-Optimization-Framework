import os
import json
import torch
from typing import Optional, Tuple

class CheckpointManager:
    @staticmethod
    def find_latest_checkpoint(output_dir: str) -> Optional[str]:
        """Find the latest checkpoint file in the directory."""
        if not os.path.exists(output_dir):
            return None
        try:
            checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_iter_') and f.endswith('.json')]
            if not checkpoints:
                return None
            checkpoints.sort()
            return os.path.join(output_dir, checkpoints[-1])
        except PermissionError:
            return None

    @staticmethod
    def get_available_iterations(output_dir: str):
        """Return a sorted list of available checkpoint iteration numbers."""
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_iter_') and f.endswith('.json')]
        iterations = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
        return sorted(iterations)

    @staticmethod
    def load_checkpoint_data(checkpoint_path: str) -> dict:
        """Load checkpoint data from a file."""
        with open(checkpoint_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def prepare_resume_data(output_dir: str, target_iteration: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load train_x, train_y, and iteration from checkpoint."""
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory not found: {output_dir}")
            
        if target_iteration is None:
            checkpoint_path = CheckpointManager.find_latest_checkpoint(output_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoints found in directory: {output_dir}")
        else:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_iter_{target_iteration:03d}.json')
            
        if not os.path.exists(checkpoint_path):
            available_iterations = CheckpointManager.get_available_iterations(output_dir)
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Available iterations: {available_iterations}")
            
        data = CheckpointManager.load_checkpoint_data(checkpoint_path)
        
        # Validate checkpoint data
        required_keys = ['train_x', 'train_y', 'iteration']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
        
        train_x = torch.tensor(data['train_x'], dtype=torch.float64)
        train_y = torch.tensor(data['train_y'], dtype=torch.float64)
        iteration = data['iteration']
        return train_x, train_y, iteration

    @staticmethod
    def validate_checkpoint_directory(output_dir: str) -> bool:
        """Check if directory contains valid checkpoints."""
        if not os.path.exists(output_dir):
            return False
        checkpoints = CheckpointManager.get_available_iterations(output_dir)
        return len(checkpoints) > 0
    
    @staticmethod
    def get_checkpoint_info(output_dir: str) -> dict:
        """Get summary information about checkpoints in directory."""
        if not os.path.exists(output_dir):
            return {"error": "Directory does not exist"}
        
        iterations = CheckpointManager.get_available_iterations(output_dir)
        if not iterations:
            return {"error": "No checkpoints found"}
        
        latest_path = CheckpointManager.find_latest_checkpoint(output_dir)
        latest_data = CheckpointManager.load_checkpoint_data(latest_path)
        
        return {
            "total_checkpoints": len(iterations),
            "available_iterations": iterations,
            "latest_iteration": max(iterations),
            "latest_evaluations": latest_data.get('n_evaluations', 'unknown'),
            "latest_timestamp": latest_data.get('timestamp', 'unknown')
        }
