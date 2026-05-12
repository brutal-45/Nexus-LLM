"""
Checkpoint Manager
===================
Handles saving and loading model checkpoints with support for:
    - Safetensors format (preferred, faster loading)
    - PyTorch binary format (fallback)
    - Optimizer state saving (for resuming training)
    - Distributed checkpointing (FSDP state dict)
    - Keeping top-K checkpoints by evaluation metric
    - Automatic cleanup of old checkpoints

Checkpoint contents:
    - model_state_dict: Model weights
    - optimizer_state_dict: Optimizer state
    - scheduler_state_dict: LR scheduler state
    - scaler_state_dict: GradScaler state (for mixed precision)
    - training_args: Training hyperparameters
    - step: Current training step
    - metrics: Evaluation metrics at this checkpoint
"""

from __future__ import annotations
import os
import json
import shutil
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

import torch
import torch.distributed as dist


@dataclass
class CheckpointState:
    """Complete checkpoint state."""
    step: int = 0
    epoch: int = 0
    global_step: int = 0
    model_state_dict: Optional[Dict] = None
    optimizer_state_dict: Optional[Dict] = None
    scheduler_state_dict: Optional[Dict] = None
    scaler_state_dict: Optional[Dict] = None
    best_metric: float = float("inf")
    metrics: Dict[str, float] = field(default_factory=dict)
    rng_state: Optional[Dict] = None


class CheckpointManager:
    """
    Manages model checkpointing for LLM training.
    
    Features:
        - Periodic checkpoint saving (every N steps)
        - Best checkpoint tracking (by evaluation metric)
        - Automatic cleanup of old checkpoints
        - FSDP-compatible distributed checkpointing
        - Safetensors format support
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 5000,
        max_checkpoints: int = 5,
        save_safetensors: bool = True,
        save_optimizer: bool = True,
        metric_mode: str = "min",  # "min" or "max"
        rank: int = 0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.save_safetensors = save_safetensors
        self.save_optimizer = save_optimizer
        self.metric_mode = metric_mode
        self.rank = rank
        self.best_metric = float("inf") if metric_mode == "min" else float("-inf")
        self.best_checkpoint_path: Optional[str] = None

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        return step > 0 and step % self.save_interval == 0

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: The model to save (may be FSDP-wrapped).
            optimizer: The optimizer.
            scheduler: The LR scheduler.
            scaler: The gradient scaler.
            step: Current training step.
            epoch: Current epoch.
            metrics: Evaluation metrics.
            is_best: Whether this is the best checkpoint.
        
        Returns:
            Path to the saved checkpoint directory.
        """
        if self.rank != 0:
            return ""  # Only rank 0 saves checkpoints
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint-step-{step:08d}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[Checkpoint] Saving checkpoint to {checkpoint_path}...")
        
        # Save model weights
        self._save_model(model, checkpoint_path)
        
        # Save optimizer state
        if optimizer is not None and self.save_optimizer:
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Save scaler state
        if scaler is not None:
            scaler_path = checkpoint_path / "scaler.pt"
            torch.save(scaler.state_dict(), scaler_path)
        
        # Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics or {},
            "is_best": is_best,
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Handle best checkpoint
        if metrics:
            metric_value = metrics.get("eval_loss", metrics.get("validation_loss", 0.0))
            self._update_best_checkpoint(metric_value, checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"[Checkpoint] Saved checkpoint at step {step}")
        return str(checkpoint_path)

    def _save_model(self, model: torch.nn.Module, checkpoint_path: Path):
        """Save model weights, handling FSDP wrapping."""
        if self.save_safetensors:
            try:
                from safetensors.torch import save_file
                
                # Check if model is FSDP-wrapped
                if hasattr(model, "_fsdp_wrapped_module"):
                    # FSDP: gather full state dict from all ranks
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                    
                    # Save only on rank 0
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    ):
                        state_dict = model.state_dict()
                        save_file(
                            state_dict,
                            checkpoint_path / "model.safetensors",
                        )
                else:
                    # Standard model: save directly
                    state_dict = model.state_dict()
                    save_file(
                        {k: v.cpu() for k, v in state_dict.items()},
                        checkpoint_path / "model.safetensors",
                    )
                return
            except ImportError:
                print("[Checkpoint] safetensors not available, falling back to PyTorch format")
        
        # Fallback: PyTorch binary format
        if hasattr(model, "_fsdp_wrapped_module"):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                torch.save(model.state_dict(), checkpoint_path / "pytorch_model.bin")
        else:
            torch.save(model.state_dict(), checkpoint_path / "pytorch_model.bin")

    def _update_best_checkpoint(self, metric_value: float, checkpoint_path: Path):
        """Track and update the best checkpoint."""
        is_better = False
        if self.metric_mode == "min" and metric_value < self.best_metric:
            is_better = True
        elif self.metric_mode == "max" and metric_value > self.best_metric:
            is_better = True
        
        if is_better:
            # Update symlink to best checkpoint
            best_path = self.checkpoint_dir / "best_checkpoint"
            if best_path.exists():
                best_path.unlink()
            best_path.symlink_to(checkpoint_path)
            self.best_metric = metric_value
            self.best_checkpoint_path = str(checkpoint_path)
            print(f"[Checkpoint] New best checkpoint: {metric_value:.4f}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-step-*"),
            key=lambda p: p.stat().st_mtime,
        )
        
        # Keep max_checkpoints most recent (plus best)
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            # Don't delete the best checkpoint
            if self.best_checkpoint_path and str(oldest) == self.best_checkpoint_path:
                continue
            print(f"[Checkpoint] Removing old checkpoint: {oldest}")
            shutil.rmtree(oldest, ignore_errors=True)

    def load_latest(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-step-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        
        return self.load(checkpoints[-1], model)

    def load_best(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best_checkpoint"
        if not best_path.exists():
            return self.load_latest(model)
        return self.load(Path(best_path.resolve()), model)

    def load(self, checkpoint_path: str, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Returns:
            Dictionary with optimizer_state_dict, scheduler_state_dict, etc.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}...")
        
        # Load model weights
        safetensors_path = checkpoint_path / "model.safetensors"
        bin_path = checkpoint_path / "pytorch_model.bin"
        
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict)
            except ImportError:
                state_dict = torch.load(bin_path, map_location="cpu")
                model.load_state_dict(state_dict)
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        # Load metadata
        result = {}
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                result["metadata"] = json.load(f)
        
        # Load optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            result["optimizer_state_dict"] = torch.load(optimizer_path, map_location="cpu")
        
        # Load scheduler
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler_path.exists():
            result["scheduler_state_dict"] = torch.load(scheduler_path, map_location="cpu")
        
        # Load scaler
        scaler_path = checkpoint_path / "scaler.pt"
        if scaler_path.exists():
            result["scaler_state_dict"] = torch.load(scaler_path, map_location="cpu")
        
        print(f"[Checkpoint] Checkpoint loaded successfully")
        return result
