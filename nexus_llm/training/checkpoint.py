"""Checkpointing: save/resume, best model tracking, checkpoint pruning, rotation."""

import os
import json
import time
import shutil
import logging
from typing import Optional, Dict, Any, List, Tuple

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpointing with best model tracking, pruning, and rotation."""

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        save_total_limit: int = 5,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_rng_state: bool = True,
    ):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_rng_state = save_rng_state

        self.best_metric_value: Optional[float] = None
        self.checkpoints: List[str] = []
        self._checkpoint_history: List[Dict[str, Any]] = []

        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            state: Dictionary containing model, optimizer, scheduler state dicts.
            step: Current training step.
            metrics: Current evaluation metrics.

        Returns:
            Path to the saved checkpoint directory.
        """
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        save_state = {}

        if "model_state_dict" in state:
            model_path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state["model_state_dict"], model_path)
            save_state["model_path"] = model_path

        if self.save_optimizer and "optimizer_state_dict" in state:
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            torch.save(state["optimizer_state_dict"], optimizer_path)
            save_state["optimizer_path"] = optimizer_path

        if self.save_scheduler and "lr_scheduler_state_dict" in state:
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            torch.save(state["lr_scheduler_state_dict"], scheduler_path)
            save_state["scheduler_path"] = scheduler_path

        if "scaler_state_dict" in state:
            scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
            torch.save(state["scaler_state_dict"], scaler_path)
            save_state["scaler_path"] = scaler_path

        if self.save_rng_state:
            rng_state = {
                "python": torch.random.get_rng_state().tolist() if torch.random.get_rng_state() is not None else None,
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            rng_path = os.path.join(checkpoint_dir, "rng_state.pt")
            torch.save(rng_state, rng_path)

        metadata = {
            "step": step,
            "epoch": state.get("epoch", 0),
            "global_step": step,
            "metrics": metrics or {},
            "timestamp": time.time(),
            "best_metric": self.best_metric_value,
        }

        meta_path = os.path.join(checkpoint_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if metrics and self.metric_for_best_model in metrics:
            is_best = self._update_best_metric(metrics)
            if is_best:
                best_dir = os.path.join(self.output_dir, "best_model")
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                shutil.copytree(checkpoint_dir, best_dir)
                logger.info(f"New best model saved at step {step} with {self.metric_for_best_model}={metrics[self.metric_for_best_model]:.4f}")

        self.checkpoints.append(checkpoint_dir)
        self._checkpoint_history.append(metadata)

        self._prune_checkpoints()

        logger.info(f"Checkpoint saved at step {step}: {checkpoint_dir}")
        return checkpoint_dir

    def _update_best_metric(self, metrics: Dict[str, float]) -> bool:
        """Update the best metric value. Returns True if this is a new best."""
        current_value = metrics.get(self.metric_for_best_model)
        if current_value is None:
            return False

        if self.best_metric_value is None:
            self.best_metric_value = current_value
            return True

        if self.greater_is_better:
            is_best = current_value > self.best_metric_value
        else:
            is_best = current_value < self.best_metric_value

        if is_best:
            self.best_metric_value = current_value

        return is_best

    def is_best(self, metrics: Dict[str, float]) -> bool:
        """Check if the current metrics represent a new best model."""
        current_value = metrics.get(self.metric_for_best_model)
        if current_value is None:
            return False
        if self.best_metric_value is None:
            return True
        if self.greater_is_better:
            return current_value > self.best_metric_value
        return current_value < self.best_metric_value

    def _prune_checkpoints(self):
        """Remove oldest checkpoints beyond save_total_limit."""
        while len(self.checkpoints) > self.save_total_limit:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest):
                shutil.rmtree(oldest)
                logger.info(f"Pruned old checkpoint: {oldest}")

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint if available."""
        if not self.checkpoints:
            all_dirs = [
                d for d in os.listdir(self.output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.output_dir, d))
            ]
            if not all_dirs:
                return None
            all_dirs.sort(key=lambda d: int(d.split("-")[-1]))
            latest_dir = os.path.join(self.output_dir, all_dirs[-1])
        else:
            latest_dir = self.checkpoints[-1]

        return self._load_from_dir(latest_dir)

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best model checkpoint."""
        best_dir = os.path.join(self.output_dir, "best_model")
        if not os.path.exists(best_dir):
            return None
        return self._load_from_dir(best_dir)

    def _load_from_dir(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Load checkpoint state from a directory."""
        state = {}

        model_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            state["model_state_dict"] = torch.load(model_path, map_location="cpu")

        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            state["optimizer_state_dict"] = torch.load(optimizer_path, map_location="cpu")

        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            state["lr_scheduler_state_dict"] = torch.load(scheduler_path, map_location="cpu")

        scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
        if os.path.exists(scaler_path):
            state["scaler_state_dict"] = torch.load(scaler_path, map_location="cpu")

        rng_path = os.path.join(checkpoint_dir, "rng_state.pt")
        if os.path.exists(rng_path):
            state["rng_state"] = torch.load(rng_path, map_location="cpu")

        meta_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                state["metadata"] = json.load(f)

        return state

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with their metadata."""
        result = []
        for checkpoint_dir in self.checkpoints:
            meta_path = os.path.join(checkpoint_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    result.append(json.load(f))
            else:
                result.append({"path": checkpoint_dir})
        return result

    def get_checkpoint_count(self) -> int:
        """Return the number of saved checkpoints."""
        return len(self.checkpoints)

    def cleanup_all(self):
        """Remove all checkpoints except the best model."""
        for checkpoint_dir in self.checkpoints:
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
        self.checkpoints = []
        logger.info("Cleaned up all checkpoints (best model retained if exists).")
