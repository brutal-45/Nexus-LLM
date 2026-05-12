"""
Training Utility Functions for Nexus LLM.

Provides essential training utilities including learning rate finding,
gradient scaling, early stopping, model EMA, stochastic weight averaging,
checkpoint management, metric tracking, time estimation, and device/seed management.

Classes:
    LearningRateFinder: Find optimal learning rate range.
    GradientScaler: Dynamic loss scaling for mixed precision.
    EarlyStopping: Early stopping with patience and min_delta.
    ModelEMA: Exponential moving average of model parameters.
    StochasticWeightAveraging: SWA for better generalization.
    CheckpointManager: Manage training checkpoints.
    MetricTracker: Track training metrics over time.
    Timer: Training time tracking and estimation.
    DeviceManager: GPU/CPU device management.
    SeedManager: Reproducible seeding.
"""

from __future__ import annotations

import abc
import copy
import json
import logging
import math
import os
import random
import shutil
import time
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class Precision(Enum):
    """Numerical precision mode."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED_FP16 = "mixed_fp16"
    MIXED_BF16 = "mixed_bf16"


class CheckpointFormat(Enum):
    """Checkpoint file format."""
    PYTORCH = auto()
    SAFE_TENSORS = auto()
    SHARDED = auto()


# ---------------------------------------------------------------------------
# Learning Rate Finder
# ---------------------------------------------------------------------------

class LearningRateFinder:
    """Find the optimal learning rate range for training.

    Implements the LR range test (Smith 2017) which trains the model
    with exponentially increasing learning rates and plots loss vs LR
    to find the optimal range.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        criterion: Loss function.
        device: Training device.
        init_lr: Initial learning rate to start from.
        final_lr: Final learning rate to end at.
        num_steps: Number of steps for the test.
        beta: Smoothing factor for loss averaging.
        skip_start: Number of initial steps to skip in analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        init_lr: float = 1e-7,
        final_lr: float = 10.0,
        num_steps: int = 200,
        beta: float = 0.98,
        skip_start: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device("cpu")
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.num_steps = num_steps
        self.beta = beta
        self.skip_start = skip_start

        self._lr_history: List[float] = []
        self._loss_history: List[float] = []
        self._smoothed_losses: List[float] = []
        self._suggested_lr: Optional[float] = None
        self._lr_multiplier = (final_lr / init_lr) ** (1.0 / num_steps)

    def _update_lr(self, step: int):
        """Update learning rate for the current step."""
        new_lr = self.init_lr * (self._lr_multiplier ** step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def run(
        self,
        dataloader: DataLoader,
        start_batch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the learning rate range test.

        Args:
            dataloader: Training data loader.
            start_batch: Batch to start from.

        Returns:
            Dictionary with LR history, loss history, and suggested LR.
        """
        self._lr_history.clear()
        self._loss_history.clear()
        self._smoothed_losses.clear()

        self.model.train()
        data_iter = iter(dataloader)

        if start_batch is not None:
            for _ in range(start_batch):
                next(data_iter, None)

        avg_loss = 0.0
        best_loss = float("inf")

        logger.info(
            f"LR Finder: testing from {self.init_lr:.2e} to {self.final_lr:.2e} "
            f"over {self.num_steps} steps"
        )

        for step in range(self.num_steps):
            lr = self._update_lr(step)
            self._lr_history.append(lr)

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get("input_ids", batch.get("inputs"))
                targets = batch.get("labels", batch.get("targets"))
            else:
                inputs, targets = batch, None

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            if targets is not None and isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if targets is not None:
                loss = self.criterion(outputs, targets)
            else:
                if isinstance(outputs, dict):
                    loss = outputs.get("loss", outputs.get("lm_loss"))
                else:
                    loss = outputs

            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=self.device)

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_val = loss.item()

            if not math.isfinite(loss_val) or loss_val > 100 * best_loss:
                logger.info(f"LR Finder: stopping at step {step}, LR={lr:.2e} (loss diverged)")
                break

            avg_loss = self.beta * avg_loss + (1 - self.beta) * loss_val
            debiased_avg = avg_loss / (1 - self.beta ** (step + 1))
            self._smoothed_losses.append(debiased_avg)
            self._loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val

            if step % 20 == 0:
                logger.info(
                    f"  Step {step}/{self.num_steps}: LR={lr:.2e}, "
                    f"loss={loss_val:.4f}, smoothed={debiased_avg:.4f}"
                )

        self._suggested_lr = self._find_suggested_lr()
        self._restore_lr()

        return {
            "lr_history": self._lr_history,
            "loss_history": self._loss_history,
            "smoothed_losses": self._smoothed_losses,
            "suggested_lr": self._suggested_lr,
            "num_steps": len(self._lr_history),
        }

    def _find_suggested_lr(self) -> float:
        """Find the suggested learning rate using the steepest descent heuristic.

        The suggested LR is the one where the loss decrease is steepest.
        """
        if len(self._smoothed_losses) < self.skip_start + 5:
            return self.init_lr * 10

        losses = self._smoothed_losses[self.skip_start:]
        lrs = self._lr_history[self.skip_start:]

        if not losses:
            return self.init_lr * 10

        min_loss_idx = 0
        min_loss = losses[0]
        for i, l in enumerate(losses):
            if l < min_loss:
                min_loss = l
                min_loss_idx = i

        gradients = []
        for i in range(1, len(losses)):
            gradient = (losses[i] - losses[i - 1]) / (lrs[i] - lrs[i - 1])
            gradients.append(gradient)

        if not gradients:
            return lrs[min_loss_idx] if lrs else self.init_lr * 10

        min_gradient_idx = 0
        min_gradient = gradients[0]
        for i, g in enumerate(gradients):
            if g < min_gradient:
                min_gradient = g
                min_gradient_idx = i

        suggested_idx = min(min_gradient_idx, min_loss_idx - 1)
        suggested_idx = max(0, min(suggested_idx, len(lrs) - 1))

        return lrs[suggested_idx]

    def _restore_lr(self):
        """Restore original learning rate."""
        if hasattr(self.optimizer, '_param_groups_backup'):
            for pg, backup in zip(self.optimizer.param_groups, self.optimizer._param_groups_backup):
                pg['lr'] = backup
        else:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.init_lr

    def get_suggested_lr(self) -> float:
        """Get the suggested learning rate."""
        return self._suggested_lr or self.init_lr * 10

    def plot_data(self) -> Dict[str, List[float]]:
        """Get data for plotting loss vs learning rate."""
        return {
            "lr": self._lr_history,
            "loss": self._loss_history,
            "smoothed_loss": self._smoothed_losses,
        }


# ---------------------------------------------------------------------------
# Gradient Scaler
# ---------------------------------------------------------------------------

class GradientScaler:
    """Dynamic loss scaling for mixed precision training.

    Prevents underflow in float16 by dynamically scaling the loss
    and adjusting the scale factor based on gradient overflow.

    Args:
        init_scale: Initial loss scale factor.
        growth_factor: Factor to increase scale when no overflow.
        backoff_factor: Factor to decrease scale on overflow.
        growth_interval: Steps between scale increases.
        max_scale: Maximum allowed scale.
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2 ** 24,
    ):
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale

        self._scale = init_scale
        self._growth_tracker = 0
        self._found_inf = False
        self._overflow_count = 0
        self._scale_history: List[float] = []

    @property
    def scale(self) -> float:
        """Current loss scale factor."""
        return self._scale

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss by the current scale factor.

        Args:
            loss: Loss tensor.

        Returns:
            Scaled loss tensor.
        """
        return loss * self._scale

    def unscale_grads(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients and check for overflow.

        Args:
            optimizer: The optimizer whose gradients to unscale.
        """
        self._found_inf = False
        inv_scale = 1.0 / max(1.0, self._scale)

        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)

                    if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                        self._found_inf = True
                        break

            if self._found_inf:
                break

    def update(self, optimizer: torch.optim.Optimizer):
        """Update scale factor based on overflow status.

        Args:
            optimizer: The optimizer.
        """
        if self._found_inf:
            self._scale = max(1.0, self._scale * self.backoff_factor)
            self._growth_tracker = 0
            self._overflow_count += 1
            self._zero_grads(optimizer)
            logger.debug(f"GradientScaler: overflow detected, reducing scale to {self._scale}")
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale = min(self._scale * self.growth_factor, self.max_scale)
                self._growth_tracker = 0

        self._scale_history.append(self._scale)

    def _zero_grads(self, optimizer: torch.optim.Optimizer):
        """Zero out all gradients."""
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state for checkpointing."""
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "overflow_count": self._overflow_count,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load scaler state from checkpoint."""
        self._scale = state.get("scale", self.init_scale)
        self._growth_tracker = state.get("growth_tracker", 0)
        self._overflow_count = state.get("overflow_count", 0)

    def get_stats(self) -> Dict[str, Any]:
        """Get scaler statistics."""
        return {
            "current_scale": self._scale,
            "overflow_count": self._overflow_count,
            "growth_tracker": self._growth_tracker,
            "growth_interval": self.growth_interval,
        }


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping with patience and min_delta.

    Monitors a metric and stops training when the metric stops improving
    for a specified number of steps.

    Args:
        mode: 'min' for loss-like metrics, 'max' for accuracy-like.
        patience: Number of steps to wait for improvement.
        min_delta: Minimum change to count as improvement.
        restore_best: Whether to track best weights for restoration.
        cooldown: Steps to wait after a restoration before monitoring.
    """

    def __init__(
        self,
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
        cooldown: int = 0,
    ):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.cooldown = cooldown

        self._counter = 0
        self._best_value: Optional[float] = None
        self._should_stop = False
        self._best_weights: Optional[Dict[str, torch.Tensor]] = None
        self._cooldown_counter = 0
        self._history: List[float] = []

        if mode == "min":
            self._is_better = lambda current, best: current < best - min_delta
        else:
            self._is_better = lambda current, best: current > best + min_delta

    def step(self, value: float, model: Optional[nn.Module] = None) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.
            model: Optional model to save best weights from.

        Returns:
            True if training should stop.
        """
        self._history.append(value)

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False

        if self._best_value is None or self._is_better(value, self._best_value):
            self._best_value = value
            self._counter = 0
            if self.restore_best and model is not None:
                self._best_weights = {
                    name: param.data.clone()
                    for name, param in model.named_parameters()
                }
            return False

        self._counter += 1

        if self._counter >= self.patience:
            self._should_stop = True
            if self._cooldown > 0:
                self._cooldown_counter = self._cooldown
            logger.info(
                f"EarlyStopping: no improvement for {self.patience} steps "
                f"(best={self._best_value:.6f}, current={value:.6f})"
            )
            return True

        return False

    def should_stop(self) -> bool:
        """Return whether training should stop."""
        return self._should_stop

    def restore_best_weights(self, model: nn.Module):
        """Restore model to best weights.

        Args:
            model: Model to restore weights into.
        """
        if self._best_weights is None:
            logger.warning("No best weights to restore")
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._best_weights:
                    param.data.copy_(self._best_weights[name])

        logger.info(f"Restored best weights (value={self._best_value:.6f})")

    def get_best_value(self) -> Optional[float]:
        """Return the best metric value seen."""
        return self._best_value

    def reset(self):
        """Reset early stopping state."""
        self._counter = 0
        self._best_value = None
        self._should_stop = False
        self._best_weights = None
        self._cooldown_counter = 0
        self._history.clear()

    def get_history(self) -> List[float]:
        """Return metric history."""
        return list(self._history)

    def state_dict(self) -> Dict[str, Any]:
        """Get early stopping state."""
        return {
            "counter": self._counter,
            "best_value": self._best_value,
            "should_stop": self._should_stop,
            "history": self._history,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load early stopping state."""
        self._counter = state.get("counter", 0)
        self._best_value = state.get("best_value")
        self._should_stop = state.get("should_stop", False)
        self._history = state.get("history", [])


# ---------------------------------------------------------------------------
# Model EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of model parameters.

    Maintains a smoothed copy of model parameters using EMA, which
    often leads to better generalization.

    Args:
        model: The model to track.
        decay: EMA decay factor (higher = slower update).
        device: Device for the EMA model.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.decay = decay
        self.device = device
        self._shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._updates = 0
        self._init_shadow()

    def _init_shadow(self):
        """Initialize shadow parameters from the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shadow_param = param.data.clone().detach()
                if self.device is not None:
                    shadow_param = shadow_param.to(self.device)
                self._shadow[name] = shadow_param

    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        self._updates += 1
        decay = min(self.decay, (1 + self._updates) / (10 + self._updates))

        for name, param in self.model.named_parameters():
            if name in self._shadow and param.requires_grad:
                shadow = self._shadow[name]
                if self.device is not None:
                    shadow = shadow.to(param.device)
                shadow.mul_(decay).add_(param.data, alpha=1.0 - decay)
                if self.device is not None:
                    self._shadow[name] = shadow.to(self.device)

    def apply_shadow(self):
        """Apply EMA parameters to the model (backup originals)."""
        self._backup.clear()
        for name, param in self.model.named_parameters():
            if name in self._shadow:
                self._backup[name] = param.data.clone()
                shadow = self._shadow[name]
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                param.data.copy_(shadow)

    def restore(self):
        """Restore original parameters from backup."""
        if not self._backup:
            logger.warning("No backup to restore")
            return
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state for checkpointing."""
        device_safe_shadow = {}
        for name, tensor in self._shadow.items():
            device_safe_shadow[name] = tensor.cpu()
        return {
            "shadow": device_safe_shadow,
            "decay": self.decay,
            "updates": self._updates,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        shadow = state.get("shadow", {})
        for name, tensor in shadow.items():
            if self.device is not None:
                self._shadow[name] = tensor.to(self.device)
            else:
                self._shadow[name] = tensor
        self.decay = state.get("decay", self.decay)
        self._updates = state.get("updates", 0)

    def get_decay(self) -> float:
        """Return current EMA decay."""
        return min(self.decay, (1 + self._updates) / (10 + self._updates))

    def num_updates(self) -> int:
        """Return number of EMA updates."""
        return self._updates


# ---------------------------------------------------------------------------
# Stochastic Weight Averaging
# ---------------------------------------------------------------------------

class StochasticWeightAveraging:
    """Stochastic Weight Averaging for better generalization.

    Averages model weights collected during the final phase of training
    to produce a smoother, better-generalizing model.

    Args:
        model: The model to train.
        start_step: Step to start collecting weights.
        freq: Frequency of weight collection.
        avg_fn: Averaging function for collected weights.
    """

    def __init__(
        self,
        model: nn.Module,
        start_step: int = 10000,
        freq: int = 500,
        avg_fn: str = "linear",
    ):
        self.model = model
        self.start_step = start_step
        self.freq = freq
        self.avg_fn = avg_fn

        self._collected_weights: List[Dict[str, torch.Tensor]] = []
        self._averaged_weights: Optional[Dict[str, torch.Tensor]] = None
        self._current_step = 0
        self._num_collected = 0

    def step(self):
        """Called after each training step."""
        self._current_step += 1

        if (self._current_step >= self.start_step
                and (self._current_step - self.start_step) % self.freq == 0):
            self._collect_weights()

    def _collect_weights(self):
        """Collect current model weights."""
        weights = {
            name: param.data.clone().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self._collected_weights.append(weights)
        self._num_collected += 1
        logger.debug(f"SWA: collected weights at step {self._current_step} "
                     f"({self._num_collected} total)")

    def average_weights(self) -> Dict[str, torch.Tensor]:
        """Compute averaged weights from collected snapshots.

        Returns:
            Dictionary of averaged parameter tensors.
        """
        if not self._collected_weights:
            logger.warning("SWA: no weights collected")
            return {}

        averaged = {}
        first = self._collected_weights[0]

        for name in first:
            if self.avg_fn == "linear":
                total = sum(
                    snapshot[name].float()
                    for snapshot in self._collected_weights
                    if name in snapshot
                )
                averaged[name] = total / self._num_collected
            else:
                total = sum(
                    snapshot[name].float()
                    for snapshot in self._collected_weights
                    if name in snapshot
                )
                averaged[name] = total / self._num_collected

        self._averaged_weights = averaged
        return averaged

    def apply_averaged_weights(self):
        """Apply averaged weights to the model."""
        if self._averaged_weights is None:
            self.average_weights()

        if self._averaged_weights is None:
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._averaged_weights:
                    param.data.copy_(self._averaged_weights[name].to(param.device))

        logger.info(
            f"SWA: applied averaged weights from {self._num_collected} snapshots"
        )

    def update_batch_norm(self, dataloader: DataLoader, device: torch.device):
        """Update batch normalization statistics with averaged weights.

        Args:
            dataloader: Data loader for BN update.
            device: Device for computation.
        """
        self.model.train()

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.running_mean.zero_()
                module.running_var.zero_()
                module.num_batches_tracked = 0

        momentum_backup = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                momentum_backup[name] = module.momentum
                module.momentum = None

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                elif isinstance(batch, dict):
                    inputs = batch.get("input_ids", batch.get("inputs")).to(device)
                else:
                    inputs = batch.to(device)
                self.model(inputs)

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = momentum_backup.get(name, 0.1)

        self.model.eval()
        logger.info("SWA: updated batch normalization statistics")

    def state_dict(self) -> Dict[str, Any]:
        """Get SWA state."""
        return {
            "current_step": self._current_step,
            "num_collected": self._num_collected,
            "start_step": self.start_step,
            "freq": self.freq,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load SWA state."""
        self._current_step = state.get("current_step", 0)
        self._num_collected = state.get("num_collected", 0)

    def get_info(self) -> Dict[str, Any]:
        """Get SWA information."""
        return {
            "current_step": self._current_step,
            "start_step": self.start_step,
            "freq": self.freq,
            "num_collected": self._num_collected,
            "has_averaged": self._averaged_weights is not None,
        }


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manage training checkpoints.

    Handles saving, loading, and pruning of model checkpoints
    with support for best-model tracking and history.

    Args:
        save_dir: Directory to save checkpoints.
        max_checkpoints: Maximum checkpoints to keep.
        checkpoint_format: Format for saving.
    """

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        max_checkpoints: int = 5,
        checkpoint_format: CheckpointFormat = CheckpointFormat.PYTORCH,
    ):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_format = checkpoint_format
        self._checkpoint_history: List[Dict[str, Any]] = []
        self._best_metric: Optional[float] = None
        self._best_checkpoint: Optional[str] = None
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            model: Model to save.
            optimizer: Optional optimizer.
            step: Training step.
            epoch: Training epoch.
            metrics: Optional metrics.
            is_best: Whether this is the best checkpoint.
            extra_state: Additional state to save.

        Returns:
            Path to saved checkpoint.
        """
        filename = f"checkpoint_step{step}_epoch{epoch}.pt"
        filepath = os.path.join(self.save_dir, filename)

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics or {},
            "format": self.checkpoint_format.name,
        }

        if self.checkpoint_format == CheckpointFormat.PYTORCH:
            checkpoint["model_state_dict"] = model.state_dict()
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if extra_state:
            checkpoint["extra"] = extra_state

        torch.save(checkpoint, filepath)

        self._checkpoint_history.append({
            "path": filepath,
            "step": step,
            "epoch": epoch,
            "metrics": metrics or {},
            "is_best": is_best,
            "timestamp": time.time(),
        })

        if is_best:
            best_path = os.path.join(self.save_dir, "best_checkpoint.pt")
            shutil.copy2(filepath, best_path)
            self._best_checkpoint = best_path
            if metrics:
                for key, value in metrics.items():
                    if self._best_metric is None or value < self._best_metric:
                        self._best_metric = value

        self._prune_checkpoints()
        logger.info(f"Checkpoint saved to {filepath} (best={is_best})")
        return filepath

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optional optimizer.
            checkpoint_path: Specific path (None for latest).
            load_best: Load best checkpoint.
            map_location: Device mapping.

        Returns:
            Checkpoint metadata.
        """
        if load_best:
            checkpoint_path = os.path.join(self.save_dir, "best_checkpoint.pt")
        elif checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model_state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(model_state, strict=False)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "path": checkpoint_path,
        }

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint."""
        checkpoints = [
            entry for entry in self._checkpoint_history
            if os.path.exists(entry["path"])
        ]
        if not checkpoints:
            patterns = ["checkpoint_step*.pt", "*.pt"]
            for pattern in patterns:
                import glob as glob_mod
                files = glob_mod.glob(os.path.join(self.save_dir, pattern))
                files = [f for f in files if "best" not in f]
                if files:
                    return max(files, key=os.path.getmtime)
            return None
        latest = max(checkpoints, key=lambda e: e.get("timestamp", 0))
        return latest["path"]

    def _prune_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self._checkpoint_history) > self.max_checkpoints:
            oldest = min(self._checkpoint_history, key=lambda e: e.get("timestamp", 0))
            path = oldest.get("path", "")
            if path and os.path.exists(path) and "best" not in path:
                try:
                    os.remove(path)
                except OSError:
                    pass
            self._checkpoint_history.remove(oldest)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return list(self._checkpoint_history)

    def get_best_metric(self) -> Optional[float]:
        """Return the best metric value."""
        return self._best_metric


# ---------------------------------------------------------------------------
# Metric Tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Track training metrics over time.

    Maintains a rolling window of metrics and provides statistics,
    history, and export functionality.

    Args:
        window_size: Rolling window size for statistics.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._all_metrics: Dict[str, List[float]] = defaultdict(list)
        self._step_metrics: Dict[str, Dict[str, float]] = {}
        self._current_step = 0

    def update(self, step: int, metrics: Dict[str, float]):
        """Record metrics for a training step.

        Args:
            step: Training step.
            metrics: Dictionary of metric values.
        """
        self._current_step = step
        self._step_metrics[step] = dict(metrics)

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._metrics[name].append(value)
                self._all_metrics[name].append(value)

    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if name in self._metrics and self._metrics[name]:
            return self._metrics[name][-1]
        return None

    def get_average(self, name: str, window: Optional[int] = None) -> float:
        """Get the average of a metric over a window.

        Args:
            name: Metric name.
            window: Window size (None for full history).

        Returns:
            Average value.
        """
        if name not in self._metrics:
            return 0.0

        data = list(self._metrics[name])
        if window:
            data = data[-window:]

        return sum(data) / max(1, len(data))

    def get_smoothed(self, name: str, beta: float = 0.9) -> Optional[float]:
        """Get exponentially smoothed value of a metric.

        Args:
            name: Metric name.
            beta: Smoothing factor.

        Returns:
            Smoothed value.
        """
        data = list(self._metrics[name])
        if not data:
            return None

        smoothed = data[0]
        for value in data[1:]:
            smoothed = beta * smoothed + (1 - beta) * value
        return smoothed

    def get_all_names(self) -> List[str]:
        """Get all tracked metric names."""
        return list(self._metrics.keys())

    def get_history(self, name: str) -> List[float]:
        """Get full history of a metric."""
        return list(self._all_metrics.get(name, []))

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name in self._metrics:
            data = list(self._metrics[name])
            if not data:
                continue
            summary[name] = {
                "latest": data[-1],
                "avg": sum(data) / len(data),
                "min": min(data),
                "max": max(data),
                "std": math.sqrt(sum((x - sum(data) / len(data)) ** 2 for x in data) / len(data))
                if len(data) > 1 else 0,
                "count": len(data),
            }
        return summary

    def reset(self):
        """Reset all tracked metrics."""
        self._metrics.clear()
        self._all_metrics.clear()
        self._step_metrics.clear()
        self._current_step = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state."""
        return {
            "current_step": self._current_step,
            "metrics": {
                name: list(data) for name, data in self._metrics.items()
            },
            "summary": self.get_summary(),
        }

    def export_json(self, path: str):
        """Export metrics to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Training time tracking and estimation.

    Tracks elapsed time, estimates remaining time, and provides
    time-based statistics for training.

    Args:
        total_steps: Expected total training steps.
    """

    def __init__(self, total_steps: int = 100000):
        self.total_steps = total_steps
        self._start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._step_durations: deque = deque(maxlen=100)
        self._current_step = 0
        self._epoch_times: List[Tuple[int, float]] = []
        self._pauses: List[Tuple[float, float]] = []

    def start(self):
        """Start the timer."""
        self._start_time = time.time()
        self._last_step_time = self._start_time
        logger.info("Timer started")

    def tick(self, step: int):
        """Record a training step.

        Args:
            step: Current training step.
        """
        now = time.time()
        if self._last_step_time is not None:
            duration = now - self._last_step_time
            self._step_durations.append(duration)
        self._last_step_time = now
        self._current_step = step

    def record_epoch(self, epoch: int):
        """Record epoch completion time."""
        self._epoch_times.append((epoch, time.time()))

    def pause(self):
        """Pause the timer."""
        self._pauses.append((time.time(), 0.0))

    def resume(self):
        """Resume the timer."""
        if self._pauses:
            pause_start, _ = self._pauses[-1]
            self._pauses[-1] = (pause_start, time.time())

    @property
    def elapsed(self) -> float:
        """Total elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def effective_elapsed(self) -> float:
        """Elapsed time minus pause durations."""
        if self._start_time is None:
            return 0.0
        total_pause = sum(
            end - start for start, end in self._pauses if end > 0
        )
        return self.elapsed - total_pause

    @property
    def avg_step_time(self) -> float:
        """Average step duration in seconds."""
        if not self._step_durations:
            return 0.0
        return sum(self._step_durations) / len(self._step_durations)

    @property
    def steps_per_second(self) -> float:
        """Average steps per second."""
        avg = self.avg_step_time
        return 1.0 / avg if avg > 0 else 0.0

    def estimate_remaining(self) -> float:
        """Estimate remaining training time in seconds."""
        remaining_steps = max(0, self.total_steps - self._current_step)
        return remaining_steps * self.avg_step_time

    def format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 0:
            return "N/A"
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)}h {int(minutes)}m"
        else:
            days = seconds / 86400
            hours = (seconds % 86400) / 3600
            return f"{int(days)}d {int(hours)}h"

    def get_eta(self) -> str:
        """Get estimated time of completion as formatted string."""
        return self.format_time(self.estimate_remaining())

    def get_progress_string(self) -> str:
        """Get a human-readable progress string."""
        elapsed_str = self.format_time(self.elapsed)
        eta_str = self.get_eta()
        progress = self._current_step / max(1, self.total_steps)
        sps = self.steps_per_second
        return (
            f"[{progress:5.1%}] Step {self._current_step}/{self.total_steps} | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
            f"{sps:.1f} steps/s"
        )

    def get_report(self) -> Dict[str, Any]:
        """Get a detailed time report."""
        return {
            "elapsed_seconds": self.elapsed,
            "effective_elapsed": self.effective_elapsed,
            "avg_step_time": self.avg_step_time,
            "steps_per_second": self.steps_per_second,
            "estimated_remaining": self.estimate_remaining(),
            "current_step": self._current_step,
            "total_steps": self.total_steps,
            "progress": self._current_step / max(1, self.total_steps),
            "num_epochs": len(self._epoch_times),
            "num_pauses": len(self._pauses),
        }


# ---------------------------------------------------------------------------
# Device Manager
# ---------------------------------------------------------------------------

class DeviceManager:
    """GPU/CPU device management.

    Provides utilities for device selection, memory management,
    and multi-GPU setup.

    Args:
        preferred_device: Preferred device string.
        memory_fraction: Fraction of GPU memory to use.
    """

    def __init__(
        self,
        preferred_device: Optional[str] = None,
        memory_fraction: float = 0.9,
    ):
        self.memory_fraction = memory_fraction
        self._device = self._select_device(preferred_device)
        self._memory_allocated = 0.0
        self._memory_reserved = 0.0

    def _select_device(self, preferred: Optional[str]) -> torch.device:
        """Select the best available device."""
        if preferred:
            return torch.device(preferred)

        if torch.cuda.is_available():
            best_gpu = self._select_best_gpu()
            return torch.device(f"cuda:{best_gpu}")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _select_best_gpu(self) -> int:
        """Select the GPU with the most free memory."""
        if not torch.cuda.device_count():
            return 0

        best_idx = 0
        best_free = 0
        for i in range(torch.cuda.device_count()):
            free = torch.cuda.get_device_properties(i).total_memory
            if free > best_free:
                best_free = free
                best_idx = i
        return best_idx

    @property
    def device(self) -> torch.device:
        """Return the selected device."""
        return self._device

    def to_device(self, data: Any) -> Any:
        """Move data to the selected device."""
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item) for item in data)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, nn.Module):
            return data.to(self._device)
        return data

    def get_memory_info(self) -> Dict[str, float]:
        """Get memory usage information."""
        info = {"device": str(self._device)}

        if self._device.type == "cuda":
            info["allocated_gb"] = torch.cuda.memory_allocated(self._device) / (1024 ** 3)
            info["reserved_gb"] = torch.cuda.memory_reserved(self._device) / (1024 ** 3)
            info["max_allocated_gb"] = torch.cuda.max_memory_allocated(self._device) / (1024 ** 3)
            props = torch.cuda.get_device_properties(self._device)
            info["total_gb"] = props.total_memory / (1024 ** 3)

        return info

    def clear_cache(self):
        """Clear GPU memory cache."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._device)

    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self._device.type == "cuda"

    def get_num_gpus(self) -> int:
        """Get number of available GPUs."""
        return torch.cuda.device_count()


# ---------------------------------------------------------------------------
# Seed Manager
# ---------------------------------------------------------------------------

class SeedManager:
    """Reproducible seeding across all random generators.

    Ensures reproducibility by setting seeds for Python random,
    NumPy random, PyTorch CPU, and PyTorch CUDA.

    Args:
        seed: Base random seed.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._worker_seeds: List[int] = []

    def set_seed(self, seed: Optional[int] = None):
        """Set seeds across all random generators.

        Args:
            seed: Seed to use (None for base seed).
        """
        s = seed if seed is not None else self.seed
        random.seed(s)
        try:
            import numpy as np
            np.random.seed(s)
        except ImportError:
            pass
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        logger.debug(f"Seeds set to {s}")

    def worker_init_fn(self, worker_id: int) -> Callable[[int], None]:
        """Get a worker initialization function for DataLoader.

        Args:
            worker_id: Base worker ID.

        Returns:
            Function to initialize workers with different seeds.
        """
        def init_fn(worker_id_: int):
            worker_seed = self.seed + worker_id_
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(worker_seed)
        return init_fn

    def get_state(self) -> Dict[str, Any]:
        """Get current random state."""
        return {
            "seed": self.seed,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.random.get_rng_state(),
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore random state."""
        self.seed = state.get("seed", self.seed)
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "torch_rng_state" in state:
            torch.random.set_rng_state(state["torch_rng_state"])


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: Only count trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters per layer.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping layer name to parameter count.
    """
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.numel()
    return params


def get_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """Get memory footprint of a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with memory statistics in MB.
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable = sum(
        p.numel() * p.element_size() for p in model.parameters()
        if p.requires_grad
    )
    buffers = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )

    return {
        "total_mb": total / (1024 * 1024),
        "trainable_mb": trainable / (1024 * 1024),
        "buffers_mb": buffers / (1024 * 1024),
        "total_params": total // 4 if total > 0 else 0,
        "trainable_params": trainable // 4 if trainable > 0 else 0,
    }


def create_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create an optimizer with common settings.

    Args:
        model: Model to optimize.
        name: Optimizer name.
        lr: Learning rate.
        weight_decay: Weight decay.
        **kwargs: Additional optimizer arguments.

    Returns:
        Configured optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, **kwargs)
    elif name == "adafactor":
        try:
            from transformers import Adafactor
            return Adafactor(params, lr=lr, **kwargs)
        except ImportError:
            logger.warning("Adafactor not available, falling back to AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    total_steps: int = 100000,
    warmup_steps: int = 1000,
    **kwargs,
) -> Any:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer.
        name: Scheduler name.
        total_steps: Total training steps.
        warmup_steps: Warmup steps.
        **kwargs: Additional scheduler arguments.

    Returns:
        Learning rate scheduler.
    """
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, **kwargs
        )
    elif name == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif name == "cosine_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif name == "warmup_decay":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=kwargs.get("max_lr", 1e-3),
            total_steps=total_steps, pct_start=warmup_steps / max(1, total_steps),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def freeze_model(model: nn.Module, freeze_bn: bool = False):
    """Freeze all model parameters.

    Args:
        model: Model to freeze.
        freeze_bn: Whether to also freeze batch norm layers.
    """
    for param in model.parameters():
        param.requires_grad = False

    if freeze_bn:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    logger.info(f"Froze all parameters ({count_parameters(model, trainable_only=True)} trainable remaining)")


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters.

    Args:
        model: Model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info(f"Unfroze all parameters ({count_parameters(model)} total)")


def get_training_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Dict[str, float],
    device: Optional[torch.device] = None,
) -> str:
    """Get a formatted training summary string.

    Args:
        model: The model.
        optimizer: The optimizer.
        step: Current step.
        metrics: Current metrics.
        device: Device for memory info.

    Returns:
        Formatted summary string.
    """
    lr = optimizer.param_groups[0]["lr"]
    parts = [f"Step {step}"]

    parts.append(f"LR: {lr:.2e}")

    for name, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{name}: {value:.4f}")
        else:
            parts.append(f"{name}: {value}")

    if device and device.type == "cuda":
        mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        parts.append(f"GPU: {mem_gb:.1f}GB")

    return " | ".join(parts)


def export_training_config(
    config: Dict[str, Any],
    path: str,
):
    """Export training configuration to JSON.

    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    serializable = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            serializable[key] = str(value)
        elif isinstance(value, (type(None), bool, int, float, str)):
            serializable[key] = value
        elif isinstance(value, (list, tuple, dict)):
            serializable[key] = value
        else:
            serializable[key] = str(value)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Training config exported to {path}")
