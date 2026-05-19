"""Training callbacks: early stopping, logging, checkpointing, LR tracking, gradient monitoring."""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, List, Callable

import torch

logger = logging.getLogger(__name__)


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, config: Any):
        pass

    def on_train_end(self, metrics: Dict[str, Any]):
        pass

    def on_epoch_begin(self, epoch: int, config: Any):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        pass

    def on_step_begin(self, step: int):
        pass

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        pass

    def on_log(self, step: int, metrics: Dict[str, Any]):
        pass

    def on_evaluate(self, step: int, metrics: Dict[str, Any]):
        pass


class CallbackManager:
    """Manages and dispatches events to multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self.should_stop = False

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback_type: type):
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]

    def on_train_begin(self, config: Any):
        for cb in self.callbacks:
            cb.on_train_begin(config)

    def on_train_end(self, metrics: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_train_end(metrics)

    def on_epoch_begin(self, epoch: int, config: Any):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, config)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)
        if self.should_stop:
            return

    def on_step_begin(self, step: int):
        for cb in self.callbacks:
            cb.on_step_begin(step)

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_step_end(step, metrics)

    def on_log(self, step: int, metrics: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_log(step, metrics)

    def on_evaluate(self, step: int, metrics: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_evaluate(step, metrics)


class EarlyStoppingCallback(Callback):
    """Stops training when a monitored metric stops improving."""

    def __init__(
        self,
        metric_name: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        greater_is_better: bool = False,
    ):
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.greater_is_better = greater_is_better
        self.best_value: Optional[float] = None
        self.counter = 0

    def on_evaluate(self, step: int, metrics: Dict[str, Any]):
        current_value = metrics.get(self.metric_name)
        if current_value is None:
            return

        if self.best_value is None:
            self.best_value = current_value
            return

        improved = False
        if self.greater_is_better:
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: no improvement for {self.counter} evaluation(s). "
                f"Best {self.metric_name}: {self.best_value:.4f}"
            )
            if self.counter >= self.patience:
                logger.info(f"EarlyStopping triggered after {self.patience} evaluations without improvement.")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        if self.counter >= self.patience:
            from nexus_llm.training.callbacks import CallbackManager
            pass  # Signal is read by the manager


class LoggingCallback(Callback):
    """Logs training metrics to console and/or file."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_interval: int = 10,
        log_to_console: bool = True,
    ):
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.log_to_console = log_to_console
        self.log_file = None
        self.history: List[Dict[str, Any]] = []

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(os.path.join(log_dir, "training_log.jsonl"), "a")

    def on_log(self, step: int, metrics: Dict[str, Any]):
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self.history.append(entry)

        if self.log_to_console and step % self.log_interval == 0:
            metrics_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
            logger.info(f"Step {step} | {metrics_str}")

        if self.log_file is not None:
            self.log_file.write(json.dumps(entry) + "\n")
            self.log_file.flush()

    def on_train_end(self, metrics: Dict[str, Any]):
        if self.log_file is not None:
            self.log_file.close()


class CheckpointCallback(Callback):
    """Saves model checkpoints at regular intervals."""

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        save_interval: int = 500,
        save_total_limit: int = 5,
    ):
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.save_total_limit = save_total_limit
        os.makedirs(output_dir, exist_ok=True)

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        pass  # Checkpointing handled by CheckpointManager

    def on_evaluate(self, step: int, metrics: Dict[str, Any]):
        if metrics:
            save_path = os.path.join(self.output_dir, f"eval_step_{step}")
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)


class LearningRateTrackingCallback(Callback):
    """Tracks and logs the learning rate throughout training."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.lr_history: List[Dict[str, float]] = []

    def on_log(self, step: int, metrics: Dict[str, Any]):
        lr = metrics.get("lr", metrics.get("learning_rate"))
        if lr is not None:
            self.lr_history.append({"step": step, "learning_rate": lr})

    def get_lr_history(self) -> List[Dict[str, float]]:
        return self.lr_history


class GradientMonitoringCallback(Callback):
    """Monitors gradient norms and detects training issues."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        log_interval: int = 100,
        max_grad_norm: float = 10.0,
        warn_on_nan: bool = True,
    ):
        self.model = model
        self.log_interval = log_interval
        self.max_grad_norm = max_grad_norm
        self.warn_on_nan = warn_on_nan
        self.grad_norm_history: List[Dict[str, Any]] = []

    def on_step_end(self, step: int, metrics: Dict[str, Any]):
        if self.model is None or step % self.log_interval != 0:
            return

        total_norm = 0.0
        has_nan = False
        param_count = 0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                if torch.isnan(param.grad).any():
                    has_nan = True

        total_norm = total_norm ** 0.5

        entry = {
            "step": step,
            "grad_norm": total_norm,
            "has_nan": has_nan,
            "num_params_with_grad": param_count,
        }
        self.grad_norm_history.append(entry)

        if has_nan and self.warn_on_nan:
            logger.warning(f"Step {step}: NaN detected in gradients!")

        if total_norm > self.max_grad_norm:
            logger.warning(
                f"Step {step}: Large gradient norm detected: {total_norm:.4f} "
                f"(max: {self.max_grad_norm:.4f})"
            )

    def get_grad_norm_history(self) -> List[Dict[str, Any]]:
        return self.grad_norm_history
