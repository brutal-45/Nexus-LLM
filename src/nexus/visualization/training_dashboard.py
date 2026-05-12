"""
Training Dashboard - Training Visualization and Monitoring
==========================================================

Comprehensive tools for monitoring LLM training including loss tracking,
throughput monitoring, gradient analysis, and text-based dashboards.

All visualizations are text-based using Unicode characters - no matplotlib/pandas.
"""

import time
import math
import json
import os
import threading
import hashlib
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Sequence, Iterator
)
from enum import Enum


# ============================================================================
# Constants and Unicode Drawing Characters
# ============================================================================

# Box-drawing characters
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_H = "─"
BOX_V = "│"
BOX_LT = "├"
BOX_RT = "┤"
BOX_BT = "┬"
BOX_BB = "┴"
BOX_CROSS = "┼"

# Progress bar characters
PROGRESS_FULL = "█"
PROGRESS_THREE_QUARTERS = "▓"
PROGRESS_HALF = "▒"
PROGRESS_QUARTER = "░"
PROGRESS_EMPTY = " "

# Block characters for heatmaps
BLOCK_FULL = "█"
BLOCK_DARK = "▓"
BLOCK_MEDIUM = "▒"
BLOCK_LIGHT = "░"

# Chart characters
CHART_LINE = "─"
CHART_UP = "╱"
CHART_DOWN = "╲"
CHART_PEAK = "⌃"
CHART_VALLEY = "⌄"
CHART_DOT = "•"
CHART_STAR = "★"

# Status indicators
STATUS_OK = "✓"
STATUS_WARN = "⚠"
STATUS_ERROR = "✗"
STATUS_INFO = "ℹ"
STATUS_RUNNING = "▶"
STATUS_PAUSED = "⏸"
STATUS_DONE = "✔"

# Arrow indicators
ARROW_UP = "↑"
ARROW_DOWN = "↓"
ARROW_RIGHT = "→"
ARROW_LEFT = "←"
ARROW_UP_RIGHT = "↗"
ARROW_DOWN_RIGHT = "↘"


class MetricUnit(Enum):
    """Units for training metrics."""
    NONE = "none"
    LOSS = "loss"
    PERCENTAGE = "percentage"
    TOKENS_PER_SEC = "tokens/sec"
    SAMPLES_PER_SEC = "samples/sec"
    MILLISECONDS = "ms"
    SECONDS = "s"
    BYTES = "bytes"
    GB = "GB"
    MB = "MB"
    STEPS = "steps"


@dataclass
class StepRecord:
    """Record of a single training step."""
    step: int
    timestamp: float
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    samples_per_sec: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    throughput_tokens: Optional[int] = None
    throughput_samples: Optional[int] = None
    batch_size: Optional[int] = None
    sequence_length: Optional[int] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "loss": self.loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "tokens_per_sec": self.tokens_per_sec,
            "samples_per_sec": self.samples_per_sec,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total,
            "gpu_utilization": self.gpu_utilization,
            "throughput_tokens": self.throughput_tokens,
            "throughput_samples": self.throughput_samples,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "custom_metrics": self.custom_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepRecord":
        """Create from dictionary."""
        custom = data.pop("custom_metrics", {})
        record = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        record.custom_metrics = custom
        return record


@dataclass
class EpochRecord:
    """Record of a completed training epoch."""
    epoch: int
    start_step: int
    end_step: int
    start_time: float
    end_time: float
    avg_loss: float = 0.0
    avg_val_loss: float = 0.0
    avg_lr: float = 0.0
    avg_grad_norm: float = 0.0
    total_tokens: int = 0
    total_samples: int = 0
    avg_tokens_per_sec: float = 0.0
    num_steps: int = 0
    best_loss: float = float("inf")
    worst_loss: float = float("-inf")
    loss_std: float = 0.0

    @property
    def duration(self) -> float:
        """Duration of the epoch in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "avg_loss": self.avg_loss,
            "avg_val_loss": self.avg_val_loss,
            "avg_lr": self.avg_lr,
            "avg_grad_norm": self.avg_grad_norm,
            "total_tokens": self.total_tokens,
            "total_samples": self.total_samples,
            "avg_tokens_per_sec": self.avg_tokens_per_sec,
            "num_steps": self.num_steps,
            "best_loss": self.best_loss,
            "worst_loss": self.worst_loss,
            "loss_std": self.loss_std,
        }


@dataclass
class TrainingStatistics:
    """Computed statistics from training history."""
    total_steps: int = 0
    total_epochs: int = 0
    total_time: float = 0.0
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    best_loss_step: Optional[int] = None
    worst_loss: Optional[float] = None
    worst_loss_step: Optional[int] = None
    avg_loss: float = 0.0
    loss_std: float = 0.0
    loss_min: float = float("inf")
    loss_max: float = float("-inf")
    avg_lr: float = 0.0
    avg_grad_norm: float = 0.0
    avg_tokens_per_sec: float = 0.0
    avg_samples_per_sec: float = 0.0
    avg_gpu_memory: float = 0.0
    peak_gpu_memory: float = 0.0
    total_tokens_processed: int = 0
    loss_reduction_pct: float = 0.0
    convergence_step: Optional[int] = None
    loss_variance: float = 0.0
    loss_trend: str = "stable"


# ============================================================================
# TrainingTracker
# ============================================================================

class TrainingTracker:
    """
    Track all training metrics per step and epoch.

    Records loss, learning rate, throughput, GPU memory, gradient norms,
    and custom metrics. Provides history retrieval and statistical computation.

    Example:
        tracker = TrainingTracker(max_history=10000)
        tracker.log_step(
            step=1,
            loss=2.5,
            learning_rate=1e-4,
            grad_norm=1.2,
            gpu_memory_used=8.5,
            gpu_memory_total=24.0,
        )
        tracker.log_epoch(epoch=1)
        stats = tracker.compute_statistics()
        print(stats.total_steps, stats.best_loss)
    """

    def __init__(
        self,
        max_history: int = 100000,
        checkpoint_interval: int = 1000,
        autosave: bool = False,
        save_path: Optional[str] = None,
    ):
        """Initialize the training tracker.

        Args:
            max_history: Maximum number of step records to keep in memory.
            checkpoint_interval: Steps between automatic checkpoint saves.
            autosave: Whether to automatically save history to disk.
            save_path: Path for saving/loading history files.
        """
        self._max_history = max_history
        self._checkpoint_interval = checkpoint_interval
        self._autosave = autosave
        self._save_path = save_path or os.path.join(
            os.getcwd(), "training_history.json"
        )
        self._lock = threading.RLock()

        # Step-level history
        self._step_records: List[StepRecord] = []
        self._step_index: Dict[int, int] = {}

        # Epoch-level history
        self._epoch_records: List[EpochRecord] = []
        self._epoch_index: Dict[int, int] = {}

        # Current epoch tracking
        self._current_epoch: int = 0
        self._epoch_start_step: int = 0
        self._epoch_start_time: float = time.time()
        self._epoch_step_losses: List[float] = []
        self._epoch_step_val_losses: List[float] = []
        self._epoch_step_lrs: List[float] = []
        self._epoch_step_grad_norms: List[float] = []
        self._epoch_tokens: int = 0
        self._epoch_samples: int = 0

        # Running statistics
        self._running_loss_ema: float = 0.0
        self._ema_alpha: float = 0.01
        self._best_loss: float = float("inf")
        self._best_loss_step: int = -1
        self._worst_loss: float = float("-inf")
        self._worst_loss_step: int = -1
        self._total_tokens: int = 0
        self._total_samples: int = 0

        # Custom metric accumulators
        self._custom_metric_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

        # Event callbacks
        self._on_step_callbacks: List[Callable[[StepRecord], None]] = []
        self._on_epoch_callbacks: List[Callable[[EpochRecord], None]] = []

    def log_step(
        self,
        step: int,
        loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
        gpu_memory_used: Optional[float] = None,
        gpu_memory_total: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        throughput_tokens: Optional[int] = None,
        throughput_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
    ) -> StepRecord:
        """Log a training step.

        Args:
            step: Current step number.
            loss: Training loss value.
            val_loss: Validation loss value.
            learning_rate: Current learning rate.
            grad_norm: Gradient norm of the step.
            tokens_per_sec: Tokens processed per second.
            samples_per_sec: Samples processed per second.
            gpu_memory_used: GPU memory used in GB.
            gpu_memory_total: Total GPU memory in GB.
            gpu_utilization: GPU utilization percentage (0-100).
            throughput_tokens: Number of tokens processed this step.
            throughput_samples: Number of samples processed this step.
            batch_size: Batch size used.
            sequence_length: Sequence length used.
            custom_metrics: Additional custom metrics.
            timestamp: Override timestamp (defaults to now).

        Returns:
            The created StepRecord.
        """
        with self._lock:
            ts = timestamp if timestamp is not None else time.time()
            record = StepRecord(
                step=step,
                timestamp=ts,
                loss=loss,
                val_loss=val_loss,
                learning_rate=learning_rate,
                grad_norm=grad_norm,
                tokens_per_sec=tokens_per_sec,
                samples_per_sec=samples_per_sec,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_utilization=gpu_utilization,
                throughput_tokens=throughput_tokens,
                throughput_samples=throughput_samples,
                batch_size=batch_size,
                sequence_length=sequence_length,
                custom_metrics=custom_metrics or {},
            )

            # Trim history if needed
            if len(self._step_records) >= self._max_history:
                removed = self._step_records[: len(self._step_records) - self._max_history + 1]
                self._step_records = self._step_records[-self._max_history + 1:]
                for r in removed:
                    self._step_index.pop(r.step, None)

            self._step_records.append(record)
            self._step_index[step] = len(self._step_records) - 1

            # Update running statistics
            if loss is not None:
                self._running_loss_ema = (
                    self._ema_alpha * loss + (1 - self._ema_alpha) * self._running_loss_ema
                )
                if loss < self._best_loss:
                    self._best_loss = loss
                    self._best_loss_step = step
                if loss > self._worst_loss:
                    self._worst_loss = loss
                    self._worst_loss_step = step
                self._epoch_step_losses.append(loss)

            if val_loss is not None:
                self._epoch_step_val_losses.append(val_loss)

            if learning_rate is not None:
                self._epoch_step_lrs.append(learning_rate)

            if grad_norm is not None:
                self._epoch_step_grad_norms.append(grad_norm)

            if throughput_tokens is not None:
                self._epoch_tokens += throughput_tokens
                self._total_tokens += throughput_tokens

            if throughput_samples is not None:
                self._epoch_samples += throughput_samples
                self._total_samples += throughput_samples

            # Store custom metrics
            if custom_metrics:
                for key, value in custom_metrics.items():
                    self._custom_metric_history[key].append((step, value))

            # Fire callbacks
            for callback in self._on_step_callbacks:
                try:
                    callback(record)
                except Exception:
                    pass

            # Auto-checkpoint
            if (
                self._autosave
                and step > 0
                and step % self._checkpoint_interval == 0
            ):
                self.save(self._save_path)

            return record

    def log_epoch(self, epoch: int) -> EpochRecord:
        """Finalize and log a completed epoch.

        Args:
            epoch: The epoch number that just completed.

        Returns:
            The created EpochRecord.
        """
        with self._lock:
            now = time.time()
            start_step = self._epoch_start_step
            end_step = self._step_records[-1].step if self._step_records else start_step

            # Compute epoch statistics
            avg_loss = 0.0
            best_loss = float("inf")
            worst_loss = float("-inf")
            loss_std = 0.0

            if self._epoch_step_losses:
                avg_loss = sum(self._epoch_step_losses) / len(self._epoch_step_losses)
                best_loss = min(self._epoch_step_losses)
                worst_loss = max(self._epoch_step_losses)
                variance = sum(
                    (x - avg_loss) ** 2 for x in self._epoch_step_losses
                ) / len(self._epoch_step_losses)
                loss_std = math.sqrt(variance)

            avg_val_loss = 0.0
            if self._epoch_step_val_losses:
                avg_val_loss = sum(self._epoch_step_val_losses) / len(
                    self._epoch_step_val_losses
                )

            avg_lr = 0.0
            if self._epoch_step_lrs:
                avg_lr = sum(self._epoch_step_lrs) / len(self._epoch_step_lrs)

            avg_grad_norm = 0.0
            if self._epoch_step_grad_norms:
                avg_grad_norm = sum(self._epoch_step_grad_norms) / len(
                    self._epoch_step_grad_norms
                )

            # Compute average throughput for the epoch
            avg_tps = 0.0
            epoch_tps_values = []
            for r in self._step_records:
                if r.step >= start_step and r.tokens_per_sec is not None:
                    epoch_tps_values.append(r.tokens_per_sec)
            if epoch_tps_values:
                avg_tps = sum(epoch_tps_values) / len(epoch_tps_values)

            record = EpochRecord(
                epoch=epoch,
                start_step=start_step,
                end_step=end_step,
                start_time=self._epoch_start_time,
                end_time=now,
                avg_loss=avg_loss,
                avg_val_loss=avg_val_loss,
                avg_lr=avg_lr,
                avg_grad_norm=avg_grad_norm,
                total_tokens=self._epoch_tokens,
                total_samples=self._epoch_samples,
                avg_tokens_per_sec=avg_tps,
                num_steps=len(self._epoch_step_losses),
                best_loss=best_loss,
                worst_loss=worst_loss,
                loss_std=loss_std,
            )

            self._epoch_records.append(record)
            self._epoch_index[epoch] = len(self._epoch_records) - 1

            # Reset epoch accumulators
            self._current_epoch = epoch
            self._epoch_start_step = end_step + 1
            self._epoch_start_time = now
            self._epoch_step_losses = []
            self._epoch_step_val_losses = []
            self._epoch_step_lrs = []
            self._epoch_step_grad_norms = []
            self._epoch_tokens = 0
            self._epoch_samples = 0

            # Fire callbacks
            for callback in self._on_epoch_callbacks:
                try:
                    callback(record)
                except Exception:
                    pass

            return record

    def get_history(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        metrics: Optional[List[str]] = None,
    ) -> List[StepRecord]:
        """Get training step history.

        Args:
            start_step: Starting step (inclusive). None for first.
            end_step: Ending step (inclusive). None for last.
            metrics: Filter to specific metric names. None for all.

        Returns:
            List of StepRecord objects.
        """
        with self._lock:
            records = self._step_records
            if start_step is not None:
                idx = self._step_index.get(start_step, 0)
                records = records[idx:]
            if end_step is not None:
                idx = self._step_index.get(end_step, len(records) - 1)
                records = records[: idx + 1]
            if metrics is not None:
                filtered = []
                for r in records:
                    d = r.to_dict()
                    new_r = StepRecord(step=r.step, timestamp=r.timestamp)
                    for m in metrics:
                        if m in d and d[m] is not None:
                            setattr(new_r, m, d[m])
                    filtered.append(new_r)
                records = filtered
            return list(records)

    def get_epoch_history(
        self,
        start_epoch: Optional[int] = None,
        end_epoch: Optional[int] = None,
    ) -> List[EpochRecord]:
        """Get training epoch history.

        Args:
            start_epoch: Starting epoch (inclusive). None for first.
            end_epoch: Ending epoch (inclusive). None for last.

        Returns:
            List of EpochRecord objects.
        """
        with self._lock:
            records = self._epoch_records
            if start_epoch is not None:
                idx = self._epoch_index.get(start_epoch, 0)
                records = records[idx:]
            if end_epoch is not None:
                idx = self._epoch_index.get(end_epoch, len(records) - 1)
                records = records[: idx + 1]
            return list(records)

    def compute_statistics(self) -> TrainingStatistics:
        """Compute summary statistics from the training history.

        Returns:
            TrainingStatistics with all computed values.
        """
        with self._lock:
            stats = TrainingStatistics()
            if not self._step_records:
                return stats

            stats.total_steps = len(self._step_records)
            stats.total_epochs = len(self._epoch_records)
            stats.total_tokens_processed = self._total_tokens

            # Time range
            first_ts = self._step_records[0].timestamp
            last_ts = self._step_records[-1].timestamp
            stats.total_time = last_ts - first_ts

            # Loss statistics
            losses = [r.loss for r in self._step_records if r.loss is not None]
            if losses:
                stats.final_loss = losses[-1]
                stats.best_loss = min(losses)
                stats.worst_loss = max(losses)
                stats.avg_loss = sum(losses) / len(losses)
                stats.loss_min = min(losses)
                stats.loss_max = max(losses)
                variance = sum((x - stats.avg_loss) ** 2 for x in losses) / len(losses)
                stats.loss_std = math.sqrt(variance)
                stats.loss_variance = variance

                # Find best/worst steps
                best_val = min(losses)
                worst_val = max(losses)
                for r in self._step_records:
                    if r.loss == best_val:
                        stats.best_loss_step = r.step
                        break
                for r in reversed(self._step_records):
                    if r.loss == worst_val:
                        stats.worst_loss_step = r.step
                        break

                # Loss reduction percentage
                if losses[0] > 0:
                    stats.loss_reduction_pct = (
                        (losses[0] - losses[-1]) / losses[0]
                    ) * 100.0

                # Determine trend
                if len(losses) >= 20:
                    recent = losses[-20:]
                    earlier = losses[:20] if len(losses) >= 40 else losses[: len(losses) // 2]
                    recent_avg = sum(recent) / len(recent)
                    earlier_avg = sum(earlier) / len(earlier)
                    diff = earlier_avg - recent_avg
                    if diff > 0.05 * earlier_avg:
                        stats.loss_trend = "decreasing"
                    elif diff < -0.05 * earlier_avg:
                        stats.loss_trend = "increasing"
                    else:
                        stats.loss_trend = "stable"

                # Detect convergence step (loss within 5% of final for 100 steps)
                target = losses[-1]
                threshold = target * 1.05
                for i in range(len(losses) - 1, -1, -1):
                    if losses[i] > threshold:
                        stats.convergence_step = (
                            self._step_records[i + 1].step if i + 1 < len(losses) else 0
                        )
                        break

            # Learning rate statistics
            lrs = [r.learning_rate for r in self._step_records if r.learning_rate is not None]
            if lrs:
                stats.avg_lr = sum(lrs) / len(lrs)

            # Gradient norm statistics
            grad_norms = [
                r.grad_norm for r in self._step_records if r.grad_norm is not None
            ]
            if grad_norms:
                stats.avg_grad_norm = sum(grad_norms) / len(grad_norms)

            # Throughput statistics
            tps_values = [
                r.tokens_per_sec for r in self._step_records if r.tokens_per_sec is not None
            ]
            if tps_values:
                stats.avg_tokens_per_sec = sum(tps_values) / len(tps_values)

            sps_values = [
                r.samples_per_sec
                for r in self._step_records
                if r.samples_per_sec is not None
            ]
            if sps_values:
                stats.avg_samples_per_sec = sum(sps_values) / len(sps_values)

            # GPU memory statistics
            gpu_values = [
                r.gpu_memory_used
                for r in self._step_records
                if r.gpu_memory_used is not None
            ]
            if gpu_values:
                stats.avg_gpu_memory = sum(gpu_values) / len(gpu_values)
                stats.peak_gpu_memory = max(gpu_values)

            return stats

    def get_latest(self) -> Optional[StepRecord]:
        """Get the most recent step record.

        Returns:
            The latest StepRecord or None if no steps logged.
        """
        with self._lock:
            return self._step_records[-1] if self._step_records else None

    def get_losses(self) -> Tuple[List[int], List[float]]:
        """Get all recorded loss values.

        Returns:
            Tuple of (steps, losses).
        """
        with self._lock:
            steps = []
            losses = []
            for r in self._step_records:
                if r.loss is not None:
                    steps.append(r.step)
                    losses.append(r.loss)
            return steps, losses

    def get_val_losses(self) -> Tuple[List[int], List[float]]:
        """Get all recorded validation loss values.

        Returns:
            Tuple of (steps, val_losses).
        """
        with self._lock:
            steps = []
            losses = []
            for r in self._step_records:
                if r.val_loss is not None:
                    steps.append(r.step)
                    losses.append(r.val_loss)
            return steps, losses

    def get_lr_history(self) -> Tuple[List[int], List[float]]:
        """Get learning rate history.

        Returns:
            Tuple of (steps, learning_rates).
        """
        with self._lock:
            steps = []
            lrs = []
            for r in self._step_records:
                if r.learning_rate is not None:
                    steps.append(r.step)
                    lrs.append(r.learning_rate)
            return steps, lrs

    def get_grad_norm_history(self) -> Tuple[List[int], List[float]]:
        """Get gradient norm history.

        Returns:
            Tuple of (steps, grad_norms).
        """
        with self._lock:
            steps = []
            norms = []
            for r in self._step_records:
                if r.grad_norm is not None:
                    steps.append(r.step)
                    norms.append(r.grad_norm)
            return steps, norms

    def get_throughput_history(self) -> Tuple[List[int], List[float]]:
        """Get tokens/sec history.

        Returns:
            Tuple of (steps, tokens_per_sec).
        """
        with self._lock:
            steps = []
            tps = []
            for r in self._step_records:
                if r.tokens_per_sec is not None:
                    steps.append(r.step)
                    tps.append(r.tokens_per_sec)
            return steps, tps

    def get_custom_metric(self, name: str) -> List[Tuple[int, float]]:
        """Get custom metric history.

        Args:
            name: Name of the custom metric.

        Returns:
            List of (step, value) tuples.
        """
        with self._lock:
            return list(self._custom_metric_history.get(name, []))

    def get_ema_loss(self) -> float:
        """Get the exponential moving average of the loss.

        Returns:
            Current EMA loss value.
        """
        return self._running_loss_ema

    def on_step(self, callback: Callable[[StepRecord], None]) -> None:
        """Register a callback to be called after each step is logged.

        Args:
            callback: Function receiving a StepRecord.
        """
        self._on_step_callbacks.append(callback)

    def on_epoch(self, callback: Callable[[EpochRecord], None]) -> None:
        """Register a callback to be called after each epoch is logged.

        Args:
            callback: Function receiving an EpochRecord.
        """
        self._on_epoch_callbacks.append(callback)

    def save(self, path: Optional[str] = None) -> None:
        """Save training history to a JSON file.

        Args:
            path: File path. Defaults to self._save_path.
        """
        path = path or self._save_path
        with self._lock:
            data = {
                "step_records": [r.to_dict() for r in self._step_records],
                "epoch_records": [r.to_dict() for r in self._epoch_records],
                "metadata": {
                    "total_tokens": self._total_tokens,
                    "total_samples": self._total_samples,
                    "best_loss": self._best_loss,
                    "best_loss_step": self._best_loss_step,
                },
            }
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """Load training history from a JSON file.

        Args:
            path: File path. Defaults to self._save_path.
        """
        path = path or self._save_path
        with self._lock:
            if not os.path.exists(path):
                return
            with open(path, "r") as f:
                data = json.load(f)

            self._step_records = []
            self._step_index = {}
            for d in data.get("step_records", []):
                record = StepRecord.from_dict(d)
                self._step_records.append(record)
                self._step_index[record.step] = len(self._step_records) - 1

            self._epoch_records = []
            self._epoch_index = {}
            for d in data.get("epoch_records", []):
                record = EpochRecord(**{k: v for k, v in d.items() if k in EpochRecord.__dataclass_fields__})
                self._epoch_records.append(record)
                self._epoch_index[record.epoch] = len(self._epoch_records) - 1

            meta = data.get("metadata", {})
            self._total_tokens = meta.get("total_tokens", 0)
            self._total_samples = meta.get("total_samples", 0)
            self._best_loss = meta.get("best_loss", float("inf"))
            self._best_loss_step = meta.get("best_loss_step", -1)

    def clear(self) -> None:
        """Clear all history and reset the tracker."""
        with self._lock:
            self._step_records = []
            self._step_index = {}
            self._epoch_records = []
            self._epoch_index = {}
            self._current_epoch = 0
            self._epoch_start_step = 0
            self._epoch_start_time = time.time()
            self._epoch_step_losses = []
            self._epoch_step_val_losses = []
            self._epoch_step_lrs = []
            self._epoch_step_grad_norms = []
            self._epoch_tokens = 0
            self._epoch_samples = 0
            self._running_loss_ema = 0.0
            self._best_loss = float("inf")
            self._best_loss_step = -1
            self._worst_loss = float("-inf")
            self._worst_loss_step = -1
            self._total_tokens = 0
            self._total_samples = 0
            self._custom_metric_history = defaultdict(list)

    def export_csv(self, path: str) -> None:
        """Export step history to CSV format.

        Args:
            path: Output file path.
        """
        with self._lock:
            if not self._step_records:
                return
            headers = list(self._step_records[0].to_dict().keys())
            lines = [",".join(headers)]
            for record in self._step_records:
                d = record.to_dict()
                row = []
                for h in headers:
                    val = d.get(h, "")
                    if isinstance(val, dict):
                        val = json.dumps(val)
                    row.append(str(val))
                lines.append(",".join(row))
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")


# ============================================================================
# LossVisualizer
# ============================================================================

class LossVisualizer:
    """
    Plot training/validation loss curves using text-based charts.

    Supports EMA smoothing, log scale, and multi-run comparison.
    All rendering uses Unicode characters for terminal display.

    Example:
        visualizer = LossVisualizer(width=80, height=20)
        chart = visualizer.plot_loss(train_losses, val_losses, smooth_window=50)
        print(chart)
    """

    def __init__(
        self,
        width: int = 80,
        height: int = 20,
        log_scale: bool = False,
        smooth_window: int = 50,
        title: str = "Training Loss",
    ):
        """Initialize the loss visualizer.

        Args:
            width: Chart width in characters.
            height: Chart height in characters.
            log_scale: Whether to use logarithmic scale for Y-axis.
            smooth_window: Window size for EMA smoothing.
            title: Default chart title.
        """
        self._width = width
        self._height = height
        self._log_scale = log_scale
        self._smooth_window = smooth_window
        self._title = title

    @staticmethod
    def _ema_smooth(values: List[float], alpha: float = 0.05) -> List[float]:
        """Apply exponential moving average smoothing.

        Args:
            values: Input values.
            alpha: Smoothing factor (0-1). Lower = smoother.

        Returns:
            Smoothed values.
        """
        if not values:
            return []
        smoothed = [values[0]]
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
        return smoothed

    @staticmethod
    def _moving_average(values: List[float], window: int) -> List[float]:
        """Apply moving average smoothing.

        Args:
            values: Input values.
            window: Window size.

        Returns:
            Smoothed values (first window-1 entries are None, then smoothed).
        """
        if not values or window <= 1:
            return list(values)
        result = [None] * (window - 1)
        window_sum = sum(values[:window])
        result.append(window_sum / window)
        for i in range(window, len(values)):
            window_sum += values[i] - values[i - window]
            result.append(window_sum / window)
        return result

    def _normalize_values(
        self, values: List[float], y_min: Optional[float] = None, y_max: Optional[float] = None
    ) -> Tuple[List[float], float, float]:
        """Normalize values to [0, 1] range for plotting.

        Args:
            values: Input values (may contain None).
            y_min: Override minimum value.
            y_max: Override maximum value.

        Returns:
            Tuple of (normalized_values, actual_min, actual_max).
        """
        valid = [v for v in values if v is not None]
        if not valid:
            return [0.0] * len(values), 0.0, 1.0

        actual_min = y_min if y_min is not None else min(valid)
        actual_max = y_max if y_max is not None else max(valid)

        if self._log_scale and actual_min > 0:
            log_min = math.log10(actual_min)
            log_max = math.log10(actual_max)
            if log_max == log_min:
                return [0.5 if v is not None else None for v in values], actual_min, actual_max
            normalized = []
            for v in values:
                if v is None:
                    normalized.append(None)
                elif v <= 0:
                    normalized.append(0.0)
                else:
                    log_v = math.log10(v)
                    normalized.append((log_v - log_min) / (log_max - log_min))
            return normalized, actual_min, actual_max

        if actual_max == actual_min:
            return [0.5 if v is not None else None for v in values], actual_min, actual_max

        normalized = []
        for v in values:
            if v is None:
                normalized.append(None)
            else:
                normalized.append((v - actual_min) / (actual_max - actual_min))
        return normalized, actual_min, actual_max

    def _render_line_chart(
        self,
        series_list: List[Tuple[str, List[float]]],
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        show_legend: bool = True,
    ) -> str:
        """Render multiple series as a line chart.

        Args:
            series_list: List of (name, values) tuples.
            y_min: Override Y-axis minimum.
            y_max: Override Y-axis maximum.
            title: Chart title.
            x_label: X-axis label.
            y_label: Y-axis label.
            show_legend: Whether to show the legend.

        Returns:
            Multi-line string of the chart.
        """
        if not series_list:
            return "(no data)"

        chart_width = self._width
        chart_height = self._height

        # Find the maximum length of any series
        max_len = max(len(values) for _, values in series_list)
        if max_len == 0:
            return "(no data)"

        # Downsample if needed
        if max_len > chart_width - 12:
            for i, (name, values) in enumerate(series_list):
                step = max_len / (chart_width - 12)
                new_values = []
                for j in range(chart_width - 12):
                    idx = int(j * step)
                    if idx < len(values):
                        new_values.append(values[idx])
                series_list[i] = (name, new_values)
            max_len = chart_width - 12

        # Normalize all series to the same scale
        all_valid = []
        for _, values in series_list:
            all_valid.extend([v for v in values if v is not None])
        if not all_valid:
            return "(no valid data)"

        global_min = y_min if y_min is not None else min(all_valid)
        global_max = y_max if y_max is not None else max(all_valid)
        if global_max == global_min:
            global_max = global_min + 1.0

        normalized_series = []
        for name, values in series_list:
            norm = []
            for v in values:
                if v is None:
                    norm.append(None)
                else:
                    if self._log_scale and global_min > 0 and v > 0:
                        log_v = math.log10(v)
                        log_min = math.log10(global_min)
                        log_max = math.log10(global_max)
                        norm.append((log_v - log_min) / (log_max - log_min))
                    else:
                        norm.append((v - global_min) / (global_max - global_min))
            normalized_series.append((name, norm))

        # Build chart grid
        plot_width = max_len
        plot_height = chart_height - 4  # Leave room for labels

        # Initialize canvas
        canvas = [[" "] * plot_width for _ in range(plot_height)]
        series_chars = ["─", "*", "+", "o", "~", "#", "@"]
        legend_items = []

        for idx, (name, norm_values) in enumerate(normalized_series):
            char = series_chars[idx % len(series_chars)]
            legend_items.append((name, char))
            prev_row = None
            prev_col = None
            for col, val in enumerate(norm_values):
                if val is not None:
                    row = int((1.0 - val) * (plot_height - 1))
                    row = max(0, min(plot_height - 1, row))
                    if 0 <= row < plot_height and 0 <= col < plot_width:
                        canvas[row][col] = char
                        # Draw connecting lines
                        if prev_row is not None and prev_col is not None:
                            r1, c1 = prev_row, prev_col
                            r2, c2 = row, col
                            if c2 == c1 + 1:
                                if r2 == r1:
                                    canvas[r1][c1] = char
                                elif r2 > r1:
                                    canvas[r1][c1] = CHART_UP
                                else:
                                    canvas[r1][c1] = CHART_DOWN
                        prev_row = row
                        prev_col = col

        # Format Y-axis labels
        y_labels = []
        num_y_ticks = min(5, plot_height)
        for i in range(num_y_ticks):
            frac = i / (num_y_ticks - 1) if num_y_ticks > 1 else 0.5
            val = global_min + frac * (global_max - global_min)
            if self._log_scale and val > 0:
                label = f"{val:.2e}"
            elif abs(val) < 0.001 or abs(val) > 10000:
                label = f"{val:.2e}"
            else:
                label = f"{val:.4f}"
            y_labels.append(label)

        # Build output
        lines = []
        # Title
        display_title = title or self._title
        if display_title:
            title_line = display_title
            if show_legend:
                legend_str = " | ".join(f"{char} {name}" for name, char in legend_items)
                title_line += "  " + legend_str
            lines.append(title_line)

        # Top border
        lines.append(BOX_TL + BOX_H * 8 + BOX_BT + BOX_H * plot_width + BOX_TR)

        # Chart rows
        for row in range(plot_height):
            y_idx = int((plot_height - 1 - row) / (plot_height - 1) * (num_y_ticks - 1)) if num_y_ticks > 1 else 0
            y_idx = min(y_idx, len(y_labels) - 1)
            y_label = y_labels[y_idx].rjust(8)
            row_content = "".join(canvas[row][:plot_width])
            lines.append(f"{BOX_V}{y_label}{BOX_V}{row_content}{BOX_V}")

        # X-axis label
        x_display = x_label or "step"
        x_label_line = f"{'':>8}{BOX_V}{x_display:^{plot_width}}{BOX_V}"
        lines.append(x_label_line)

        # Bottom border
        lines.append(BOX_BL + BOX_H * 8 + BOX_BB + BOX_H * plot_width + BOX_BR)

        # Y-axis label
        y_display = y_label or ("loss (log)" if self._log_scale else "loss")
        lines.append(f"{'':>8}  {y_display}")

        return "\n".join(lines)

    def plot_loss(
        self,
        losses: List[float],
        val_losses: Optional[List[float]] = None,
        smooth_window: Optional[int] = None,
        log_scale: Optional[bool] = None,
        title: Optional[str] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> str:
        """Plot training and validation loss curves.

        Args:
            losses: Training loss values per step.
            val_losses: Validation loss values (aligned with training steps).
            smooth_window: Override smoothing window.
            log_scale: Override log scale setting.
            title: Chart title.
            y_min: Y-axis minimum.
            y_max: Y-axis maximum.

        Returns:
            Multi-line string chart.
        """
        if not losses:
            return "(no loss data)"

        window = smooth_window or self._smooth_window
        original_log = self._log_scale
        if log_scale is not None:
            self._log_scale = log_scale

        # Smooth losses
        smoothed = self._moving_average(losses, window)
        series_list = [("train", smoothed)]

        if val_losses is not None:
            smoothed_val = self._moving_average(val_losses, window)
            series_list.append(("val", smoothed_val))

        chart = self._render_line_chart(
            series_list,
            y_min=y_min,
            y_max=y_max,
            title=title or "Training & Validation Loss",
            x_label="step",
            y_label="loss (log)" if self._log_scale else "loss",
        )

        self._log_scale = original_log
        return chart

    def plot_lr(
        self,
        lr_history: List[float],
        title: Optional[str] = None,
    ) -> str:
        """Plot learning rate schedule.

        Args:
            lr_history: Learning rate values per step.
            title: Chart title.

        Returns:
            Multi-line string chart.
        """
        if not lr_history:
            return "(no LR data)"

        return self._render_line_chart(
            [("lr", lr_history)],
            title=title or "Learning Rate Schedule",
            x_label="step",
            y_label="learning rate",
        )

    def annotate_best(
        self,
        losses: List[float],
        epochs: Optional[List[int]] = None,
        top_k: int = 5,
    ) -> str:
        """Annotate the best (lowest) loss values.

        Args:
            losses: Loss values.
            epochs: Optional epoch labels (defaults to indices).
            top_k: Number of best checkpoints to show.

        Returns:
            Formatted string with best loss annotations.
        """
        if not losses:
            return "(no data)"

        indexed = list(enumerate(losses))
        indexed.sort(key=lambda x: x[1])

        lines = []
        lines.append(f"Top {top_k} Best Loss Checkpoints:")
        lines.append(BOX_TL + BOX_H * 50 + BOX_TR)
        lines.append(f"{BOX_V}{'Rank':>6}{BOX_V}{'Step':>8}{BOX_V}{'Loss':>14}{BOX_V}{'Delta':>14}{BOX_V}")
        lines.append(f"{BOX_V}{CHART_H*6}{BOX_V}{CHART_H*8}{BOX_V}{CHART_H*14}{BOX_V}{CHART_H*14}{BOX_V}")

        best_loss = indexed[0][1]
        for rank, (idx, loss_val) in enumerate(indexed[:top_k]):
            epoch_label = epochs[idx] if epochs else idx
            delta = loss_val - best_loss
            delta_str = f"+{delta:.6f}" if delta > 0 else "  0.000000"
            marker = STATUS_STAR if rank == 0 else "  "
            lines.append(
                f"{BOX_V}{marker}{rank+1:>4}{BOX_V}{epoch_label:>8}{BOX_V}"
                f"{loss_val:>14.6f}{BOX_V}{delta_str:>14}{BOX_V}"
            )

        lines.append(BOX_BL + BOX_H * 50 + BOX_BR)
        return "\n".join(lines)

    def plot_multi_run(
        self,
        runs: Dict[str, List[float]],
        smooth_window: Optional[int] = None,
        title: Optional[str] = None,
    ) -> str:
        """Compare loss curves from multiple runs.

        Args:
            runs: Dictionary mapping run names to loss lists.
            smooth_window: Override smoothing window.
            title: Chart title.

        Returns:
            Multi-line string chart.
        """
        if not runs:
            return "(no run data)"

        window = smooth_window or self._smooth_window
        series_list = []
        for name, losses in runs.items():
            smoothed = self._moving_average(losses, window)
            series_list.append((name, smoothed))

        return self._render_line_chart(
            series_list,
            title=title or "Multi-Run Loss Comparison",
            x_label="step",
            y_label="loss",
        )


# ============================================================================
# ThroughputMonitor
# ============================================================================

class ThroughputMonitor:
    """
    Monitor training throughput: tokens/sec, samples/sec, GPU utilization.

    Provides running averages, peak values, and formatted reports.

    Example:
        monitor = ThroughputMonitor(window_size=100)
        monitor.update(batch_size=32, seq_len=512, time_taken=0.5)
        print(monitor.report())
    """

    def __init__(
        self,
        window_size: int = 100,
        history_size: int = 10000,
    ):
        """Initialize the throughput monitor.

        Args:
            window_size: Window size for rolling averages.
            history_size: Maximum history entries to keep.
        """
        self._window_size = window_size
        self._history_size = history_size
        self._lock = threading.Lock()

        # Throughput history
        self._token_throughputs: deque = deque(maxlen=history_size)
        self._sample_throughputs: deque = deque(maxlen=history_size)
        self._gpu_utils: deque = deque(maxlen=history_size)
        self._step_times: deque = deque(maxlen=history_size)

        # Summary statistics
        self._total_tokens: int = 0
        self._total_samples: int = 0
        self._total_time: float = 0.0
        self._num_updates: int = 0
        self._peak_tokens_per_sec: float = 0.0
        self._peak_samples_per_sec: float = 0.0
        self._start_time: float = time.time()

    def update(
        self,
        batch_size: int,
        seq_len: int,
        time_taken: float,
        gpu_utilization: Optional[float] = None,
        tokens_per_batch: Optional[int] = None,
    ) -> Dict[str, float]:
        """Update throughput with a new step measurement.

        Args:
            batch_size: Number of samples in the batch.
            seq_len: Sequence length per sample.
            time_taken: Time for this step in seconds.
            gpu_utilization: GPU utilization percentage (0-100).
            tokens_per_batch: Override total tokens (defaults to batch_size * seq_len).

        Returns:
            Dictionary with current throughput metrics.
        """
        tokens = tokens_per_batch if tokens_per_batch is not None else batch_size * seq_len

        with self._lock:
            tps = tokens / time_taken if time_taken > 0 else 0.0
            sps = batch_size / time_taken if time_taken > 0 else 0.0

            self._token_throughputs.append(tps)
            self._sample_throughputs.append(sps)
            self._step_times.append(time_taken)
            self._total_tokens += tokens
            self._total_samples += batch_size
            self._total_time += time_taken
            self._num_updates += 1

            if tps > self._peak_tokens_per_sec:
                self._peak_tokens_per_sec = tps
            if sps > self._peak_samples_per_sec:
                self._peak_samples_per_sec = sps

            if gpu_utilization is not None:
                self._gpu_utils.append(gpu_utilization)

            return {
                "tokens_per_sec": tps,
                "samples_per_sec": sps,
                "time_taken": time_taken,
                "gpu_utilization": gpu_utilization,
            }

    def get_average(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get average throughput over a window.

        Args:
            window: Window size. None for configured default.

        Returns:
            Dictionary with averaged metrics.
        """
        w = window or self._window_size
        with self._lock:
            recent_tps = list(self._token_throughputs)[-w:]
            recent_sps = list(self._sample_throughputs)[-w:]
            recent_times = list(self._step_times)[-w:]
            recent_gpu = list(self._gpu_utils)[-w:]

            avg_tps = sum(recent_tps) / len(recent_tps) if recent_tps else 0.0
            avg_sps = sum(recent_sps) / len(recent_sps) if recent_sps else 0.0
            avg_time = sum(recent_times) / len(recent_times) if recent_times else 0.0
            avg_gpu = sum(recent_gpu) / len(recent_gpu) if recent_gpu else 0.0

            return {
                "avg_tokens_per_sec": avg_tps,
                "avg_samples_per_sec": avg_sps,
                "avg_step_time": avg_time,
                "avg_gpu_utilization": avg_gpu,
                "window_size": len(recent_tps),
            }

    def get_all_time_average(self) -> Dict[str, float]:
        """Get all-time average throughput.

        Returns:
            Dictionary with all-time averaged metrics.
        """
        with self._lock:
            elapsed = time.time() - self._start_time
            return {
                "all_time_tokens_per_sec": self._total_tokens / elapsed if elapsed > 0 else 0.0,
                "all_time_samples_per_sec": self._total_samples / elapsed if elapsed > 0 else 0.0,
                "total_tokens": self._total_tokens,
                "total_samples": self._total_samples,
                "total_time": self._total_time,
                "num_steps": self._num_updates,
                "elapsed_wall_time": elapsed,
            }

    def report(self, window: Optional[int] = None) -> str:
        """Generate a formatted throughput report.

        Args:
            window: Window size for recent averages.

        Returns:
            Multi-line formatted report string.
        """
        recent = self.get_average(window)
        all_time = self.get_all_time_average()

        lines = []
        lines.append(f"{BOX_TL}{'Throughput Report':^58}{BOX_TR}")
        lines.append(f"{BOX_V}{'Metric':<30}{'Recent':>12}{'All-Time':>14}{BOX_V}")
        lines.append(f"{BOX_V}{BOX_H*30}{BOX_CROSS}{BOX_H*12}{BOX_CROSS}{BOX_H*14}{BOX_V}")

        def fmt_val(v, unit=""):
            if abs(v) >= 1e6:
                return f"{v/1e6:.2f}M{unit}"
            elif abs(v) >= 1e3:
                return f"{v/1e3:.2f}K{unit}"
            else:
                return f"{v:.2f}{unit}"

        lines.append(
            f"{BOX_V}{'Tokens/sec':<30}{fmt_val(recent['avg_tokens_per_sec']):>12}"
            f"{fmt_val(all_time['all_time_tokens_per_sec']):>14}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{'Samples/sec':<30}{fmt_val(recent['avg_samples_per_sec']):>12}"
            f"{fmt_val(all_time['all_time_samples_per_sec']):>14}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{'Step time (ms)':<30}{recent['avg_step_time']*1000:>12.1f}"
            f"{'N/A':>14}{BOX_V}"
        )
        if recent["avg_gpu_utilization"] > 0:
            lines.append(
                f"{BOX_V}{'GPU utilization':<30}{recent['avg_gpu_utilization']:>11.1f}%"
                f"{'N/A':>14}{BOX_V}"
            )
        lines.append(
            f"{BOX_V}{'Peak tokens/sec':<30}{fmt_val(self._peak_tokens_per_sec):>12}"
            f"{'N/A':>14}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{'Peak samples/sec':<30}{fmt_val(self._peak_samples_per_sec):>12}"
            f"{'N/A':>14}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{'Total tokens':<30}{'N/A':>12}{fmt_val(all_time['total_tokens']):>14}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{'Wall time (s)':<30}{'N/A':>12}{all_time['elapsed_wall_time']:>13.1f}{BOX_V}"
        )
        lines.append(BOX_BL + BOX_H * 58 + BOX_BR)

        return "\n".join(lines)

    def estimate_training_time(
        self,
        remaining_steps: int,
        window: Optional[int] = None,
    ) -> Dict[str, Union[float, str]]:
        """Estimate time to complete remaining training steps.

        Args:
            remaining_steps: Number of training steps remaining.
            window: Window for recent average step time.

        Returns:
            Dictionary with time estimates.
        """
        recent = self.get_average(window)
        avg_step_time = recent["avg_step_time"]
        if avg_step_time <= 0:
            return {"remaining_seconds": 0.0, "eta": "unknown"}

        remaining_seconds = remaining_steps * avg_step_time
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        seconds = int(remaining_seconds % 60)
        eta_str = f"{hours}h {minutes}m {seconds}s"

        return {
            "remaining_seconds": remaining_seconds,
            "remaining_hours": remaining_seconds / 3600,
            "eta": eta_str,
            "avg_step_time": avg_step_time,
        }


# ============================================================================
# GradientMonitor
# ============================================================================

class GradientMonitor:
    """
    Monitor gradient norms and detect training health issues.

    Tracks gradient norms per layer and across the model, detects
    exploding/vanishing gradients, and provides health diagnostics.

    Example:
        monitor = GradientMonitor()
        norms = monitor.log_gradients(model)  # or pass dict of norms
        health = monitor.check_health()
        print(health.summary())
    """

    def __init__(
        self,
        max_norm_threshold: float = 100.0,
        min_norm_threshold: float = 1e-7,
        history_size: int = 10000,
        alert_window: int = 50,
    ):
        """Initialize the gradient monitor.

        Args:
            max_norm_threshold: Norm above which is considered exploding.
            min_norm_threshold: Norm below which is considered vanishing.
            history_size: Maximum history entries.
            alert_window: Window for trend analysis.
        """
        self._max_norm = max_norm_threshold
        self._min_norm = min_norm_threshold
        self._history_size = history_size
        self._alert_window = alert_window
        self._lock = threading.Lock()

        # Per-step total gradient norm
        self._norm_history: deque = deque(maxlen=history_size)
        self._step_indices: deque = deque(maxlen=history_size)

        # Per-layer norms (latest)
        self._layer_norms: Dict[str, float] = {}
        self._layer_norm_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))

        # Health tracking
        self._exploding_count: int = 0
        self._vanishing_count: int = 0
        self._total_steps: int = 0
        self._recent_exploding: deque = deque(maxlen=alert_window)
        self._recent_vanishing: deque = deque(maxlen=alert_window)

        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, float, float], None]] = []

    def log_gradients(
        self,
        model: Optional[Any] = None,
        grad_norms: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Log gradient norms.

        Either pass a model object with named_parameters() or a dictionary
        of layer_name -> gradient_norm.

        Args:
            model: Model with named_parameters() method.
            grad_norms: Dictionary of layer_name -> gradient_norm.
            step: Optional step number.

        Returns:
            Dictionary of layer_name -> gradient_norm.
        """
        with self._lock:
            norms = {}

            if grad_norms is not None:
                norms = dict(grad_norms)
            elif model is not None:
                if hasattr(model, "named_parameters"):
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad = param.grad
                            norm = float(torch.norm(grad).item()) if hasattr(torch, "norm") else 0.0
                            if hasattr(grad, "norm"):
                                norm = float(grad.norm().item())
                            elif hasattr(grad, "data"):
                                norm = float(torch.norm(grad.data).item()) if hasattr(torch, "norm") else 0.0
                            else:
                                flat = []
                                _collect_nested(grad, flat)
                                norm = math.sqrt(sum(x * x for x in flat)) if flat else 0.0
                            norms[name] = norm
                elif hasattr(model, "parameters"):
                    idx = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            norm = 0.0
                            if hasattr(param.grad, "norm"):
                                norm = float(param.grad.norm().item())
                            norms[f"layer_{idx}"] = norm
                            idx += 1

            if not norms:
                return {}

            step_num = step if step is not None else self._total_steps
            self._total_steps += 1

            # Compute total norm
            total_norm = math.sqrt(sum(v * v for v in norms.values()))

            self._norm_history.append(total_norm)
            self._step_indices.append(step_num)

            # Update per-layer tracking
            self._layer_norms = dict(norms)
            for name, norm_val in norms.items():
                self._layer_norm_history[name].append(norm_val)

            # Check for issues
            is_exploding = total_norm > self._max_norm
            is_vanishing = total_norm < self._min_norm

            if is_exploding:
                self._exploding_count += 1
                self._recent_exploding.append(step_num)
                for cb in self._alert_callbacks:
                    try:
                        cb("exploding", total_norm, step_num)
                    except Exception:
                        pass

            if is_vanishing:
                self._vanishing_count += 1
                self._recent_vanishing.append(step_num)
                for cb in self._alert_callbacks:
                    try:
                        cb("vanishing", total_norm, step_num)
                    except Exception:
                        pass

            return norms

    def check_health(self) -> "GradientHealthReport":
        """Check gradient health based on recent history.

        Returns:
            GradientHealthReport with health assessment.
        """
        with self._lock:
            report = GradientHealthReport()

            if not self._norm_history:
                report.status = "no_data"
                report.summary_text = "No gradient data recorded yet."
                return report

            recent = list(self._norm_history)[-self._alert_window:]
            current_norm = recent[-1]
            avg_norm = sum(recent) / len(recent)

            # Check norms of individual layers for vanishing
            vanishing_layers = []
            exploding_layers = []
            for name, norm_val in self._layer_norms.items():
                if norm_val < self._min_norm:
                    vanishing_layers.append((name, norm_val))
                elif norm_val > self._max_norm:
                    exploding_layers.append((name, norm_val))

            report.current_norm = current_norm
            report.average_norm = avg_norm
            report.max_norm = max(recent)
            report.min_norm = min(recent)
            report.norm_std = math.sqrt(
                sum((x - avg_norm) ** 2 for x in recent) / len(recent)
            ) if len(recent) > 1 else 0.0

            # Trend analysis
            if len(recent) >= 10:
                first_half = recent[: len(recent) // 2]
                second_half = recent[len(recent) // 2 :]
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                if first_avg > 0:
                    trend_ratio = second_avg / first_avg
                else:
                    trend_ratio = 1.0
                report.trend_ratio = trend_ratio
                if trend_ratio > 2.0:
                    report.trend = "exploding"
                elif trend_ratio < 0.5:
                    report.trend = "vanishing"
                elif trend_ratio > 1.3:
                    report.trend = "increasing"
                elif trend_ratio < 0.7:
                    report.trend = "decreasing"
                else:
                    report.trend = "stable"

            # Exploding/vanishing rate
            exploding_rate = len(self._recent_exploding) / max(len(recent), 1)
            vanishing_rate = len(self._recent_vanishing) / max(len(recent), 1)

            # Determine overall status
            if exploding_rate > 0.2:
                report.status = "exploding"
            elif vanishing_rate > 0.2:
                report.status = "vanishing"
            elif exploding_layers and len(exploding_layers) > len(self._layer_norms) * 0.5:
                report.status = "layer_exploding"
            elif vanishing_layers and len(vanishing_layers) > len(self._layer_norms) * 0.5:
                report.status = "layer_vanishing"
            elif exploding_rate > 0.05 or vanishing_rate > 0.05:
                report.status = "warning"
            elif report.trend in ("exploding", "vanishing"):
                report.status = "warning"
            else:
                report.status = "healthy"

            report.exploding_count = self._exploding_count
            report.vanishing_count = self._vanishing_count
            report.exploding_rate = exploding_rate
            report.vanishing_rate = vanishing_rate
            report.exploding_layers = exploding_layers
            report.vanishing_layers = vanishing_layers
            report.total_steps = self._total_steps

            # Build summary
            summary_parts = []
            if report.status == "healthy":
                summary_parts.append(f"{STATUS_OK} Gradients healthy")
            elif report.status == "warning":
                summary_parts.append(f"{STATUS_WARN} Gradient issues detected")
            elif "exploding" in report.status:
                summary_parts.append(f"{STATUS_ERROR} Exploding gradients detected")
            elif "vanishing" in report.status:
                summary_parts.append(f"{STATUS_ERROR} Vanishing gradients detected")

            summary_parts.append(f"  Norm: {current_norm:.6f} (avg: {avg_norm:.6f})")
            summary_parts.append(f"  Trend: {report.trend} (ratio: {report.trend_ratio:.2f})")
            if exploding_layers:
                summary_parts.append(f"  Exploding layers: {len(exploding_layers)}")
            if vanishing_layers:
                summary_parts.append(f"  Vanishing layers: {len(vanishing_layers)}")

            report.summary_text = "\n".join(summary_parts)
            return report

    def plot_gradient_norms(
        self,
        width: int = 80,
        height: int = 15,
    ) -> str:
        """Plot gradient norm history as a text chart.

        Args:
            width: Chart width.
            height: Chart height.

        Returns:
            Multi-line chart string.
        """
        with self._lock:
            norms = list(self._norm_history)
            if not norms:
                return "(no gradient data)"

            # Downsample if needed
            max_points = width - 14
            if len(norms) > max_points:
                step = len(norms) / max_points
                downsampled = []
                for i in range(max_points):
                    idx = int(i * step)
                    downsampled.append(norms[idx])
                norms = downsampled

            # Build chart
            log_norms = [math.log10(max(v, 1e-10)) for v in norms]
            y_min = min(log_norms)
            y_max = max(log_norms)
            if y_max == y_min:
                y_max = y_min + 1.0

            chart_width = len(norms)
            chart_height = height - 4

            canvas = [[" "] * chart_width for _ in range(chart_height)]

            # Threshold lines
            if y_max >= math.log10(self._max_norm) >= y_min:
                thresh_row = int(
                    (1.0 - (math.log10(self._max_norm) - y_min) / (y_max - y_min))
                    * (chart_height - 1)
                )
                thresh_row = max(0, min(chart_height - 1, thresh_row))
                for col in range(chart_width):
                    canvas[thresh_row][col] = "─"

            if y_min <= math.log10(self._min_norm) <= y_max:
                thresh_row = int(
                    (1.0 - (math.log10(self._min_norm) - y_min) / (y_max - y_min))
                    * (chart_height - 1)
                )
                thresh_row = max(0, min(chart_height - 1, thresh_row))
                for col in range(chart_width):
                    if canvas[thresh_row][col] == " ":
                        canvas[thresh_row][col] = "·"

            # Plot norms
            for col, val in enumerate(log_norms):
                row = int((1.0 - (val - y_min) / (y_max - y_min)) * (chart_height - 1))
                row = max(0, min(chart_height - 1, row))
                char = "*" if val > math.log10(self._max_norm) else (
                    "." if val < math.log10(self._min_norm) else CHART_DOT
                )
                canvas[row][col] = char

            # Build output
            lines = []
            lines.append("Gradient Norm History (log scale)")
            lines.append(BOX_TL + BOX_H * 10 + BOX_BT + BOX_H * chart_width + BOX_TR)

            for row in range(chart_height):
                val = 10 ** (y_max - (row / (chart_height - 1)) * (y_max - y_min))
                label = f"{val:>9.1e}"
                row_str = "".join(canvas[row])
                lines.append(f"{BOX_V}{label}{BOX_V}{row_str}{BOX_V}")

            lines.append(BOX_BL + BOX_H * 10 + BOX_BB + BOX_H * chart_width + BOX_BR)
            lines.append(f"  step (max={self._max_norm:.0f}, min={self._min_norm:.0e})")
            return "\n".join(lines)

    def get_layer_norms(self) -> Dict[str, float]:
        """Get the latest per-layer gradient norms.

        Returns:
            Dictionary of layer_name -> gradient_norm.
        """
        with self._lock:
            return dict(self._layer_norms)

    def on_alert(self, callback: Callable[[str, float, float], None]) -> None:
        """Register callback for gradient alerts.

        Args:
            callback: Function receiving (alert_type, norm_value, step).
        """
        self._alert_callbacks.append(callback)


def _collect_nested(obj, flat_list):
    """Recursively collect numeric values from nested objects."""
    if hasattr(obj, "tolist"):
        flat_list.extend(obj.tolist())
    elif hasattr(obj, "numpy"):
        flat_list.extend(obj.numpy().tolist())
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_nested(item, flat_list)
    elif isinstance(obj, (int, float)):
        flat_list.append(float(obj))
    elif hasattr(obj, "item"):
        flat_list.append(float(obj.item()))


@dataclass
class GradientHealthReport:
    """Report on gradient health status."""
    status: str = "no_data"
    current_norm: float = 0.0
    average_norm: float = 0.0
    max_norm: float = 0.0
    min_norm: float = 0.0
    norm_std: float = 0.0
    trend: str = "unknown"
    trend_ratio: float = 1.0
    exploding_count: int = 0
    vanishing_count: int = 0
    exploding_rate: float = 0.0
    vanishing_rate: float = 0.0
    exploding_layers: List[Tuple[str, float]] = field(default_factory=list)
    vanishing_layers: List[Tuple[str, float]] = field(default_factory=list)
    total_steps: int = 0
    summary_text: str = ""

    def summary(self) -> str:
        """Get formatted health summary.

        Returns:
            Multi-line summary string.
        """
        return self.summary_text


# ============================================================================
# CheckpointVisualizer
# ============================================================================

class CheckpointVisualizer:
    """
    Compare model checkpoints and visualize parameter drift.

    Analyzes parameter changes between checkpoints to detect training
    issues and understand model evolution.

    Example:
        viz = CheckpointVisualizer()
        drift = viz.compare_checkpoints(params_1, params_2)
        print(viz.format_drift_report(drift))
    """

    def __init__(self, layer_filter: Optional[List[str]] = None):
        """Initialize the checkpoint visualizer.

        Args:
            layer_filter: Optional list of layer name patterns to include.
        """
        self._layer_filter = layer_filter
        self._checkpoint_cache: Dict[str, Dict[str, List[float]]] = {}

    def compare_checkpoints(
        self,
        params_a: Dict[str, Any],
        params_b: Dict[str, Any],
        label_a: str = "checkpoint_a",
        label_b: str = "checkpoint_b",
    ) -> "CheckpointDiff":
        """Compare parameter states between two checkpoints.

        Args:
            params_a: First checkpoint parameters (name -> values).
            params_b: Second checkpoint parameters (name -> values).
            label_a: Label for first checkpoint.
            label_b: Label for second checkpoint.

        Returns:
            CheckpointDiff with comparison results.
        """
        diff = CheckpointDiff(label_a=label_a, label_b=label_b)

        all_names = set(list(params_a.keys()) + list(params_b.keys()))
        filtered_names = self._filter_layers(all_names)

        for name in filtered_names:
            a = params_a.get(name)
            b = params_b.get(name)

            if a is None and b is None:
                continue
            elif a is None:
                diff.added_layers.append(name)
                continue
            elif b is None:
                diff.removed_layers.append(name)
                continue

            # Extract flat values
            vals_a = self._flatten(a)
            vals_b = self._flatten(b)

            if not vals_a or not vals_b:
                continue

            # Compute statistics
            changes = [va - vb for va, vb in zip(vals_a, vals_b)]
            abs_changes = [abs(c) for c in changes]

            mean_change = sum(changes) / len(changes) if changes else 0.0
            mean_abs_change = sum(abs_changes) / len(abs_changes) if abs_changes else 0.0
            max_abs_change = max(abs_changes) if abs_changes else 0.0

            # Cosine similarity
            dot = sum(va * vb for va, vb in zip(vals_a, vals_b))
            norm_a = math.sqrt(sum(va * va for va in vals_a))
            norm_b = math.sqrt(sum(vb * vb for vb in vals_b))
            cosine_sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

            # L2 distance
            l2_dist = math.sqrt(sum(c * c for c in changes))

            layer_diff = LayerParameterDiff(
                name=name,
                mean_change=mean_change,
                mean_abs_change=mean_abs_change,
                max_abs_change=max_abs_change,
                cosine_similarity=cosine_sim,
                l2_distance=l2_dist,
                num_parameters=len(vals_a),
            )
            diff.layer_diffs.append(layer_diff)

        # Compute overall statistics
        if diff.layer_diffs:
            diff.avg_cosine_similarity = sum(
                ld.cosine_similarity for ld in diff.layer_diffs
            ) / len(diff.layer_diffs)
            diff.avg_l2_distance = sum(
                ld.l2_distance for ld in diff.layer_diffs
            ) / len(diff.layer_diffs)
            diff.max_l2_distance = max(
                ld.l2_distance for ld in diff.layer_diffs
            )
            diff.total_parameters = sum(
                ld.num_parameters for ld in diff.layer_diffs
            )

        return diff

    def format_drift_report(self, diff: "CheckpointDiff", top_k: int = 10) -> str:
        """Format a checkpoint comparison as a readable report.

        Args:
            diff: CheckpointDiff from compare_checkpoints.
            top_k: Number of most-changed layers to show.

        Returns:
            Multi-line report string.
        """
        lines = []
        lines.append(f"{BOX_TL}{'Checkpoint Drift Report':^72}{BOX_TR}")
        lines.append(
            f"{BOX_V} {diff.label_a} --> {diff.label_b}"
            f"{'':>{72 - len(diff.label_a) - len(diff.label_b) - 5}}{BOX_V}"
        )
        lines.append(f"{BOX_V}{BOX_H * 72}{BOX_V}")

        if diff.added_layers:
            lines.append(f"{BOX_V} Added layers: {len(diff.added_layers)}{BOX_V}")
        if diff.removed_layers:
            lines.append(f"{BOX_V} Removed layers: {len(diff.removed_layers)}{BOX_V}")

        lines.append(
            f"{BOX_V} Overall cosine similarity: {diff.avg_cosine_similarity:.6f}"
            f"{'':>{72 - 35}}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Average L2 distance:        {diff.avg_l2_distance:.6f}"
            f"{'':>{72 - 35}}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Max L2 distance:            {diff.max_l2_distance:.6f}"
            f"{'':>{72 - 35}}{BOX_V}"
        )
        lines.append(f"{BOX_V}{BOX_H * 72}{BOX_V}")

        # Top-k most changed layers
        sorted_layers = sorted(
            diff.layer_diffs, key=lambda x: x.l2_distance, reverse=True
        )[:top_k]

        if sorted_layers:
            header = f"  Top {top_k} Most Changed Layers:"
            lines.append(f"{BOX_V}{header:<72}{BOX_V}")
            lines.append(
                f"{BOX_V}{'Layer':<40}{'L2 Dist':>12}{'Cosine':>12}{'Params':>8}{BOX_V}"
            )
            lines.append(
                f"{BOX_V}{BOX_H*40}{BOX_CROSS}{BOX_H*12}{BOX_CROSS}{BOX_H*12}{BOX_CROSS}{BOX_H*8}{BOX_V}"
            )

            for ld in sorted_layers:
                name = ld.name[-39:] if len(ld.name) > 39 else ld.name
                lines.append(
                    f"{BOX_V}{name:<40}{ld.l2_distance:>12.4f}"
                    f"{ld.cosine_similarity:>12.4f}{ld.num_parameters:>8}{BOX_V}"
                )

        lines.append(BOX_BL + BOX_H * 72 + BOX_BR)
        return "\n".join(lines)

    def parameter_drift_chart(
        self,
        checkpoint_diffs: List["CheckpointDiff"],
        width: int = 80,
        height: int = 15,
    ) -> str:
        """Create a text chart of parameter drift over multiple checkpoints.

        Args:
            checkpoint_diffs: List of CheckpointDiff objects (consecutive).
            width: Chart width.
            height: Chart height.

        Returns:
            Multi-line chart string.
        """
        if not checkpoint_diffs:
            return "(no checkpoint data)"

        l2_values = [cd.avg_l2_distance for cd in checkpoint_diffs]
        cosine_values = [cd.avg_cosine_similarity for cd in checkpoint_diffs]

        # L2 distance chart
        chart_h = height // 2 - 1
        if l2_values:
            lines = ["Parameter Drift Over Time"]
            lines.append(self._mini_chart(l2_values, "L2 Distance", width - 2, chart_h))
            lines.append(self._mini_chart(cosine_values, "Cosine Similarity", width - 2, chart_h))
            return "\n".join(lines)
        return "(no data)"

    def _mini_chart(
        self, values: List[float], label: str, width: int, height: int
    ) -> str:
        """Create a mini sparkline chart.

        Args:
            values: Data values.
            label: Chart label.
            width: Chart width.
            height: Chart height.

        Returns:
            Mini chart string.
        """
        if not values:
            return f"{label}: (no data)"

        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            v_max = v_min + 1.0

        # Downsample to fit width
        if len(values) > width:
            step = len(values) / width
            sampled = []
            for i in range(width):
                idx = int(i * step)
                sampled.append(values[idx])
            values = sampled

        canvas = [[" "] * len(values) for _ in range(height)]
        for col, val in enumerate(values):
            row = int((1.0 - (val - v_min) / (v_max - v_min)) * (height - 1))
            row = max(0, min(height - 1, row))
            canvas[row][col] = CHART_DOT

        lines = [f"  {label}: [{v_min:.4f} - {v_max:.4f}]"]
        for row in range(height):
            lines.append("  " + "".join(canvas[row]))
        return "\n".join(lines)

    def _flatten(self, obj: Any) -> List[float]:
        """Flatten a parameter tensor-like object to a list of floats.

        Args:
            obj: Parameter object (tensor, list, etc.).

        Returns:
            List of float values.
        """
        result = []
        _collect_nested(obj, result)
        return result

    def _filter_layers(self, names: set) -> List[str]:
        """Filter layer names based on configured patterns.

        Args:
            names: Set of all layer names.

        Returns:
            Filtered list of layer names.
        """
        if self._layer_filter is None:
            return sorted(names)
        filtered = []
        for name in sorted(names):
            for pattern in self._layer_filter:
                if pattern in name:
                    filtered.append(name)
                    break
        return filtered


@dataclass
class LayerParameterDiff:
    """Difference in a single layer's parameters between checkpoints."""
    name: str = ""
    mean_change: float = 0.0
    mean_abs_change: float = 0.0
    max_abs_change: float = 0.0
    cosine_similarity: float = 1.0
    l2_distance: float = 0.0
    num_parameters: int = 0


@dataclass
class CheckpointDiff:
    """Comparison between two model checkpoints."""
    label_a: str = "a"
    label_b: str = "b"
    layer_diffs: List[LayerParameterDiff] = field(default_factory=list)
    added_layers: List[str] = field(default_factory=list)
    removed_layers: List[str] = field(default_factory=list)
    avg_cosine_similarity: float = 1.0
    avg_l2_distance: float = 0.0
    max_l2_distance: float = 0.0
    total_parameters: int = 0


# ============================================================================
# DashboardFormatter
# ============================================================================

class DashboardFormatter:
    """
    Text-based dashboard for terminal display.

    Uses Unicode box-drawing characters, progress bars, and formatted
    tables to create a comprehensive training dashboard without matplotlib.

    Example:
        fmt = DashboardFormatter(width=100)
        dashboard = fmt.format_dashboard(tracker, throughput_monitor, grad_monitor)
        print(dashboard)
    """

    def __init__(self, width: int = 100, refresh_interval: float = 1.0):
        """Initialize the dashboard formatter.

        Args:
            width: Dashboard width in characters.
            refresh_interval: Suggested refresh interval in seconds.
        """
        self._width = width
        self._refresh_interval = refresh_interval

    @staticmethod
    def progress_bar(
        value: float,
        maximum: float = 1.0,
        width: int = 30,
        fill_char: str = PROGRESS_FULL,
        empty_char: str = PROGRESS_EMPTY,
        show_pct: bool = True,
    ) -> str:
        """Create a progress bar string.

        Args:
            value: Current value.
            maximum: Maximum value.
            width: Bar width in characters.
            fill_char: Character for filled portion.
            empty_char: Character for empty portion.
            show_pct: Whether to show percentage.

        Returns:
            Progress bar string.
        """
        if maximum <= 0:
            pct = 1.0
        else:
            pct = min(max(value / maximum, 0.0), 1.0)

        filled = int(pct * width)
        bar = fill_char * filled + empty_char * (width - filled)

        if show_pct:
            return f"[{bar}] {pct * 100:6.2f}%"
        return f"[{bar}]"

    @staticmethod
    def sparkline(values: List[float], width: int = 40) -> str:
        """Create a sparkline from values.

        Args:
            values: Data values.
            width: Sparkline width.

        Returns:
            Sparkline string.
        """
        if not values:
            return " " * width

        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
            values = sampled

        chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return chars[0] * len(values)

        result = []
        for v in values:
            idx = int((v - v_min) / (v_max - v_min) * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            result.append(chars[idx])
        return "".join(result)

    @staticmethod
    def heatmap(
        values: List[List[float]],
        width_chars: int = 60,
        height_chars: int = 15,
        col_labels: Optional[List[str]] = None,
        row_labels: Optional[List[str]] = None,
    ) -> str:
        """Create a text-based heatmap.

        Args:
            values: 2D array of float values (0-1 normalized or auto-normalized).
            width_chars: Maximum width.
            height_chars: Maximum height.
            col_labels: Optional column labels.
            row_labels: Optional row labels.

        Returns:
            Multi-line heatmap string.
        """
        if not values or not values[0]:
            return "(empty)"

        # Auto-normalize
        flat = [v for row in values for v in row]
        v_min = min(flat)
        v_max = max(flat)
        if v_max == v_min:
            v_max = v_min + 1.0

        # Downsample if needed
        rows = len(values)
        cols = len(values[0])
        row_label_width = max((len(l) for l in row_labels), default=0) if row_labels else 0

        available_width = width_chars - row_label_width - 3
        available_height = height_chars - 2

        row_step = max(1, rows // available_height) if rows > available_height else 1
        col_step = max(1, cols // available_width) if cols > available_width else 1

        block_chars = [" ", "░", "▒", "▓", "█"]
        lines = []

        # Header with column labels
        header = " " * (row_label_width + 1)
        if col_labels:
            for c in range(0, cols, col_step):
                label = col_labels[c] if c < len(col_labels) else ""
                header += label[:2].ljust(2)
        lines.append(header)

        for r in range(0, rows, row_step):
            line = ""
            if row_labels and r < len(row_labels):
                line = row_labels[r].ljust(row_label_width) + " "
            else:
                line = " " * (row_label_width + 1)

            for c in range(0, cols, col_step):
                val = values[r][c]
                norm = (val - v_min) / (v_max - v_min)
                idx = int(norm * (len(block_chars) - 1))
                idx = max(0, min(len(block_chars) - 1, idx))
                line += block_chars[idx] * 2
            lines.append(line)

        return "\n".join(lines)

    def format_dashboard(
        self,
        tracker: Optional[TrainingTracker] = None,
        throughput: Optional[ThroughputMonitor] = None,
        grad_monitor: Optional[GradientMonitor] = None,
        title: str = "Nexus LLM Training Dashboard",
        extra_sections: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Format a complete training dashboard.

        Args:
            tracker: TrainingTracker with step data.
            throughput: ThroughputMonitor with throughput data.
            grad_monitor: GradientMonitor with gradient data.
            title: Dashboard title.
            extra_sections: Additional (section_title, content) pairs.

        Returns:
            Multi-line dashboard string.
        """
        w = self._width
        sections = []

        # Title bar
        sections.append(f"{BOX_TL}{BOX_H * (w - 2)}{BOX_TR}")
        title_padded = title.center(w - 4)
        sections.append(f"{BOX_V} {title_padded} {BOX_V}")
        sections.append(f"{BOX_V}{BOX_H * (w - 2)}{BOX_V}")

        # Training progress section
        if tracker is not None:
            latest = tracker.get_latest()
            stats = tracker.compute_statistics()
            progress_section = self._format_progress_section(latest, stats, w)
            sections.append(progress_section)

        # Throughput section
        if throughput is not None:
            tp_report = throughput.report()
            for line in tp_report.split("\n"):
                padded = line[: w - 2].ljust(w - 2)
                sections.append(f"{BOX_V}{padded}{BOX_V}")
            sections.append(f"{BOX_V}{BOX_H * (w - 2)}{BOX_V}")

        # Gradient health section
        if grad_monitor is not None:
            health = grad_monitor.check_health()
            for line in health.summary().split("\n"):
                padded = line[: w - 2].ljust(w - 2)
                sections.append(f"{BOX_V}{padded}{BOX_V}")
            sections.append(f"{BOX_V}{BOX_H * (w - 2)}{BOX_V}")

        # Loss sparkline
        if tracker is not None:
            steps, losses = tracker.get_losses()
            if losses:
                sparkline = self.sparkline(losses[-w + 10:])
                sparkline_line = f"  Loss: {sparkline}"
                padded = sparkline_line[: w - 2].ljust(w - 2)
                sections.append(f"{BOX_V}{padded}{BOX_V}")

        # Extra sections
        if extra_sections:
            for sec_title, sec_content in extra_sections:
                sections.append(f"{BOX_V}{BOX_H * (w - 2)}{BOX_V}")
                header = f"  {sec_title}"
                padded = header[: w - 2].ljust(w - 2)
                sections.append(f"{BOX_V}{padded}{BOX_V}")
                for line in sec_content.split("\n"):
                    padded = f"  {line}"[: w - 2].ljust(w - 2)
                    sections.append(f"{BOX_V}{padded}{BOX_V}")

        # Bottom border
        sections.append(f"{BOX_BL}{BOX_H * (w - 2)}{BOX_BR}")

        return "\n".join(sections)

    def _format_progress_section(
        self,
        latest: Optional[StepRecord],
        stats: TrainingStatistics,
        width: int,
    ) -> List[str]:
        """Format the training progress section.

        Args:
            latest: Latest step record.
            stats: Training statistics.
            width: Dashboard width.

        Returns:
            List of formatted lines.
        """
        lines = []

        # Step info
        step_str = f"  Step: {stats.total_steps}"
        if stats.total_epochs > 0:
            step_str += f" | Epoch: {stats.total_epochs}"

        padded = step_str[: width - 2].ljust(width - 2)
        lines.append(f"{BOX_V}{padded}{BOX_V}")

        # Loss info
        if stats.final_loss is not None:
            loss_str = (
                f"  Loss: {stats.final_loss:.6f} "
                f"(best: {stats.best_loss:.6f} @ step {stats.best_loss_step})"
            )
            trend_indicator = {
                "decreasing": f" {ARROW_DOWN} {STATUS_OK}",
                "increasing": f" {ARROW_UP} {STATUS_WARN}",
                "stable": f" {ARROW_RIGHT}",
            }.get(stats.loss_trend, "")
            loss_str += trend_indicator
            padded = loss_str[: width - 2].ljust(width - 2)
            lines.append(f"{BOX_V}{padded}{BOX_V}")

        # Learning rate
        if stats.avg_lr > 0:
            lr_str = f"  LR: {stats.avg_lr:.2e}"
            if latest and latest.learning_rate is not None:
                lr_str = f"  LR: {latest.learning_rate:.2e}"
            padded = lr_str[: width - 2].ljust(width - 2)
            lines.append(f"{BOX_V}{padded}{BOX_V}")

        # Throughput
        if stats.avg_tokens_per_sec > 0:
            tp_str = f"  Throughput: {stats.avg_tokens_per_sec:.0f} tok/s"
            if stats.avg_samples_per_sec > 0:
                tp_str += f" | {stats.avg_samples_per_sec:.1f} samples/s"
            padded = tp_str[: width - 2].ljust(width - 2)
            lines.append(f"{BOX_V}{padded}{BOX_V}")

        # GPU
        if stats.peak_gpu_memory > 0:
            gpu_str = f"  GPU Memory: {stats.avg_gpu_memory:.1f} GB (peak: {stats.peak_gpu_memory:.1f} GB)"
            padded = gpu_str[: width - 2].ljust(width - 2)
            lines.append(f"{BOX_V}{padded}{BOX_V}")

        # Gradient norm
        if stats.avg_grad_norm > 0:
            gn_str = f"  Grad Norm: {stats.avg_grad_norm:.4f}"
            padded = gn_str[: width - 2].ljust(width - 2)
            lines.append(f"{BOX_V}{padded}{BOX_V}")

        # Time
        hours = int(stats.total_time // 3600)
        minutes = int((stats.total_time % 3600) // 60)
        time_str = f"  Elapsed: {hours}h {minutes}m"
        if stats.loss_reduction_pct != 0:
            time_str += f" | Loss reduction: {stats.loss_reduction_pct:.1f}%"
        padded = time_str[: width - 2].ljust(width - 2)
        lines.append(f"{BOX_V}{padded}{BOX_V}")

        lines.append(f"{BOX_V}{BOX_H * (width - 2)}{BOX_V}")
        return lines


# ============================================================================
# ExperimentComparison
# ============================================================================

class ExperimentComparison:
    """
    Compare multiple experiment runs.

    Provides side-by-side comparison of training metrics across
    different experiment configurations.

    Example:
        comp = ExperimentComparison()
        comp.add_run("exp_1", tracker_1)
        comp.add_run("exp_2", tracker_2)
        report = comp.compare()
        print(report)
    """

    def __init__(self):
        """Initialize the experiment comparator."""
        self._runs: Dict[str, TrainingTracker] = {}
        self._run_stats: Dict[str, TrainingStatistics] = {}

    def add_run(self, name: str, tracker: TrainingTracker) -> None:
        """Add an experiment run.

        Args:
            name: Experiment name.
            tracker: TrainingTracker with the run's data.
        """
        self._runs[name] = tracker
        self._run_stats[name] = tracker.compute_statistics()

    def remove_run(self, name: str) -> None:
        """Remove an experiment run.

        Args:
            name: Experiment name to remove.
        """
        self._runs.pop(name, None)
        self._run_stats.pop(name, None)

    def compare(self, metric: Optional[str] = None) -> str:
        """Generate a comparison report of all runs.

        Args:
            metric: Specific metric to focus on. None for all.

        Returns:
            Multi-line comparison report.
        """
        if not self._runs:
            return "(no experiment runs added)"

        lines = []
        run_names = list(self._runs.keys())
        col_width = max(18, max(len(n) for n in run_names) + 2)
        total_width = col_width + 16 * len(run_names) + 4

        lines.append(f"{BOX_TL}{'Experiment Comparison':^{total_width - 2}}{BOX_TR}")

        # Header
        header = f"{'Metric':<16}"
        for name in run_names:
            header += f"{name:>{col_width}}"
        lines.append(f"{BOX_V}{header}{BOX_V}")
        sep = BOX_H * 16
        for _ in run_names:
            sep += BOX_CROSS + BOX_H * col_width
        lines.append(f"{BOX_V}{sep}{BOX_V}")

        # Metrics
        metrics_to_show = [
            ("Steps", lambda s: str(s.total_steps)),
            ("Epochs", lambda s: str(s.total_epochs)),
            ("Final Loss", lambda s: f"{s.final_loss:.6f}" if s.final_loss else "N/A"),
            ("Best Loss", lambda s: f"{s.best_loss:.6f}" if s.best_loss else "N/A"),
            ("Loss Std", lambda s: f"{s.loss_std:.6f}"),
            ("Loss Trend", lambda s: s.loss_trend),
            ("Reduction %", lambda s: f"{s.loss_reduction_pct:.1f}"),
            ("Avg LR", lambda s: f"{s.avg_lr:.2e}"),
            ("Avg Grad Norm", lambda s: f"{s.avg_grad_norm:.4f}"),
            ("Avg tok/s", lambda s: f"{s.avg_tokens_per_sec:.0f}"),
            ("Avg samp/s", lambda s: f"{s.avg_samples_per_sec:.1f}"),
            ("Peak GPU (GB)", lambda s: f"{s.peak_gpu_memory:.1f}"),
            ("Total Tokens", lambda s: f"{s.total_tokens_processed:,}"),
            ("Time (s)", lambda s: f"{s.total_time:.0f}"),
        ]

        if metric is not None:
            metrics_to_show = [m for m in metrics_to_show if metric.lower() in m[0].lower()]

        for metric_name, extractor in metrics_to_show:
            row = f"{metric_name:<16}"
            for name in run_names:
                stats = self._run_stats[name]
                val = extractor(stats)
                row += f"{val:>{col_width}}"
            lines.append(f"{BOX_V}{row}{BOX_V}")

        lines.append(BOX_BL + BOX_H * (total_width - 2) + BOX_BR)

        # Winner per metric
        if len(self._runs) > 1:
            lines.append("")
            lines.append("Best per metric:")
            win_metrics = [
                ("Best Loss", "best_loss", True),
                ("Loss Std (lowest)", "loss_std", True),
                ("Avg tok/s", "avg_tokens_per_sec", False),
                ("Peak GPU (lowest)", "peak_gpu_memory", True),
                ("Total Time (lowest)", "total_time", True),
            ]
            for label, attr, lower_is_better in win_metrics:
                best_name = None
                best_val = None
                for name, stats in self._run_stats.items():
                    val = getattr(stats, attr, None)
                    if val is None:
                        continue
                    if best_val is None:
                        best_val = val
                        best_name = name
                    elif lower_is_better and val < best_val:
                        best_val = val
                        best_name = name
                    elif not lower_is_better and val > best_val:
                        best_val = val
                        best_name = name
                if best_name:
                    lines.append(f"  {label}: {best_name} ({best_val:.4f})")

        return "\n".join(lines)

    def loss_comparison_chart(self, width: int = 80, height: int = 15) -> str:
        """Generate a multi-run loss comparison chart.

        Args:
            width: Chart width.
            height: Chart height.

        Returns:
            Multi-line chart string.
        """
        runs_data = {}
        for name, tracker in self._runs.items():
            _, losses = tracker.get_losses()
            if losses:
                runs_data[name] = losses

        if not runs_data:
            return "(no loss data)"

        visualizer = LossVisualizer(width=width, height=height)
        return visualizer.plot_multi_run(runs_data)

    def get_ranking(self, metric: str = "best_loss", lower_is_better: bool = True) -> List[Tuple[str, float]]:
        """Rank experiments by a specific metric.

        Args:
            metric: Attribute name from TrainingStatistics.
            lower_is_better: Whether lower values are better.

        Returns:
            List of (name, value) sorted by ranking.
        """
        rankings = []
        for name, stats in self._run_stats.items():
            val = getattr(stats, metric, None)
            if val is not None:
                rankings.append((name, val))

        rankings.sort(key=lambda x: x[1], reverse=not lower_is_better)
        return rankings
