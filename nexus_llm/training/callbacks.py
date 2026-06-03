"""Training callbacks for progress tracking and logging.

Provides Rich-based progress display, file logging, early stopping,
checkpoint management, and metrics collection.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_llm.core.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback implementations
# ---------------------------------------------------------------------------

class _ProgressCallback:
    """Rich-based progress bar callback for HuggingFace Trainer."""

    def __init__(self) -> None:
        self._progress = None
        self._task_id = None
        self._start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
            self._progress.start()
            self._task_id = self._progress.add_task("Training", total=state.max_steps)
            self._start_time = time.monotonic()
        except ImportError:
            logger.info("Rich not available; progress display disabled.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, completed=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self._progress:
            if self._task_id is not None:
                self._progress.update(self._task_id, completed=state.global_step)
            self._progress.stop()
            elapsed = time.monotonic() - self._start_time if self._start_time else 0
            logger.info("Training finished in %.1f seconds", elapsed)


class _FileLoggingCallback:
    """Logs training metrics to a JSONL file."""

    def __init__(self, log_file: str) -> None:
        self._log_path = Path(log_file)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "global_step": state.global_step,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))},
        }
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


class _EarlyStoppingCallback:
    """Simple early stopping based on evaluation loss."""

    def __init__(self, patience: int = 3, threshold: float = 0.001) -> None:
        self._patience = patience
        self._threshold = threshold
        self._best_loss: Optional[float] = None
        self._wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if self._best_loss is None or eval_loss < self._best_loss - self._threshold:
            self._best_loss = eval_loss
            self._wait = 0
        else:
            self._wait += 1
            logger.info(
                "Early stopping: no improvement for %d/%d evals (best=%.4f, current=%.4f)",
                self._wait, self._patience, self._best_loss, eval_loss,
            )
            if self._wait >= self._patience:
                logger.info("Early stopping triggered!")
                control.should_training_stop = True


class _MetricsTrackerCallback:
    """Collects training and evaluation metrics over time."""

    def __init__(self) -> None:
        self.train_metrics: List[Dict[str, Any]] = []
        self.eval_metrics: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, **logs}
        if any(k.startswith("eval_") for k in logs):
            self.eval_metrics.append(entry)
        else:
            self.train_metrics.append(entry)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of collected metrics."""
        summary: Dict[str, Any] = {
            "total_train_entries": len(self.train_metrics),
            "total_eval_entries": len(self.eval_metrics),
        }
        if self.train_metrics:
            losses = [m.get("loss") for m in self.train_metrics if "loss" in m]
            if losses:
                summary["train_loss_min"] = min(losses)
                summary["train_loss_max"] = max(losses)
                summary["train_loss_last"] = losses[-1]
        if self.eval_metrics:
            eval_losses = [m.get("eval_loss") for m in self.eval_metrics if "eval_loss" in m]
            if eval_losses:
                summary["eval_loss_min"] = min(eval_losses)
                summary["eval_loss_best"] = min(eval_losses)
        return summary


class _CheckpointCallback:
    """Ensures checkpoints are saved at configured intervals."""

    def __init__(self, save_steps: int = 500, save_total_limit: int = 3) -> None:
        self._save_steps = save_steps
        self._save_total_limit = save_total_limit

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self._save_steps == 0 and state.global_step > 0:
            control.should_save = True


# ---------------------------------------------------------------------------
# TrainingCallbacks – aggregator
# ---------------------------------------------------------------------------

class TrainingCallbacks:
    """Aggregates and builds training callbacks.

    Usage::

        cb = TrainingCallbacks(settings)
        hf_callbacks = cb.build_callbacks()
        # pass hf_callbacks to HuggingFace Trainer
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        self._progress = _ProgressCallback()
        self._file_logger: Optional[_FileLoggingCallback] = None
        self._early_stopper: Optional[_EarlyStoppingCallback] = None
        self._metrics_tracker = _MetricsTrackerCallback()
        self._checkpoint: Optional[_CheckpointCallback] = None

        # Set up file logger
        log_dir = Path(self._settings.log_file).parent / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._file_logger = _FileLoggingCallback(str(log_dir / "training_metrics.jsonl"))

        # Set up early stopping (patience=3 by default)
        self._early_stopper = _EarlyStoppingCallback(patience=3)

        # Set up checkpoint callback
        self._checkpoint = _CheckpointCallback(
            save_steps=self._settings.training.save_steps,
            save_total_limit=3,
        )

    @property
    def metrics_tracker(self) -> _MetricsTrackerCallback:
        """Access the metrics tracker for post-training analysis."""
        return self._metrics_tracker

    def build_callbacks(self) -> List[Any]:
        """Build and return a list of HuggingFace-compatible callbacks.

        Returns:
            List of callback objects to pass to ``Trainer(callbacks=...)``.
        """
        try:
            from transformers import TrainerCallback
        except ImportError:
            logger.warning("transformers not installed; returning raw callback objects")
            return [
                self._progress,
                self._file_logger,
                self._early_stopper,
                self._metrics_tracker,
                self._checkpoint,
            ]

        # Wrap each internal callback as a proper TrainerCallback
        callbacks: List[TrainerCallback] = []
        for cb in (
            self._progress,
            self._file_logger,
            self._early_stopper,
            self._metrics_tracker,
            self._checkpoint,
        ):
            callbacks.append(_CallbackAdapter(cb))

        return callbacks

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a summary of collected training metrics."""
        return self._metrics_tracker.get_summary()


# ---------------------------------------------------------------------------
# Adapter to make our callbacks compatible with HuggingFace TrainerCallback
# ---------------------------------------------------------------------------

class _CallbackAdapter:
    """Adapts our lightweight callbacks to the HuggingFace TrainerCallback interface."""

    def __init__(self, callback: Any) -> None:
        self._callback = callback

    def on_train_begin(self, args, state, control, **kwargs):
        if hasattr(self._callback, "on_train_begin"):
            self._callback.on_train_begin(args, state, control, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(self._callback, "on_train_end"):
            self._callback.on_train_end(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(self._callback, "on_log"):
            self._callback.on_log(args, state, control, logs=logs, **kwargs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if hasattr(self._callback, "on_evaluate"):
            self._callback.on_evaluate(args, state, control, metrics=metrics, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self._callback, "on_step_end"):
            self._callback.on_step_end(args, state, control, **kwargs)
