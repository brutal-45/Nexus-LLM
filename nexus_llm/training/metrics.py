"""Training metrics: loss, perplexity, accuracy, learning rate, throughput tracking."""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Container for a single metrics snapshot."""
    step: int = 0
    loss: float = 0.0
    perplexity: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0
    epoch: int = 0
    timestamp: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "perplexity": self.perplexity,
            "accuracy": self.accuracy,
            "learning_rate": self.learning_rate,
            "throughput": self.throughput,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "extra": self.extra,
        }


class MetricsTracker:
    """Tracks and computes training metrics including loss, perplexity, accuracy, and throughput."""

    def __init__(self):
        self.history: List[Metrics] = []
        self._start_time: Optional[float] = None
        self._total_tokens: int = 0
        self._window_size: int = 100
        self._loss_window: List[float] = []
        self._best_loss: Optional[float] = None
        self._best_step: Optional[int] = None

    def start_timer(self):
        """Start the training timer."""
        self._start_time = time.time()

    def stop_timer(self):
        """Stop the training timer."""
        pass

    def log(
        self,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        accuracy: Optional[float] = None,
        perplexity: Optional[float] = None,
        throughput: Optional[float] = None,
        epoch: int = 0,
        tokens: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log metrics for a given step."""
        if loss is not None:
            self._loss_window.append(loss)
            if len(self._loss_window) > self._window_size:
                self._loss_window.pop(0)

            if perplexity is None:
                perplexity = self._compute_perplexity(loss)

            if self._best_loss is None or loss < self._best_loss:
                self._best_loss = loss
                self._best_step = step

        if throughput is None and self._start_time is not None and tokens > 0:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                throughput = tokens / elapsed

        metrics = Metrics(
            step=step,
            loss=loss if loss is not None else 0.0,
            perplexity=perplexity if perplexity is not None else 0.0,
            accuracy=accuracy if accuracy is not None else 0.0,
            learning_rate=learning_rate if learning_rate is not None else 0.0,
            throughput=throughput if throughput is not None else 0.0,
            epoch=epoch,
            timestamp=time.time(),
            extra=extra or {},
        )

        self.history.append(metrics)
        self._total_tokens += tokens

    @staticmethod
    def _compute_perplexity(loss: float) -> float:
        """Compute perplexity from cross-entropy loss."""
        if loss <= 0:
            return float("inf")
        try:
            return math.exp(loss)
        except OverflowError:
            return float("inf")

    def compute_accuracy(self, predictions, references) -> float:
        """Compute token-level accuracy."""
        if len(predictions) != len(references):
            logger.warning("Predictions and references have different lengths.")
            return 0.0

        correct = 0
        total = 0
        for pred, ref in zip(predictions, references):
            if hasattr(pred, "__len__") and hasattr(ref, "__len__"):
                min_len = min(len(pred), len(ref))
                for i in range(min_len):
                    if pred[i] == ref[i]:
                        correct += 1
                    total += 1
            else:
                if pred == ref:
                    correct += 1
                total += 1

        return correct / max(total, 1)

    def get_current_loss(self) -> Optional[float]:
        """Get the most recent loss value."""
        if self.history:
            return self.history[-1].loss
        return None

    def get_smoothed_loss(self) -> Optional[float]:
        """Get the exponentially smoothed loss over the recent window."""
        if not self._loss_window:
            return None
        return sum(self._loss_window) / len(self._loss_window)

    def get_best_loss(self) -> Optional[float]:
        """Get the best (lowest) loss recorded."""
        return self._best_loss

    def get_best_step(self) -> Optional[int]:
        """Get the step at which the best loss was recorded."""
        return self._best_step

    def get_throughput(self) -> float:
        """Compute overall throughput in tokens/second."""
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_tokens / elapsed

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked metrics."""
        if not self.history:
            return {}

        latest = self.history[-1]
        summary = {
            "total_steps": latest.step,
            "final_loss": latest.loss,
            "best_loss": self._best_loss,
            "best_step": self._best_step,
            "smoothed_loss": self.get_smoothed_loss(),
            "final_perplexity": latest.perplexity,
            "final_accuracy": latest.accuracy,
            "final_learning_rate": latest.learning_rate,
            "throughput_tokens_per_sec": self.get_throughput(),
            "total_tokens": self._total_tokens,
            "num_logged_points": len(self.history),
        }

        if self._start_time is not None:
            summary["total_time_seconds"] = time.time() - self._start_time

        return summary

    def get_loss_history(self) -> List[float]:
        """Get the full loss history."""
        return [m.loss for m in self.history if m.loss is not None]

    def get_lr_history(self) -> List[float]:
        """Get the full learning rate history."""
        return [m.learning_rate for m in self.history]

    def reset(self):
        """Reset the metrics tracker."""
        self.history = []
        self._start_time = None
        self._total_tokens = 0
        self._loss_window = []
        self._best_loss = None
        self._best_step = None

    def export(self, path: str):
        """Export metrics history to a JSON file."""
        import json
        data = [m.to_dict() for m in self.history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported metrics to {path}")

    def print_summary(self):
        """Print a formatted summary to the logger."""
        summary = self.get_summary()
        if not summary:
            logger.info("No metrics recorded yet.")
            return
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
