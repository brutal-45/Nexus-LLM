"""Nexus-LLM Performance Optimizer.

Provides the Optimizer class that monitors performance metrics and applies
optimization strategies to improve throughput and reduce latency.
"""

import enum
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OptimizationLevel(enum.Enum):
    """Optimization aggressiveness level."""

    NONE = "none"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceMetrics:
    """Snapshot of performance metrics.

    Attributes:
        avg_latency_ms: Average latency in milliseconds.
        p50_latency_ms: 50th percentile latency.
        p95_latency_ms: 95th percentile latency.
        p99_latency_ms: 99th percentile latency.
        throughput_per_sec: Operations per second.
        error_rate: Error rate as a fraction (0.0 - 1.0).
        memory_usage_mb: Memory usage in MB.
        sample_count: Number of samples in the current window.
    """

    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    sample_count: int = 0


@dataclass
class OptimizationAction:
    """An optimization action taken by the optimizer.

    Attributes:
        name: Name of the optimization.
        description: Human-readable description.
        before: Metric value before the action.
        after: Metric value after the action (if measured).
        timestamp: When the action was taken.
    """

    name: str
    description: str
    before: float = 0.0
    after: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class Optimizer:
    """Performance optimizer that monitors and improves system metrics.

    The Optimizer collects latency and throughput data, computes rolling
    statistics, and applies registered optimization strategies when
    performance degrades beyond configured thresholds.

    Attributes:
        level: Current optimization level.
        metrics: Latest performance metrics snapshot.
    """

    def __init__(
        self,
        level: OptimizationLevel = OptimizationLevel.MODERATE,
        window_size: int = 1000,
        target_latency_ms: float = 200.0,
        target_error_rate: float = 0.01,
    ) -> None:
        self._level = level
        self._window_size = window_size
        self._target_latency_ms = target_latency_ms
        self._target_error_rate = target_error_rate
        self._latencies: deque = deque(maxlen=window_size)
        self._errors: deque = deque(maxlen=window_size)
        self._throughputs: deque = deque(maxlen=window_size)
        self._strategies: Dict[str, Callable[[PerformanceMetrics], Optional[OptimizationAction]]] = {}
        self._actions: List[OptimizationAction] = []
        self._last_optimization_time: float = 0.0
        logger.info(
            "Optimizer initialized: level=%s, target_latency=%.1fms",
            level.value,
            target_latency_ms,
        )

    @property
    def level(self) -> OptimizationLevel:
        """Current optimization level."""
        return self._level

    @property
    def metrics(self) -> PerformanceMetrics:
        """Latest computed performance metrics."""
        return self._compute_metrics()

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds.
        """
        self._latencies.append(latency_ms)

    def record_error(self) -> None:
        """Record an error occurrence."""
        self._errors.append(1.0)

    def record_success(self) -> None:
        """Record a successful operation."""
        self._errors.append(0.0)

    def record_throughput(self, ops_per_sec: float) -> None:
        """Record a throughput measurement.

        Args:
            ops_per_sec: Operations per second.
        """
        self._throughputs.append(ops_per_sec)

    def register_strategy(
        self,
        name: str,
        strategy: Callable[[PerformanceMetrics], Optional[OptimizationAction]],
    ) -> None:
        """Register an optimization strategy.

        Args:
            name: Unique name for the strategy.
            strategy: Callable that receives metrics and optionally returns
                     an OptimizationAction if it was applied.
        """
        self._strategies[name] = strategy
        logger.debug("Registered optimization strategy: %s", name)

    def set_level(self, level: OptimizationLevel) -> None:
        """Change the optimization level.

        Args:
            level: New optimization level.
        """
        self._level = level
        logger.info("Optimization level changed to: %s", level.value)

    def optimize(self) -> List[OptimizationAction]:
        """Run all optimization strategies if performance is degrading.

        Only runs strategies if at least min_samples data points have been
        collected and the cooldown period has elapsed.

        Returns:
            List of OptimizationActions that were applied.
        """
        if self._level == OptimizationLevel.NONE:
            return []

        min_samples = 10
        if len(self._latencies) < min_samples:
            logger.debug("Not enough samples for optimization (%d/%d)", len(self._latencies), min_samples)
            return []

        # Cooldown between optimization rounds
        cooldown_seconds = {
            OptimizationLevel.CONSERVATIVE: 60.0,
            OptimizationLevel.MODERATE: 30.0,
            OptimizationLevel.AGGRESSIVE: 10.0,
        }.get(self._level, 30.0)

        now = time.time()
        if now - self._last_optimization_time < cooldown_seconds:
            return []

        metrics = self._compute_metrics()
        applied: List[OptimizationAction] = []

        # Check if optimization is needed
        needs_optimization = (
            metrics.avg_latency_ms > self._target_latency_ms
            or metrics.error_rate > self._target_error_rate
        )

        if not needs_optimization and self._level != OptimizationLevel.AGGRESSIVE:
            return []

        for name, strategy in self._strategies.items():
            try:
                action = strategy(metrics)
                if action is not None:
                    applied.append(action)
                    self._actions.append(action)
                    logger.info("Applied optimization: %s - %s", name, action.description)
            except Exception as exc:
                logger.warning("Strategy '%s' failed: %s", name, exc)

        self._last_optimization_time = now
        return applied

    def get_actions(self, limit: int = 50) -> List[OptimizationAction]:
        """Return recent optimization actions.

        Args:
            limit: Maximum number of actions to return.

        Returns:
            List of recent OptimizationActions.
        """
        return self._actions[-limit:]

    def reset(self) -> None:
        """Clear all collected metrics and action history."""
        self._latencies.clear()
        self._errors.clear()
        self._throughputs.clear()
        self._actions.clear()
        logger.info("Optimizer metrics reset")

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the optimizer."""
        metrics = self._compute_metrics()
        return {
            "status": "healthy",
            "level": self._level.value,
            "sample_count": metrics.sample_count,
            "avg_latency_ms": metrics.avg_latency_ms,
            "error_rate": metrics.error_rate,
            "actions_taken": len(self._actions),
        }

    def _compute_metrics(self) -> PerformanceMetrics:
        """Compute performance metrics from collected data."""
        metrics = PerformanceMetrics()

        if not self._latencies:
            return metrics

        latencies = sorted(self._latencies)
        metrics.sample_count = len(latencies)
        metrics.avg_latency_ms = sum(latencies) / len(latencies)
        metrics.p50_latency_ms = self._percentile(latencies, 50)
        metrics.p95_latency_ms = self._percentile(latencies, 95)
        metrics.p99_latency_ms = self._percentile(latencies, 99)

        if self._errors:
            metrics.error_rate = sum(self._errors) / len(self._errors)

        if self._throughputs:
            metrics.throughput_per_sec = sum(self._throughputs) / len(self._throughputs)

        return metrics

    @staticmethod
    def _percentile(sorted_data: List[float], pct: int) -> float:
        """Compute a percentile from sorted data."""
        if not sorted_data:
            return 0.0
        idx = (pct / 100.0) * (len(sorted_data) - 1)
        lower = int(idx)
        upper = lower + 1
        if upper >= len(sorted_data):
            return sorted_data[-1]
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac
