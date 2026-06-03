"""Metrics collection for Nexus-LLM monitoring."""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates runtime metrics for Nexus-LLM.

    Tracks key operational metrics including request counts, latency,
    tokens generated, memory usage, and error rates. Supports tagging
    for dimensional analysis and time-windowed aggregation.
    """

    # Well-known metric names
    REQUEST_COUNT = "request_count"
    LATENCY = "latency"
    TOKENS_GENERATED = "tokens_generated"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"

    def __init__(self, max_points_per_metric: int = 10000) -> None:
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._max_points = max_points_per_metric
        self._lock = threading.Lock()

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value with optional tags.

        Args:
            name: Metric name (e.g. 'request_count', 'latency').
            value: Numeric value of the metric.
            tags: Optional key-value tags for dimensional analysis.
        """
        point = MetricPoint(
            timestamp=time.time(),
            value=float(value),
            tags=tags or {},
        )
        with self._lock:
            store = self._metrics[name]
            store.append(point)
            # Evict oldest points if we exceed the cap
            if len(store) > self._max_points:
                excess = len(store) - self._max_points
                del store[:excess]

    def get_metric(self, name: str) -> List[Tuple[float, float]]:
        """Retrieve all recorded values for a metric.

        Args:
            name: Metric name to look up.

        Returns:
            List of (timestamp, value) tuples sorted chronologically.
        """
        with self._lock:
            return [(p.timestamp, p.value) for p in self._metrics.get(name, [])]

    def get_aggregated(
        self,
        name: str,
        window: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute aggregated statistics for a metric.

        Args:
            name: Metric name to aggregate.
            window: Optional time window in seconds. Only points within
                    the window are considered. ``None`` means all points.

        Returns:
            Dictionary with keys: count, min, max, avg, p50, p95, p99.
            Returns zeros if no data is available.
        """
        with self._lock:
            points = list(self._metrics.get(name, []))

        if window is not None:
            cutoff = time.time() - window
            points = [p for p in points if p.timestamp >= cutoff]

        if not points:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        values = sorted(p.value for p in points)
        count = len(values)

        def _percentile(sorted_vals: List[float], pct: float) -> float:
            if not sorted_vals:
                return 0.0
            idx = (pct / 100.0) * (len(sorted_vals) - 1)
            lower = int(idx)
            upper = lower + 1
            if upper >= len(sorted_vals):
                return sorted_vals[-1]
            fraction = idx - lower
            return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])

        return {
            "count": count,
            "min": values[0],
            "max": values[-1],
            "avg": sum(values) / count,
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "p99": _percentile(values, 99),
        }

    def list_metrics(self) -> List[str]:
        """Return names of all metrics that have been recorded."""
        with self._lock:
            return list(self._metrics.keys())

    def clear(self, name: Optional[str] = None) -> None:
        """Clear metric data.

        Args:
            name: If provided, only clear that metric. Otherwise clear all.
        """
        with self._lock:
            if name is not None:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear()

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Convenience method to increment a counter metric.

        Args:
            name: Metric name.
            value: Amount to add (default 1.0).
            tags: Optional tags.
        """
        self.record_metric(name, value, tags)
