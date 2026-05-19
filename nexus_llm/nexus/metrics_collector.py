"""Nexus-LLM Metrics Collector.

Provides collection, aggregation, and reporting of system metrics
including request counts, latency measurements, token usage,
and error rates.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point.

    Attributes:
        name: Metric name.
        value: Metric value.
        timestamp: When the metric was recorded.
        tags: Optional tags for filtering/grouping.
    """

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates system metrics.

    Supports counters, gauges, and histograms with configurable
    retention periods and aggregation windows.

    Example::

        collector = MetricsCollector()
        collector.increment("requests.total")
        collector.observe("request.duration_ms", 150.0)
        collector.set_gauge("active_connections", 42)
        summary = collector.get_summary()
    """

    def __init__(self, retention_seconds: float = 3600.0) -> None:
        """Initialize the MetricsCollector.

        Args:
            retention_seconds: How long to retain metric data points.
        """
        self._retention_seconds = retention_seconds
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._lock = threading.RLock()
        logger.debug("MetricsCollector initialized with retention=%.0fs", retention_seconds)

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.

        Args:
            name: Counter name.
            value: Amount to increment by.
            tags: Optional tags.
        """
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value.

        Args:
            name: Gauge name.
            value: Current value.
            tags: Optional tags.
        """
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record an observation for histogram tracking.

        Args:
            name: Histogram name.
            value: Observed value.
            tags: Optional tags.
        """
        with self._lock:
            point = MetricPoint(name=name, value=value, tags=tags or {})
            self._histograms[name].append(point)

    def get_counter(self, name: str) -> float:
        """Get the current value of a counter.

        Args:
            name: Counter name.

        Returns:
            Current counter value.
        """
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get the current value of a gauge.

        Args:
            name: Gauge name.

        Returns:
            Current gauge value, or None if not set.
        """
        return self._gauges.get(name)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a histogram metric.

        Args:
            name: Histogram name.

        Returns:
            Dictionary with min, max, mean, count, and percentiles.
        """
        with self._lock:
            points = self._histograms.get(name, [])
            if not points:
                return {"count": 0}

            values = sorted(p.value for p in points)
            count = len(values)
            mean = sum(values) / count
            p50 = values[int(count * 0.5)]
            p95 = values[int(count * 0.95)] if count > 1 else values[0]
            p99 = values[int(count * 0.99)] if count > 1 else values[0]

            return {
                "count": count,
                "min": values[0],
                "max": values[-1],
                "mean": mean,
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "sum": sum(values),
            }

    def get_all_counters(self) -> Dict[str, float]:
        """Get all counter values."""
        return dict(self._counters)

    def get_all_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        return dict(self._gauges)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dictionary with counters, gauges, and histogram summaries.
        """
        with self._lock:
            histograms = {}
            for name in self._histograms:
                histograms[name] = self.get_histogram_stats(name)

            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histograms,
                "timestamp": time.time(),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
        logger.info("Metrics reset")

    def cleanup(self) -> int:
        """Remove expired metric data points.

        Returns:
            Number of expired points removed.
        """
        now = time.time()
        cutoff = now - self._retention_seconds
        removed = 0

        with self._lock:
            for name in list(self._histograms.keys()):
                points = self._histograms[name]
                original_len = len(points)
                self._histograms[name] = [p for p in points if p.timestamp >= cutoff]
                removed += original_len - len(self._histograms[name])

        if removed:
            logger.debug("Cleaned up %d expired metric points", removed)
        return removed

    def time_it(self, name: str):
        """Context manager/decorator for timing operations.

        Args:
            name: Histogram name for the timing metric.

        Returns:
            A context manager that records duration.

        Example::

            with collector.time_it("request.duration_ms"):
                # do work
                pass
        """
        class _Timer:
            def __init__(self, collector, metric_name):
                self._collector = collector
                self._name = metric_name
                self._start = 0.0

            def __enter__(self):
                self._start = time.perf_counter()
                return self

            def __exit__(self, *args):
                duration_ms = (time.perf_counter() - self._start) * 1000
                self._collector.observe(self._name, duration_ms)
                self._collector.increment(f"{self._name}.count")

        return _Timer(self, name)
