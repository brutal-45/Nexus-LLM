"""Backend metrics for Nexus-LLM.

Provides Prometheus-style metrics, request counters, latency histograms,
and throughput gauges for monitoring the inference backend.
"""

import time
import threading
import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """A single metric sample."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """A monotonically increasing counter (Prometheus-style).

    Usage: track total requests, total tokens generated, total errors, etc.
    """

    def __init__(self, name: str, description: str = "", label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.RLock()
        self._created_at = time.time()

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter by the given amount."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._values[key] += amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current counter value."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def get_all(self) -> Dict[Tuple[str, ...], float]:
        """Get all labeled values."""
        with self._lock:
            return dict(self._values)

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if labels is None:
            return ()
        return tuple(sorted(labels.items()))

    def to_prometheus(self) -> str:
        """Export in Prometheus text format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} counter"]
        for key, value in self._values.items():
            if key:
                label_str = ",".join(f'{k}="{v}"' for k, v in key)
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Gauge:
    """A value that can go up and down (Prometheus-style).

    Usage: track current memory usage, active requests, queue depth, etc.
    """

    def __init__(self, name: str, description: str = "", label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.RLock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge to a specific value."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        self.inc(-amount, labels)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current gauge value."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if labels is None:
            return ()
        return tuple(sorted(labels.items()))

    def to_prometheus(self) -> str:
        """Export in Prometheus text format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} gauge"]
        for key, value in self._values.items():
            if key:
                label_str = ",".join(f'{k}="{v}"' for k, v in key)
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Histogram:
    """A histogram for observing value distributions (Prometheus-style).

    Usage: track request latency, token generation time, batch sizes, etc.
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.label_names = label_names or []
        self._counts: Dict[Tuple[str, ...], Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._total_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._lock = threading.RLock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value and add it to the histogram."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._sums[key] += value
            self._total_counts[key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1
            self._counts[key][float("inf")] += 1

    def observe_many(self, values: List[float], labels: Optional[Dict[str, str]] = None) -> None:
        """Observe multiple values."""
        for v in values:
            self.observe(v, labels)

    def get_percentile(self, percentile: float, labels: Optional[Dict[str, str]] = None) -> float:
        """Compute an approximate percentile from the histogram."""
        key = self._labels_to_key(labels)
        total = self._total_counts.get(key, 0)
        if total == 0:
            return 0.0

        target_count = math.ceil(total * percentile / 100.0)
        cumulative = 0

        with self._lock:
            for i, bucket in enumerate(self.buckets):
                cumulative += self._counts[key].get(bucket, 0)
                if cumulative >= target_count:
                    if i == 0:
                        return bucket
                    prev_bucket = self.buckets[i - 1]
                    prev_cumulative = cumulative - self._counts[key].get(bucket, 0)
                    fraction = (target_count - prev_cumulative) / max(1, self._counts[key].get(bucket, 0))
                    return prev_bucket + fraction * (bucket - prev_bucket)

        return self.buckets[-1] if self.buckets else 0.0

    def get_stats(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._labels_to_key(labels)
        total = self._total_counts.get(key, 0)
        total_sum = self._sums.get(key, 0.0)
        mean = total_sum / total if total > 0 else 0.0

        return {
            "count": total,
            "sum": total_sum,
            "mean": mean,
            "p50": self.get_percentile(50, labels),
            "p90": self.get_percentile(90, labels),
            "p95": self.get_percentile(95, labels),
            "p99": self.get_percentile(99, labels),
        }

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        if labels is None:
            return ()
        return tuple(sorted(labels.items()))

    def to_prometheus(self) -> str:
        """Export in Prometheus text format."""
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} histogram"]

        for key in self._counts:
            label_prefix = ""
            if key:
                label_str = ",".join(f'{k}="{v}"' for k, v in key)
                label_prefix = f"{{{label_str},"

            cumulative = 0
            for bucket in sorted(self.buckets):
                cumulative += self._counts[key].get(bucket, 0)
                le_label = f'{label_prefix}le="{bucket}"' if key else f'{{le="{bucket}"'
                lines.append(f"{self.name}_bucket{le_label}}} {cumulative}")

            inf_count = self._total_counts.get(key, 0)
            le_label = f'{label_prefix}le="+Inf"' if key else '{{le="+Inf"'
            lines.append(f"{self.name}_bucket{le_label}}} {inf_count}")

            total = self._total_counts.get(key, 0)
            sum_val = self._sums.get(key, 0.0)
            if key:
                label_str = ",".join(f'{k}="{v}"' for k, v in key)
                lines.append(f"{self.name}_count{{{label_str}}} {total}")
                lines.append(f"{self.name}_sum{{{label_str}}} {sum_val}")
            else:
                lines.append(f"{self.name}_count {total}")
                lines.append(f"{self.name}_sum {sum_val}")

        return "\n".join(lines)


class MetricsRegistry:
    """Central registry for all backend metrics."""

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.RLock()

    def counter(self, name: str, description: str = "", label_names: Optional[List[str]] = None) -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description, label_names)
            return self._counters[name]

    def gauge(self, name: str, description: str = "", label_names: Optional[List[str]] = None) -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, label_names)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        label_names: Optional[List[str]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets, label_names)
            return self._histograms[name]

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        parts = []
        for counter in self._counters.values():
            parts.append(counter.to_prometheus())
        for gauge in self._gauges.values():
            parts.append(gauge.to_prometheus())
        for histogram in self._histograms.values():
            parts.append(histogram.to_prometheus())
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as a dictionary."""
        result = {
            "counters": {},
            "gauges": {},
            "histograms": {},
        }
        for name, counter in self._counters.items():
            result["counters"][name] = dict(counter.get_all())
        for name, gauge in self._gauges.items():
            result["gauges"][name] = dict(gauge.get_all())
        for name, histogram in self._histograms.items():
            key = ()
            result["histograms"][name] = histogram.get_stats()
        return result


class BackendMetrics:
    """Pre-configured backend metrics for common use cases."""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()

        self.request_count = self.registry.counter(
            "nexus_request_total", "Total number of inference requests", ["model", "status"]
        )
        self.request_latency = self.registry.histogram(
            "nexus_request_latency_seconds", "Request latency in seconds",
            label_names=["model", "endpoint"]
        )
        self.tokens_generated = self.registry.counter(
            "nexus_tokens_generated_total", "Total tokens generated", ["model"]
        )
        self.tokens_per_second = self.registry.gauge(
            "nexus_tokens_per_second", "Current tokens per second", ["model"]
        )
        self.active_requests = self.registry.gauge(
            "nexus_active_requests", "Number of active requests"
        )
        self.queue_depth = self.registry.gauge(
            "nexus_queue_depth", "Current request queue depth"
        )
        self.gpu_memory_used = self.registry.gauge(
            "nexus_gpu_memory_used_mb", "GPU memory used in MB", ["gpu"]
        )
        self.gpu_memory_total = self.registry.gauge(
            "nexus_gpu_memory_total_mb", "GPU memory total in MB", ["gpu"]
        )
        self.model_memory = self.registry.gauge(
            "nexus_model_memory_mb", "Model memory usage in MB", ["model"]
        )
        self.batch_size = self.registry.histogram(
            "nexus_batch_size", "Batch size distribution",
            buckets=(1, 2, 4, 8, 16, 32),
            label_names=["model"]
        )
        self.cache_utilization = self.registry.gauge(
            "nexus_cache_utilization", "KV cache utilization", ["model"]
        )
        self.errors = self.registry.counter(
            "nexus_errors_total", "Total errors", ["type", "model"]
        )

    def record_request(self, model: str, latency: float, tokens: int, status: str = "success", endpoint: str = "/generate") -> None:
        """Record a completed inference request."""
        self.request_count.inc(labels={"model": model, "status": status})
        self.request_latency.observe(latency, labels={"model": model, "endpoint": endpoint})
        if status == "success":
            self.tokens_generated.inc(amount=tokens, labels={"model": model})
            if latency > 0:
                self.tokens_per_second.set(tokens / latency, labels={"model": model})

    def update_gpu_metrics(self) -> None:
        """Update GPU memory metrics from current state."""
        import torch
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            self.gpu_memory_used.set(used / (1024 * 1024), labels={"gpu": str(i)})
            self.gpu_memory_total.set(total / (1024 * 1024), labels={"gpu": str(i)})

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return self.registry.to_dict()

    def get_prometheus(self) -> str:
        """Get metrics in Prometheus text format."""
        return self.registry.to_prometheus()
