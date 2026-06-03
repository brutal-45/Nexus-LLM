"""Nexus-LLM Resource Tracking Over Time.

Provides time-series tracking of system resources including CPU, memory,
GPU utilization, and inference throughput. Supports data export,
aggregation, and retention management.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ResourceSample:
    """A single resource usage sample.

    Attributes:
        timestamp: When the sample was collected.
        cpu_percent: CPU utilization percentage.
        memory_percent: Memory utilization percentage.
        memory_used_gb: Used memory in GB.
        gpu_utilization: GPU compute utilization percentage.
        gpu_memory_percent: GPU memory utilization percentage.
        gpu_memory_used_mb: Used GPU memory in MB.
        gpu_temperature_c: GPU temperature in Celsius.
        disk_percent: Disk utilization percentage.
        network_bytes_sent: Network bytes sent since last sample.
        network_bytes_recv: Network bytes received since last sample.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_temperature_c: float = 0.0
    disk_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_gb": round(self.memory_used_gb, 3),
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_percent": self.gpu_memory_percent,
            "gpu_memory_used_mb": round(self.gpu_memory_used_mb, 1),
            "gpu_temperature_c": self.gpu_temperature_c,
            "disk_percent": self.disk_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
        }


@dataclass
class AggregatedSample:
    """Aggregated statistics for a time window.

    Attributes:
        start_time: Start of the aggregation window.
        end_time: End of the aggregation window.
        count: Number of samples in the window.
        avg_cpu: Average CPU utilization.
        max_cpu: Maximum CPU utilization.
        avg_memory: Average memory utilization.
        max_memory: Maximum memory utilization.
        avg_gpu: Average GPU utilization.
        max_gpu: Maximum GPU utilization.
        avg_gpu_memory: Average GPU memory utilization.
        max_gpu_memory: Maximum GPU memory utilization.
    """

    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    count: int = 0
    avg_cpu: float = 0.0
    max_cpu: float = 0.0
    avg_memory: float = 0.0
    max_memory: float = 0.0
    avg_gpu: float = 0.0
    max_gpu: float = 0.0
    avg_gpu_memory: float = 0.0
    max_gpu_memory: float = 0.0


class ResourceTracker:
    """Time-series resource tracker for Nexus-LLM.

    Periodically samples system resource metrics and stores them in
    memory with configurable retention. Supports aggregation, export,
    and integration with the alert system.

    Attributes:
        sample_interval: Time between samples in seconds.
        retention_hours: How long to keep samples.
    """

    def __init__(
        self,
        sample_interval: float = 5.0,
        retention_hours: int = 24,
        max_samples: int = 50000,
    ) -> None:
        """Initialize the resource tracker.

        Args:
            sample_interval: Seconds between resource samples.
            retention_hours: Hours of data to retain.
            max_samples: Maximum number of samples to keep.
        """
        self.sample_interval = sample_interval
        self.retention_hours = retention_hours
        self.max_samples = max_samples

        self._samples: Deque[ResourceSample] = deque(maxlen=max_samples)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_net_io: Optional[Tuple[int, int]] = None

    def sample(self) -> ResourceSample:
        """Collect a single resource sample.

        Returns:
            ResourceSample with current system metrics.
        """
        s = ResourceSample()

        # CPU and memory
        try:
            import psutil
            s.cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            s.memory_percent = mem.percent
            s.memory_used_gb = mem.used / (1024 ** 3)
            s.disk_percent = psutil.disk_usage("/").percent

            net_io = psutil.net_io_counters()
            if self._last_net_io is not None:
                s.network_bytes_sent = max(0, net_io.bytes_sent - self._last_net_io[0])
                s.network_bytes_recv = max(0, net_io.bytes_recv - self._last_net_io[1])
            self._last_net_io = (net_io.bytes_sent, net_io.bytes_recv)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"psutil collection error: {e}")

        # GPU metrics
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            s.gpu_utilization = util.gpu
            s.gpu_memory_percent = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0
            s.gpu_memory_used_mb = mem_info.used / (1024 ** 2)
            s.gpu_temperature_c = temp
            pynvml.nvmlShutdown()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"GPU collection error: {e}")

        with self._lock:
            self._samples.append(s)

        return s

    def start(self) -> None:
        """Start periodic resource sampling in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
        logger.info(f"Resource tracker started (interval={self.sample_interval}s)")

    def stop(self) -> None:
        """Stop periodic resource sampling."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("Resource tracker stopped")

    def _sampling_loop(self) -> None:
        """Background sampling loop."""
        while self._running:
            try:
                self.sample()
                self._cleanup_old_samples()
            except Exception as e:
                logger.error(f"Resource sampling error: {e}")
            time.sleep(self.sample_interval)

    def _cleanup_old_samples(self) -> None:
        """Remove samples older than the retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        with self._lock:
            while self._samples and self._samples[0].timestamp < cutoff:
                self._samples.popleft()

    def get_samples(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[ResourceSample]:
        """Get resource samples within a time range.

        Args:
            since: Start time filter.
            until: End time filter.
            limit: Maximum number of samples.

        Returns:
            List of ResourceSample objects.
        """
        with self._lock:
            samples = list(self._samples)

        if since:
            samples = [s for s in samples if s.timestamp >= since]
        if until:
            samples = [s for s in samples if s.timestamp <= until]
        if limit:
            samples = samples[-limit:]

        return samples

    def get_latest(self) -> Optional[ResourceSample]:
        """Get the most recent sample.

        Returns:
            Latest ResourceSample, or None if no samples exist.
        """
        with self._lock:
            return self._samples[-1] if self._samples else None

    def aggregate(
        self,
        window_minutes: int = 5,
        since: Optional[datetime] = None,
    ) -> List[AggregatedSample]:
        """Aggregate samples into fixed-size time windows.

        Args:
            window_minutes: Size of each aggregation window.
            since: Only aggregate samples after this time.

        Returns:
            List of AggregatedSample objects.
        """
        samples = self.get_samples(since=since)
        if not samples:
            return []

        window_delta = timedelta(minutes=window_minutes)
        results: List[AggregatedSample] = []

        window_start = samples[0].timestamp
        window_samples: List[ResourceSample] = []

        for sample in samples:
            if sample.timestamp >= window_start + window_delta:
                if window_samples:
                    results.append(self._aggregate_window(window_start, window_start + window_delta, window_samples))
                window_start = window_start + window_delta
                window_samples = [sample]
            else:
                window_samples.append(sample)

        if window_samples:
            results.append(self._aggregate_window(window_start, window_start + window_delta, window_samples))

        return results

    def _aggregate_window(
        self,
        start: datetime,
        end: datetime,
        samples: List[ResourceSample],
    ) -> AggregatedSample:
        """Aggregate a list of samples into a single AggregatedSample."""
        n = len(samples)
        return AggregatedSample(
            start_time=start,
            end_time=end,
            count=n,
            avg_cpu=sum(s.cpu_percent for s in samples) / n,
            max_cpu=max(s.cpu_percent for s in samples),
            avg_memory=sum(s.memory_percent for s in samples) / n,
            max_memory=max(s.memory_percent for s in samples),
            avg_gpu=sum(s.gpu_utilization for s in samples) / n,
            max_gpu=max(s.gpu_utilization for s in samples),
            avg_gpu_memory=sum(s.gpu_memory_percent for s in samples) / n,
            max_gpu_memory=max(s.gpu_memory_percent for s in samples),
        )

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics as a flat dictionary.

        Suitable for passing to the AlertManager.evaluate() method.

        Returns:
            Dictionary of metric name to current value.
        """
        latest = self.get_latest()
        if latest is None:
            return {}

        return {
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "gpu_utilization": latest.gpu_utilization,
            "gpu_memory_percent": latest.gpu_memory_percent,
            "gpu_temperature_c": latest.gpu_temperature_c,
            "disk_percent": latest.disk_percent,
        }

    def export_json(self, output_path: str, since: Optional[datetime] = None) -> int:
        """Export samples to a JSON file.

        Args:
            output_path: Path for the output file.
            since: Only export samples after this time.

        Returns:
            Number of samples exported.
        """
        samples = self.get_samples(since=since)
        data = [s.to_dict() for s in samples]

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(samples)} samples to {output_path}")
        return len(samples)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked resource data.

        Returns:
            Dictionary with sample counts, time range, and averages.
        """
        with self._lock:
            count = len(self._samples)
            if count == 0:
                return {"total_samples": 0}

            first = self._samples[0]
            last = self._samples[-1]

            avg_cpu = sum(s.cpu_percent for s in self._samples) / count
            avg_memory = sum(s.memory_percent for s in self._samples) / count
            max_cpu = max(s.cpu_percent for s in self._samples)
            max_memory = max(s.memory_percent for s in self._samples)

            gpu_samples = [s for s in self._samples if s.gpu_utilization > 0]

        result: Dict[str, Any] = {
            "total_samples": count,
            "time_range": {
                "start": first.timestamp.isoformat(),
                "end": last.timestamp.isoformat(),
            },
            "avg_cpu": round(avg_cpu, 2),
            "max_cpu": round(max_cpu, 2),
            "avg_memory": round(avg_memory, 2),
            "max_memory": round(max_memory, 2),
        }

        if gpu_samples:
            result["avg_gpu"] = round(sum(s.gpu_utilization for s in gpu_samples) / len(gpu_samples), 2)
            result["max_gpu"] = round(max(s.gpu_utilization for s in gpu_samples), 2)

        return result
