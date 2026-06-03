"""Health checking for Nexus-LLM."""

import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SystemHealth:
    """System resource health snapshot."""
    cpu_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_total_mb: Optional[float]
    gpu_memory_used_mb: Optional[float]
    gpu_utilization_percent: Optional[float]

    @property
    def is_healthy(self) -> bool:
        """Return True if all resource usage is below critical thresholds."""
        return (
            self.cpu_percent < 95.0
            and self.memory_percent < 95.0
            and self.disk_percent < 95.0
        )


@dataclass
class ModelHealth:
    """Model serving health status."""
    model_loaded: bool
    model_name: Optional[str]
    last_inference_time: Optional[float]
    inference_count: int
    error_count: int
    avg_latency_ms: Optional[float]

    @property
    def is_healthy(self) -> bool:
        """Return True if the model is loaded and has acceptable error rate."""
        if not self.model_loaded:
            return False
        if self.inference_count > 0:
            error_rate = self.error_count / self.inference_count
            return error_rate < 0.5
        return True


@dataclass
class HealthReport:
    """Aggregated health report."""
    healthy: bool
    timestamp: float
    model_health: ModelHealth
    system_health: SystemHealth
    custom_checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Performs health checks on model and system resources.

    Supports registering custom health check functions that are
    evaluated alongside built-in checks.
    """

    def __init__(self) -> None:
        self._custom_checks: Dict[str, Callable[[], bool]] = {}
        self._model_state: Dict[str, Any] = {
            "loaded": False,
            "name": None,
            "last_inference": None,
            "inference_count": 0,
            "error_count": 0,
            "avg_latency_ms": None,
        }
        self._lock = threading.Lock()

    # -- Public API -----------------------------------------------------------

    def check_health(self) -> HealthReport:
        """Run all health checks and return a comprehensive report.

        Returns:
            ``HealthReport`` with model, system, and custom check results.
        """
        model_health = self.check_model_health()
        system_health = self.check_system_health()

        custom_results: Dict[str, bool] = {}
        for name, check_fn in list(self._custom_checks.items()):
            try:
                custom_results[name] = bool(check_fn())
            except Exception:
                custom_results[name] = False

        all_healthy = (
            model_health.is_healthy
            and system_health.is_healthy
            and all(custom_results.values())
        )

        return HealthReport(
            healthy=all_healthy,
            timestamp=time.time(),
            model_health=model_health,
            system_health=system_health,
            custom_checks=custom_results,
        )

    def check_model_health(self) -> ModelHealth:
        """Check the health of the loaded model.

        Returns:
            ``ModelHealth`` with model status details.
        """
        with self._lock:
            state = dict(self._model_state)

        return ModelHealth(
            model_loaded=state["loaded"],
            model_name=state["name"],
            last_inference_time=state["last_inference"],
            inference_count=state["inference_count"],
            error_count=state["error_count"],
            avg_latency_ms=state["avg_latency_ms"],
        )

    def check_system_health(self) -> SystemHealth:
        """Check system resource health (CPU, RAM, disk, GPU).

        Returns:
            ``SystemHealth`` with resource utilization data.
        """
        # CPU
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            memory_total_gb = mem.total / (1024 ** 3)
            memory_used_gb = mem.used / (1024 ** 3)
            memory_percent = mem.percent
            disk_total_gb = disk.total / (1024 ** 3)
            disk_used_gb = disk.used / (1024 ** 3)
            disk_percent = disk.percent
        except ImportError:
            # Fallback if psutil is unavailable
            cpu_percent = 0.0
            memory_total_gb = 0.0
            memory_used_gb = 0.0
            memory_percent = 0.0
            disk_total_gb = 0.0
            disk_used_gb = 0.0
            disk_percent = 0.0

        # GPU
        gpu_available = False
        gpu_name = None
        gpu_memory_total_mb: Optional[float] = None
        gpu_memory_used_mb: Optional[float] = None
        gpu_utilization_percent: Optional[float] = None

        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_total_mb = gpu_props.total_mem / (1024 ** 2)
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
                gpu_memory_used_mb = gpu_mem_allocated
                gpu_utilization_percent = (gpu_mem_allocated / (gpu_props.total_mem / (1024 ** 2))) * 100
        except (ImportError, RuntimeError):
            pass

        return SystemHealth(
            cpu_percent=cpu_percent,
            memory_total_gb=round(memory_total_gb, 2),
            memory_used_gb=round(memory_used_gb, 2),
            memory_percent=round(memory_percent, 1),
            disk_total_gb=round(disk_total_gb, 2),
            disk_used_gb=round(disk_used_gb, 2),
            disk_percent=round(disk_percent, 1),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_total_mb=round(gpu_memory_total_mb, 1) if gpu_memory_total_mb else None,
            gpu_memory_used_mb=round(gpu_memory_used_mb, 1) if gpu_memory_used_mb else None,
            gpu_utilization_percent=round(gpu_utilization_percent, 1) if gpu_utilization_percent else None,
        )

    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a custom health check function.

        Args:
            name: Unique name for the check.
            check_func: Callable returning ``True`` for healthy.
        """
        self._custom_checks[name] = check_func

    def unregister_check(self, name: str) -> None:
        """Remove a previously registered custom check."""
        self._custom_checks.pop(name, None)

    def update_model_state(self, **kwargs: Any) -> None:
        """Update internal model state used by ``check_model_health``.

        Accepted keyword arguments: ``loaded``, ``name``,
        ``last_inference``, ``inference_count``, ``error_count``,
        ``avg_latency_ms``.
        """
        with self._lock:
            for key, value in kwargs.items():
                if key in self._model_state:
                    self._model_state[key] = value
