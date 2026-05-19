"""Health checks for Nexus-LLM backend.

Provides model health, GPU health, memory health, and overall service
health checking with detailed status reporting.
"""

import torch
import time
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 2),
        }


class ModelHealthCheck:
    """Checks model health: loading status, inference capability, parameter integrity."""

    def __init__(self, model_manager=None):
        self._model_manager = model_manager

    def check(self, model_id: Optional[str] = None) -> HealthCheckResult:
        """Run model health check."""
        start = time.time()
        result = HealthCheckResult(name="model_health")

        if self._model_manager is None:
            result.status = HealthStatus.UNKNOWN
            result.message = "No model manager configured"
            result.latency_ms = (time.time() - start) * 1000
            return result

        active_id = model_id or self._model_manager.get_active_model_id()
        if active_id is None:
            result.status = HealthStatus.UNHEALTHY
            result.message = "No model is currently loaded"
            result.latency_ms = (time.time() - start) * 1000
            return result

        model_info = self._model_manager.get_model_info(active_id)
        if model_info is None:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"Model info not found for '{active_id}'"
            result.latency_ms = (time.time() - start) * 1000
            return result

        try:
            model = self._model_manager.get_model(active_id)
            tokenizer = self._model_manager.get_tokenizer(active_id)

            has_parameters = sum(1 for _ in model.parameters()) > 0
            if not has_parameters:
                result.status = HealthStatus.UNHEALTHY
                result.message = "Model has no parameters"
                result.latency_ms = (time.time() - start) * 1000
                return result

            device = next(model.parameters()).device
            test_input = tokenizer.encode("test", return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(test_input)

            has_output = output.logits is not None and output.logits.shape[-1] > 0
            if not has_output:
                result.status = HealthStatus.UNHEALTHY
                result.message = "Model inference returned invalid output"
                result.latency_ms = (time.time() - start) * 1000
                return result

            result.status = HealthStatus.HEALTHY
            result.message = f"Model '{active_id}' is healthy"
            result.details = {
                "model_id": active_id,
                "device": str(device),
                "num_parameters": model_info.num_parameters,
                "architecture": model_info.architecture,
                "vocab_size": len(tokenizer),
                "inference_test": "passed",
            }

        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"Model health check failed: {str(e)}"
            result.details = {"error": str(e)}

        result.latency_ms = (time.time() - start) * 1000
        return result


class GPUHealthCheck:
    """Checks GPU health: availability, memory, temperature, compute capability."""

    def check(self, gpu_id: Optional[int] = None) -> HealthCheckResult:
        """Run GPU health check."""
        start = time.time()
        result = HealthCheckResult(name="gpu_health")

        if not torch.cuda.is_available():
            result.status = HealthStatus.DEGRADED
            result.message = "CUDA is not available, running on CPU only"
            result.details = {"cuda_available": False}
            result.latency_ms = (time.time() - start) * 1000
            return result

        try:
            gpu_ids = [gpu_id] if gpu_id is not None else range(torch.cuda.device_count())
            gpu_details = {}

            for gid in gpu_ids:
                props = torch.cuda.get_device_properties(gid)
                free_mem, total_mem = torch.cuda.mem_get_info(gid)
                used_mem = total_mem - free_mem
                utilization_pct = (used_mem / total_mem) * 100

                gpu_info = {
                    "name": props.name,
                    "total_memory_mb": round(total_mem / (1024 * 1024), 2),
                    "used_memory_mb": round(used_mem / (1024 * 1024), 2),
                    "free_memory_mb": round(free_mem / (1024 * 1024), 2),
                    "utilization_pct": round(utilization_pct, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }

                reserved_mb = torch.cuda.memory_reserved(gid) / (1024 * 1024)
                gpu_info["reserved_memory_mb"] = round(reserved_mb, 2)

                test_tensor = torch.randn(100, 100, device=f"cuda:{gid}")
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.synchronize(gid)
                gpu_info["compute_test"] = "passed"

                gpu_details[f"gpu_{gid}"] = gpu_info

            all_healthy = all(
                gpu_details[f"gpu_{gid}"]["compute_test"] == "passed" and
                gpu_details[f"gpu_{gid}"]["utilization_pct"] < 95
                for gid in gpu_ids
            )

            if all_healthy:
                result.status = HealthStatus.HEALTHY
                result.message = f"All {len(gpu_ids)} GPU(s) healthy"
            else:
                result.status = HealthStatus.DEGRADED
                result.message = "One or more GPUs are under pressure (>95% memory utilization)"

            result.details = gpu_details

        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"GPU health check failed: {str(e)}"
            result.details = {"error": str(e)}

        result.latency_ms = (time.time() - start) * 1000
        return result


class MemoryHealthCheck:
    """Checks memory health: RAM and GPU memory availability."""

    def check(self, warning_threshold_pct: float = 80.0) -> HealthCheckResult:
        """Run memory health check."""
        start = time.time()
        result = HealthCheckResult(name="memory_health")

        details = {}

        try:
            import psutil
            mem = psutil.virtual_memory()
            details["ram"] = {
                "total_mb": round(mem.total / (1024 * 1024), 2),
                "available_mb": round(mem.available / (1024 * 1024), 2),
                "used_pct": round(mem.percent, 2),
            }
            ram_healthy = mem.percent < warning_threshold_pct
        except ImportError:
            details["ram"] = {"status": "psutil not available"}
            ram_healthy = True

        if torch.cuda.is_available():
            gpu_mem = {}
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used_pct = ((total - free) / total) * 100
                gpu_mem[f"gpu_{i}"] = {
                    "total_mb": round(total / (1024 * 1024), 2),
                    "used_pct": round(used_pct, 2),
                    "free_mb": round(free / (1024 * 1024), 2),
                }
            details["gpu_memory"] = gpu_mem
            gpu_healthy = all(
                v["used_pct"] < warning_threshold_pct
                for v in gpu_mem.values()
            )
        else:
            gpu_healthy = True
            details["gpu_memory"] = {"status": "no GPU available"}

        if ram_healthy and gpu_healthy:
            result.status = HealthStatus.HEALTHY
            result.message = "Memory usage is within healthy limits"
        elif ram_healthy or gpu_healthy:
            result.status = HealthStatus.DEGRADED
            result.message = "Some memory resources are under pressure"
        else:
            result.status = HealthStatus.UNHEALTHY
            result.message = "Memory resources critically low"

        result.details = details
        result.latency_ms = (time.time() - start) * 1000
        return result


class ServiceHealthCheck:
    """Overall service health check combining all component checks."""

    def __init__(self, model_manager=None):
        self._model_health = ModelHealthCheck(model_manager)
        self._gpu_health = GPUHealthCheck()
        self._memory_health = MemoryHealthCheck()
        self._model_manager = model_manager

    def check_all(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Run all health checks and return a comprehensive status report."""
        start = time.time()

        results = {
            "model": self._model_health.check(model_id),
            "gpu": self._gpu_health.check(),
            "memory": self._memory_health.check(),
        }

        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
            overall_message = "All systems healthy"
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
            unhealthy = [n for n, r in results.items() if r.status == HealthStatus.UNHEALTHY]
            overall_message = f"Unhealthy components: {', '.join(unhealthy)}"
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
            degraded = [n for n, r in results.items() if r.status == HealthStatus.DEGRADED]
            overall_message = f"Degraded components: {', '.join(degraded)}"
        else:
            overall = HealthStatus.UNKNOWN
            overall_message = "Health status unknown"

        report = {
            "status": overall.value,
            "message": overall_message,
            "timestamp": time.time(),
            "total_latency_ms": round((time.time() - start) * 1000, 2),
            "checks": {name: r.to_dict() for name, r in results.items()},
            "loaded_models": self._model_manager.list_loaded_models() if self._model_manager else [],
        }

        return report

    def liveness_check(self) -> Dict[str, Any]:
        """Quick liveness check: is the service running?"""
        return {
            "status": "alive",
            "timestamp": time.time(),
        }

    def readiness_check(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Readiness check: is the service ready to accept requests?"""
        if self._model_manager is None:
            return {
                "status": "not_ready",
                "message": "No model manager configured",
                "timestamp": time.time(),
            }

        active_id = model_id or self._model_manager.get_active_model_id()
        if active_id is None:
            return {
                "status": "not_ready",
                "message": "No model loaded",
                "timestamp": time.time(),
            }

        return {
            "status": "ready",
            "model_id": active_id,
            "timestamp": time.time(),
        }
