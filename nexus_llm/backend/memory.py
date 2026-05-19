"""Memory management for Nexus-LLM backend.

Provides GPU memory tracking, memory estimation, garbage collection,
and memory-mapped loading for efficient model memory usage.
"""

import torch
import gc
import os
import mmap
import struct
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import threading
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of current memory state."""
    timestamp: float = field(default_factory=time.time)
    gpu_total_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_free_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    cpu_total_mb: float = 0.0
    cpu_used_mb: float = 0.0
    cpu_available_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    model_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gpu_total_mb": round(self.gpu_total_mb, 2),
            "gpu_used_mb": round(self.gpu_used_mb, 2),
            "gpu_free_mb": round(self.gpu_free_mb, 2),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 2),
            "cpu_total_mb": round(self.cpu_total_mb, 2),
            "cpu_used_mb": round(self.cpu_used_mb, 2),
            "cpu_available_mb": round(self.cpu_available_mb, 2),
            "ram_total_mb": round(self.ram_total_mb, 2),
            "ram_available_mb": round(self.ram_available_mb, 2),
            "model_memory_mb": round(self.model_memory_mb, 2),
            "cache_memory_mb": round(self.cache_memory_mb, 2),
        }


class MemoryTracker:
    """Tracks GPU and CPU memory usage over time."""

    def __init__(self, history_size: int = 100):
        self._history: List[MemorySnapshot] = []
        self._history_size = history_size
        self._lock = threading.RLock()
        self._model_memory_mb: float = 0.0

    def set_model_memory(self, memory_mb: float) -> None:
        """Set the current model memory usage."""
        self._model_memory_mb = memory_mb

    def take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory state."""
        snapshot = MemorySnapshot()

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                snapshot.gpu_total_mb += total / (1024 * 1024)
                snapshot.gpu_used_mb += used / (1024 * 1024)
                snapshot.gpu_free_mb += free / (1024 * 1024)

            if snapshot.gpu_total_mb > 0:
                snapshot.gpu_utilization_pct = (snapshot.gpu_used_mb / snapshot.gpu_total_mb) * 100

        try:
            import psutil
            mem = psutil.virtual_memory()
            snapshot.ram_total_mb = mem.total / (1024 * 1024)
            snapshot.ram_available_mb = mem.available / (1024 * 1024)
        except ImportError:
            pass

        snapshot.model_memory_mb = self._model_memory_mb

        if torch.cuda.is_available():
            snapshot.cache_memory_mb = torch.cuda.memory_reserved() / (1024 * 1024)

        with self._lock:
            self._history.append(snapshot)
            if len(self._history) > self._history_size:
                self._history.pop(0)

        return snapshot

    def get_history(self) -> List[MemorySnapshot]:
        """Get memory usage history."""
        with self._lock:
            return list(self._history)

    def get_peak_usage(self) -> Optional[MemorySnapshot]:
        """Get the snapshot with peak GPU memory usage."""
        if not self._history:
            return None
        return max(self._history, key=lambda s: s.gpu_used_mb)

    def get_average_usage(self) -> Optional[MemorySnapshot]:
        """Get average memory usage over history."""
        if not self._history:
            return None

        n = len(self._history)
        avg = MemorySnapshot()
        avg.gpu_total_mb = sum(s.gpu_total_mb for s in self._history) / n
        avg.gpu_used_mb = sum(s.gpu_used_mb for s in self._history) / n
        avg.gpu_free_mb = sum(s.gpu_free_mb for s in self._history) / n
        avg.gpu_utilization_pct = sum(s.gpu_utilization_pct for s in self._history) / n
        avg.ram_available_mb = sum(s.ram_available_mb for s in self._history) / n
        avg.ram_total_mb = sum(s.ram_total_mb for s in self._history) / n
        avg.model_memory_mb = sum(s.model_memory_mb for s in self._history) / n
        avg.cache_memory_mb = sum(s.cache_memory_mb for s in self._history) / n
        return avg


class MemoryEstimator:
    """Estimates memory requirements for models and operations."""

    @staticmethod
    def estimate_model_memory(
        num_parameters: int,
        dtype: str = "float16",
        overhead_pct: float = 0.1,
    ) -> float:
        """Estimate model memory in MB.

        Args:
            num_parameters: Total number of model parameters.
            dtype: Data type string (float32, float16, bfloat16, int8, int4).
            overhead_pct: Additional overhead as percentage (optimizer states, etc.).

        Returns:
            Estimated memory in MB.
        """
        bytes_per_param = {
            "float32": 4, "float16": 2, "bfloat16": 2,
            "int8": 1, "int4": 0.5, "nf4": 0.5, "fp4": 0.5,
        }
        b = bytes_per_param.get(dtype, 2)
        base_mb = num_parameters * b / (1024 * 1024)
        return base_mb * (1 + overhead_pct)

    @staticmethod
    def estimate_kv_cache_memory(
        batch_size: int,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: str = "float16",
    ) -> float:
        """Estimate KV cache memory in MB."""
        bytes_per_element = {"float32": 4, "float16": 2, "bfloat16": 2}.get(dtype, 2)
        per_token_per_layer = 2 * num_heads * head_dim * bytes_per_element
        total_bytes = batch_size * seq_length * num_layers * per_token_per_layer
        return total_bytes / (1024 * 1024)

    @staticmethod
    def estimate_inference_memory(
        num_parameters: int,
        dtype: str = "float16",
        batch_size: int = 1,
        seq_length: int = 2048,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> Dict[str, float]:
        """Estimate total inference memory requirements in MB."""
        model_mb = MemoryEstimator.estimate_model_memory(num_parameters, dtype)
        kv_cache_mb = MemoryEstimator.estimate_kv_cache_memory(
            batch_size, seq_length, num_layers, num_heads, head_dim, dtype
        )
        activation_mb = batch_size * seq_length * num_heads * head_dim * 2 / (1024 * 1024)
        overhead_mb = (model_mb + kv_cache_mb) * 0.05

        return {
            "model_mb": model_mb,
            "kv_cache_mb": kv_cache_mb,
            "activations_mb": activation_mb,
            "overhead_mb": overhead_mb,
            "total_mb": model_mb + kv_cache_mb + activation_mb + overhead_mb,
        }

    @staticmethod
    def estimate_training_memory(
        num_parameters: int,
        dtype: str = "float32",
        optimizer: str = "adam",
        gradient_checkpointing: bool = False,
    ) -> Dict[str, float]:
        """Estimate training memory requirements in MB."""
        bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2}.get(dtype, 4)
        model_mb = num_parameters * bytes_per_param / (1024 * 1024)
        gradients_mb = model_mb

        optimizer_states_mb = 0
        if optimizer == "adam":
            optimizer_states_mb = model_mb * 2
        elif optimizer == "sgd":
            optimizer_states_mb = model_mb * 0.5

        activation_mb = model_mb * 2 if not gradient_checkpointing else model_mb * 0.5

        return {
            "model_mb": model_mb,
            "gradients_mb": gradients_mb,
            "optimizer_mb": optimizer_states_mb,
            "activations_mb": activation_mb,
            "total_mb": model_mb + gradients_mb + optimizer_states_mb + activation_mb,
        }


class GarbageCollector:
    """Manages garbage collection for GPU and CPU memory."""

    @staticmethod
    def collect_gpu() -> Dict[int, Dict[str, float]]:
        """Run garbage collection on all GPUs."""
        results = {}
        if not torch.cuda.is_available():
            return results

        for i in range(torch.cuda.device_count()):
            before_free = torch.cuda.mem_get_info(i)[0]
            torch.cuda.empty_cache()
            after_free = torch.cuda.mem_get_info(i)[0]
            freed_mb = (after_free - before_free) / (1024 * 1024)

            results[i] = {
                "freed_mb": freed_mb,
                "free_mb_after": after_free / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024),
            }

        return results

    @staticmethod
    def collect_cpu() -> Dict[str, float]:
        """Run garbage collection on CPU."""
        before = 0
        try:
            import psutil
            before = psutil.virtual_memory().available
        except ImportError:
            pass

        gc.collect()
        gc.collect()
        gc.collect()

        after = 0
        try:
            import psutil
            after = psutil.virtual_memory().available
        except ImportError:
            pass

        return {
            "freed_mb": (after - before) / (1024 * 1024) if before and after else 0,
            "available_mb_after": after / (1024 * 1024) if after else 0,
        }

    @staticmethod
    def collect_all() -> Dict[str, Any]:
        """Run full garbage collection on GPU and CPU."""
        gpu_results = GarbageCollector.collect_gpu()
        cpu_results = GarbageCollector.collect_cpu()
        return {"gpu": gpu_results, "cpu": cpu_results}


class MemoryMappedLoader:
    """Memory-mapped loading for large model files.

    Enables loading model weights without fully reading them into memory,
    useful for models larger than available RAM.
    """

    def __init__(self, file_path: str, mode: str = "r"):
        self._file_path = file_path
        self._mode = mode
        self._mmap = None
        self._file = None

    def open(self) -> None:
        """Open the file with memory mapping."""
        self._file = open(self._file_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        logger.info(f"Memory-mapped file opened: {self._file_path} ({os.path.getsize(self._file_path) / (1024**3):.2f} GB)")

    def close(self) -> None:
        """Close the memory-mapped file."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def read_tensor(self, offset: int, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Read a tensor from a specific offset in the memory-mapped file."""
        if self._mmap is None:
            raise RuntimeError("Memory-mapped file is not open")

        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        byte_size = num_elements * element_size

        self._mmap.seek(offset)
        data = self._mmap.read(byte_size)

        tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
        return tensor.clone()

    def load_weights_lazily(self, model: Any) -> None:
        """Load model weights using memory-mapped access for lazy loading."""
        if self._file_path.endswith(".safetensors"):
            self._load_safetensors_lazily(model)
        elif self._file_path.endswith(".bin") or self._file_path.endswith(".pt"):
            self._load_pytorch_lazily(model)
        else:
            logger.warning(f"Unsupported format for memory-mapped loading: {self._file_path}")

    def _load_safetensors_lazily(self, model: Any) -> None:
        """Load weights from a safetensors file using memory mapping."""
        try:
            from safetensors.torch import load_file
            state_dict = load_file(self._file_path)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded safetensors weights via memory mapping: {self._file_path}")
        except ImportError:
            logger.error("safetensors library required for memory-mapped loading")

    def _load_pytorch_lazily(self, model: Any) -> None:
        """Load weights from a PyTorch file using memory mapping."""
        state_dict = torch.load(self._file_path, map_location="cpu", mmap=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded PyTorch weights via memory mapping: {self._file_path}")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def size_bytes(self) -> int:
        """Total size of the memory-mapped file."""
        if os.path.exists(self._file_path):
            return os.path.getsize(self._file_path)
        return 0
