"""CPU and disk offloading for Nexus-LLM backend.

Implements layer-wise CPU offloading, disk offloading, and automatic
offloading based on available GPU memory.
"""

import torch
import os
import json
import tempfile
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)


class OffloadTarget(Enum):
    """Offloading destination."""
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"


class OffloadStrategy(Enum):
    """Strategy for deciding which layers to offload."""
    MANUAL = "manual"
    AUTO = "auto"
    BALANCED = "balanced"
    GREEDY = "greedy"


@dataclass
class LayerPlacement:
    """Placement information for a single model layer."""
    layer_name: str
    target: OffloadTarget = OffloadTarget.GPU
    device: str = "cuda:0"
    disk_path: Optional[str] = None
    size_bytes: int = 0
    is_loaded: bool = False


@dataclass
class OffloadConfig:
    """Configuration for model offloading."""
    strategy: OffloadStrategy = OffloadStrategy.AUTO
    gpu_memory_limit_gb: Optional[float] = None
    cpu_memory_limit_gb: Optional[float] = None
    disk_offload_dir: Optional[str] = None
    offload_layers: Optional[List[str]] = None
    keep_on_gpu: Optional[List[str]] = None
    prefetch_layers: int = 1
    pin_memory: bool = True


class LayerOffloader:
    """Manages layer-wise offloading between GPU, CPU, and disk.

    Supports:
    - Moving individual layers between devices
    - Automatic offloading based on memory constraints
    - Disk offloading for very large models
    - Layer prefetching for sequential access patterns
    """

    def __init__(self, model: Any, config: Optional[OffloadConfig] = None):
        self._model = model
        self._config = config or OffloadConfig()
        self._placements: Dict[str, LayerPlacement] = {}
        self._layer_order: List[str] = []
        self._lock = threading.RLock()
        self._disk_dir = self._config.disk_offload_dir or tempfile.mkdtemp(prefix="nexus_offload_")

    def analyze_model(self) -> Dict[str, int]:
        """Analyze model layers and their sizes in bytes."""
        layer_sizes = {}
        for name, param in self._model.named_parameters():
            size = param.numel() * param.element_size()
            layer_prefix = name.rsplit(".", 1)[0] if "." in name else name
            layer_sizes[layer_prefix] = layer_sizes.get(layer_prefix, 0) + size

        for name, buf in self._model.named_buffers():
            size = buf.numel() * buf.element_size()
            layer_prefix = name.rsplit(".", 1)[0] if "." in name else name
            layer_sizes[layer_prefix] = layer_sizes.get(layer_prefix, 0) + size

        return layer_sizes

    def compute_device_map(self) -> Dict[str, str]:
        """Compute an optimal device map based on available memory and strategy."""
        layer_sizes = self.analyze_model()
        sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[0])

        self._layer_order = [name for name, _ in sorted_layers]
        device_map: Dict[str, str] = {}

        if self._config.strategy == OffloadStrategy.MANUAL:
            return self._compute_manual_device_map(layer_sizes)
        elif self._config.strategy == OffloadStrategy.AUTO:
            return self._compute_auto_device_map(layer_sizes)
        elif self._config.strategy == OffloadStrategy.BALANCED:
            return self._compute_balanced_device_map(layer_sizes)
        elif self._config.strategy == OffloadStrategy.GREEDY:
            return self._compute_greedy_device_map(layer_sizes)
        return device_map

    def _compute_manual_device_map(self, layer_sizes: Dict[str, int]) -> Dict[str, str]:
        """Compute device map based on manually specified layers."""
        device_map = {}
        gpu_layers = set(self._config.keep_on_gpu or [])
        offload_layers = set(self._config.offload_layers or [])

        for layer_name in layer_sizes:
            if layer_name in gpu_layers:
                device_map[layer_name] = "cuda:0"
            elif layer_name in offload_layers:
                device_map[layer_name] = "cpu"
            else:
                device_map[layer_name] = "cuda:0"

        return device_map

    def _compute_auto_device_map(self, layer_sizes: Dict[str, int]) -> Dict[str, str]:
        """Automatically compute device map based on available GPU memory."""
        device_map = {}
        if not torch.cuda.is_available():
            for name in layer_sizes:
                device_map[name] = "cpu"
            return device_map

        gpu_limit = self._config.gpu_memory_limit_gb
        if gpu_limit is None:
            gpu_limit = torch.cuda.get_device_properties(0).total_mem / (1024**3) * 0.85

        gpu_limit_bytes = int(gpu_limit * (1024**3))
        current_gpu_usage = 0

        keep_on_gpu = set(self._config.keep_on_gpu or [])

        sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[0])
        for layer_name, size in sorted_layers:
            if layer_name in keep_on_gpu or current_gpu_usage + size <= gpu_limit_bytes:
                device_map[layer_name] = "cuda:0"
                current_gpu_usage += size
            else:
                device_map[layer_name] = "cpu"

        return device_map

    def _compute_balanced_device_map(self, layer_sizes: Dict[str, int]) -> Dict[str, str]:
        """Compute a balanced device map distributing layers across GPU and CPU."""
        device_map = {}
        sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[0])
        total_size = sum(s for _, s in sorted_layers)

        if torch.cuda.is_available():
            gpu_limit = self._config.gpu_memory_limit_gb
            if gpu_limit is None:
                gpu_limit = torch.cuda.get_device_properties(0).total_mem / (1024**3) * 0.7
            gpu_fraction = min(1.0, (gpu_limit * 1024**3) / total_size)
        else:
            gpu_fraction = 0.0

        gpu_layers = max(1, int(len(sorted_layers) * gpu_fraction))
        for i, (name, _) in enumerate(sorted_layers):
            if i < gpu_layers:
                device_map[name] = "cuda:0"
            else:
                device_map[name] = "cpu"

        return device_map

    def _compute_greedy_device_map(self, layer_sizes: Dict[str, int]) -> Dict[str, str]:
        """Greedy strategy: fill GPU until full, then CPU, then disk."""
        device_map = {}
        sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[0])

        if not torch.cuda.is_available():
            for name in layer_sizes:
                device_map[name] = "cpu"
            return device_map

        gpu_limit = self._config.gpu_memory_limit_gb or (
            torch.cuda.get_device_properties(0).total_mem / (1024**3) * 0.85
        )
        gpu_limit_bytes = int(gpu_limit * (1024**3))

        cpu_limit = self._config.cpu_memory_limit_gb or 64.0
        cpu_limit_bytes = int(cpu_limit * (1024**3))

        gpu_used = 0
        cpu_used = 0

        for name, size in sorted_layers:
            if gpu_used + size <= gpu_limit_bytes:
                device_map[name] = "cuda:0"
                gpu_used += size
            elif cpu_used + size <= cpu_limit_bytes:
                device_map[name] = "cpu"
                cpu_used += size
            else:
                device_map[name] = "disk"
                self._placements[name] = LayerPlacement(
                    layer_name=name,
                    target=OffloadTarget.DISK,
                    disk_path=os.path.join(self._disk_dir, f"{name}.pt"),
                    size_bytes=size,
                )

        return device_map

    def apply_device_map(self, device_map: Dict[str, str]) -> None:
        """Apply a device map to the model, moving layers to their target devices."""
        with self._lock:
            for name, param in self._model.named_parameters():
                layer_prefix = name.rsplit(".", 1)[0] if "." in name else name
                target = device_map.get(layer_prefix, "cuda:0")

                if target.startswith("cuda"):
                    if param.device.type != "cuda":
                        param.data = param.data.to(target)
                elif target == "cpu":
                    if param.device.type != "cpu":
                        param.data = param.data.cpu()
                elif target == "disk":
                    self._save_layer_to_disk(layer_prefix)

                self._placements[layer_prefix] = LayerPlacement(
                    layer_name=layer_prefix,
                    target=OffloadTarget(target) if target in ("gpu", "cpu", "disk") else OffloadTarget.GPU,
                    device=target,
                    size_bytes=param.numel() * param.element_size(),
                    is_loaded=(target != "disk"),
                )

    def _save_layer_to_disk(self, layer_name: str) -> None:
        """Save a model layer to disk."""
        layer_state = {}
        for name, param in self._model.named_parameters():
            prefix = name.rsplit(".", 1)[0] if "." in name else name
            if prefix == layer_name:
                layer_state[name] = param.data.cpu()

        for name, buf in self._model.named_buffers():
            prefix = name.rsplit(".", 1)[0] if "." in name else name
            if prefix == layer_name:
                layer_state[name] = buf.data.cpu()

        if layer_state:
            path = os.path.join(self._disk_dir, f"{layer_name}.pt")
            torch.save(layer_state, path)
            logger.debug(f"Saved layer '{layer_name}' to disk: {path}")

    def _load_layer_from_disk(self, layer_name: str, device: str = "cpu") -> None:
        """Load a model layer from disk."""
        path = os.path.join(self._disk_dir, f"{layer_name}.pt")
        if not os.path.exists(path):
            logger.warning(f"Disk offload file not found for layer '{layer_name}'")
            return

        layer_state = torch.load(path, map_location=device)
        model_state = self._model.state_dict()
        for name, tensor in layer_state.items():
            if name in model_state:
                model_state[name] = tensor

        self._model.load_state_dict(model_state, strict=False)

        if layer_name in self._placements:
            self._placements[layer_name].is_loaded = True

        logger.debug(f"Loaded layer '{layer_name}' from disk to {device}")

    def swap_layer_to_device(self, layer_name: str, target_device: str) -> None:
        """Move a specific layer to a different device (swap in/out)."""
        with self._lock:
            placement = self._placements.get(layer_name)
            if placement is None:
                return

            if placement.target == OffloadTarget.DISK and not placement.is_loaded:
                self._load_layer_from_disk(layer_name, target_device)
                return

            for name, param in self._model.named_parameters():
                prefix = name.rsplit(".", 1)[0] if "." in name else name
                if prefix == layer_name:
                    param.data = param.data.to(target_device)

            if target_device.startswith("cuda") and self._config.pin_memory:
                pass

            placement.device = target_device
            placement.is_loaded = True

    def prefetch_next_layer(self, current_layer_idx: int) -> None:
        """Prefetch the next N layers to GPU for sequential access."""
        if not self._layer_order:
            return

        for i in range(1, self._config.prefetch_layers + 1):
            next_idx = current_layer_idx + i
            if next_idx < len(self._layer_order):
                next_layer = self._layer_order[next_idx]
                placement = self._placements.get(next_layer)
                if placement and not placement.is_loaded:
                    self._load_layer_from_disk(next_layer, "cpu")

    def get_offload_summary(self) -> Dict[str, Any]:
        """Get a summary of current offloading state."""
        gpu_count = sum(1 for p in self._placements.values() if p.device.startswith("cuda"))
        cpu_count = sum(1 for p in self._placements.values() if p.device == "cpu")
        disk_count = sum(1 for p in self._placements.values() if p.target == OffloadTarget.DISK)
        gpu_bytes = sum(p.size_bytes for p in self._placements.values() if p.device.startswith("cuda"))
        cpu_bytes = sum(p.size_bytes for p in self._placements.values() if p.device == "cpu")
        disk_bytes = sum(p.size_bytes for p in self._placements.values() if p.target == OffloadTarget.DISK)

        return {
            "total_layers": len(self._placements),
            "gpu_layers": gpu_count,
            "cpu_layers": cpu_count,
            "disk_layers": disk_count,
            "gpu_memory_mb": gpu_bytes / (1024 * 1024),
            "cpu_memory_mb": cpu_bytes / (1024 * 1024),
            "disk_memory_mb": disk_bytes / (1024 * 1024),
            "strategy": self._config.strategy.value,
        }

    def offload_all_to_cpu(self) -> None:
        """Move all model layers to CPU."""
        if not torch.cuda.is_available():
            return
        self._model = self._model.cpu()
        for name, placement in self._placements.items():
            placement.device = "cpu"
            placement.target = OffloadTarget.CPU
        torch.cuda.empty_cache()
        logger.info("All layers offloaded to CPU")

    def cleanup_disk(self) -> None:
        """Clean up disk-offloaded files."""
        if os.path.exists(self._disk_dir):
            for f in os.listdir(self._disk_dir):
                if f.endswith(".pt"):
                    os.remove(os.path.join(self._disk_dir, f))
            logger.info(f"Cleaned up disk offload directory: {self._disk_dir}")
