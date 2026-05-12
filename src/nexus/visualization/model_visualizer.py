"""
Model Visualizer - Model Architecture Visualization
====================================================

Comprehensive tools for analyzing LLM model architecture including
parameter counting, FLOPs estimation, memory profiling, architecture
tree display, parameter distributions, and layer inspection.

All implementations use Python stdlib only.
"""

import math
import time
import json
import os
import hashlib
import threading
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Sequence, Set
)
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_H = "─"
BOX_V = "│"
BOX_LT = "├"
BOX_RT = "┤"
BOX_BT = "┬"
BOX_BB = "┴"
BOX_CROSS = "┼"

BRANCH_T = "├──"
BRANCH_L = "└──"
BRANCH_V = "│  "
BRANCH_S = "   "

PROGRESS_FULL = "█"
PROGRESS_THREE_QUARTERS = "▓"
PROGRESS_HALF = "▒"
PROGRESS_QUARTER = "░"
PROGRESS_EMPTY = " "


class ParameterType(Enum):
    """Types of model parameters."""
    WEIGHT = "weight"
    BIAS = "bias"
    EMBEDDING = "embedding"
    LAYERNORM = "layernorm"
    POSITION = "position"
    OTHER = "other"


class LayerType(Enum):
    """Types of model layers."""
    LINEAR = "linear"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"
    FEEDFORWARD = "feedforward"
    EMBEDDING = "embedding"
    LAYERNORM = "layernorm"
    ACTIVATION = "activation"
    DROPOUT = "dropout"
    POOLING = "pooling"
    RNN = "rnn"
    CUSTOM = "custom"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LayerInfo:
    """Information about a single model layer."""
    name: str
    layer_type: str = "unknown"
    num_parameters: int = 0
    trainable_parameters: int = 0
    output_shape: Optional[Tuple[int, ...]] = None
    input_shape: Optional[Tuple[int, ...]] = None
    parameter_type: str = "other"
    dtype: str = "float32"
    memory_bytes: int = 0
    flops: float = 0.0
    depth: int = 0
    parent: str = ""
    children: List[str] = field(default_factory=list)
    activation_stats: Optional[Dict[str, float]] = None


@dataclass
class ModelProfile:
    """Complete model profile."""
    name: str = "model"
    total_parameters: int = 0
    trainable_parameters: int = 0
    frozen_parameters: int = 0
    parameter_types: Dict[str, int] = field(default_factory=dict)
    layers: List[LayerInfo] = field(default_factory=list)
    total_flops: float = 0.0
    total_memory_bytes: int = 0
    dtype: str = "float32"
    model_size_mb: float = 0.0
    model_size_gb: float = 0.0
    depth: int = 0
    num_layers: int = 0
    profile_time: float = 0.0

    def summary(self) -> str:
        """Get a text summary of the model profile."""
        lines = [
            f"Model Profile: {self.name}",
            f"  Total parameters:    {self.total_parameters:>15,}",
            f"  Trainable parameters:{self.trainable_parameters:>15,}",
            f"  Frozen parameters:   {self.frozen_parameters:>15,}",
            f"  Model size:          {self.model_size_mb:>12.2f} MB ({self.model_size_gb:.4f} GB)",
            f"  Estimated FLOPs:     {self.total_flops:>15,.0f}",
            f"  Memory (params):     {self.total_memory_bytes / (1024**2):>12.2f} MB",
            f"  Number of layers:    {self.num_layers:>15,}",
            f"  Max depth:           {self.depth:>15,}",
            f"  Dtype:               {self.dtype:>15}",
        ]
        if self.parameter_types:
            lines.append("  Parameter breakdown:")
            for ptype, count in sorted(self.parameter_types.items(), key=lambda x: -x[1]):
                pct = (count / self.total_parameters * 100) if self.total_parameters > 0 else 0
                lines.append(f"    {ptype:<20} {count:>15,} ({pct:>5.1f}%)")
        return "\n".join(lines)


@dataclass
class DistributionStats:
    """Statistical distribution of parameter values."""
    name: str = ""
    min_val: float = 0.0
    max_val: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    num_zeros: int = 0
    num_parameters: int = 0
    sparsity: float = 0.0
    norm: float = 0.0
    histogram: List[int] = field(default_factory=list)


# ============================================================================
# ModelProfiler
# ============================================================================

class ModelProfiler:
    """
    Profile model architecture: parameter counts, FLOPs, memory estimation.

    Supports any model object that exposes parameters via named_parameters(),
    state_dict(), or __dict__ with nested module attributes.

    Example:
        profiler = ModelProfiler()
        profile = profiler.profile(model)
        print(profile.summary())
    """

    def __init__(self, dtype_bytes: Optional[Dict[str, int]] = None):
        """Initialize the model profiler.

        Args:
            dtype_bytes: Mapping from dtype name to bytes per element.
        """
        self._dtype_bytes = dtype_bytes or {
            "float32": 4, "float64": 8, "float16": 2, "bfloat16": 2,
            "int32": 4, "int64": 8, "int16": 2, "int8": 1,
            "uint8": 1, "bool": 1, "float": 4, "double": 8,
        }

    def profile(self, model: Any, input_size: Optional[Tuple[int, ...]] = None) -> ModelProfile:
        """Profile a model's architecture and compute estimates.

        Args:
            model: Model to profile.
            input_size: Optional input shape for FLOPs estimation (batch, seq_len, ...).

        Returns:
            ModelProfile with comprehensive analysis.
        """
        start_time = time.time()
        profile = ModelProfile()
        profile.name = getattr(model, "__class__", type(model)).__name__

        # Extract parameter information
        param_info = self._extract_parameters(model)
        profile.total_parameters = param_info["total"]
        profile.trainable_parameters = param_info["trainable"]
        profile.frozen_parameters = param_info["total"] - param_info["trainable"]
        profile.parameter_types = param_info["types"]
        profile.dtype = param_info["dtype"]

        # Memory estimation
        bytes_per_param = self._dtype_bytes.get(profile.dtype, 4)
        profile.total_memory_bytes = profile.total_parameters * bytes_per_param
        profile.model_size_mb = profile.total_memory_bytes / (1024 * 1024)
        profile.model_size_gb = profile.model_size_mb / 1024

        # Extract layer structure
        layers = self._extract_layers(model)
        profile.layers = layers
        profile.num_layers = len(layers)
        if layers:
            profile.depth = max(l.depth for l in layers) + 1

        # FLOPs estimation
        if input_size is not None:
            profile.total_flops = self.estimate_flops(model, input_size, profile)
        else:
            profile.total_flops = sum(l.flops for l in layers)

        profile.profile_time = time.time() - start_time
        return profile

    def count_parameters(
        self,
        model: Any,
        trainable_only: bool = False,
    ) -> int:
        """Count model parameters.

        Args:
            model: Model to count parameters for.
            trainable_only: Whether to count only trainable parameters.

        Returns:
            Number of parameters.
        """
        param_info = self._extract_parameters(model)
        if trainable_only:
            return param_info["trainable"]
        return param_info["total"]

    def estimate_flops(
        self,
        model: Any,
        input_size: Tuple[int, ...],
        profile: Optional[ModelProfile] = None,
    ) -> float:
        """Estimate FLOPs for a forward pass.

        Uses simplified analytical estimation based on layer types and dimensions.

        Args:
            model: Model to estimate FLOPs for.
            input_size: Input shape (batch_size, sequence_length, ...).
            profile: Optional pre-computed ModelProfile.

        Returns:
            Estimated FLOPs per forward pass.
        """
        if profile is None:
            profile = self.profile(model, input_size)

        total_flops = 0.0
        batch_size = input_size[0] if len(input_size) > 0 else 1
        seq_len = input_size[1] if len(input_size) > 1 else 1

        for layer in profile.layers:
            layer_flops = self._estimate_layer_flops(
                layer, batch_size, seq_len, input_size
            )
            total_flops += layer_flops

        return total_flops

    def estimate_memory(
        self,
        model: Any,
        batch_size: int = 1,
        seq_len: int = 512,
        precision: str = "float32",
    ) -> Dict[str, float]:
        """Estimate memory usage for inference.

        Args:
            model: Model to estimate memory for.
            batch_size: Batch size.
            seq_len: Sequence length.
            precision: Model precision.

        Returns:
            Dictionary with memory estimates in GB.
        """
        profile = self.profile(model)
        bytes_per_param = self._dtype_bytes.get(precision, 4)

        # Parameter memory
        param_memory_gb = (profile.total_parameters * bytes_per_param) / (1024 ** 3)

        # Activation memory (rough estimate: 2x parameter memory per layer for
        # activations in the forward pass, with some factor for sequence length)
        num_layers = max(profile.num_layers, 1)
        activation_per_layer = (batch_size * seq_len * profile.total_parameters // num_layers * bytes_per_param)
        activation_memory_gb = min(activation_per_layer * 2 / (1024 ** 3), param_memory_gb * 4)

        # KV cache memory for transformer models (per layer: 2 * batch * heads * seq * head_dim * bytes)
        # Approximate: num_layers * 2 * batch_size * seq_len * hidden_dim * bytes_per_param
        hidden_dim = self._estimate_hidden_dim(profile)
        kv_cache_gb = (num_layers * 2 * batch_size * seq_len * hidden_dim * bytes_per_param) / (1024 ** 3)

        # Gradient memory (same as parameter memory for training)
        gradient_memory_gb = param_memory_gb

        # Optimizer memory (2x parameter memory for Adam: momentum + variance)
        optimizer_memory_gb = param_memory_gb * 2

        total_inference = param_memory_gb + activation_memory_gb
        total_training = param_memory_gb * 4 + activation_memory_gb + optimizer_memory_gb

        return {
            "parameters_gb": param_memory_gb,
            "activations_gb": activation_memory_gb,
            "kv_cache_gb": kv_cache_gb,
            "gradients_gb": gradient_memory_gb,
            "optimizer_gb": optimizer_memory_gb,
            "total_inference_gb": total_inference + kv_cache_gb,
            "total_training_gb": total_training,
            "batch_size": float(batch_size),
            "sequence_length": float(seq_len),
        }

    def _extract_parameters(self, model: Any) -> Dict[str, Any]:
        """Extract parameter information from a model.

        Args:
            model: Model object.

        Returns:
            Dictionary with total, trainable, types, and dtype.
        """
        total = 0
        trainable = 0
        types: Dict[str, int] = defaultdict(int)
        dtype = "float32"

        params = self._get_parameters(model)
        for name, param in params:
            num = self._param_count(param)
            total += num

            is_trainable = True
            if hasattr(param, "requires_grad"):
                is_trainable = bool(param.requires_grad)
            if is_trainable:
                trainable += num

            # Determine parameter type
            name_lower = name.lower()
            if "weight" in name_lower:
                types["weight"] += num
            elif "bias" in name_lower:
                types["bias"] += num
            elif "embed" in name_lower:
                types["embedding"] += num
            elif "norm" in name_lower or "ln_" in name_lower:
                types["layernorm"] += num
            elif "position" in name_lower or "pos" in name_lower:
                types["position"] += num
            else:
                types["other"] += num

            # Detect dtype
            if hasattr(param, "dtype"):
                dtype_str = str(param.dtype)
                for known_dtype in self._dtype_bytes:
                    if known_dtype in dtype_str:
                        dtype = known_dtype
                        break

        return {
            "total": total,
            "trainable": trainable,
            "types": dict(types),
            "dtype": dtype,
        }

    def _get_parameters(self, model: Any) -> List[Tuple[str, Any]]:
        """Get parameters from a model object.

        Supports PyTorch-like models with named_parameters() or state_dict(),
        as well as generic Python objects with attributes.

        Args:
            model: Model object.

        Returns:
            List of (name, parameter) tuples.
        """
        params = []

        # Try named_parameters (PyTorch)
        if hasattr(model, "named_parameters"):
            try:
                for name, param in model.named_parameters():
                    params.append((name, param))
                return params
            except Exception:
                pass

        # Try state_dict
        if hasattr(model, "state_dict"):
            try:
                sd = model.state_dict()
                for name, param in sd.items():
                    params.append((name, param))
                return params
            except Exception:
                pass

        # Try parameters()
        if hasattr(model, "parameters"):
            try:
                for idx, param in enumerate(model.parameters()):
                    params.append((f"param_{idx}", param))
                return params
            except Exception:
                pass

        # Generic attribute traversal
        self._traverse_attributes(model, "", params, set())
        return params

    def _traverse_attributes(
        self,
        obj: Any,
        prefix: str,
        params: List[Tuple[str, Any]],
        visited: Set[int],
    ) -> None:
        """Recursively traverse object attributes to find parameters.

        Args:
            obj: Object to traverse.
            prefix: Name prefix.
            params: Accumulator for found parameters.
            visited: Set of visited object IDs to prevent cycles.
        """
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, (int, float, str, bool, type(None))):
            return

        # Check if this looks like a parameter/tensor
        if hasattr(obj, "shape") and hasattr(obj, "numel"):
            if prefix:
                params.append((prefix, obj))
            return

        if hasattr(obj, "size") and hasattr(obj, "dim"):
            if prefix:
                params.append((prefix, obj))
            return

        # Try dictionary-like access
        if hasattr(obj, "items"):
            try:
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else str(key)
                    self._traverse_attributes(value, new_prefix, params, visited)
            except (TypeError, AttributeError):
                pass

        # Try named attribute access
        if hasattr(obj, "__dict__"):
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._traverse_attributes(value, new_prefix, params, visited)

        # Try modules list (PyTorch Sequential, ModuleList)
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
            try:
                for idx, child in enumerate(obj):
                    new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                    self._traverse_attributes(child, new_prefix, params, visited)
            except (TypeError, KeyError):
                pass

    def _extract_layers(self, model: Any) -> List[LayerInfo]:
        """Extract layer information from a model.

        Args:
            model: Model object.

        Returns:
            List of LayerInfo objects.
        """
        layers = []

        # Try named_modules (PyTorch)
        if hasattr(model, "named_modules"):
            try:
                name_map = {}
                for name, module in model.named_modules():
                    name_map[name] = module
                    layer = self._build_layer_info(name, module, name_map)
                    layers.append(layer)
                return layers
            except Exception:
                pass

        # Generic traversal
        self._traverse_layers(model, "", layers, 0, set(), "")
        return layers

    def _build_layer_info(
        self, name: str, module: Any, name_map: Dict[str, Any]
    ) -> LayerInfo:
        """Build LayerInfo from a named module.

        Args:
            name: Module name.
            module: Module object.
            name_map: Map of all module names.

        Returns:
            LayerInfo object.
        """
        info = LayerInfo(name=name)

        # Count parameters in this module
        total_params = 0
        trainable_params = 0
        if hasattr(module, "parameters"):
            try:
                for p in module.parameters():
                    count = self._param_count(p)
                    total_params += count
                    if hasattr(p, "requires_grad") and p.requires_grad:
                        trainable_params += count
            except Exception:
                pass

        # Subtract child parameters to get only direct parameters
        child_prefix = name + "."
        child_params = 0
        for child_name in name_map:
            if child_name.startswith(child_prefix) and child_name != name:
                child_module = name_map[child_name]
                if hasattr(child_module, "parameters"):
                    try:
                        for p in child_module.parameters():
                            child_params += self._param_count(p)
                    except Exception:
                        pass

        info.num_parameters = max(total_params - child_params, 0)
        info.trainable_parameters = info.num_parameters  # Approximate
        info.memory_bytes = info.num_parameters * self._dtype_bytes.get(info.dtype, 4)

        # Detect layer type
        class_name = module.__class__.__name__.lower() if hasattr(module, "__class__") else ""
        if "linear" in class_name or "dense" in class_name:
            info.layer_type = "linear"
            info.flops = self._estimate_linear_flops(info.num_parameters)
        elif "attention" in class_name:
            info.layer_type = "attention"
        elif "mlp" in class_name or "feedforward" in class_name or "ffn" in class_name:
            info.layer_type = "feedforward"
        elif "embed" in class_name:
            info.layer_type = "embedding"
        elif "norm" in class_name or "layernorm" in class_name or "rmsnorm" in class_name:
            info.layer_type = "layernorm"
        elif "conv" in class_name:
            info.layer_type = "convolution"
        elif "rnn" in class_name or "lstm" in class_name or "gru" in class_name:
            info.layer_type = "rnn"
        elif "dropout" in class_name:
            info.layer_type = "dropout"
        elif "relu" in class_name or "gelu" in class_name or "sigmoid" in class_name:
            info.layer_type = "activation"
        else:
            info.layer_type = class_name or "custom"

        # Compute depth
        info.depth = name.count(".")

        # Output shape
        if hasattr(module, "output_shape"):
            info.output_shape = tuple(module.output_shape)
        elif hasattr(module, "out_features"):
            info.output_shape = (module.out_features,)

        return info

    def _traverse_layers(
        self,
        obj: Any,
        prefix: str,
        layers: List[LayerInfo],
        depth: int,
        visited: Set[int],
        parent: str,
    ) -> None:
        """Recursively traverse to find layers.

        Args:
            obj: Object to traverse.
            prefix: Name prefix.
            layers: Accumulator for LayerInfo.
            depth: Current depth.
            visited: Visited objects set.
            parent: Parent name.
        """
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, (int, float, str, bool, type(None))):
            return

        # Check if this is a leaf layer (has parameters but no child modules)
        has_params = False
        if hasattr(obj, "parameters"):
            try:
                param_list = list(obj.parameters())
                has_params = len(param_list) > 0
            except Exception:
                pass

        if hasattr(obj, "__dict__") and has_params and prefix:
            info = LayerInfo(
                name=prefix,
                depth=depth,
                parent=parent,
                layer_type=getattr(obj.__class__, "__name__", "unknown").lower(),
            )
            total_params = 0
            if hasattr(obj, "parameters"):
                try:
                    for p in obj.parameters():
                        total_params += self._param_count(p)
                except Exception:
                    pass
            info.num_parameters = total_params
            info.trainable_parameters = total_params
            info.memory_bytes = total_params * self._dtype_bytes.get(info.dtype, 4)
            layers.append(info)
            return

        # Traverse children
        if hasattr(obj, "__dict__"):
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._traverse_layers(value, new_prefix, layers, depth + 1, visited, prefix)

        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
            try:
                for idx, child in enumerate(obj):
                    new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                    self._traverse_layers(child, new_prefix, layers, depth + 1, visited, prefix)
            except (TypeError, KeyError):
                pass

    def _estimate_layer_flops(
        self,
        layer: LayerInfo,
        batch_size: int,
        seq_len: int,
        input_size: Tuple[int, ...],
    ) -> float:
        """Estimate FLOPs for a single layer.

        Args:
            layer: Layer information.
            batch_size: Batch size.
            seq_len: Sequence length.
            input_size: Input dimensions.

        Returns:
            Estimated FLOPs.
        """
        if layer.flops > 0:
            return layer.flops * batch_size * seq_len

        params = layer.num_parameters
        if params == 0:
            return 0.0

        ltype = layer.layer_type.lower()

        if "linear" in ltype or "dense" in ltype:
            return self._estimate_linear_flops(params) * batch_size * seq_len
        elif "attention" in ltype:
            # Attention: Q*K^T (d*d*seq), softmax, A*V (d*seq*d)
            return params * seq_len * 4 * batch_size
        elif "embed" in ltype:
            return params * batch_size * seq_len  # Lookup is cheap, count as 1 op
        elif "norm" in ltype or "layernorm" in ltype:
            return params * batch_size * seq_len * 4
        elif "conv" in ltype:
            return params * batch_size * seq_len * 2
        elif "rnn" in ltype or "lstm" in ltype:
            return params * batch_size * seq_len * 4
        else:
            return params * batch_size * seq_len * 2

    @staticmethod
    def _estimate_linear_flops(num_params: int) -> float:
        """Estimate FLOPs for a linear layer given parameter count.

        For a linear layer with in_features x out_features weight matrix:
        FLOPs = 2 * in_features * out_features (multiply-add).

        Args:
            num_params: Number of parameters (including bias).

        Returns:
            Estimated FLOPs.
        """
        if num_params <= 0:
            return 0.0
        # Assume roughly square or rectangular weight matrix
        # FLOPs ≈ 2 * weight_elements (for matmul) + bias_add
        return float(num_params * 2)

    @staticmethod
    def _param_count(param: Any) -> int:
        """Get the number of elements in a parameter.

        Args:
            param: Parameter/tensor object.

        Returns:
            Number of elements.
        """
        if hasattr(param, "numel"):
            try:
                return int(param.numel())
            except Exception:
                pass
        if hasattr(param, "shape"):
            try:
                count = 1
                for dim in param.shape:
                    count *= int(dim)
                return count
            except Exception:
                pass
        if hasattr(param, "size"):
            try:
                count = 1
                for dim in param.size():
                    count *= int(dim)
                return count
            except Exception:
                pass
        if isinstance(param, (list, tuple)):
            count = 0
            for item in param:
                if isinstance(item, (list, tuple)):
                    count += ModelProfiler._param_count(item)
                else:
                    count += 1
            return count
        if isinstance(param, (int, float)):
            return 1
        return 0

    @staticmethod
    def _estimate_hidden_dim(profile: ModelProfile) -> int:
        """Estimate hidden dimension from model profile.

        Args:
            profile: Model profile.

        Returns:
            Estimated hidden dimension.
        """
        # Look at linear layers to estimate hidden dim
        for layer in profile.layers:
            if "linear" in layer.layer_type.lower() or "dense" in layer.layer_type.lower():
                if layer.output_shape:
                    return layer.output_shape[-1]
        # Fallback: total params / (12 * num_layers) for typical transformer
        if profile.num_layers > 0 and profile.total_parameters > 0:
            return profile.total_parameters // (12 * profile.num_layers)
        return 512  # Reasonable default


# ============================================================================
# ArchitecturePrinter
# ============================================================================

class ArchitecturePrinter:
    """
    Print model architecture as a text-based tree.

    Creates a hierarchical tree representation of model layers
    with parameter counts and type annotations.

    Example:
        printer = ArchitecturePrinter(max_depth=4, show_params=True)
        tree = printer.print_model(model)
        print(tree)
    """

    def __init__(
        self,
        max_depth: int = 6,
        show_params: bool = True,
        show_shapes: bool = True,
        indent: str = "  ",
        verbose: bool = False,
    ):
        """Initialize the architecture printer.

        Args:
            max_depth: Maximum depth to print.
            show_params: Whether to show parameter counts.
            show_shapes: Whether to show output shapes.
            indent: Indentation string per level.
            verbose: Show additional details.
        """
        self._max_depth = max_depth
        self._show_params = show_params
        self._show_shapes = show_shapes
        self._indent = indent
        self._verbose = verbose

    def print_model(self, model: Any, max_depth: Optional[int] = None) -> str:
        """Print the model architecture as a tree.

        Args:
            model: Model to visualize.
            max_depth: Override max depth.

        Returns:
            Multi-line tree string.
        """
        depth = max_depth if max_depth is not None else self._max_depth
        lines = []

        # Try named_modules (PyTorch)
        if hasattr(model, "named_modules"):
            try:
                modules = list(model.named_modules())
                if modules:
                    return self._print_from_modules(modules, depth)
            except Exception:
                pass

        # Generic object traversal
        class_name = getattr(model, "__class__", type(model)).__name__
        lines.append(f"{class_name}")
        self._print_object_tree(model, "", True, 0, depth, lines)
        return "\n".join(lines)

    def layer_summary(self, layer: Any) -> str:
        """Get a summary string for a single layer.

        Args:
            layer: Layer object.

        Returns:
            Summary string.
        """
        parts = []
        class_name = getattr(layer, "__class__", type(layer)).__name__
        parts.append(class_name)

        # Parameters
        num_params = 0
        if hasattr(layer, "parameters"):
            try:
                for p in layer.parameters():
                    num_params += ModelProfiler._param_count(p)
            except Exception:
                pass
        if self._show_params and num_params > 0:
            parts.append(f"[{num_params:,} params]")

        # Output shape
        if self._show_shapes:
            shape = None
            if hasattr(layer, "output_shape"):
                shape = layer.output_shape
            elif hasattr(layer, "out_features"):
                shape = (layer.out_features,)
            elif hasattr(layer, "out_channels"):
                shape = (layer.out_channels,)
            if shape:
                parts.append(f"-> {tuple(shape)}")

        return " ".join(parts)

    def parameter_table(self, model: Any, sort_by: str = "name") -> str:
        """Generate a table of all parameters.

        Args:
            model: Model with parameters.
            sort_by: Sort by "name", "size", or "type".

        Returns:
            Multi-line table string.
        """
        params = []
        if hasattr(model, "named_parameters"):
            try:
                for name, param in model.named_parameters():
                    count = ModelProfiler._param_count(param)
                    dtype = str(getattr(param, "dtype", "unknown"))
                    trainable = hasattr(param, "requires_grad") and param.requires_grad
                    params.append({
                        "name": name,
                        "count": count,
                        "dtype": dtype,
                        "trainable": trainable,
                    })
            except Exception:
                pass

        if not params:
            return "(no parameters found)"

        if sort_by == "size":
            params.sort(key=lambda x: -x["count"])
        elif sort_by == "type":
            params.sort(key=lambda x: x["dtype"])

        total = sum(p["count"] for p in params)

        # Format table
        name_width = max(len(p["name"]) for p in params)
        name_width = min(name_width, 60)
        count_width = 15
        dtype_width = 12

        lines = []
        lines.append(f"{'Name':<{name_width}} {'Count':>{count_width}} {'Trainable':>10} {'Dtype':>{dtype_width}}")
        lines.append(f"{'-' * name_width} {'-' * count_width} {'-' * 10} {'-' * dtype_width}")

        for p in params:
            name = p["name"][:name_width - 3] + "..." if len(p["name"]) > name_width else p["name"]
            trainable = "Y" if p["trainable"] else "N"
            lines.append(f"{name:<{name_width}} {p['count']:>{count_width},} {trainable:>10} {p['dtype']:>{dtype_width}}")

        lines.append(f"{'-' * name_width} {'-' * count_width} {'-' * 10} {'-' * dtype_width}")
        lines.append(f"{'TOTAL':<{name_width}} {total:>{count_width},}")

        return "\n".join(lines)

    def _print_from_modules(
        self,
        modules: List[Tuple[str, Any]],
        max_depth: int,
    ) -> str:
        """Print architecture from named_modules list.

        Args:
            modules: List of (name, module) tuples.
            max_depth: Maximum depth.

        Returns:
            Tree string.
        """
        lines = []
        # Build tree structure
        prev_depth = -1
        stack: List[str] = []

        for i, (name, module) in enumerate(modules):
            depth = name.count(".")
            if depth > max_depth:
                continue

            # Determine prefix
            prefix = ""
            is_last = i == len(modules) - 1

            # Look ahead to determine if this is the last at this depth
            next_same_depth = False
            for j in range(i + 1, len(modules)):
                next_depth = modules[j][0].count(".")
                if next_depth <= depth:
                    if next_depth == depth:
                        next_same_depth = True
                    break

            # Build tree prefix
            name_parts = name.split(".") if name else []
            display_name = name_parts[-1] if name_parts else "root"

            # Build indentation based on parent structure
            indent_str = ""
            for d in range(depth):
                # Check if parent at depth d has more children
                parent_prefix = ".".join(name_parts[:d + 1]) if name_parts else ""
                has_more_children = any(
                    modules[k][0].count(".") == d + 1
                    and modules[k][0].startswith(parent_prefix + ".")
                    and k > i
                    for k in range(i)
                )
                indent_str += BRANCH_V if has_more_children else BRANCH_S

            if depth == 0:
                connector = ""
            elif not next_same_depth:
                connector = BRANCH_L
            else:
                connector = BRANCH_T

            # Class name and info
            class_name = module.__class__.__name__

            # Count parameters
            num_params = 0
            if hasattr(module, "parameters"):
                try:
                    num_params = sum(ModelProfiler._param_count(p) for p in module.parameters())
                except Exception:
                    pass

            info_str = class_name
            if self._show_params and num_params > 0:
                info_str += f" [{num_params:,}]"

            line = f"{indent_str}{connector}{display_name} ({info_str})"
            lines.append(line)

        return "\n".join(lines)

    def _print_object_tree(
        self,
        obj: Any,
        prefix: str,
        is_last: bool,
        depth: int,
        max_depth: int,
        lines: List[str],
    ) -> None:
        """Recursively print object tree.

        Args:
            obj: Object to print.
            prefix: Tree prefix string.
            is_last: Whether this is the last child.
            depth: Current depth.
            max_depth: Maximum depth.
            lines: Output line accumulator.
        """
        if depth > max_depth:
            return

        connector = BRANCH_L if is_last else BRANCH_T
        extension = BRANCH_S if is_last else BRANCH_V

        if hasattr(obj, "__class__"):
            class_name = obj.__class__.__name__
        else:
            class_name = type(obj).__name__

        info = self.layer_summary(obj)
        lines.append(f"{prefix}{connector}{info}")

        children = []
        if hasattr(obj, "__dict__"):
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, (int, float, str, bool, type(None))):
                    continue
                children.append((key, value))

        for i, (key, child) in enumerate(children):
            is_child_last = i == len(children) - 1
            self._print_object_tree(
                child,
                prefix + extension,
                is_child_last,
                depth + 1,
                max_depth,
                lines,
            )


# ============================================================================
# ParameterDistributionAnalyzer
# ============================================================================

class ParameterDistributionAnalyzer:
    """
    Analyze parameter value distributions and sparsity.

    Computes histograms, statistical measures, and identifies
    potential issues in model parameter distributions.

    Example:
        analyzer = ParameterDistributionAnalyzer(num_bins=50)
        stats = analyzer.analyze_layer(layer.weight)
        print(stats.summary())
    """

    def __init__(
        self,
        num_bins: int = 50,
        detect_outliers: bool = True,
        outlier_threshold: float = 4.0,
    ):
        """Initialize the distribution analyzer.

        Args:
            num_bins: Number of histogram bins.
            detect_outliers: Whether to detect outlier values.
            outlier_threshold: Standard deviations for outlier detection.
        """
        self._num_bins = num_bins
        self._detect_outliers = detect_outliers
        self._outlier_threshold = outlier_threshold

    def analyze_distribution(
        self,
        values: List[float],
        name: str = "layer",
    ) -> DistributionStats:
        """Analyze the distribution of parameter values.

        Args:
            values: Flattened list of parameter values.
            name: Name for the distribution.

        Returns:
            DistributionStats with computed statistics.
        """
        if not values:
            return DistributionStats(name=name)

        stats = DistributionStats(name=name, num_parameters=len(values))

        # Basic statistics
        stats.min_val = min(values)
        stats.max_val = max(values)
        stats.mean = sum(values) / len(values)

        variance = sum((x - stats.mean) ** 2 for x in values) / len(values)
        stats.std = math.sqrt(variance) if variance > 0 else 0.0

        # Sorted values for percentiles
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        stats.median = sorted_vals[n // 2]
        stats.q1 = sorted_vals[n // 4]
        stats.q3 = sorted_vals[3 * n // 4]

        # Count zeros (for sparsity)
        stats.num_zeros = sum(1 for v in values if v == 0.0 or abs(v) < 1e-10)
        stats.sparsity = stats.num_zeros / n if n > 0 else 0.0

        # Norm
        stats.norm = math.sqrt(sum(v * v for v in values))

        # Histogram
        stats.histogram = self._compute_histogram(values, stats.min_val, stats.max_val)

        # Higher-order statistics
        if stats.std > 0:
            # Skewness
            m3 = sum((x - stats.mean) ** 3 for x in values) / n
            stats.skewness = m3 / (stats.std ** 3)

            # Kurtosis
            m4 = sum((x - stats.mean) ** 4 for x in values) / n
            stats.kurtosis = m4 / (stats.std ** 4) - 3.0

        return stats

    def analyze_layer(self, param: Any, name: str = "layer") -> DistributionStats:
        """Analyze distribution of a single layer's parameters.

        Args:
            param: Parameter/tensor object.
            name: Layer name.

        Returns:
            DistributionStats.
        """
        values = self._flatten_param(param)
        return self.analyze_distribution(values, name)

    def analyze_model(
        self,
        model: Any,
        max_layers: int = 50,
    ) -> Dict[str, DistributionStats]:
        """Analyze all layers in a model.

        Args:
            model: Model to analyze.
            max_layers: Maximum layers to analyze.

        Returns:
            Dictionary mapping layer name to DistributionStats.
        """
        results = {}

        params = []
        if hasattr(model, "named_parameters"):
            try:
                params = list(model.named_parameters())
            except Exception:
                pass

        for name, param in params[:max_layers]:
            try:
                stats = self.analyze_layer(param, name)
                results[name] = stats
            except Exception:
                continue

        return results

    def sparsity_report(self, model: Any) -> str:
        """Generate a sparsity analysis report for the model.

        Args:
            model: Model to analyze.

        Returns:
            Multi-line report string.
        """
        all_stats = self.analyze_model(model)
        if not all_stats:
            return "(no parameters found)"

        lines = []
        lines.append(f"{BOX_TL}{'Parameter Sparsity Report':^60}{BOX_TR}")
        lines.append(
            f"{BOX_V}{'Layer':<40}{'Params':>10}{'Sparse':>8}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{BOX_H*40}{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*8}{BOX_V}"
        )

        total_params = 0
        total_sparse = 0

        for name, stats in sorted(all_stats.items(), key=lambda x: -x[1].num_parameters):
            display_name = name[-39:] if len(name) > 39 else name
            sparse_pct = stats.sparsity * 100
            lines.append(
                f"{BOX_V}{display_name:<40}{stats.num_parameters:>10,}{sparse_pct:>7.1f}%{BOX_V}"
            )
            total_params += stats.num_parameters
            total_sparse += stats.num_zeros

        overall_sparsity = (total_sparse / total_params * 100) if total_params > 0 else 0
        lines.append(
            f"{BOX_V}{BOX_H*40}{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*8}{BOX_V}"
        )
        lines.append(
            f"{BOX_V}{'TOTAL':<40}{total_params:>10,}{overall_sparsity:>7.1f}%{BOX_V}"
        )
        lines.append(BOX_BL + BOX_H * 60 + BOX_BR)

        return "\n".join(lines)

    def distribution_report(
        self,
        stats: DistributionStats,
        width: int = 60,
    ) -> str:
        """Format a distribution report with text histogram.

        Args:
            stats: DistributionStats to report.
            width: Histogram width.

        Returns:
            Multi-line report string.
        """
        lines = []
        lines.append(f"Distribution: {stats.name}")
        lines.append(f"  Count:    {stats.num_parameters:>15,}")
        lines.append(f"  Min:      {stats.min_val:>15.6f}")
        lines.append(f"  Max:      {stats.max_val:>15.6f}")
        lines.append(f"  Mean:     {stats.mean:>15.6f}")
        lines.append(f"  Std:      {stats.std:>15.6f}")
        lines.append(f"  Median:   {stats.median:>15.6f}")
        lines.append(f"  Q1:       {stats.q1:>15.6f}")
        lines.append(f"  Q3:       {stats.q3:>15.6f}")
        lines.append(f"  Skewness: {stats.skewness:>15.6f}")
        lines.append(f"  Kurtosis: {stats.kurtosis:>15.6f}")
        lines.append(f"  Sparsity: {stats.sparsity * 100:>14.2f}%")
        lines.append(f"  L2 Norm:  {stats.norm:>15.6f}")

        # Text histogram
        if stats.histogram:
            lines.append("")
            hist_lines = self._render_histogram(stats.histogram, stats.min_val, stats.max_val, width)
            lines.extend(hist_lines)

        return "\n".join(lines)

    def _compute_histogram(
        self, values: List[float], v_min: float, v_max: float
    ) -> List[int]:
        """Compute histogram bin counts.

        Args:
            values: Data values.
            v_min: Minimum value.
            v_max: Maximum value.

        Returns:
            List of bin counts.
        """
        bins = [0] * self._num_bins
        if v_max == v_min:
            bins[0] = len(values)
            return bins

        for v in values:
            idx = int((v - v_min) / (v_max - v_min) * self._num_bins)
            idx = max(0, min(self._num_bins - 1, idx))
            bins[idx] += 1

        return bins

    def _render_histogram(
        self, bins: List[int], v_min: float, v_max: float, width: int
    ) -> List[str]:
        """Render histogram as text.

        Args:
            bins: Bin counts.
            v_min: Minimum value.
            v_max: Maximum value.
            width: Chart width.

        Returns:
            List of text lines.
        """
        if not bins:
            return []

        max_count = max(bins)
        if max_count == 0:
            return ["(all zeros)"]

        num_rows = 8
        lines = []

        for row in range(num_rows):
            threshold = max_count * (num_rows - row) / num_rows
            line = ""
            for count in bins:
                filled = count / max_count * (width // len(bins))
                filled_int = int(filled)
                if count >= threshold:
                    if filled_int >= 1:
                        line += PROGRESS_FULL * filled_int
                    else:
                        line += PROGRESS_QUARTER
                else:
                    line += " " * max(filled_int, 1)
            lines.append(f"  {line}")

        # X-axis labels
        x_min = f"{v_min:.4f}"
        x_max = f"{v_max:.4f}"
        x_label = f"  {x_min}{' ' * (width - len(x_min) - len(x_max))}{x_max}"
        lines.append(x_label)

        return lines

    @staticmethod
    def _flatten_param(param: Any) -> List[float]:
        """Flatten a parameter to a list of floats.

        Args:
            param: Parameter/tensor.

        Returns:
            List of float values.
        """
        values = []

        if hasattr(param, "tolist"):
            try:
                flat = param.tolist()
                if isinstance(flat, list):
                    ParameterDistributionAnalyzer._flatten_list(flat, values)
                else:
                    values.append(float(flat))
                return values
            except Exception:
                pass

        if hasattr(param, "numpy"):
            try:
                arr = param.numpy()
                flat = arr.flatten().tolist()
                return [float(x) for x in flat]
            except Exception:
                pass

        if hasattr(param, "flatten"):
            try:
                flat = param.flatten()
                return ParameterDistributionAnalyzer._flatten_param(flat)
            except Exception:
                pass

        if isinstance(param, (list, tuple)):
            ParameterDistributionAnalyzer._flatten_list(param, values)
            return values

        if isinstance(param, (int, float)):
            return [float(param)]

        return values

    @staticmethod
    def _flatten_list(obj: Any, values: List[float]) -> None:
        """Recursively flatten a nested list.

        Args:
            obj: Object to flatten.
            values: Output accumulator.
        """
        if isinstance(obj, (list, tuple)):
            for item in obj:
                ParameterDistributionAnalyzer._flatten_list(item, values)
        elif isinstance(obj, (int, float)):
            values.append(float(obj))
        elif hasattr(obj, "item"):
            try:
                values.append(float(obj.item()))
            except Exception:
                pass


# ============================================================================
# ModelDiff
# ============================================================================

class ModelDiff:
    """
    Compare two model architectures and parameters.

    Identifies structural differences, parameter changes, and
    computes similarity metrics between two models.

    Example:
        differ = ModelDiff()
        diff = differ.compare(model_a, model_b)
        print(differ.format_report(diff))
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize the model differ.

        Args:
            tolerance: Numerical tolerance for considering parameters equal.
        """
        self._tolerance = tolerance

    def compare(
        self,
        model_a: Any,
        model_b: Any,
        label_a: str = "model_a",
        label_b: str = "model_b",
    ) -> "ModelDiffResult":
        """Compare two models.

        Args:
            model_a: First model.
            model_b: Second model.
            label_a: Label for first model.
            label_b: Label for second model.

        Returns:
            ModelDiffResult with comparison results.
        """
        result = ModelDiffResult(label_a=label_a, label_b=label_b)

        # Profile both models
        profiler = ModelProfiler()
        profile_a = profiler.profile(model_a)
        profile_b = profiler.profile(model_b)

        # Compare parameter counts
        result.params_a = profile_a.total_parameters
        result.params_b = profile_b.total_parameters
        result.param_difference = profile_a.total_parameters - profile_b.total_parameters
        result.param_diff_pct = (
            abs(result.param_difference) / max(profile_a.total_parameters, 1) * 100
        )

        # Compare layers
        layers_a = {l.name: l for l in profile_a.layers}
        layers_b = {l.name: l for l in profile_b.layers}

        all_names = set(list(layers_a.keys()) + list(layers_b.keys()))

        for name in sorted(all_names):
            la = layers_a.get(name)
            lb = layers_b.get(name)

            if la is None:
                result.added_layers.append(name)
            elif lb is None:
                result.removed_layers.append(name)
            else:
                if la.layer_type != lb.layer_type:
                    result.changed_types.append((name, la.layer_type, lb.layer_type))
                if la.num_parameters != lb.num_parameters:
                    result.changed_sizes.append((
                        name, la.num_parameters, lb.num_parameters
                    ))

        # Compare parameter values if possible
        params_a = self._get_state_dict(model_a)
        params_b = self._get_state_dict(model_b)

        common_keys = set(params_a.keys()) & set(params_b.keys())
        total_cosine = 0.0
        total_l2 = 0.0
        compared = 0

        for key in common_keys:
            vals_a = ParameterDistributionAnalyzer._flatten_param(params_a[key])
            vals_b = ParameterDistributionAnalyzer._flatten_param(params_b[key])

            if len(vals_a) != len(vals_b):
                result.shape_mismatches.append(key)
                continue

            if not vals_a:
                continue

            # L2 distance
            l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(vals_a, vals_b)))
            total_l2 += l2

            # Cosine similarity
            dot = sum(a * b for a, b in zip(vals_a, vals_b))
            norm_a = math.sqrt(sum(a * a for a in vals_a))
            norm_b = math.sqrt(sum(b * b for b in vals_b))
            cosine = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
            total_cosine += cosine

            compared += 1

            # Check if significantly different
            if l2 > self._tolerance:
                result.diff_parameters.append((key, l2, cosine))

        result.avg_cosine_similarity = total_cosine / compared if compared > 0 else 1.0
        result.avg_l2_distance = total_l2 / compared if compared > 0 else 0.0
        result.num_compared_layers = compared

        # Overall similarity assessment
        if result.avg_cosine_similarity > 0.999:
            result.similarity = "identical"
        elif result.avg_cosine_similarity > 0.99:
            result.similarity = "very similar"
        elif result.avg_cosine_similarity > 0.95:
            result.similarity = "similar"
        elif result.avg_cosine_similarity > 0.8:
            result.similarity = "somewhat similar"
        else:
            result.similarity = "different"

        return result

    def format_report(self, result: "ModelDiffResult", verbose: bool = False) -> str:
        """Format a comparison report.

        Args:
            result: ModelDiffResult from compare().
            verbose: Include detailed per-parameter differences.

        Returns:
            Multi-line report string.
        """
        lines = []
        lines.append(f"{BOX_TL}{'Model Comparison Report':^66}{BOX_TR}")
        lines.append(
            f"{BOX_V} {result.label_a} vs {result.label_b}"
            f"{'':>{66 - len(result.label_a) - len(result.label_b) - 5}}{BOX_V}"
        )
        lines.append(f"{BOX_V}{BOX_H * 66}{BOX_V}")

        # Similarity
        sim_indicator = {
            "identical": "✓",
            "very similar": "✓",
            "similar": "~",
            "somewhat similar": "!",
            "different": "✗",
        }.get(result.similarity, "?")

        lines.append(
            f"{BOX_V} Similarity: {result.similarity} {sim_indicator}"
            f"{'':>{66 - len(result.similarity) - 16}}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Avg cosine similarity: {result.avg_cosine_similarity:.8f}"
            f"{'':>{66 - 30}}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Avg L2 distance:        {result.avg_l2_distance:.8f}"
            f"{'':>{66 - 30}}{BOX_V}"
        )
        lines.append(f"{BOX_V}{BOX_H * 66}{BOX_V}")

        # Parameter counts
        lines.append(
            f"{BOX_V} Parameters: {result.params_a:>15,} vs {result.params_b:>15,}"
            f"{'':>{66 - 50}}{BOX_V}"
        )
        lines.append(
            f"{BOX_V} Difference: {result.param_difference:>+15,} ({result.param_diff_pct:.2f}%)"
            f"{'':>{66 - 48}}{BOX_V}"
        )
        lines.append(f"{BOX_V}{BOX_H * 66}{BOX_V}")

        # Structural differences
        if result.added_layers:
            lines.append(f"{BOX_V} Added layers: {len(result.added_layers)}{BOX_V}")
            for name in result.added_layers[:10]:
                lines.append(f"{BOX_V}   + {name}{BOX_V}")

        if result.removed_layers:
            lines.append(f"{BOX_V} Removed layers: {len(result.removed_layers)}{BOX_V}")
            for name in result.removed_layers[:10]:
                lines.append(f"{BOX_V}   - {name}{BOX_V}")

        if result.changed_types:
            lines.append(f"{BOX_V} Changed types: {len(result.changed_types)}{BOX_V}")

        if result.changed_sizes:
            lines.append(f"{BOX_V} Changed sizes: {len(result.changed_sizes)}{BOX_V}")

        if result.shape_mismatches:
            lines.append(f"{BOX_V} Shape mismatches: {len(result.shape_mismatches)}{BOX_V}")

        # Parameter differences
        if verbose and result.diff_parameters:
            lines.append(f"{BOX_V}{BOX_H * 66}{BOX_V}")
            lines.append(f"{BOX_V} Parameter differences (top 20):{BOX_V}")
            sorted_diffs = sorted(result.diff_parameters, key=lambda x: -x[1])[:20]
            for name, l2, cosine in sorted_diffs:
                lines.append(
                    f"{BOX_V}  {name[:50]:<50} L2={l2:.6f} cos={cosine:.4f}{BOX_V}"
                )

        lines.append(BOX_BL + BOX_H * 66 + BOX_BR)
        return "\n".join(lines)

    @staticmethod
    def _get_state_dict(model: Any) -> Dict[str, Any]:
        """Get state dict from a model.

        Args:
            model: Model object.

        Returns:
            Dictionary of parameter name to value.
        """
        if hasattr(model, "state_dict"):
            try:
                return dict(model.state_dict())
            except Exception:
                pass

        result = {}
        if hasattr(model, "named_parameters"):
            try:
                for name, param in model.named_parameters():
                    result[name] = param
                return result
            except Exception:
                pass

        if hasattr(model, "__dict__"):
            for key, value in model.__dict__.items():
                if hasattr(value, "shape") or hasattr(value, "numel"):
                    result[key] = value

        return result


@dataclass
class ModelDiffResult:
    """Result of comparing two models."""
    label_a: str = "model_a"
    label_b: str = "model_b"
    params_a: int = 0
    params_b: int = 0
    param_difference: int = 0
    param_diff_pct: float = 0.0
    added_layers: List[str] = field(default_factory=list)
    removed_layers: List[str] = field(default_factory=list)
    changed_types: List[Tuple[str, str, str]] = field(default_factory=list)
    changed_sizes: List[Tuple[str, int, int]] = field(default_factory=list)
    shape_mismatches: List[str] = field(default_factory=list)
    diff_parameters: List[Tuple[str, float, float]] = field(default_factory=list)
    avg_cosine_similarity: float = 1.0
    avg_l2_distance: float = 0.0
    num_compared_layers: int = 0
    similarity: str = "unknown"


# ============================================================================
# LayerInspector
# ============================================================================

class LayerInspector:
    """
    Detailed analysis of individual model layers.

    Provides input/output shape analysis, parameter statistics,
    activation analysis, and layer health diagnostics.

    Example:
        inspector = LayerInspector()
        report = inspector.inspect(layer, input_tensor)
        print(inspector.format_report(report))
    """

    def __init__(
        self,
        compute_activation_stats: bool = True,
        detect_dead_neurons: bool = True,
        dead_neuron_threshold: float = 1e-8,
    ):
        """Initialize the layer inspector.

        Args:
            compute_activation_stats: Whether to compute activation statistics.
            detect_dead_neurons: Whether to detect dead neurons.
            dead_neuron_threshold: Threshold below which a neuron is considered dead.
        """
        self._compute_activation_stats = compute_activation_stats
        self._detect_dead_neurons = detect_dead_neurons
        self._dead_threshold = dead_neuron_threshold

    def inspect(
        self,
        layer: Any,
        input_data: Optional[Any] = None,
        name: str = "layer",
    ) -> "LayerInspectionReport":
        """Inspect a single layer.

        Args:
            layer: Layer object to inspect.
            input_data: Optional input data for forward pass analysis.
            name: Layer name.

        Returns:
            LayerInspectionReport with analysis results.
        """
        report = LayerInspectionReport(name=name)

        # Basic info
        report.class_name = getattr(layer, "__class__", type(layer)).__name__
        report.layer_type = self._detect_layer_type(layer)

        # Parameter info
        params = self._get_layer_params(layer)
        report.parameters = params
        report.total_params = sum(p["count"] for p in params)
        report.parameter_names = [p["name"] for p in params]

        # Shapes
        report.input_shape = self._get_input_shape(layer)
        report.output_shape = self._get_output_shape(layer)

        # Weight statistics
        for param_info in params:
            if "weight" in param_info["name"].lower():
                values = ParameterDistributionAnalyzer._flatten_param(param_info["value"])
                if values:
                    stats = {
                        "mean": sum(values) / len(values),
                        "std": math.sqrt(sum((x - sum(values)/len(values))**2 for x in values) / len(values)) if values else 0,
                        "min": min(values),
                        "max": max(values),
                        "numel": len(values),
                    }
                    stats["norm"] = math.sqrt(sum(v*v for v in values))
                    zeros = sum(1 for v in values if abs(v) < 1e-10)
                    stats["sparsity"] = zeros / len(values) if values else 0
                    report.weight_stats[param_info["name"]] = stats

        # Activation analysis
        if input_data is not None and self._compute_activation_stats:
            report.activation_stats = self._analyze_activations(layer, input_data)

        # Dead neuron detection
        if self._detect_dead_neurons and report.weight_stats:
            for wname, wstats in report.weight_stats.items():
                if wstats.get("std", 1) < self._dead_threshold:
                    report.potential_issues.append(
                        f"Near-zero weights in {wname} (std={wstats['std']:.2e})"
                    )

        # Health check
        report.health = self._check_layer_health(report)

        return report

    def format_report(self, report: "LayerInspectionReport") -> str:
        """Format an inspection report.

        Args:
            report: LayerInspectionReport.

        Returns:
            Multi-line report string.
        """
        lines = []
        lines.append(f"Layer Inspection: {report.name}")
        lines.append(f"  Class:      {report.class_name}")
        lines.append(f"  Type:       {report.layer_type}")
        lines.append(f"  Parameters: {report.total_params:,}")

        if report.input_shape:
            lines.append(f"  Input:      {report.input_shape}")
        if report.output_shape:
            lines.append(f"  Output:     {report.output_shape}")

        # Weight statistics
        if report.weight_stats:
            lines.append("  Weight Statistics:")
            for wname, wstats in report.weight_stats.items():
                short_name = wname.split(".")[-1]
                lines.append(f"    {short_name}:")
                lines.append(f"      shape:    {wstats.get('numel', 'N/A'):>15,}")
                lines.append(f"      mean:     {wstats.get('mean', 0):>15.6f}")
                lines.append(f"      std:      {wstats.get('std', 0):>15.6f}")
                lines.append(f"      min:      {wstats.get('min', 0):>15.6f}")
                lines.append(f"      max:      {wstats.get('max', 0):>15.6f}")
                lines.append(f"      norm:     {wstats.get('norm', 0):>15.6f}")
                lines.append(f"      sparsity: {wstats.get('sparsity', 0) * 100:>14.2f}%")

        # Activation statistics
        if report.activation_stats:
            lines.append("  Activation Statistics:")
            for key, val in report.activation_stats.items():
                if isinstance(val, float):
                    lines.append(f"    {key}: {val:.6f}")
                else:
                    lines.append(f"    {key}: {val}")

        # Health and issues
        if report.potential_issues:
            lines.append(f"  Issues ({len(report.potential_issues)}):")
            for issue in report.potential_issues:
                lines.append(f"    ! {issue}")

        health_indicator = {"healthy": "✓", "warning": "⚠", "unhealthy": "✗"}.get(
            report.health, "?"
        )
        lines.append(f"  Health: {report.health} {health_indicator}")

        return "\n".join(lines)

    def _detect_layer_type(self, layer: Any) -> str:
        """Detect the type of a layer.

        Args:
            layer: Layer object.

        Returns:
            Layer type string.
        """
        class_name = getattr(layer, "__class__", type(layer)).__name__.lower()

        type_map = {
            "linear": ("linear", "dense"),
            "conv": ("convolution", "conv1d", "conv2d", "conv3d"),
            "attention": ("attention", "multiheadattention", "selfattention"),
            "embedding": ("embedding",),
            "layernorm": ("layernorm", "rmsnorm", "batchnorm", "instancenorm"),
            "dropout": ("dropout",),
            "relu": ("relu", "gelu", "sigmoid", "tanh", "silu", "mish"),
            "rnn": ("rnn", "lstm", "gru"),
            "pooling": ("pooling", "maxpool", "avgpool", "adaptivepool"),
        }

        for ltype, patterns in type_map.items():
            for pattern in patterns:
                if pattern in class_name:
                    return ltype

        return "custom"

    def _get_layer_params(self, layer: Any) -> List[Dict[str, Any]]:
        """Get parameters of a layer.

        Args:
            layer: Layer object.

        Returns:
            List of parameter info dictionaries.
        """
        params = []

        if hasattr(layer, "named_parameters"):
            try:
                for name, param in layer.named_parameters():
                    params.append({
                        "name": name,
                        "value": param,
                        "count": ModelProfiler._param_count(param),
                        "dtype": str(getattr(param, "dtype", "unknown")),
                        "trainable": bool(getattr(param, "requires_grad", True)),
                    })
                return params
            except Exception:
                pass

        if hasattr(layer, "__dict__"):
            for key, value in layer.__dict__.items():
                if hasattr(value, "shape") or hasattr(value, "numel"):
                    params.append({
                        "name": key,
                        "value": value,
                        "count": ModelProfiler._param_count(value),
                        "dtype": str(getattr(value, "dtype", "unknown")),
                        "trainable": True,
                    })

        return params

    def _get_input_shape(self, layer: Any) -> Optional[Tuple[int, ...]]:
        """Get input shape of a layer.

        Args:
            layer: Layer object.

        Returns:
            Input shape tuple or None.
        """
        if hasattr(layer, "in_features"):
            return (layer.in_features,)
        if hasattr(layer, "in_channels"):
            return (layer.in_channels,)
        if hasattr(layer, "input_shape"):
            shape = layer.input_shape
            if hasattr(shape, "__iter__"):
                return tuple(shape)
            return (shape,)
        return None

    def _get_output_shape(self, layer: Any) -> Optional[Tuple[int, ...]]:
        """Get output shape of a layer.

        Args:
            layer: Layer object.

        Returns:
            Output shape tuple or None.
        """
        if hasattr(layer, "out_features"):
            return (layer.out_features,)
        if hasattr(layer, "out_channels"):
            return (layer.out_channels,)
        if hasattr(layer, "output_shape"):
            shape = layer.output_shape
            if hasattr(shape, "__iter__"):
                return tuple(shape)
            return (shape,)
        return None

    def _analyze_activations(self, layer: Any, input_data: Any) -> Dict[str, Any]:
        """Analyze layer activations by running a forward pass.

        Args:
            layer: Layer object.
            input_data: Input data.

        Returns:
            Dictionary with activation statistics.
        """
        try:
            if not hasattr(layer, "__call__") and not hasattr(layer, "forward"):
                return {"error": "Layer is not callable"}

            forward_fn = getattr(layer, "forward", layer)
            output = forward_fn(input_data)

            output_values = ParameterDistributionAnalyzer._flatten_param(output)
            if not output_values:
                return {"error": "Could not extract activation values"}

            stats = {}
            stats["output_mean"] = sum(output_values) / len(output_values)
            stats["output_std"] = math.sqrt(
                sum((x - stats["output_mean"]) ** 2 for x in output_values) / len(output_values)
            )
            stats["output_min"] = min(output_values)
            stats["output_max"] = max(output_values)
            stats["output_abs_mean"] = sum(abs(x) for x in output_values) / len(output_values)

            # Dead neurons (outputs very close to zero)
            if self._detect_dead_neurons:
                dead_count = sum(
                    1 for x in output_values if abs(x) < self._dead_threshold
                )
                stats["dead_output_ratio"] = dead_count / len(output_values)

                # Variance per output neuron
                if hasattr(output, "shape") and len(output.shape) >= 2:
                    try:
                        output_list = output.tolist() if hasattr(output, "tolist") else list(output)
                        if isinstance(output_list[0], list):
                            num_neurons = len(output_list[0])
                            neuron_vars = []
                            for n in range(min(num_neurons, 1000)):
                                col_vals = [row[n] for row in output_list if n < len(row)]
                                if col_vals:
                                    col_mean = sum(col_vals) / len(col_vals)
                                    col_var = sum((x - col_mean)**2 for x in col_vals) / len(col_vals)
                                    neuron_vars.append(col_var)
                            if neuron_vars:
                                dead_neurons = sum(1 for v in neuron_vars if v < self._dead_threshold)
                                stats["dead_neurons"] = dead_neurons
                                stats["total_neurons"] = len(neuron_vars)
                                stats["dead_neuron_ratio"] = dead_neurons / len(neuron_vars)
                    except Exception:
                        pass

            return stats
        except Exception as e:
            return {"error": str(e)}

    def _check_layer_health(self, report: "LayerInspectionReport") -> str:
        """Check layer health based on inspection results.

        Args:
            report: Layer inspection report.

        Returns:
            Health status string.
        """
        if report.potential_issues:
            return "unhealthy"

        for wname, wstats in report.weight_stats.items():
            if wstats.get("std", 1) < 1e-6:
                return "warning"
            if math.isnan(wstats.get("mean", 0)) or math.isinf(wstats.get("mean", 0)):
                return "unhealthy"

        if report.activation_stats:
            if "error" in report.activation_stats:
                return "warning"
            if report.activation_stats.get("dead_neuron_ratio", 0) > 0.1:
                return "warning"

        return "healthy"


@dataclass
class LayerInspectionReport:
    """Report from inspecting a single layer."""
    name: str = "layer"
    class_name: str = "Unknown"
    layer_type: str = "unknown"
    total_params: int = 0
    parameter_names: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    weight_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    activation_stats: Dict[str, Any] = field(default_factory=dict)
    potential_issues: List[str] = field(default_factory=list)
    health: str = "unknown"
