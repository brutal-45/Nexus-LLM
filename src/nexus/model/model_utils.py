"""
Nexus Model Utilities
========================
Comprehensive model utilities for profiling, initialization, checkpointing,
exporting, gradient management, and architecture analysis.

Modules provided:
    - ModelProfiler: Count parameters, estimate FLOPs and memory
    - WeightInitializer: Xavier, Kaiming, truncated normal, scaled init
    - ModelCheckpoint: Save/load/checkpoint management
    - ModelExporter: Export to TorchScript, ONNX
    - ParameterSharing: Tie parameters between layers
    - GradientClipper: max_norm, value, norm_type clipping with logging
    - ActivationRecorder: Record and analyze activations
    - HookManager: Register forward/backward hooks on specific layers
    - ModelComparator: Compare two models
    - ArchitectureSearch: Enumerate architecture variants
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import logging
import hashlib
import copy
import warnings
from typing import (
    Optional, Tuple, List, Dict, Any, Union, Callable, Set,
    NamedTuple, TypeVar, Generic, Iterator,
)
from dataclasses import dataclass, field, asdict
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases and Constants
# =============================================================================

T = TypeVar("T")
ModuleType = TypeVar("ModuleType", bound=nn.Module)
HookFunc = Callable[[nn.Module, Tuple, Tuple], None]

# Standard activation function mapping
ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "hardsigmoid": nn.Hardsigmoid,
    "hardswish": nn.Hardswish,
    "prelu": nn.PReLU,
}

# FLOPs reference: multiply-accumulate = 2 FLOPs
MAC_TO_FLOP = 2


# =============================================================================
# ModelProfiler
# =============================================================================


class FLOPsEstimate(NamedTuple):
    """Result of FLOPs estimation."""
    total_flops: float
    total_mac: float
    forward_flops: float
    backward_flops: float
    per_layer_flops: Dict[str, float]


class MemoryEstimate(NamedTuple):
    """Result of memory estimation."""
    parameter_memory_bytes: int
    activation_memory_bytes: int
    gradient_memory_bytes: int
    optimizer_memory_bytes: int
    total_memory_bytes: int
    parameter_memory_mb: float
    activation_memory_mb: float
    gradient_memory_mb: float
    optimizer_memory_mb: float
    total_memory_mb: float


class ParameterStats(NamedTuple):
    """Parameter statistics."""
    total: int
    trainable: int
    frozen: int
    total_bytes: int
    trainable_bytes: int
    frozen_bytes: int


@dataclass
class LayerProfile:
    """Profile information for a single layer."""
    name: str
    type: str
    num_params: int
    num_trainable: int
    output_shape: Optional[Tuple[int, ...]]
    flops: float
    memory_bytes: int


class ModelProfiler:
    """Comprehensive model profiler for parameter counting, FLOP estimation,
    and memory analysis.

    Usage:
        profiler = ModelProfiler(model)
        stats = profiler.profile()
        print(profiler.parameter_table())

    Args:
        model: The neural network model to profile.
        dtype: Data type for memory estimation.
        device: Device for intermediate computation.
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.dtype = dtype
        self.device = device
        self._layer_profiles: List[LayerProfile] = []
        self._hook_handles: List[RemovableHandle] = []

    def count_parameters(
        self,
        trainable_only: bool = False,
    ) -> int:
        """Count model parameters.

        Args:
            trainable_only: Count only trainable parameters.

        Returns:
            Total parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def parameter_stats(self) -> ParameterStats:
        """Get detailed parameter statistics.

        Returns:
            ParameterStats with total, trainable, frozen counts and sizes.
        """
        total = 0
        trainable = 0
        frozen = 0
        total_bytes = 0
        trainable_bytes = 0
        frozen_bytes = 0

        for p in self.model.parameters():
            numel = p.numel()
            size_bytes = numel * p.element_size()
            total += numel
            total_bytes += size_bytes
            if p.requires_grad:
                trainable += numel
                trainable_bytes += size_bytes
            else:
                frozen += numel
                frozen_bytes += size_bytes

        return ParameterStats(
            total=total,
            trainable=trainable,
            frozen=frozen,
            total_bytes=total_bytes,
            trainable_bytes=trainable_bytes,
            frozen_bytes=frozen_bytes,
        )

    def estimate_flops(
        self,
        input_size: Optional[Tuple[int, ...]] = None,
        custom_ops: Optional[Dict[type, Callable]] = None,
    ) -> FLOPsEstimate:
        """Estimate FLOPs for a forward pass.

        Uses analytical estimation based on layer types. For more accurate
        estimates, provide actual input size.

        Args:
            input_size: Input tensor shape for size-dependent estimation.
            custom_ops: Custom FLOP computation for specific layer types.

        Returns:
            FLOPsEstimate with forward/backward FLOPs.
        """
        total_flops = 0.0
        per_layer: Dict[str, float] = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                layer_flops = 2.0 * in_features * out_features
                if module.bias is not None:
                    layer_flops += out_features
                total_flops += layer_flops
                per_layer[name] = layer_flops

            elif isinstance(module, nn.Conv1d):
                out_h = 1
                if input_size and len(input_size) >= 2:
                    out_h = input_size[-1]
                    kernel_size = module.kernel_size[0]
                    padding = module.padding[0]
                    stride = module.stride[0]
                    out_h = (out_h + 2 * padding - kernel_size) // stride + 1
                layer_flops = (
                    2.0 * module.in_channels * module.out_channels *
                    module.kernel_size[0] * out_h
                )
                total_flops += layer_flops
                per_layer[name] = layer_flops

            elif isinstance(module, nn.Conv2d):
                if input_size and len(input_size) >= 3:
                    in_h, in_w = input_size[-2], input_size[-1]
                    kh, kw = module.kernel_size
                    ph, pw = module.padding
                    sh, sw = module.stride
                    out_h = (in_h + 2 * ph - kh) // sh + 1
                    out_w = (in_w + 2 * pw - kw) // sw + 1
                else:
                    out_h, out_w = 1, 1

                layer_flops = (
                    2.0 * module.in_channels * module.out_channels *
                    kh * kw * out_h * out_w / module.groups
                )
                total_flops += layer_flops
                per_layer[name] = layer_flops

            elif isinstance(module, nn.MultiheadAttention):
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                if input_size and len(input_size) >= 2:
                    seq_len = input_size[1]
                else:
                    seq_len = 128
                # Q, K, V projections
                proj_flops = 3 * 2 * embed_dim * embed_dim * seq_len
                # Attention: Q @ K^T, softmax, @ V
                attn_flops = 2 * num_heads * (seq_len * seq_len * (embed_dim // num_heads))
                # Output projection
                out_proj_flops = 2 * embed_dim * embed_dim * seq_len
                layer_flops = proj_flops + attn_flops + out_proj_flops
                total_flops += layer_flops
                per_layer[name] = layer_flops

            elif isinstance(module, nn.LayerNorm):
                if input_size and len(input_size) >= 2:
                    num_elements = 1
                    for s in input_size[1:]:
                        num_elements *= s
                    layer_flops = 2.0 * module.normalized_shape[0] * num_elements
                    total_flops += layer_flops
                    per_layer[name] = layer_flops

            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid)):
                if input_size and len(input_size) >= 2:
                    num_elements = 1
                    for s in input_size[1:]:
                        num_elements *= s
                    layer_flops = float(num_elements)
                    total_flops += layer_flops
                    per_layer[name] = layer_flops

            elif isinstance(module, nn.Embedding):
                if input_size and len(input_size) >= 1:
                    layer_flops = float(input_size[0] * (input_size[1] if len(input_size) > 1 else 1))
                    total_flops += layer_flops
                    per_layer[name] = layer_flops

            # Custom operation handlers
            if custom_ops and type(module) in custom_ops:
                custom_flops = custom_ops[type(module)](module, input_size)
                per_layer[name] = custom_flops
                total_flops = total_flops + custom_flops

        forward_flops = total_flops
        backward_flops = total_flops * 2.0  # Backward is ~2x forward
        total_flops = forward_flops + backward_flops

        return FLOPsEstimate(
            total_flops=total_flops,
            total_mac=total_flops / MAC_TO_FLOP,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            per_layer_flops=per_layer,
        )

    def memory_estimate(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        dtype: Optional[torch.dtype] = None,
    ) -> MemoryEstimate:
        """Estimate memory usage in bytes and MB.

        Estimates include:
            - Parameter memory
            - Activation memory (approximate)
            - Gradient memory
            - Optimizer state memory (AdamW: 8 bytes per param)

        Args:
            batch_size: Batch size for activation estimation.
            seq_len: Sequence length for activation estimation.
            dtype: Data type for calculation.

        Returns:
            MemoryEstimate with detailed memory breakdown.
        """
        if dtype is None:
            dtype = self.dtype

        elem_size = torch.finfo(dtype).bits // 8  # bytes per element

        # Parameter memory
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        # Gradient memory (same as parameters)
        grad_bytes = sum(p.numel() * elem_size for p in self.model.parameters() if p.requires_grad)

        # Optimizer memory (AdamW: 2 states per param, each same size as param)
        opt_bytes = grad_bytes * 2

        # Activation memory estimation
        # Rough: for each layer with params, store output activations
        activation_bytes = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                activation_bytes += batch_size * seq_len * module.out_features * elem_size
            elif isinstance(module, nn.Embedding):
                activation_bytes += batch_size * seq_len * module.embedding_dim * elem_size
            elif isinstance(module, nn.LayerNorm):
                activation_bytes += batch_size * seq_len * module.normalized_shape[0] * elem_size

        total_bytes = param_bytes + activation_bytes + grad_bytes + opt_bytes

        mb = 1024 * 1024

        return MemoryEstimate(
            parameter_memory_bytes=param_bytes,
            activation_memory_bytes=activation_bytes,
            gradient_memory_bytes=grad_bytes,
            optimizer_memory_bytes=opt_bytes,
            total_memory_bytes=total_bytes,
            parameter_memory_mb=param_bytes / mb,
            activation_memory_mb=activation_bytes / mb,
            gradient_memory_mb=grad_bytes / mb,
            optimizer_memory_mb=opt_bytes / mb,
            total_memory_mb=total_bytes / mb,
        )

    def parameter_table(
        self,
        max_rows: int = 50,
        trainable_only: bool = False,
    ) -> str:
        """Generate a formatted table of parameter counts per layer.

        Args:
            max_rows: Maximum number of rows to display.
            trainable_only: Show only trainable parameters.

        Returns:
            Formatted string table.
        """
        rows = []
        header = f"{'Layer':<60} {'Type':<25} {'Params':>15} {'Trainable':>12}"
        sep = "-" * len(header)
        rows.append(header)
        rows.append(sep)

        total_params = 0
        total_trainable = 0
        count = 0

        for name, module in self.model.named_modules():
            if not list(module.children()):
                n_params = sum(p.numel() for p in module.parameters(recurse=False))
                n_trainable = sum(
                    p.numel() for p in module.parameters(recurse=False)
                    if p.requires_grad
                )

                if trainable_only and n_trainable == 0:
                    continue

                layer_type = module.__class__.__name__
                rows.append(
                    f"{name:<60} {layer_type:<25} {n_params:>15,} {str(n_trainable > 0):>12}"
                )

                total_params += n_params
                total_trainable += n_trainable
                count += 1

                if count >= max_rows:
                    rows.append(f"... ({max_rows} of {count} layers shown)")
                    break

        rows.append(sep)
        rows.append(
            f"{'Total':<60} {'':<25} {total_params:>15,} {total_trainable:>12,}"
        )

        return "\n".join(rows)

    def profile(
        self,
        input_size: Optional[Tuple[int, ...]] = None,
        batch_size: int = 1,
        seq_len: int = 128,
    ) -> Dict[str, Any]:
        """Run a full profiling pass.

        Args:
            input_size: Input tensor shape for FLOP estimation.
            batch_size: Batch size for memory estimation.
            seq_len: Sequence length for memory estimation.

        Returns:
            Dictionary with all profiling results.
        """
        stats = self.parameter_stats()
        flops = self.estimate_flops(input_size)
        memory = self.memory_estimate(batch_size, seq_len)

        return {
            "parameter_stats": {
                "total": stats.total,
                "trainable": stats.trainable,
                "frozen": stats.frozen,
                "total_millions": stats.total / 1e6,
                "trainable_millions": stats.trainable / 1e6,
                "total_billions": stats.total / 1e9,
                "total_mb": stats.total_bytes / (1024 * 1024),
            },
            "flops_estimate": {
                "forward_flops": flops.forward_flops,
                "backward_flops": flops.backward_flops,
                "total_flops": flops.total_flops,
                "total_gflops": flops.total_flops / 1e9,
                "forward_gflops": flops.forward_flops / 1e9,
                "total_tflops": flops.total_flops / 1e12,
            },
            "memory_estimate": {
                "parameters_mb": memory.parameter_memory_mb,
                "activations_mb": memory.activation_memory_mb,
                "gradients_mb": memory.gradient_memory_mb,
                "optimizer_mb": memory.optimizer_memory_mb,
                "total_mb": memory.total_memory_mb,
                "total_gb": memory.total_memory_mb / 1024,
            },
            "parameter_table": self.parameter_table(),
        }

    def print_summary(
        self,
        input_size: Optional[Tuple[int, ...]] = None,
    ):
        """Print a formatted profiling summary to the logger.

        Args:
            input_size: Input tensor shape for FLOP estimation.
        """
        profile = self.profile(input_size)

        logger.info("=" * 70)
        logger.info("Model Profiling Summary")
        logger.info("=" * 70)

        ps = profile["parameter_stats"]
        logger.info(
            f"Parameters: {ps['total_millions']:.2f}M total, "
            f"{ps['trainable_millions']:.2f}M trainable, "
            f"{ps['frozen']} frozen"
        )
        logger.info(f"Parameter size: {ps['total_mb']:.2f} MB")

        fe = profile["flops_estimate"]
        logger.info(
            f"FLOPs: {fe['forward_gflops']:.2f} GFLOPs forward, "
            f"{fe['total_gflops']:.2f} GFLOPs total (incl. backward)"
        )

        me = profile["memory_estimate"]
        logger.info(
            f"Memory: {me['parameters_mb']:.1f}MB params, "
            f"{me['activations_mb']:.1f}MB activations, "
            f"{me['gradients_mb']:.1f}MB gradients, "
            f"{me['optimizer_mb']:.1f}MB optimizer, "
            f"{me['total_gb']:.2f}GB total"
        )

        logger.info("=" * 70)


# =============================================================================
# WeightInitializer
# =============================================================================


@dataclass
class InitConfig:
    """Configuration for weight initialization.

    Attributes:
        strategy: Initialization strategy name.
        gain: Gain value for Xavier/LeCun init.
        mode: Xavier mode ('fan_in' or 'fan_out').
        nonlinearity: Nonlinearity for Xavier init.
        std: Standard deviation for normal init.
        a: Negative slope for LeakyReLU (Kaiming).
        distribution: Distribution for Kaiming ('normal' or 'uniform').
        mean: Mean for truncated normal init.
        trunc_range: Truncation range as multiples of std.
        init_bias: Whether to initialize bias terms.
        bias_value: Value for bias initialization.
        init_norm: Whether to initialize normalization layers.
        init_embedding: Whether to initialize embedding layers.
        embedding_std: Standard deviation for embeddings.
        scaled_init_scale: Scale factor for scaled initialization.
        exclude_keywords: Layer name keywords to exclude from init.
        include_keywords: Layer name keywords to include (if set).
    """

    strategy: str = "normal"
    gain: float = 1.0
    mode: str = "fan_in"
    nonlinearity: str = "leaky_relu"
    std: float = 0.02
    a: float = 0.0
    distribution: str = "normal"
    mean: float = 0.0
    trunc_range: float = 3.0
    init_bias: bool = True
    bias_value: float = 0.0
    init_norm: bool = True
    init_embedding: bool = True
    embedding_std: float = 0.02
    scaled_init_scale: float = 1.0
    exclude_keywords: Tuple[str, ...] = ()
    include_keywords: Tuple[str, ...] = ()


class WeightInitializer:
    """Weight initialization utility supporting multiple strategies.

    Strategies:
        - 'normal': Normal distribution with given std
        - 'xavier_uniform': Xavier/Glorot uniform initialization
        - 'xavier_normal': Xavier/Glorot normal initialization
        - 'kaiming_uniform': Kaiming/He uniform initialization
        - 'kaiming_normal': Kaiming/He normal initialization
        - 'truncated_normal': Truncated normal distribution
        - 'scaled': Scale by 1/sqrt(depth) for deep networks
        - 'orthogonal': Orthogonal initialization
        - 'zeros': Initialize all weights to zero
        - 'ones': Initialize all weights to one

    Usage:
        initializer = WeightInitializer(strategy="xavier_normal")
        initializer.init_weights(model)
        # Or per-layer:
        initializer.init_layer(model.fc, strategy="kaiming_normal")

    Args:
        strategy: Default initialization strategy.
        config: Optional InitConfig with detailed settings.
    """

    STRATEGIES = {
        "normal", "xavier_uniform", "xavier_normal",
        "kaiming_uniform", "kaiming_normal", "truncated_normal",
        "scaled", "orthogonal", "zeros", "ones", "lecun_normal",
        "lecun_uniform", "sparse",
    }

    def __init__(
        self,
        strategy: str = "normal",
        config: Optional[InitConfig] = None,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown initialization strategy: {strategy}. "
                f"Available: {self.STRATEGIES}"
            )
        self.strategy = strategy
        self.config = config or InitConfig(strategy=strategy)

    def init_weights(
        self,
        model: nn.Module,
        strategy: Optional[str] = None,
    ):
        """Initialize all weights in a model.

        Args:
            model: Model to initialize.
            strategy: Override strategy (uses default if None).
        """
        strat = strategy or self.strategy

        for name, module in model.named_modules():
            if self.config.exclude_keywords:
                if any(kw in name for kw in self.config.exclude_keywords):
                    continue
            if self.config.include_keywords:
                if not any(kw in name for kw in self.config.include_keywords):
                    continue

            self.init_layer(module, strat, name=name)

    def init_layer(
        self,
        layer: nn.Module,
        strategy: Optional[str] = None,
        name: str = "",
    ):
        """Initialize a single layer.

        Args:
            layer: Layer to initialize.
            strategy: Override strategy.
            name: Layer name for logging.
        """
        strat = strategy or self.strategy

        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self._init_linear_like(layer, strat)
        elif isinstance(layer, nn.Embedding):
            if self.config.init_embedding:
                self._init_embedding(layer)
        elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if self.config.init_norm:
                self._init_norm(layer)
        elif isinstance(layer, nn.RNNBase):
            self._init_rnn(layer, strat)

    def _init_linear_like(self, layer: nn.Module, strategy: str):
        """Initialize linear-like layers (Linear, Conv).

        Args:
            layer: Layer with weight and optional bias.
            strategy: Initialization strategy.
        """
        if not hasattr(layer, 'weight'):
            return

        weight = layer.weight
        if strategy == "normal":
            nn.init.normal_(weight, mean=self.config.mean, std=self.config.std)
        elif strategy == "xavier_uniform":
            nn.init.xavier_uniform_(
                weight, gain=self.config.gain
            )
        elif strategy == "xavier_normal":
            nn.init.xavier_normal_(
                weight, gain=self.config.gain
            )
        elif strategy == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                weight, a=self.config.a, mode=self.config.mode,
                nonlinearity=self.config.nonlinearity,
            )
        elif strategy == "kaiming_normal":
            nn.init.kaiming_normal_(
                weight, a=self.config.a, mode=self.config.mode,
                nonlinearity=self.config.nonlinearity,
            )
        elif strategy == "truncated_normal":
            nn.init.trunc_normal_(
                weight,
                mean=self.config.mean,
                std=self.config.std,
                a=-self.config.trunc_range * self.config.std,
                b=self.config.trunc_range * self.config.std,
            )
        elif strategy == "scaled":
            fan_in = nn.init._calculate_correct_fan(weight, self.config.mode)
            std = self.config.std * math.sqrt(self.config.scaled_init_scale / fan_in)
            nn.init.normal_(weight, mean=0.0, std=std)
        elif strategy == "orthogonal":
            nn.init.orthogonal_(weight, gain=self.config.gain)
        elif strategy == "zeros":
            nn.init.zeros_(weight)
        elif strategy == "ones":
            nn.init.ones_(weight)
        elif strategy == "lecun_normal":
            fan_in = nn.init._calculate_correct_fan(weight, "fan_in")
            std = math.sqrt(1.0 / fan_in)
            nn.init.normal_(weight, mean=0.0, std=std)
        elif strategy == "lecun_uniform":
            fan_in = nn.init._calculate_correct_fan(weight, "fan_in")
            bound = math.sqrt(3.0 / fan_in)
            nn.init.uniform_(weight, -bound, bound)
        elif strategy == "sparse":
            nn.init.sparse_(
                weight, sparsity=0.1, std=self.config.std
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if hasattr(layer, 'bias') and layer.bias is not None:
            if self.config.init_bias:
                if strategy == "zeros":
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.constant_(layer.bias, self.config.bias_value)

    def _init_embedding(self, layer: nn.Embedding):
        """Initialize embedding layer.

        Args:
            layer: Embedding layer.
        """
        nn.init.normal_(
            layer.weight, mean=0.0, std=self.config.embedding_std
        )
        if layer.padding_idx is not None:
            with torch.no_grad():
                layer.weight[layer.padding_idx].fill_(0)

    def _init_norm(self, layer: nn.Module):
        """Initialize normalization layer.

        Args:
            layer: Normalization layer.
        """
        if hasattr(layer, 'weight'):
            nn.init.ones_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _init_rnn(self, layer: nn.RNNBase, strategy: str):
        """Initialize RNN layer.

        Args:
            layer: RNN layer.
            strategy: Initialization strategy.
        """
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for LSTM
                if isinstance(layer, nn.LSTM):
                    hidden_size = layer.hidden_size
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)


# =============================================================================
# ModelCheckpoint
# =============================================================================


class CheckpointInfo(NamedTuple):
    """Information about a saved checkpoint."""
    path: str
    step: int
    timestamp: float
    file_size_bytes: int
    is_best: bool
    metric_value: Optional[float]


class ModelCheckpoint:
    """Checkpoint management for model saving and loading.

    Supports:
        - Regular checkpointing by step
        - Best model tracking by metric
        - Checkpoint pruning (keep top-k)
        - SafeTensors and PyTorch formats
        - Resume from checkpoint with optimizer state

    Usage:
        checkpoint = ModelCheckpoint(
            save_dir="checkpoints/",
            max_keep=5,
            monitor="val_loss",
            mode="min",
        )
        checkpoint.save(model, optimizer, step=100, metrics={"val_loss": 0.5})

    Args:
        save_dir: Directory to save checkpoints.
        max_keep: Maximum number of checkpoints to keep.
        monitor: Metric name to track for best model.
        mode: 'min' or 'max' for metric tracking.
        save_format: 'safetensors' or 'pytorch'.
        filename_template: Template for checkpoint filenames.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        max_keep: int = 5,
        monitor: Optional[str] = None,
        mode: str = "min",
        save_format: str = "safetensors",
        filename_template: str = "checkpoint_step_{step}",
    ):
        self.save_dir = Path(save_dir)
        self.max_keep = max_keep
        self.monitor = monitor
        self.mode = mode
        self.save_format = save_format
        self.filename_template = filename_template

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_step = -1
        self.checkpoint_history: List[CheckpointInfo] = []

        if self.save_format == "safetensors":
            try:
                import safetensors
                self._has_safetensors = True
            except ImportError:
                logger.warning(
                    "safetensors not available, falling back to pytorch format"
                )
                self.save_format = "pytorch"
                self._has_safetensors = False
        else:
            self._has_safetensors = False

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ):
        """Save a checkpoint.

        Args:
            model: Model to save.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            step: Training step number.
            metrics: Optional metrics dictionary.
            is_best: Force marking as best checkpoint.
        """
        metrics = metrics or {}

        # Check if this is the best checkpoint
        if self.monitor and self.monitor in metrics:
            value = metrics[self.monitor]
            if self._is_better(value):
                self.best_value = value
                self.best_step = step
                is_best = True

        # Build checkpoint data
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "timestamp": time.time(),
            "is_best": is_best,
        }

        # Save model state dict
        if self.save_format == "safetensors" and self._has_safetensors:
            weights_path = self.save_dir / f"{self.filename_template.format(step=step)}.safetensors"
            state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
            from safetensors.torch import save_file
            save_file(state_dict, str(weights_path))
            checkpoint["weights_path"] = str(weights_path)
        else:
            checkpoint["model_state_dict"] = {
                k: v.cpu() for k, v in model.state_dict().items()
            }

        # Save optimizer state
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = {
                k: v.cpu() if torch.is_tensor(v) else v
                for k, v in optimizer.state_dict().items()
            }

        # Save scheduler state
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save checkpoint file
        ckpt_path = self.save_dir / f"{self.filename_template.format(step=step)}.ckpt"
        torch.save(checkpoint, str(ckpt_path))

        # Track checkpoint
        info = CheckpointInfo(
            path=str(ckpt_path),
            step=step,
            timestamp=checkpoint["timestamp"],
            file_size_bytes=ckpt_path.stat().st_size,
            is_best=is_best,
            metric_value=metrics.get(self.monitor),
        )
        self.checkpoint_history.append(info)

        # Save best model symlink
        if is_best:
            best_path = self.save_dir / "best.ckpt"
            if best_path.exists():
                best_path.unlink()
            best_path.symlink_to(ckpt_path.name)

        # Prune old checkpoints
        self._prune_checkpoints()

        logger.info(
            f"Saved checkpoint at step {step}"
            + (f" (best, {self.monitor}={metrics.get(self.monitor):.6f})" if is_best else "")
        )

    def load(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model to load weights into.
            checkpoint_path: Specific checkpoint path (None = load best).
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            map_location: Device to map tensors to.
            strict: Strict loading of state dict.

        Returns:
            Checkpoint metadata dictionary.
        """
        if checkpoint_path is None:
            # Try to load best
            best_path = self.save_dir / "best.ckpt"
            if best_path.exists():
                checkpoint_path = str(best_path)
            else:
                # Load latest
                checkpoints = self._list_checkpoints()
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoint_path = checkpoints[-1]

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load weights
        if "weights_path" in checkpoint and os.path.exists(checkpoint["weights_path"]):
            if self._has_safetensors:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint["weights_path"])
            else:
                state_dict = torch.load(checkpoint["weights_path"], map_location=map_location)
            model.load_state_dict(state_dict, strict=strict)
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Load optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer_state = checkpoint["optimizer_state_dict"]
            # Move tensors to device
            optimizer_state = {
                k: v.to(map_location) if torch.is_tensor(v) else v
                for k, v in optimizer_state.items()
            }
            optimizer.load_state_dict(optimizer_state)

        # Load scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def _is_better(self, value: float) -> bool:
        """Check if value is better than current best.

        Args:
            value: New metric value.

        Returns:
            True if value is better.
        """
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def _list_checkpoints(self) -> List[str]:
        """List available checkpoint files.

        Returns:
            Sorted list of checkpoint paths.
        """
        checkpoints = sorted(
            str(p) for p in self.save_dir.glob("*.ckpt")
        )
        return checkpoints

    def _prune_checkpoints(self):
        """Remove old checkpoints beyond max_keep."""
        if self.max_keep <= 0:
            return

        checkpoints = sorted(
            self.save_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
        )

        while len(checkpoints) > self.max_keep:
            oldest = checkpoints.pop(0)
            if oldest.name != "best.ckpt":
                # Remove associated safetensors file
                ckpt = torch.load(str(oldest), map_location="cpu")
                if "weights_path" in ckpt:
                    wp = Path(ckpt["weights_path"])
                    if wp.exists():
                        wp.unlink()
                oldest.unlink()
                logger.info(f"Pruned old checkpoint: {oldest.name}")

    def list_available(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with info.

        Returns:
            List of checkpoint info dictionaries.
        """
        results = []
        for ckpt_path in self._list_checkpoints():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            results.append({
                "path": ckpt_path,
                "step": ckpt.get("step", -1),
                "is_best": ckpt.get("is_best", False),
                "metrics": ckpt.get("metrics", {}),
                "file_size_mb": os.path.getsize(ckpt_path) / (1024 * 1024),
            })
        return results


# =============================================================================
# ModelExporter
# =============================================================================


class ModelExporter:
    """Export models to various formats.

    Supports:
        - TorchScript (via tracing or scripting)
        - ONNX
        - CoreML (if available)
        - Quantized models

    Usage:
        exporter = ModelExporter(model)
        exporter.export_torchscript(sample_input, "model.ts")
        exporter.export_onnx(sample_input, "model.onnx")

    Args:
        model: Model to export.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def export_torchscript(
        self,
        sample_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output_path: str,
        method: str = "trace",
        optimize: bool = True,
        **kwargs,
    ) -> torch.jit.ScriptModule:
        """Export model to TorchScript.

        Args:
            sample_input: Sample input for tracing/scripting.
            output_path: Output file path.
            method: 'trace' or 'script'.
            optimize: Apply TorchScript optimizations.
            **kwargs: Additional arguments for trace/script.

        Returns:
            TorchScript module.
        """
        self.model.eval()

        with torch.no_grad():
            if method == "trace":
                scripted = torch.jit.trace(self.model, sample_input, **kwargs)
            elif method == "script":
                scripted = torch.jit.script(self.model, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            if optimize:
                scripted = torch.jit.optimize_for_inference(scripted)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        scripted.save(output_path)
        logger.info(f"Exported TorchScript model to {output_path}")

        return scripted

    def export_onnx(
        self,
        sample_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        do_constant_folding: bool = True,
        **kwargs,
    ):
        """Export model to ONNX format.

        Args:
            sample_input: Sample input for export.
            output_path: Output ONNX file path.
            opset_version: ONNX opset version.
            dynamic_axes: Dynamic axis specifications.
            do_constant_folding: Apply constant folding optimization.
            **kwargs: Additional arguments for onnx.export.
        """
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "ONNX not installed. Install with: pip install onnx"
            )

        self.model.eval()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                sample_input,
                output_path,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                **kwargs,
            )

        # Validate the model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        logger.info(f"Exported ONNX model to {output_path}")

    def export_quantized(
        self,
        sample_input: torch.Tensor,
        output_path: str,
        quant_type: str = "dynamic",
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """Export quantized model.

        Args:
            sample_input: Sample input for calibration.
            output_path: Output path.
            quant_type: 'dynamic' or 'static' quantization.
            dtype: Quantization data type.

        Returns:
            Quantized model.
        """
        self.model.eval()

        if quant_type == "dynamic":
            quantized = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=dtype,
            )
        elif quant_type == "static":
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            prepared = torch.quantization.prepare(self.model)

            # Calibration pass
            with torch.no_grad():
                prepared(sample_input)

            quantized = torch.quantization.convert(prepared)
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(quantized.state_dict(), output_path)
        logger.info(f"Exported quantized model to {output_path}")

        return quantized

    def get_model_hash(self) -> str:
        """Compute a hash of the model's architecture and parameters.

        Returns:
            SHA256 hash string.
        """
        hasher = hashlib.sha256()

        # Hash architecture
        arch_str = str(self.model)
        hasher.update(arch_str.encode())

        # Hash parameter shapes and first few values
        for name, param in self.model.named_parameters():
            hasher.update(name.encode())
            hasher.update(str(param.shape).encode())
            hasher.update(str(param.dtype).encode())
            # Include first 10 values as a fingerprint
            flat = param.data.flatten()[:10]
            for v in flat:
                hasher.update(f"{v.item():.8f}".encode())

        return hasher.hexdigest()[:16]


# =============================================================================
# Parameter Sharing
# =============================================================================


class ParameterSharing:
    """Manage parameter sharing (tying) between model layers.

    Usage:
        sharing = ParameterSharing(model)
        sharing.tie("layer.0.mlp", "layer.2.mlp")  # Share all params
        sharing.tie("layer.0.attn.q_proj", "layer.0.attn.k_proj")  # Specific params
        sharing.untie("layer.0.mlp")
        sharing.print_tied_parameters()

    Args:
        model: The model to manage.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._ties: Dict[str, str] = {}

    def tie(
        self,
        source_path: str,
        target_path: str,
        strict: bool = True,
    ):
        """Tie parameters from source to target.

        After tying, modifying source parameters also modifies target
        parameters since they share the same tensor.

        Args:
            source_path: Dot-separated path to the source module.
            target_path: Dot-separated path to the target module.
            strict: If True, raise error if shapes don't match.
        """
        source = self._get_module(source_path)
        target = self._get_module(target_path)

        # Share all parameters
        for (src_name, src_param), (tgt_name, tgt_param) in zip(
            source.named_parameters(),
            target.named_parameters(),
        ):
            if strict and src_param.shape != tgt_param.shape:
                raise ValueError(
                    f"Shape mismatch: {source_path}.{src_name} has shape "
                    f"{src_param.shape} but {target_path}.{tgt_name} has "
                    f"shape {tgt_param.shape}"
                )
            if src_param.shape == tgt_param.shape:
                target.register_parameter(tgt_name, src_param)
                if hasattr(target, tgt_name):
                    setattr(target, tgt_name, src_param)

        self._ties[target_path] = source_path
        logger.info(f"Tied {target_path} -> {source_path}")

    def tie_parameter(
        self,
        source_path: str,
        target_path: str,
        param_name: str,
    ):
        """Tie a specific parameter between modules.

        Args:
            source_path: Dot-separated path to source module.
            target_path: Dot-separated path to target module.
            param_name: Name of the parameter to tie.
        """
        source = self._get_module(source_path)
        target = self._get_module(target_path)

        if not hasattr(source, param_name):
            raise AttributeError(f"{source_path} has no parameter '{param_name}'")
        if not hasattr(target, param_name):
            raise AttributeError(f"{target_path} has no parameter '{param_name}'")

        src_param = getattr(source, param_name)
        tgt_param = getattr(target, param_name)

        if src_param.shape != tgt_param.shape:
            raise ValueError(
                f"Shape mismatch: {src_param.shape} vs {tgt_param.shape}"
            )

        target.register_parameter(param_name, src_param)
        setattr(target, param_name, src_param)

        tie_key = f"{target_path}.{param_name}"
        self._ties[tie_key] = f"{source_path}.{param_name}"

    def untie(self, path: str):
        """Untie parameters at the given path.

        Creates independent copies of tied parameters.

        Args:
            path: Dot-separated module path to untie.
        """
        if path in self._ties:
            del self._ties[path]
            logger.info(f"Untied {path}")
        else:
            # Check for parameter-level ties
            keys_to_remove = [k for k in self._ties if k.startswith(f"{path}.")]
            for key in keys_to_remove:
                del self._ties[key]
                logger.info(f"Untied {key}")

    def _get_module(self, path: str) -> nn.Module:
        """Get a module by dot-separated path.

        Args:
            path: Dot-separated path string.

        Returns:
            Module at the specified path.
        """
        parts = path.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, '__getitem__') and part.isdigit():
                module = module[int(part)]
            else:
                raise AttributeError(f"Cannot resolve path '{path}' at '{part}'")
        return module

    def get_tied_groups(self) -> Dict[str, List[str]]:
        """Get groups of tied parameters.

        Returns:
            Dictionary mapping canonical parameter paths to all tied paths.
        """
        groups: Dict[str, List[str]] = {}
        for target, source in self._ties.items():
            if source not in groups:
                groups[source] = [source]
            groups[source].append(target)
        return groups

    def print_tied_parameters(self):
        """Print all tied parameter groups."""
        groups = self.get_tied_groups()
        if not groups:
            logger.info("No tied parameters")
            return

        for source, targets in groups.items():
            logger.info(f"  {source} <- {', '.join(targets[1:])}")

    def count_shared_parameters(self) -> int:
        """Count the number of parameters that are shared.

        Returns:
            Number of parameters involved in sharing.
        """
        count = 0
        for target_path in self._ties:
            module = self._get_module(target_path.rsplit(".", 1)[0])
            param_name = target_path.rsplit(".", 1)[1]
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if isinstance(param, nn.Parameter):
                    count += param.numel()
        return count


# =============================================================================
# GradientClipper
# =============================================================================


@dataclass
class ClippingStats:
    """Statistics from gradient clipping."""
    step: int
    total_norm_before: float
    total_norm_after: float
    max_grad_norm: float
    num_clipped: int
    clipping_ratio: float
    per_layer_norms: Dict[str, float]
    time_ms: float


class GradientClipper:
    """Gradient clipping with monitoring and logging.

    Supports multiple clipping strategies:
        - 'norm': Clip by global norm (default)
        - 'value': Clip by absolute value
        - 'agc': Adaptive Gradient Clipping
        - 'layerwise': Clip each layer independently

    Usage:
        clipper = GradientClipper(max_norm=1.0, strategy="norm")
        stats = clipper(model)
        print(f"Clipped {stats.num_clipped} gradients")

    Args:
        max_norm: Maximum gradient norm.
        strategy: Clipping strategy.
        norm_type: Type of norm for norm-based clipping.
        log_frequency: Log stats every N steps.
        record_history: Whether to record clipping history.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        strategy: str = "norm",
        norm_type: float = 2.0,
        log_frequency: int = 100,
        record_history: bool = True,
    ):
        self.max_norm = max_norm
        self.strategy = strategy
        self.norm_type = norm_type
        self.log_frequency = log_frequency
        self.record_history = record_history

        self.step_count = 0
        self.history: List[ClippingStats] = []

    def __call__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> ClippingStats:
        """Apply gradient clipping to model parameters.

        Args:
            model: Model with gradients.
            optimizer: Optional optimizer (for unscale before clipping).

        Returns:
            ClippingStats with clipping statistics.
        """
        start_time = time.time()
        self.step_count += 1

        if self.strategy == "norm":
            stats = self._clip_by_norm(model)
        elif self.strategy == "value":
            stats = self._clip_by_value(model)
        elif self.strategy == "agc":
            stats = self._clip_agc(model)
        elif self.strategy == "layerwise":
            stats = self._clip_layerwise(model)
        else:
            raise ValueError(f"Unknown clipping strategy: {self.strategy}")

        stats.time_ms = (time.time() - start_time) * 1000

        if self.record_history:
            self.history.append(stats)

        if self.step_count % self.log_frequency == 0:
            self._log_stats(stats)

        return stats

    def _clip_by_norm(self, model: nn.Module) -> ClippingStats:
        """Clip gradients by global norm.

        Args:
            model: Model with gradients.

        Returns:
            ClippingStats.
        """
        per_layer = {}
        all_grads = []
        all_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(self.norm_type).item()
                per_layer[name] = grad_norm
                all_grads.append(param.grad.data)
                all_norms.append(grad_norm)

        total_norm = math.sqrt(sum(n ** self.norm_type for n in all_norms))

        clip_coef = self.max_norm / (total_norm + 1e-6)
        num_clipped = int(clip_coef < 1.0)

        if clip_coef < 1.0:
            for grad in all_grads:
                grad.mul_(clip_coef)

        clipped_norm = math.sqrt(sum(
            g.data.norm(self.norm_type).item() ** self.norm_type
            for g in all_grads
        ))

        return ClippingStats(
            step=self.step_count,
            total_norm_before=total_norm,
            total_norm_after=clipped_norm,
            max_grad_norm=self.max_norm,
            num_clipped=num_clipped,
            clipping_ratio=clip_coef if clip_coef < 1 else 1.0,
            per_layer_norms=per_layer,
            time_ms=0,
        )

    def _clip_by_value(self, model: nn.Module) -> ClippingStats:
        """Clip gradients by absolute value.

        Args:
            model: Model with gradients.

        Returns:
            ClippingStats.
        """
        per_layer = {}
        total_sq = 0.0
        num_clipped = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                before_norm = param.grad.data.norm(self.norm_type).item()
                per_layer[name] = before_norm
                total_sq += before_norm ** self.norm_type

                torch.clamp_(param.grad.data, -self.max_norm, self.max_norm)
                after_norm = param.grad.data.norm(self.norm_type).item()

                if after_norm < before_norm:
                    num_clipped += 1

        total_norm = math.sqrt(total_sq)

        return ClippingStats(
            step=self.step_count,
            total_norm_before=total_norm,
            total_norm_after=total_norm,
            max_grad_norm=self.max_norm,
            num_clipped=num_clipped,
            clipping_ratio=1.0,
            per_layer_norms=per_layer,
            time_ms=0,
        )

    def _clip_agc(self, model: nn.Module, eps: float = 1e-3) -> ClippingStats:
        """Adaptive Gradient Clipping.

        Clips gradients based on the ratio of gradient norm to weight norm:
            G = min(1, ||W|| / (||G|| + eps)) * G

        Args:
            model: Model with gradients.
            eps: Small constant for numerical stability.

        Returns:
            ClippingStats.
        """
        per_layer = {}
        total_sq_before = 0.0
        num_clipped = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(self.norm_type).item()
                weight_norm = param.data.norm(self.norm_type).item()
                per_layer[name] = grad_norm
                total_sq_before += grad_norm ** self.norm_type

                clip_coef = weight_norm / (grad_norm + eps)
                if clip_coef < 1.0:
                    param.grad.data.mul_(clip_coef)
                    num_clipped += 1

        total_norm = math.sqrt(total_sq_before)

        total_sq_after = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_sq_after += param.grad.data.norm(self.norm_type).item() ** self.norm_type

        return ClippingStats(
            step=self.step_count,
            total_norm_before=total_norm,
            total_norm_after=math.sqrt(total_sq_after),
            max_grad_norm=self.max_norm,
            num_clipped=num_clipped,
            clipping_ratio=1.0,
            per_layer_norms=per_layer,
            time_ms=0,
        )

    def _clip_layerwise(self, model: nn.Module) -> ClippingStats:
        """Clip each layer's gradients independently.

        Args:
            model: Model with gradients.

        Returns:
            ClippingStats.
        """
        per_layer = {}
        total_sq_before = 0.0
        total_sq_after = 0.0
        num_clipped = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(self.norm_type).item()
                per_layer[name] = grad_norm
                total_sq_before += grad_norm ** self.norm_type

                if grad_norm > self.max_norm:
                    clip_coef = self.max_norm / (grad_norm + 1e-6)
                    param.grad.data.mul_(clip_coef)
                    num_clipped += 1

                total_sq_after += param.grad.data.norm(self.norm_type).item() ** self.norm_type

        return ClippingStats(
            step=self.step_count,
            total_norm_before=math.sqrt(total_sq_before),
            total_norm_after=math.sqrt(total_sq_after),
            max_grad_norm=self.max_norm,
            num_clipped=num_clipped,
            clipping_ratio=1.0,
            per_layer_norms=per_layer,
            time_ms=0,
        )

    def _log_stats(self, stats: ClippingStats):
        """Log clipping statistics.

        Args:
            stats: Clipping statistics.
        """
        logger.info(
            f"[Step {stats.step}] Gradient clipping ({self.strategy}): "
            f"norm {stats.total_norm_before:.4f} -> {stats.total_norm_after:.4f}, "
            f"clipped={stats.num_clipped}, "
            f"time={stats.time_ms:.1f}ms"
        )

    def get_history(self) -> List[ClippingStats]:
        """Get clipping history.

        Returns:
            List of historical ClippingStats.
        """
        return self.history

    def get_clipping_rate(self) -> float:
        """Get the fraction of steps where clipping was applied.

        Returns:
            Clipping rate (0.0 to 1.0).
        """
        if not self.history:
            return 0.0
        return sum(1 for s in self.history if s.num_clipped > 0) / len(self.history)


# =============================================================================
# ActivationRecorder
# =============================================================================


@dataclass
class ActivationRecord:
    """Recorded activation data."""
    layer_name: str
    input_mean: float
    input_std: float
    input_min: float
    input_max: float
    output_mean: float
    output_std: float
    output_min: float
    output_max: float
    output_abs_mean: float
    output_sparsity: float
    num_zeros: int
    total_elements: int
    grad_mean: Optional[float] = None
    grad_std: Optional[float] = None
    has_nan: bool = False
    has_inf: bool = False


class ActivationRecorder:
    """Record and analyze activations during forward pass.

    Attaches forward hooks to specified layers and records statistics
    about their inputs and outputs. Useful for debugging activation
    issues like vanishing/exploding gradients, dead neurons, etc.

    Usage:
        recorder = ActivationRecorder(model)
        recorder.register_layers(["layer.0", "layer.1.mlp"])
        model(input)  # Forward pass records activations
        report = recorder.get_report()
        recorder.clear()

    Args:
        model: Model to record activations from.
        record_gradients: Whether to also record gradient statistics.
        max_records: Maximum number of records to keep per layer.
    """

    def __init__(
        self,
        model: nn.Module,
        record_gradients: bool = False,
        max_records: int = 100,
    ):
        self.model = model
        self.record_gradients = record_gradients
        self.max_records = max_records

        self._records: Dict[str, List[ActivationRecord]] = defaultdict(list)
        self._hooks: List[RemovableHandle] = []
        self._registered_layers: Set[str] = set()

    def register_layer(self, layer_name: str):
        """Register a single layer for activation recording.

        Args:
            layer_name: Dot-separated path to the layer.
        """
        module = self._resolve_path(layer_name)
        self._attach_hook(layer_name, module)
        self._registered_layers.add(layer_name)

    def register_layers(self, layer_names: List[str]):
        """Register multiple layers for activation recording.

        Args:
            layer_names: List of dot-separated layer paths.
        """
        for name in layer_names:
            self.register_layer(name)

    def register_by_type(self, layer_type: type):
        """Register all layers of a specific type.

        Args:
            layer_type: Type of layers to register (e.g., nn.Linear).
        """
        for name, module in self.model.named_modules():
            if isinstance(module, layer_type):
                self._attach_hook(name, module)
                self._registered_layers.add(name)

    def register_all(self):
        """Register all leaf modules (those without children)."""
        for name, module in self.model.named_modules():
            if not list(module.children()):
                self._attach_hook(name, module)
                self._registered_layers.add(name)

    def _attach_hook(self, name: str, module: nn.Module):
        """Attach a forward hook to a module.

        Args:
            name: Layer name.
            module: Module to hook.
        """
        recorder = self
        record_grads = self.record_gradients
        records_dict = self._records
        max_rec = self.max_records

        def hook(module, inp, out):
            inp_tensor = None
            if isinstance(inp, tuple) and len(inp) > 0:
                inp_tensor = inp[0]
            if inp_tensor is None or not isinstance(inp_tensor, torch.Tensor):
                return

            out_tensor = out
            if isinstance(out, tuple):
                out_tensor = out[0]
            if not isinstance(out_tensor, torch.Tensor):
                return

            with torch.no_grad():
                record = ActivationRecord(
                    layer_name=name,
                    input_mean=inp_tensor.float().mean().item(),
                    input_std=inp_tensor.float().std().item() if inp_tensor.numel() > 1 else 0.0,
                    input_min=inp_tensor.float().min().item(),
                    input_max=inp_tensor.float().max().item(),
                    output_mean=out_tensor.float().mean().item(),
                    output_std=out_tensor.float().std().item() if out_tensor.numel() > 1 else 0.0,
                    output_min=out_tensor.float().min().item(),
                    output_max=out_tensor.float().max().item(),
                    output_abs_mean=out_tensor.float().abs().mean().item(),
                    output_sparsity=float((out_tensor == 0).sum()) / out_tensor.numel(),
                    num_zeros=int((out_tensor == 0).sum()),
                    total_elements=out_tensor.numel(),
                    has_nan=torch.isnan(out_tensor).any().item(),
                    has_inf=torch.isinf(out_tensor).any().item(),
                )

            records_dict[name].append(record)
            if len(records_dict[name]) > max_rec:
                records_dict[name] = records_dict[name][-max_rec:]

            if record_grads and out_tensor.requires_grad:
                out_tensor.register_hook(
                    lambda grad, n=name: recorder._record_gradient(n, grad)
                )

        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)

    def _record_gradient(self, name: str, grad: torch.Tensor):
        """Record gradient statistics for a layer.

        Args:
            name: Layer name.
            grad: Gradient tensor.
        """
        if name in self._records and self._records[name]:
            self._records[name][-1].grad_mean = grad.float().mean().item()
            self._records[name][-1].grad_std = grad.float().std().item()

    def _resolve_path(self, path: str) -> nn.Module:
        """Resolve dot-separated path to a module.

        Args:
            path: Dot-separated path string.

        Returns:
            Module at the path.
        """
        parts = path.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit():
                module = module[int(part)]
            else:
                raise AttributeError(f"Cannot resolve '{path}' at '{part}'")
        return module

    def get_records(self, layer_name: Optional[str] = None) -> Dict[str, List[ActivationRecord]]:
        """Get recorded activation data.

        Args:
            layer_name: Specific layer name, or None for all.

        Returns:
            Dictionary of layer names to activation records.
        """
        if layer_name is not None:
            return {layer_name: self._records.get(layer_name, [])}
        return dict(self._records)

    def get_report(self) -> Dict[str, Any]:
        """Generate a comprehensive activation analysis report.

        Returns:
            Report dictionary with statistics.
        """
        report = {
            "registered_layers": list(self._registered_layers),
            "layers_with_records": list(self._records.keys()),
            "issues": [],
        }

        for name, records in self._records.items():
            if not records:
                continue

            latest = records[-1]
            layer_report = {
                "name": name,
                "num_records": len(records),
                "latest": {
                    "output_mean": latest.output_mean,
                    "output_std": latest.output_std,
                    "output_abs_mean": latest.output_abs_mean,
                    "sparsity": latest.output_sparsity,
                    "has_nan": latest.has_nan,
                    "has_inf": latest.has_inf,
                },
            }

            # Check for issues
            if latest.has_nan:
                report["issues"].append(f"{name}: NaN in output")
            if latest.has_inf:
                report["issues"].append(f"{name}: Inf in output")
            if latest.output_std < 1e-6:
                report["issues"].append(f"{name}: Very low output variance ({latest.output_std:.2e})")
            if latest.output_std > 100:
                report["issues"].append(f"{name}: Very high output variance ({latest.output_std:.2f})")
            if latest.output_sparsity > 0.9:
                report["issues"].append(f"{name}: High sparsity ({latest.output_sparsity:.2%})")

            report[name] = layer_report

        return report

    def clear(self):
        """Remove all hooks and clear recorded data."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._records.clear()
        self._registered_layers.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.clear()


# =============================================================================
# HookManager
# =============================================================================


class HookManager:
    """Manage forward and backward hooks on model layers.

    Provides a clean interface for attaching, managing, and removing
    hooks on specific layers of a model.

    Usage:
        manager = HookManager(model)
        handle = manager.register_forward_hook("layer.0", my_hook_fn)
        model(input)  # Hook is called during forward
        manager.remove_hook(handle)
        manager.remove_all()

    Args:
        model: The model to manage hooks on.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: Dict[str, RemovableHandle] = {}

    def register_forward_hook(
        self,
        layer_path: str,
        hook_fn: Callable,
    ) -> RemovableHandle:
        """Register a forward hook on a specific layer.

        Args:
            layer_path: Dot-separated path to the layer.
            hook_fn: Hook function (module, input, output) -> None or modified output.

        Returns:
            Removable handle.
        """
        module = self._resolve_path(layer_path)
        handle = module.register_forward_hook(hook_fn)
        self._hooks[f"forward:{layer_path}:{id(hook_fn)}"] = handle
        return handle

    def register_forward_pre_hook(
        self,
        layer_path: str,
        hook_fn: Callable,
    ) -> RemovableHandle:
        """Register a forward pre-hook on a specific layer.

        Args:
            layer_path: Dot-separated path to the layer.
            hook_fn: Hook function (module, input) -> None or modified input.

        Returns:
            Removable handle.
        """
        module = self._resolve_path(layer_path)
        handle = module.register_forward_pre_hook(hook_fn)
        self._hooks[f"forward_pre:{layer_path}:{id(hook_fn)}"] = handle
        return handle

    def register_backward_hook(
        self,
        layer_path: str,
        hook_fn: Callable,
    ) -> RemovableHandle:
        """Register a backward hook on a specific layer.

        Args:
            layer_path: Dot-separated path to the layer.
            hook_fn: Hook function (module, grad_input, grad_output) -> None.

        Returns:
            Removable handle.
        """
        module = self._resolve_path(layer_path)
        handle = module.register_full_backward_hook(hook_fn)
        self._hooks[f"backward:{layer_path}:{id(hook_fn)}"] = handle
        return handle

    def register_by_type(
        self,
        layer_type: type,
        hook_fn: Callable,
        hook_type: str = "forward",
    ) -> List[RemovableHandle]:
        """Register a hook on all layers of a specific type.

        Args:
            layer_type: Type of layers (e.g., nn.Linear).
            hook_fn: Hook function.
            hook_type: 'forward', 'forward_pre', or 'backward'.

        Returns:
            List of removable handles.
        """
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, layer_type):
                if hook_type == "forward":
                    handle = module.register_forward_hook(hook_fn)
                elif hook_type == "forward_pre":
                    handle = module.register_forward_pre_hook(hook_fn)
                elif hook_type == "backward":
                    handle = module.register_full_backward_hook(hook_fn)
                else:
                    raise ValueError(f"Unknown hook type: {hook_type}")
                self._hooks[f"{hook_type}:{name}:{id(hook_fn)}"] = handle
                handles.append(handle)
        return handles

    def remove_hook(self, handle: RemovableHandle):
        """Remove a specific hook.

        Args:
            handle: Hook handle to remove.
        """
        handle.remove()
        keys_to_remove = [
            k for k, v in self._hooks.items() if v is handle
        ]
        for key in keys_to_remove:
            del self._hooks[key]

    def remove_all(self):
        """Remove all registered hooks."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()

    def _resolve_path(self, path: str) -> nn.Module:
        """Resolve a dot-separated path to a module.

        Args:
            path: Path string.

        Returns:
            Resolved module.
        """
        parts = path.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit():
                module = module[int(part)]
            else:
                raise AttributeError(f"Cannot resolve '{path}' at '{part}'")
        return module

    def num_hooks(self) -> int:
        """Get the number of registered hooks.

        Returns:
            Number of active hooks.
        """
        return len(self._hooks)

    def __len__(self) -> int:
        return self.num_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()

    def __del__(self):
        self.remove_all()


# =============================================================================
# ModelComparator
# =============================================================================


class ComparisonResult(NamedTuple):
    """Result of model comparison."""
    are_identical: bool
    architecture_match: bool
    parameter_count_match: bool
    parameter_value_match: bool
    output_match: bool
    max_output_diff: float
    mean_output_diff: float
    parameter_diffs: Dict[str, float]
    shape_mismatches: List[str]
    missing_keys: List[str]
    extra_keys: List[str]


class ModelComparator:
    """Compare two models for architecture and parameter differences.

    Useful for verifying that model modifications preserve behavior,
    or for comparing trained vs. reloaded models.

    Usage:
        comparator = ModelComparator(model_a, model_b)
        result = comparator.compare(sample_input)
        print(result.are_identical)

    Args:
        model_a: First model.
        model_b: Second model.
        rtol: Relative tolerance for value comparison.
        atol: Absolute tolerance for value comparison.
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.rtol = rtol
        self.atol = atol

    def compare_architecture(self) -> Tuple[bool, List[str]]:
        """Compare model architectures.

        Returns:
            Tuple of (match, list of differences).
        """
        differences = []

        arch_a = str(self.model_a)
        arch_b = str(self.model_b)

        if arch_a == arch_b:
            return True, []

        # Check named module structures
        modules_a = dict(self.model_a.named_modules())
        modules_b = dict(self.model_b.named_modules())

        keys_a = set(modules_a.keys())
        keys_b = set(modules_b.keys())

        for key in sorted(keys_a - keys_b):
            differences.append(f"Module in A but not B: {key}")
        for key in sorted(keys_b - keys_a):
            differences.append(f"Module in B but not A: {key}")

        # Check parameter shapes
        params_a = {k: v.shape for k, v in self.model_a.named_parameters()}
        params_b = {k: v.shape for k, v in self.model_b.named_parameters()}

        keys_pa = set(params_a.keys())
        keys_pb = set(params_b.keys())

        for key in sorted(keys_pa - keys_pb):
            differences.append(f"Parameter in A but not B: {key}")
        for key in sorted(keys_pb - keys_pa):
            differences.append(f"Parameter in B but not A: {key}")

        for key in sorted(keys_pa & keys_pb):
            if params_a[key] != params_b[key]:
                differences.append(
                    f"Shape mismatch at {key}: {params_a[key]} vs {params_b[key]}"
                )

        return len(differences) == 0, differences

    def compare_parameters(self) -> Tuple[bool, Dict[str, float]]:
        """Compare parameter values between models.

        Returns:
            Tuple of (match, per-parameter max differences).
        """
        diffs = {}
        all_close = True

        params_a = dict(self.model_a.named_parameters())
        params_b = dict(self.model_b.named_parameters())

        for name in params_a:
            if name not in params_b:
                continue
            if params_a[name].shape != params_b[name].shape:
                diffs[name] = float('inf')
                all_close = False
                continue

            with torch.no_grad():
                diff = (params_a[name] - params_b[name]).abs().max().item()
                diffs[name] = diff
                if diff > self.atol:
                    all_close = False

        return all_close, diffs

    def compare_outputs(
        self,
        sample_input: Union[torch.Tensor, Tuple],
        atol: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """Compare model outputs for the same input.

        Args:
            sample_input: Input tensor(s) for both models.
            atol: Override absolute tolerance.

        Returns:
            Tuple of (match, max_diff, mean_diff).
        """
        self.model_a.eval()
        self.model_b.eval()

        tol = atol or self.atol

        with torch.no_grad():
            output_a = self.model_a(sample_input)
            output_b = self.model_b(sample_input)

        if isinstance(output_a, dict):
            output_a = output_a.get("logits", output_a.get("output", list(output_a.values())[0]))
        if isinstance(output_b, dict):
            output_b = output_b.get("logits", output_b.get("output", list(output_b.values())[0]))

        if isinstance(output_a, tuple):
            output_a = output_a[0]
        if isinstance(output_b, tuple):
            output_b = output_b[0]

        with torch.no_grad():
            diff = (output_a - output_b).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            match = max_diff < tol

        return match, max_diff, mean_diff

    def compare(
        self,
        sample_input: Optional[Union[torch.Tensor, Tuple]] = None,
    ) -> ComparisonResult:
        """Run a full comparison.

        Args:
            sample_input: Optional input for output comparison.

        Returns:
            ComparisonResult with all comparison metrics.
        """
        arch_match, arch_diffs = self.compare_architecture()
        param_match, param_diffs = self.compare_parameters()

        if sample_input is not None:
            output_match, max_out_diff, mean_out_diff = self.compare_outputs(sample_input)
        else:
            output_match = True
            max_out_diff = 0.0
            mean_out_diff = 0.0

        # Compute missing and extra keys
        params_a = set(self.model_a.named_parameters())
        params_b = set(self.model_b.named_parameters())
        names_a = {k for k, _ in params_a}
        names_b = {k for k, _ in params_b}

        param_count_match = (
            sum(p.numel() for _, p in params_a) ==
            sum(p.numel() for _, p in params_b)
        )

        are_identical = (
            arch_match and param_match and output_match and param_count_match
        )

        return ComparisonResult(
            are_identical=are_identical,
            architecture_match=arch_match,
            parameter_count_match=param_count_match,
            parameter_value_match=param_match,
            output_match=output_match,
            max_output_diff=max_out_diff,
            mean_output_diff=mean_out_diff,
            parameter_diffs=param_diffs,
            shape_mismatches=[d for d in arch_diffs if "Shape mismatch" in d],
            missing_keys=sorted(names_a - names_b),
            extra_keys=sorted(names_b - names_a),
        )

    def print_comparison(self, result: Optional[ComparisonResult] = None):
        """Print a formatted comparison report.

        Args:
            result: Optional pre-computed result.
        """
        if result is None:
            result = self.compare()

        logger.info("=" * 60)
        logger.info("Model Comparison Report")
        logger.info("=" * 60)
        logger.info(f"Identical: {result.are_identical}")
        logger.info(f"Architecture match: {result.architecture_match}")
        logger.info(f"Parameters match: {result.parameter_value_match}")
        logger.info(f"Outputs match: {result.output_match}")

        if result.max_output_diff > 0:
            logger.info(
                f"Output diff: max={result.max_output_diff:.2e}, "
                f"mean={result.mean_output_diff:.2e}"
            )

        if result.missing_keys:
            logger.info(f"Missing keys: {result.missing_keys[:5]}...")
        if result.extra_keys:
            logger.info(f"Extra keys: {result.extra_keys[:5]}...")

        # Top parameter differences
        sorted_diffs = sorted(
            result.parameter_diffs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        if sorted_diffs and sorted_diffs[0][1] > 0:
            logger.info("Largest parameter differences:")
            for name, diff in sorted_diffs:
                logger.info(f"  {name}: {diff:.2e}")

        logger.info("=" * 60)


# =============================================================================
# ArchitectureSearch
# =============================================================================


@dataclass
class ArchitectureVariant:
    """Description of an architecture variant."""
    name: str
    config: Dict[str, Any]
    estimated_params: int
    estimated_flops: float
    estimated_memory_mb: float


class ArchitectureSearch:
    """Enumerate and evaluate architecture variants.

    Generates different model configurations by varying hyperparameters
    and estimates their computational cost (parameters, FLOPs, memory).

    Usage:
        search = ArchitectureSearch(base_config)
        variants = search.generate_variants(
            hidden_sizes=[1024, 2048, 4096],
            num_layers=[12, 24, 36],
            num_heads=[8, 16, 32],
        )
        for v in variants:
            print(f"{v.name}: {v.estimated_params/1e6:.1f}M params")

    Args:
        base_config: Base configuration dictionary.
        d_model: Model hidden dimension.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        d_model: int = 768,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
    ):
        self.base_config = base_config or {}
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def estimate_params(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        vocab_size: Optional[int] = None,
        kv_heads: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 8,
    ) -> int:
        """Estimate total parameter count.

        Args:
            hidden_size: Hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            intermediate_size: FFN intermediate size.
            vocab_size: Vocabulary size.
            kv_heads: Number of KV heads (GQA).
            use_moe: Whether to use MoE.
            num_experts: Number of experts.

        Returns:
            Estimated total parameter count.
        """
        vs = vocab_size or self.vocab_size
        kh = kv_heads or num_heads
        head_dim = hidden_size // num_heads

        # Embedding
        embed_params = vs * hidden_size

        # Per-layer params
        # Attention: Q, K, V, O projections
        attn_params = (
            hidden_size * num_heads * head_dim +     # Q
            hidden_size * kh * head_dim +            # K
            hidden_size * kh * head_dim +            # V
            num_heads * head_dim * hidden_size        # O
        )

        # FFN params
        if use_moe:
            # MoE: each expert has its own up/down proj, shared router
            expert_ffn = 2 * hidden_size * intermediate_size  # per expert
            ffn_params = expert_ffn * num_experts + hidden_size * num_experts  # router
        else:
            # Standard FFN (SwiGLU has 3 projections)
            ffn_params = 3 * hidden_size * intermediate_size

        # Norm params
        norm_params = 2 * hidden_size  # 2 LayerNorms per layer

        per_layer = attn_params + ffn_params + norm_params
        layer_params = per_layer * num_layers

        # Final norm + LM head
        final_params = hidden_size + hidden_size * vs

        total = embed_params + layer_params + final_params
        return total

    def estimate_flops_per_token(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        seq_len: int,
        kv_heads: Optional[int] = None,
    ) -> float:
        """Estimate FLOPs per token for forward pass.

        Args:
            hidden_size: Hidden dimension.
            num_layers: Number of layers.
            num_heads: Number of attention heads.
            intermediate_size: FFN intermediate size.
            seq_len: Sequence length.
            kv_heads: Number of KV heads.

        Returns:
            Estimated FLOPs per token.
        """
        kh = kv_heads or num_heads
        head_dim = hidden_size // num_heads

        # Attention FLOPs per token
        qkv_proj_flops = 3 * 2 * hidden_size * (num_heads + 2 * kh) * head_dim
        attn_score_flops = 2 * seq_len * head_dim * num_heads  # Q @ K^T
        attn_weight_flops = seq_len * num_heads  # softmax
        attn_value_flops = 2 * seq_len * head_dim * num_heads  # weights @ V
        out_proj_flops = 2 * hidden_size * hidden_size
        attn_total = qkv_proj_flops + attn_score_flops + attn_weight_flops + attn_value_flops + out_proj_flops

        # FFN FLOPs per token (SwiGLU: 3 projections)
        ffn_flops = 3 * 2 * hidden_size * intermediate_size

        # Norm FLOPs
        norm_flops = 2 * 2 * hidden_size

        per_layer = attn_total + ffn_flops + norm_flops
        total = per_layer * num_layers

        return total

    def generate_variants(
        self,
        hidden_sizes: Optional[List[int]] = None,
        num_layers_list: Optional[List[int]] = None,
        num_heads_list: Optional[List[int]] = None,
        expansion_ratios: Optional[List[int]] = None,
        kv_heads_list: Optional[List[int]] = None,
    ) -> List[ArchitectureVariant]:
        """Generate architecture variants by grid search.

        Args:
            hidden_sizes: Hidden dimensions to try.
            num_layers_list: Number of layers to try.
            num_heads_list: Attention heads to try.
            expansion_ratios: FFN expansion ratios to try.
            kv_heads_list: KV heads to try.

        Returns:
            List of ArchitectureVariant objects.
        """
        hidden_sizes = hidden_sizes or [768, 1024, 2048]
        num_layers_list = num_layers_list or [12, 24]
        num_heads_list = num_heads_list or [12, 16, 32]
        expansion_ratios = expansion_ratios or [4]
        kv_heads_list = kv_heads_list or [None]

        variants = []

        for h in hidden_sizes:
            for l in num_layers_list:
                for n_h in num_heads_list:
                    if h % n_h != 0:
                        continue
                    for er in expansion_ratios:
                        inter = h * er
                        for kv_h in kv_heads_list:
                            if kv_h is not None and n_h % kv_h != 0:
                                continue

                            config = {
                                "hidden_size": h,
                                "num_layers": l,
                                "num_heads": n_h,
                                "intermediate_size": inter,
                                "head_dim": h // n_h,
                                "expansion_ratio": er,
                            }
                            if kv_h is not None:
                                config["kv_heads"] = kv_h

                            params = self.estimate_params(
                                hidden_size=h,
                                num_layers=l,
                                num_heads=n_h,
                                intermediate_size=inter,
                                kv_heads=kv_h,
                            )

                            flops = self.estimate_flops_per_token(
                                hidden_size=h,
                                num_layers=l,
                                num_heads=n_h,
                                intermediate_size=inter,
                                seq_len=self.max_seq_len,
                                kv_heads=kv_h,
                            )

                            # Rough memory: 4 bytes per param + 4 for gradients + 8 for optimizer
                            mem_mb = (params * 16) / (1024 * 1024)

                            name = (
                                f"h{h}_l{l}_nh{n_h}_er{er}"
                                f"{'_kv' + str(kv_h) if kv_h else ''}"
                            )

                            variant = ArchitectureVariant(
                                name=name,
                                config=config,
                                estimated_params=params,
                                estimated_flops=flops * self.max_seq_len,  # total for sequence
                                estimated_memory_mb=mem_mb,
                            )
                            variants.append(variant)

        return variants

    def find_optimal(
        self,
        variants: List[ArchitectureVariant],
        max_params: Optional[int] = None,
        max_memory_mb: Optional[float] = None,
        max_flops: Optional[float] = None,
    ) -> List[ArchitectureVariant]:
        """Filter variants by constraints and sort by efficiency.

        Args:
            variants: List of variants to filter.
            max_params: Maximum parameter count.
            max_memory_mb: Maximum memory in MB.
            max_flops: Maximum FLOPs.

        Returns:
            Filtered and sorted list of variants.
        """
        filtered = []
        for v in variants:
            if max_params and v.estimated_params > max_params:
                continue
            if max_memory_mb and v.estimated_memory_mb > max_memory_mb:
                continue
            if max_flops and v.estimated_flops > max_flops:
                continue
            filtered.append(v)

        # Sort by FLOPs/parameter ratio (efficiency)
        filtered.sort(key=lambda v: v.estimated_flops / max(v.estimated_params, 1))

        return filtered

    def print_variants(
        self,
        variants: List[ArchitectureVariant],
        max_rows: int = 20,
    ):
        """Print a formatted table of architecture variants.

        Args:
            variants: List of variants to display.
            max_rows: Maximum number of rows.
        """
        header = (
            f"{'Name':<40} {'Params':>12} {'FLOPs':>14} {'Memory':>10}"
        )
        sep = "-" * len(header)
        logger.info(header)
        logger.info(sep)

        for v in variants[:max_rows]:
            params_b = v.estimated_params / 1e9
            flops_t = v.estimated_flops / 1e12
            mem_gb = v.estimated_memory_mb / 1024

            if params_b >= 1:
                param_str = f"{params_b:.2f}B"
            else:
                param_str = f"{params_b * 1000:.1f}M"

            if flops_t >= 1:
                flops_str = f"{flops_t:.2f}T"
            else:
                flops_str = f"{flops_t * 1000:.1f}G"

            logger.info(
                f"{v.name:<40} {param_str:>12} {flops_str:>14} {mem_gb:>8.2f}GB"
            )

        logger.info(sep)
        logger.info(f"Total variants: {len(variants)}")


# =============================================================================
# Utility Functions
# =============================================================================


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """Count parameters in a model.

    Convenience function wrapping ModelProfiler.

    Args:
        model: Model to count.
        trainable_only: Count only trainable parameters.

    Returns:
        Total parameter count.
    """
    profiler = ModelProfiler(model)
    return profiler.count_parameters(trainable_only=trainable_only)


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of the first parameter in a model.

    Args:
        model: Model to inspect.

    Returns:
        Device of the model parameters.
    """
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def set_model_device(model: nn.Module, device: Union[str, torch.device]):
    """Move model to a specific device.

    Args:
        model: Model to move.
        device: Target device.
    """
    model.to(device)


def freeze_parameters(
    model: nn.Module,
    exclude_keywords: Optional[List[str]] = None,
    include_keywords: Optional[List[str]] = None,
):
    """Freeze model parameters.

    Args:
        model: Model to freeze.
        exclude_keywords: Parameter name keywords to exclude from freezing.
        include_keywords: Parameter name keywords to include for freezing.
    """
    for name, param in model.named_parameters():
        if exclude_keywords and any(kw in name for kw in exclude_keywords):
            continue
        if include_keywords and not any(kw in name for kw in include_keywords):
            continue
        param.requires_grad = False


def unfreeze_parameters(
    model: nn.Module,
    exclude_keywords: Optional[List[str]] = None,
    include_keywords: Optional[List[str]] = None,
):
    """Unfreeze model parameters.

    Args:
        model: Model to unfreeze.
        exclude_keywords: Keywords to exclude from unfreezing.
        include_keywords: Keywords to include for unfreezing.
    """
    for name, param in model.named_parameters():
        if exclude_keywords and any(kw in name for kw in exclude_keywords):
            continue
        if include_keywords and not any(kw in name for kw in include_keywords):
            continue
        param.requires_grad = True


def get_model_fingerprint(model: nn.Module) -> str:
    """Generate a unique fingerprint for a model's state.

    Useful for checking if model weights have changed.

    Args:
        model: Model to fingerprint.

    Returns:
        Hexadecimal fingerprint string.
    """
    hasher = hashlib.md5()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(str(param.shape).encode())
        hasher.update(str(param.dtype).encode())
        flat = param.data.cpu().flatten()
        hasher.update(flat[:100].numpy().tobytes())
    return hasher.hexdigest()


def print_model_summary(
    model: nn.Module,
    input_size: Optional[Tuple[int, ...]] = None,
):
    """Print a comprehensive model summary.

    Args:
        model: Model to summarize.
        input_size: Input shape for FLOP estimation.
    """
    profiler = ModelProfiler(model)
    profiler.print_summary(input_size)


def create_activation_fn(name: str, **kwargs) -> nn.Module:
    """Create an activation function by name.

    Args:
        name: Activation function name.
        **kwargs: Arguments for the activation function.

    Returns:
        Activation module.
    """
    name_lower = name.lower().replace("-", "_")
    if name_lower not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(ACTIVATION_MAP.keys())}"
        )
    return ACTIVATION_MAP[name_lower](**kwargs)


def merge_checkpoints(
    checkpoint_paths: List[str],
    output_path: str,
    merge_strategy: str = "average",
):
    """Merge multiple checkpoints into one.

    Args:
        checkpoint_paths: Paths to checkpoint files.
        output_path: Output checkpoint path.
        merge_strategy: 'average' or 'sum'.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    # Load all checkpoints
    checkpoints = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu")
        if "model_state_dict" in ckpt:
            checkpoints.append(ckpt["model_state_dict"])
        else:
            checkpoints.append(ckpt)

    # Merge
    merged = OrderedDict()
    for key in checkpoints[0]:
        tensors = [ckpt[key] for ckpt in checkpoints if key in ckpt]
        if not tensors:
            continue
        stacked = torch.stack(tensors)
        if merge_strategy == "average":
            merged[key] = stacked.mean(dim=0)
        elif merge_strategy == "sum":
            merged[key] = stacked.sum(dim=0)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    torch.save(merged, output_path)
    logger.info(f"Merged {len(checkpoint_paths)} checkpoints to {output_path}")


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    "ModelProfiler",
    "FLOPsEstimate",
    "MemoryEstimate",
    "ParameterStats",
    "LayerProfile",
    "WeightInitializer",
    "InitConfig",
    "ModelCheckpoint",
    "CheckpointInfo",
    "ModelExporter",
    "ParameterSharing",
    "GradientClipper",
    "ClippingStats",
    "ActivationRecorder",
    "ActivationRecord",
    "HookManager",
    "ModelComparator",
    "ComparisonResult",
    "ArchitectureSearch",
    "ArchitectureVariant",
    "count_parameters",
    "get_model_device",
    "set_model_device",
    "freeze_parameters",
    "unfreeze_parameters",
    "get_model_fingerprint",
    "print_model_summary",
    "create_activation_fn",
    "merge_checkpoints",
    "ACTIVATION_MAP",
]
