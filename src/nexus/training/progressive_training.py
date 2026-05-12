"""
Progressive Training Strategies for Nexus LLM.

Implements progressive training techniques where model complexity, data
difficulty, batch sizes, and other training parameters are gradually
increased over the course of training for better convergence and efficiency.

Classes:
    ProgressiveResolver: Progressively increase model resolution/size.
    ProgressiveData: Start with simple data, progressively add harder data.
    LayerFreezing: Freeze early layers initially, gradually unfreeze.
    GradualUnfreezer: Schedule for unfreezing layers based on progress.
    ProgressiveBatchSize: Gradually increase batch size during training.
    ProgressiveSequenceLength: Start with short sequences, increase over time.
    StageScheduler: Manage multi-stage training with different configs.
    ProgressiveTrainer: Main trainer orchestrating progressive strategies.
"""

from __future__ import annotations

import abc
import copy
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class FreezeStrategy(Enum):
    """Strategy for layer freezing."""
    BOTTOM_UP = auto()
    TOP_DOWN = auto()
    ALTERNATING = auto()
    BLOCKS = auto()
    CUSTOM = auto()


class GrowthFunction(Enum):
    """Function for progressive growth."""
    LINEAR = auto()
    EXPONENTIAL = auto()
    LOGARITHMIC = auto()
    STEP = auto()
    COSINE = auto()
    ROOT = auto()


class StageType(Enum):
    """Type of training stage."""
    WARMUP = auto()
    GROWTH = auto()
    FINE_TUNE = auto()
    ANNEALING = auto()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ProgressiveConfig:
    """Configuration for progressive training."""
    total_steps: int = 100000
    total_epochs: int = -1
    num_stages: int = 3
    warmup_fraction: float = 0.05
    initial_batch_size: int = 8
    final_batch_size: int = 256
    initial_sequence_length: int = 16
    final_sequence_length: int = 4096
    initial_learning_rate: float = 1e-4
    final_learning_rate: float = 1e-6
    initial_model_fraction: float = 0.25
    final_model_fraction: float = 1.0
    growth_function: GrowthFunction = GrowthFunction.LINEAR
    freeze_strategy: FreezeStrategy = FreezeStrategy.BOTTOM_UP
    initial_frozen_fraction: float = 0.75
    final_frozen_fraction: float = 0.0
    stage_boundaries: Optional[List[float]] = None
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "num_stages": self.num_stages,
            "warmup_fraction": self.warmup_fraction,
            "initial_batch_size": self.initial_batch_size,
            "final_batch_size": self.final_batch_size,
            "initial_sequence_length": self.initial_sequence_length,
            "final_sequence_length": self.final_sequence_length,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "initial_model_fraction": self.initial_model_fraction,
            "final_model_fraction": self.final_model_fraction,
            "growth_function": self.growth_function.name,
            "freeze_strategy": self.freeze_strategy.name,
            "initial_frozen_fraction": self.initial_frozen_fraction,
            "final_frozen_fraction": self.final_frozen_fraction,
            "seed": self.seed,
        }


@dataclass
class StageConfig:
    """Configuration for a single training stage."""
    name: str
    stage_type: StageType
    start_step: int
    end_step: int
    batch_size: int = 32
    learning_rate: float = 3e-4
    sequence_length: int = 2048
    frozen_layers: float = 0.0
    model_fraction: float = 1.0
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 0
    data_config: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stage_type": self.stage_type.name,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "frozen_layers": self.frozen_layers,
            "model_fraction": self.model_fraction,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "warmup_steps": self.warmup_steps,
        }


@dataclass
class ProgressiveState:
    """Current state of progressive training."""
    current_step: int = 0
    current_epoch: int = 0
    current_stage: int = 0
    current_batch_size: int = 32
    current_lr: float = 3e-4
    current_seq_len: int = 2048
    frozen_fraction: float = 0.0
    model_fraction: float = 1.0
    stage_transitions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "current_stage": self.current_stage,
            "current_batch_size": self.current_batch_size,
            "current_lr": self.current_lr,
            "current_seq_len": self.current_seq_len,
            "frozen_fraction": self.frozen_fraction,
            "model_fraction": self.model_fraction,
            "num_transitions": len(self.stage_transitions),
        }


# ---------------------------------------------------------------------------
# Growth Function Utilities
# ---------------------------------------------------------------------------

def compute_growth(
    progress: float,
    initial: float,
    final: float,
    function: GrowthFunction = GrowthFunction.LINEAR,
) -> float:
    """Compute a value based on training progress using a growth function.

    Args:
        progress: Training progress in [0, 1].
        initial: Starting value.
        final: Ending value.
        function: Growth function to use.

    Returns:
        Interpolated value.
    """
    progress = max(0.0, min(1.0, progress))

    if function == GrowthFunction.LINEAR:
        return initial + progress * (final - initial)

    elif function == GrowthFunction.EXPONENTIAL:
        log_initial = math.log(max(1e-10, initial))
        log_final = math.log(max(1e-10, final))
        log_val = log_initial + progress * (log_final - log_initial)
        return math.exp(log_val)

    elif function == GrowthFunction.LOGARITHMIC:
        if progress == 0:
            return initial
        val = initial + (final - initial) * math.log(1 + progress * 9) / math.log(10)
        return val

    elif function == GrowthFunction.STEP:
        n_steps = 5
        step_idx = int(progress * n_steps)
        step_val = step_idx / n_steps
        return initial + step_val * (final - initial)

    elif function == GrowthFunction.COSINE:
        return initial + 0.5 * (final - initial) * (1 - math.cos(math.pi * progress))

    elif function == GrowthFunction.ROOT:
        return initial + math.sqrt(progress) * (final - initial)

    return initial + progress * (final - initial)


# ---------------------------------------------------------------------------
# Progressive Resolver
# ---------------------------------------------------------------------------

class ProgressiveResolver:
    """Progressively increase model resolution/size during training.

    Starts with a smaller/fractional model and gradually activates more
    parameters until the full model is in use. This allows stable
    training of large models by starting simple.

    Args:
        model: The full model to train progressively.
        config: Progressive training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ProgressiveConfig,
    ):
        self.model = model
        self.config = config
        self._current_fraction = config.initial_model_fraction
        self._layers = self._identify_layers()
        self._num_layers = len(self._layers)
        self._active_count = max(1, int(self._num_layers * self._current_fraction))

    def _identify_layers(self) -> List[nn.Module]:
        """Identify all layers in the model in order."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                   nn.Embedding, nn.LayerNorm,
                                   nn.MultiheadAttention)):
                layers.append(module)
        return layers

    def set_active_fraction(self, fraction: float):
        """Set the fraction of model layers that are active.

        Inactive layers have their parameters frozen and are optionally
        replaced with identity/skip connections.
        """
        self._current_fraction = max(0.01, min(1.0, fraction))
        new_active = max(1, int(self._num_layers * self._current_fraction))

        for i, layer in enumerate(self._layers):
            if i < new_active:
                self._unfreeze_layer(layer)
            else:
                self._freeze_layer(layer)

        if new_active != self._active_count:
            logger.info(
                f"ProgressiveResolver: activating {new_active}/{self._num_layers} "
                f"layers ({self._current_fraction:.1%})"
            )
        self._active_count = new_active

    def _freeze_layer(self, layer: nn.Module):
        """Freeze all parameters in a layer."""
        for param in layer.parameters():
            param.requires_grad = False

    def _unfreeze_layer(self, layer: nn.Module):
        """Unfreeze all parameters in a layer."""
        for param in layer.parameters():
            param.requires_grad = True

    def get_active_fraction(self) -> float:
        """Return the current active model fraction."""
        return self._current_fraction

    def update(self, step: int):
        """Update active fraction based on training step."""
        progress = step / max(1, self.config.total_steps)
        fraction = compute_growth(
            progress,
            self.config.initial_model_fraction,
            self.config.final_model_fraction,
            self.config.growth_function,
        )
        self.set_active_fraction(fraction)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def get_active_info(self) -> Dict[str, Any]:
        """Get information about active model state."""
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        return {
            "active_fraction": self._current_fraction,
            "active_layers": self._active_count,
            "total_layers": self._num_layers,
            "trainable_params": trainable,
            "total_params": total,
            "param_utilization": trainable / max(1, total),
        }


# ---------------------------------------------------------------------------
# Progressive Data
# ---------------------------------------------------------------------------

class ProgressiveData:
    """Start with small/simple data, progressively add harder data.

    Manages a data curriculum where easier examples are used first,
    with harder examples gradually introduced.

    Args:
        datasets: List of datasets ordered from easy to hard.
        schedule: List of (step_threshold, dataset_idx) for introduction.
        seed: Random seed.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        schedule: Optional[List[Tuple[int, int]]] = None,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.seed = seed
        self._rng = random.Random(seed)
        self._current_datasets: List[int] = [0] if datasets else []

        if schedule is not None:
            self._schedule = sorted(schedule, key=lambda x: x[0])
        elif len(datasets) > 1:
            self._schedule = []
            for i in range(1, len(datasets)):
                fraction = i / len(datasets)
                step = int(fraction * 100000)
                self._schedule.append((step, i))
        else:
            self._schedule = []

    def get_active_datasets(self, step: int) -> List[Dataset]:
        """Get datasets that should be active at the given step."""
        active_indices = {0}
        for threshold, dataset_idx in self._schedule:
            if step >= threshold:
                active_indices.add(dataset_idx)

        if active_indices != set(self._current_datasets):
            new_indices = active_indices - set(self._current_datasets)
            self._current_datasets = sorted(active_indices)
            for idx in new_indices:
                if idx < len(self.datasets):
                    logger.info(f"ProgressiveData: adding dataset {idx} at step {step}")

        return [self.datasets[i] for i in sorted(active_indices) if i < len(self.datasets)]

    def get_active_dataset_size(self, step: int) -> int:
        """Get total size of active datasets."""
        return sum(len(ds) for ds in self.get_active_datasets(step))

    def sample_batch(
        self, step: int, batch_size: int, weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Sample a batch from active datasets.

        Args:
            step: Current training step.
            batch_size: Number of samples to draw.
            weights: Optional weights for each dataset.

        Returns:
            List of sample dictionaries.
        """
        active = self.get_active_datasets(step)
        if not active:
            return []

        if weights is None:
            weights = [1.0 / len(active)] * len(active)

        total = sum(weights)
        probs = [w / total for w in weights]
        counts = [max(1, int(batch_size * p)) for p in probs]

        remainder = batch_size - sum(counts)
        if remainder > 0:
            idx = self._rng.choices(range(len(active)), weights=probs, k=remainder)
            for i in idx:
                counts[i] += 1

        batch = []
        for dataset, count in zip(active, counts):
            indices = self._rng.sample(range(len(dataset)), min(count, len(dataset)))
            for idx in indices:
                item = dataset[idx]
                if isinstance(item, dict):
                    batch.append(item)
                elif isinstance(item, (list, tuple)):
                    batch.append({"data": item})
                else:
                    batch.append({"data": item})

        self._rng.shuffle(batch)
        return batch[:batch_size]


# ---------------------------------------------------------------------------
# Layer Freezing
# ---------------------------------------------------------------------------

class LayerFreezing:
    """Freeze early layers initially, gradually unfreeze during training.

    Supports multiple freezing strategies including bottom-up, top-down,
    alternating, and block-based unfreezing.

    Args:
        model: The model whose layers to freeze/unfreeze.
        strategy: Freezing/unfreezing strategy.
        initial_frozen_fraction: Fraction of layers frozen initially.
        final_frozen_fraction: Target fraction frozen at end.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: FreezeStrategy = FreezeStrategy.BOTTOM_UP,
        initial_frozen_fraction: float = 0.75,
        final_frozen_fraction: float = 0.0,
    ):
        self.model = model
        self.strategy = strategy
        self.initial_frozen_fraction = initial_frozen_fraction
        self.final_frozen_fraction = final_frozen_fraction
        self._layers = self._enumerate_layers()
        self._num_layers = len(self._layers)
        self._layer_names = [name for name, _ in self._layers]
        self._frozen_indices: Set[int] = set()
        self._initial_freeze()

    def _enumerate_layers(self) -> List[Tuple[str, nn.Module]]:
        """Enumerate all trainable layers in order."""
        layers = []
        for name, module in self.model.named_modules():
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                layers.append((name, module))
        return layers

    def _initial_freeze(self):
        """Apply initial freezing based on strategy."""
        n_freeze = int(self._num_layers * self.initial_frozen_fraction)
        self._set_frozen_count(n_freeze)

    def _set_frozen_count(self, n_freeze: int):
        """Freeze exactly n_freeze layers according to strategy."""
        self._unfreeze_all()
        self._frozen_indices.clear()
        n_freeze = max(0, min(self._num_layers, n_freeze))

        if self.strategy == FreezeStrategy.BOTTOM_UP:
            for i in range(n_freeze):
                self._freeze_at(i)

        elif self.strategy == FreezeStrategy.TOP_DOWN:
            for i in range(self._num_layers - n_freeze, self._num_layers):
                self._freeze_at(i)

        elif self.strategy == FreezeStrategy.ALTERNATING:
            frozen = 0
            i = 0
            while frozen < n_freeze and i < self._num_layers:
                self._freeze_at(i)
                frozen += 1
                i += 2
            i = 1
            while frozen < n_freeze and i < self._num_layers:
                self._freeze_at(i)
                frozen += 1
                i += 2

        elif self.strategy == FreezeStrategy.BLOCKS:
            block_size = max(1, self._num_layers // max(1, n_freeze))
            frozen = 0
            for i in range(0, self._num_layers, block_size):
                for j in range(block_size):
                    if frozen >= n_freeze:
                        break
                    if i + j < self._num_layers:
                        self._freeze_at(i + j)
                        frozen += 1

    def _freeze_at(self, idx: int):
        """Freeze layer at index."""
        if 0 <= idx < self._num_layers:
            name, module = self._layers[idx]
            for param in module.parameters():
                param.requires_grad = False
            self._frozen_indices.add(idx)

    def _unfreeze_at(self, idx: int):
        """Unfreeze layer at index."""
        if 0 <= idx < self._num_layers:
            name, module = self._layers[idx]
            for param in module.parameters():
                param.requires_grad = True
            self._frozen_indices.discard(idx)

    def _unfreeze_all(self):
        """Unfreeze all layers."""
        for _, module in self._layers:
            for param in module.parameters():
                param.requires_grad = True
        self._frozen_indices.clear()

    def update(self, progress: float):
        """Update freezing based on training progress.

        Args:
            progress: Training progress in [0, 1].
        """
        current_fraction = self.initial_frozen_fraction + progress * (
            self.final_frozen_fraction - self.initial_frozen_fraction
        )
        n_freeze = int(self._num_layers * max(0, min(1, current_fraction)))

        prev_count = len(self._frozen_indices)
        if n_freeze != prev_count:
            self._set_frozen_count(n_freeze)

    def get_frozen_count(self) -> int:
        """Return number of frozen layers."""
        return len(self._frozen_indices)

    def get_frozen_fraction(self) -> float:
        """Return fraction of frozen layers."""
        return len(self._frozen_indices) / max(1, self._num_layers)

    def get_frozen_layer_names(self) -> List[str]:
        """Return names of frozen layers."""
        return [self._layer_names[i] for i in sorted(self._frozen_indices)]

    def get_trainable_layer_names(self) -> List[str]:
        """Return names of trainable layers."""
        return [
            self._layer_names[i] for i in range(self._num_layers)
            if i not in self._frozen_indices
        ]

    def get_info(self) -> Dict[str, Any]:
        """Get freezing information."""
        return {
            "strategy": self.strategy.name,
            "total_layers": self._num_layers,
            "frozen_layers": len(self._frozen_indices),
            "trainable_layers": self._num_layers - len(self._frozen_indices),
            "frozen_fraction": self.get_frozen_fraction(),
            "frozen_names": self.get_frozen_layer_names(),
        }


# ---------------------------------------------------------------------------
# Gradual Unfreezer
# ---------------------------------------------------------------------------

class GradualUnfreezer:
    """Schedule for unfreezing layers based on training progress.

    Provides fine-grained control over which layers are unfrozen
    and when, supporting milestones and callbacks.

    Args:
        model: The model to manage freezing for.
        total_steps: Total training steps.
        milestones: List of (step, num_layers_to_unfreeze) tuples.
    """

    def __init__(
        self,
        model: nn.Module,
        total_steps: int,
        milestones: Optional[List[Tuple[int, int]]] = None,
    ):
        self.model = model
        self.total_steps = total_steps
        self._layer_freezer = LayerFreezing(model, FreezeStrategy.BOTTOM_UP, 1.0, 0.0)
        self._unfreeze_schedule: List[Tuple[int, int]] = []

        if milestones is not None:
            self._unfreeze_schedule = sorted(milestones, key=lambda x: x[0])
        else:
            num_layers = self._layer_freezer._num_layers
            steps_between = max(1, total_steps // max(1, num_layers))
            for i in range(num_layers):
                step = i * steps_between
                self._unfreeze_schedule.append((step, i + 1))

        self._current_step = 0
        self._unfreeze_count = 0
        self._callbacks: List[Callable[[int, int], None]] = []

    def register_callback(self, callback: Callable[[int, int], None]):
        """Register a callback for unfreeze events (step, unfrozen_count)."""
        self._callbacks.append(callback)

    def step(self, step: int):
        """Advance to a step and unfreeze layers as needed."""
        self._current_step = step
        progress = step / max(1, self.total_steps)

        n_unfreeze = 0
        for milestone_step, count in self._unfreeze_schedule:
            if step >= milestone_step:
                n_unfreeze = max(n_unfreeze, count)

        n_freeze = self._layer_freezer._num_layers - n_unfreeze
        if n_freeze != self._layer_freezer.get_frozen_count():
            self._layer_freezer._set_frozen_count(max(0, n_freeze))
            if n_unfreeze != self._unfreeze_count:
                self._unfreeze_count = n_unfreeze
                for cb in self._callbacks:
                    try:
                        cb(step, n_unfreeze)
                    except Exception as e:
                        logger.warning(f"Unfreeze callback failed: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get current unfreezing state."""
        return {
            "current_step": self._current_step,
            "unfreeze_count": self._unfreeze_count,
            "total_layers": self._layer_freezer._num_layers,
            "progress": self._current_step / max(1, self.total_steps),
        }


# ---------------------------------------------------------------------------
# Progressive Batch Size
# ---------------------------------------------------------------------------

class ProgressiveBatchSize:
    """Gradually increase batch size during training (warmup).

    Starts with a small batch size for stability and gradually increases
    to the target batch size. Implements gradient accumulation to
    maintain effective batch size when GPU memory is limited.

    Args:
        initial_batch_size: Starting batch size.
        final_batch_size: Target batch size.
        total_steps: Total training steps for the warmup.
        growth_fn: How to grow batch size.
        micro_batch_size: Maximum batch that fits in GPU memory.
    """

    def __init__(
        self,
        initial_batch_size: int = 8,
        final_batch_size: int = 256,
        total_steps: int = 10000,
        growth_fn: GrowthFunction = GrowthFunction.LINEAR,
        micro_batch_size: Optional[int] = None,
    ):
        self.initial_batch_size = initial_batch_size
        self.final_batch_size = final_batch_size
        self.total_steps = total_steps
        self.growth_fn = growth_fn
        self.micro_batch_size = micro_batch_size or final_batch_size
        self._current_batch_size = initial_batch_size
        self._current_accumulation = 1

    def get_batch_size(self, step: int) -> int:
        """Get the batch size at a given step.

        Args:
            step: Current training step.

        Returns:
            Micro batch size (single GPU forward pass).
        """
        progress = min(1.0, step / max(1, self.total_steps))
        effective = compute_growth(
            progress, self.initial_batch_size, self.final_batch_size, self.growth_fn
        )
        effective = int(max(1, round(effective)))

        if effective <= self.micro_batch_size:
            self._current_batch_size = effective
            self._current_accumulation = 1
        else:
            self._current_accumulation = max(1, effective // self.micro_batch_size)
            self._current_batch_size = self.micro_batch_size

        return self._current_batch_size

    def get_accumulation_steps(self, step: int) -> int:
        """Get gradient accumulation steps at a given step."""
        self.get_batch_size(step)
        return self._current_accumulation

    def get_effective_batch_size(self, step: int) -> int:
        """Get the effective batch size (micro * accumulation)."""
        return self.get_batch_size(step) * self.get_accumulation_steps(step)

    def get_lr_scale(self, step: int) -> float:
        """Get learning rate scaling factor for batch size change.

        When batch size doubles, LR should be scaled by sqrt(2) for
        optimal convergence (linear scaling rule).
        """
        effective = self.get_effective_batch_size(step)
        scale = math.sqrt(effective / max(1, self.initial_batch_size))
        return scale

    def get_info(self, step: int) -> Dict[str, Any]:
        """Get batch size information at a given step."""
        return {
            "micro_batch_size": self.get_batch_size(step),
            "accumulation_steps": self.get_accumulation_steps(step),
            "effective_batch_size": self.get_effective_batch_size(step),
            "lr_scale": self.get_lr_scale(step),
            "progress": min(1.0, step / max(1, self.total_steps)),
        }


# ---------------------------------------------------------------------------
# Progressive Sequence Length
# ---------------------------------------------------------------------------

class ProgressiveSequenceLength:
    """Start with short sequences, increase over time.

    Gradually increases the maximum sequence length used during training.
    This improves training stability and memory efficiency.

    Args:
        initial_length: Starting sequence length.
        final_length: Target maximum sequence length.
        total_steps: Total training steps.
        growth_fn: How to grow sequence length.
        bucket_size: Round to nearest bucket for efficiency.
    """

    def __init__(
        self,
        initial_length: int = 16,
        final_length: int = 4096,
        total_steps: int = 100000,
        growth_fn: GrowthFunction = GrowthFunction.LINEAR,
        bucket_size: int = 16,
    ):
        self.initial_length = initial_length
        self.final_length = final_length
        self.total_steps = total_steps
        self.growth_fn = growth_fn
        self.bucket_size = bucket_size
        self._current_length = initial_length
        self._length_history: List[Tuple[int, int]] = []

    def get_length(self, step: int) -> int:
        """Get the sequence length at a given step.

        Args:
            step: Current training step.

        Returns:
            Current maximum sequence length.
        """
        progress = min(1.0, step / max(1, self.total_steps))
        length = compute_growth(
            progress, self.initial_length, self.final_length, self.growth_fn
        )

        if self.bucket_size > 1:
            length = max(
                self.initial_length,
                int(math.ceil(length / self.bucket_size)) * self.bucket_size,
            )

        length = int(min(self.final_length, length))
        self._current_length = length

        if self._length_history and self._length_history[-1][1] != length:
            self._length_history.append((step, length))

        return length

    def truncate(self, sequence: torch.Tensor, step: int) -> torch.Tensor:
        """Truncate a sequence to the current maximum length.

        Args:
            sequence: Input tensor.
            step: Current training step.

        Returns:
            Potentially truncated tensor.
        """
        max_len = self.get_length(step)
        if sequence.size(-1) > max_len:
            return sequence[..., :max_len]
        return sequence

    def pad(
        self, sequence: torch.Tensor, step: int, pad_value: float = 0.0
    ) -> torch.Tensor:
        """Pad sequence to current max length if needed."""
        max_len = self.get_length(step)
        if sequence.size(-1) < max_len:
            pad_size = max_len - sequence.size(-1)
            padding = torch.full(
                (*sequence.shape[:-1], pad_size),
                pad_value,
                dtype=sequence.dtype,
                device=sequence.device,
            )
            return torch.cat([sequence, padding], dim=-1)
        return sequence

    def get_info(self, step: int) -> Dict[str, Any]:
        """Get sequence length info."""
        return {
            "current_length": self.get_length(step),
            "initial_length": self.initial_length,
            "final_length": self.final_length,
            "progress": min(1.0, step / max(1, self.total_steps)),
            "length_changes": len(self._length_history),
        }


# ---------------------------------------------------------------------------
# Stage Scheduler
# ---------------------------------------------------------------------------

class StageScheduler:
    """Manage multi-stage training with different configurations per stage.

    Each stage has its own hyperparameters, and the scheduler handles
    transitions between stages.

    Args:
        stages: List of StageConfig objects.
    """

    def __init__(self, stages: Optional[List[StageConfig]] = None):
        self._stages = stages or []
        self._current_stage_idx = 0
        self._transition_callbacks: List[Callable[[StageConfig, StageConfig], None]] = []

    def add_stage(self, stage: StageConfig):
        """Add a stage to the scheduler."""
        self._stages.append(stage)
        self._stages.sort(key=lambda s: s.start_step)

    def remove_stage(self, name: str):
        """Remove a stage by name."""
        self._stages = [s for s in self._stages if s.name != name]

    def get_current_stage(self, step: int) -> StageConfig:
        """Get the stage active at a given step."""
        for i, stage in enumerate(self._stages):
            if step >= stage.start_step and (step < stage.end_step or stage.end_step < 0):
                if i != self._current_stage_idx:
                    old_stage = self._stages[self._current_stage_idx] if self._current_stage_idx < len(self._stages) else None
                    self._current_stage_idx = i
                    if old_stage and old_stage != stage:
                        for cb in self._transition_callbacks:
                            try:
                                cb(old_stage, stage)
                            except Exception as e:
                                logger.warning(f"Stage transition callback failed: {e}")
                return stage
        if self._stages:
            return self._stages[-1]
        raise RuntimeError("No stages defined")

    def get_stage_config(self, step: int) -> Dict[str, Any]:
        """Get all hyperparameters for the current stage."""
        stage = self.get_current_stage(step)
        return stage.to_dict()

    def register_transition_callback(
        self, callback: Callable[[StageConfig, StageConfig], None]
    ):
        """Register a callback for stage transitions."""
        self._transition_callbacks.append(callback)

    def interpolate_config(self, step: int) -> Dict[str, Any]:
        """Get interpolated config at the current step.

        For numeric parameters, interpolates between the current and next stage.
        """
        stage = self.get_current_stage(step)
        stage_progress = 0.0
        if stage.end_step > stage.start_step:
            stage_progress = (step - stage.start_step) / (stage.end_step - stage.start_step)
        stage_progress = max(0.0, min(1.0, stage_progress))

        next_stage_idx = self._current_stage_idx + 1
        next_stage = self._stages[next_stage_idx] if next_stage_idx < len(self._stages) else None

        config = stage.to_dict()
        if next_stage:
            numeric_keys = ["batch_size", "learning_rate", "sequence_length", "weight_decay", "grad_clip"]
            for key in numeric_keys:
                if key in config and key in next_stage.to_dict():
                    current_val = config[key]
                    next_val = next_stage.to_dict()[key]
                    if isinstance(current_val, (int, float)) and isinstance(next_val, (int, float)):
                        if key == "learning_rate" or key == "weight_decay":
                            log_curr = math.log(max(1e-10, current_val))
                            log_next = math.log(max(1e-10, next_val))
                            config[key] = math.exp(log_curr + stage_progress * (log_next - log_curr))
                        else:
                            config[key] = current_val + stage_progress * (next_val - current_val)

        return config

    def get_all_stages(self) -> List[StageConfig]:
        """Return all stages."""
        return list(self._stages)

    def get_num_stages(self) -> int:
        """Return number of stages."""
        return len(self._stages)

    @classmethod
    def from_progressive_config(
        cls, config: ProgressiveConfig
    ) -> "StageScheduler":
        """Create a StageScheduler from a ProgressiveConfig."""
        stages = []
        stage_names = ["warmup", "growth", "finetune"]

        if config.stage_boundaries:
            boundaries = [0.0] + config.stage_boundaries + [1.0]
        else:
            boundaries = [
                0.0,
                config.warmup_fraction,
                1.0 - config.warmup_fraction * 0.5,
                1.0,
            ]

        for i in range(len(boundaries) - 1):
            start = int(boundaries[i] * config.total_steps)
            end = int(boundaries[i + 1] * config.total_steps)

            start_progress = boundaries[i]
            end_progress = boundaries[i + 1]

            bs_start = int(compute_growth(
                start_progress, config.initial_batch_size,
                config.final_batch_size, config.growth_function,
            ))
            bs_end = int(compute_growth(
                end_progress, config.initial_batch_size,
                config.final_batch_size, config.growth_function,
            ))

            lr_start = compute_growth(
                start_progress, config.initial_learning_rate,
                config.final_learning_rate, GrowthFunction.LOGARITHMIC,
            )
            lr_end = compute_growth(
                end_progress, config.initial_learning_rate,
                config.final_learning_rate, GrowthFunction.LOGARITHMIC,
            )

            sl_start = int(compute_growth(
                start_progress, config.initial_sequence_length,
                config.final_sequence_length, config.growth_function,
            ))
            sl_end = int(compute_growth(
                end_progress, config.initial_sequence_length,
                config.final_sequence_length, config.growth_function,
            ))

            frozen_start = config.initial_frozen_fraction + start_progress * (
                config.final_frozen_fraction - config.initial_frozen_fraction
            )
            frozen_end = config.initial_frozen_fraction + end_progress * (
                config.final_frozen_fraction - config.initial_frozen_fraction
            )

            name = stage_names[i] if i < len(stage_names) else f"stage_{i}"
            stage_type = {
                0: StageType.WARMUP,
                1: StageType.GROWTH,
                2: StageType.FINE_TUNE,
            }.get(i, StageType.GROWTH)

            stages.append(StageConfig(
                name=name,
                stage_type=stage_type,
                start_step=start,
                end_step=end,
                batch_size=(bs_start + bs_end) // 2,
                learning_rate=lr_start,
                sequence_length=(sl_start + sl_end) // 2,
                frozen_layers=frozen_start,
            ))

        return cls(stages)


# ---------------------------------------------------------------------------
# Progressive Trainer (Main Orchestrator)
# ---------------------------------------------------------------------------

class ProgressiveTrainer:
    """Main trainer that orchestrates all progressive strategies.

    Combines progressive model resolution, data difficulty, batch size,
    sequence length, layer freezing, and stage scheduling into a unified
    training loop.

    Args:
        model: The model to train.
        config: Progressive training configuration.
        optimizer: Optional optimizer (created if None).
        tokenizer: Optional tokenizer.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ProgressiveConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.config = config or ProgressiveConfig()
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        self.stage_scheduler = StageScheduler.from_progressive_config(self.config)
        self.layer_freezer = LayerFreezing(
            model, self.config.freeze_strategy,
            self.config.initial_frozen_fraction,
            self.config.final_frozen_fraction,
        )
        self.batch_scheduler = ProgressiveBatchSize(
            self.config.initial_batch_size,
            self.config.final_batch_size,
            int(self.config.total_steps * 0.3),
            self.config.growth_function,
        )
        self.seq_scheduler = ProgressiveSequenceLength(
            self.config.initial_sequence_length,
            self.config.final_sequence_length,
            self.config.total_steps,
            self.config.growth_function,
        )
        self.resolver = ProgressiveResolver(model, self.config)

        self._state = ProgressiveState()
        self._step_count = 0
        self._epoch_count = 0
        self._history: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[ProgressiveState, Dict], None]] = []
        self._total_loss = 0.0
        self._loss_count = 0

    def register_callback(
        self, callback: Callable[[ProgressiveState, Dict], None]
    ):
        """Register a callback called after each step."""
        self._callbacks.append(callback)

    def step(self, loss: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Execute one training step with progressive updates.

        Args:
            loss: Current loss value.
            **kwargs: Additional information.

        Returns:
            Dictionary with current training state.
        """
        self._step_count += 1
        step = self._step_count
        progress = step / max(1, self.config.total_steps)

        self.resolver.update(step)
        self.layer_freezer.update(progress)
        batch_size = self.batch_scheduler.get_batch_size(step)
        seq_length = self.seq_scheduler.get_length(step)

        current_lr = compute_growth(
            progress,
            self.config.initial_learning_rate,
            self.config.final_learning_rate,
            GrowthFunction.LOGARITHMIC,
        )

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        self._state.current_step = step
        self._state.current_batch_size = batch_size
        self._state.current_lr = current_lr
        self._state.current_seq_len = seq_length
        self._state.frozen_fraction = self.layer_freezer.get_frozen_fraction()
        self._state.model_fraction = self.resolver.get_active_fraction()

        try:
            stage = self.stage_scheduler.get_current_stage(step)
            self._state.current_stage = self.stage_scheduler._current_stage_idx
        except RuntimeError:
            pass

        if loss > 0:
            self._total_loss += loss
            self._loss_count += 1

        metrics = {
            "step": step,
            "progress": progress,
            "batch_size": batch_size,
            "accumulation_steps": self.batch_scheduler.get_accumulation_steps(step),
            "effective_batch_size": self.batch_scheduler.get_effective_batch_size(step),
            "sequence_length": seq_length,
            "learning_rate": current_lr,
            "frozen_fraction": self._state.frozen_fraction,
            "model_fraction": self._state.model_fraction,
            "trainable_params": self.resolver.get_trainable_params(),
            "avg_loss": self._total_loss / max(1, self._loss_count),
        }

        if step % 100 == 0:
            self._history.append(metrics)
            if step % 1000 == 0:
                logger.info(
                    f"Step {step}: lr={current_lr:.2e}, bs={batch_size}, "
                    f"seq={seq_length}, frozen={self._state.frozen_fraction:.1%}, "
                    f"model={self._state.model_fraction:.1%}"
                )

        for cb in self._callbacks:
            try:
                cb(copy.deepcopy(self._state), metrics)
            except Exception as e:
                logger.warning(f"Progressive trainer callback failed: {e}")

        return metrics

    def get_state(self) -> ProgressiveState:
        """Get current progressive training state."""
        return copy.deepcopy(self._state)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            "state": self._state.to_dict(),
            "config": self.config.to_dict(),
            "steps_completed": self._step_count,
            "progress": self._step_count / max(1, self.config.total_steps),
            "avg_loss": self._total_loss / max(1, self._loss_count),
            "resolver_info": self.resolver.get_active_info(),
            "freezer_info": self.layer_freezer.get_info(),
            "batch_info": self.batch_scheduler.get_info(self._step_count),
            "seq_info": self.seq_scheduler.get_info(self._step_count),
            "num_stages": self.stage_scheduler.get_num_stages(),
            "history_length": len(self._history),
        }

    def save_state(self, path: str):
        """Save training state to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "state": self._state.to_dict(),
            "config": self.config.to_dict(),
            "step_count": self._step_count,
            "avg_loss": self._total_loss / max(1, self._loss_count),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: str):
        """Load training state from file."""
        with open(path, "r") as f:
            data = json.load(f)
        self._step_count = data.get("step_count", 0)
        self._state = ProgressiveState()
        state_data = data.get("state", {})
        for key, value in state_data.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)

    def reset(self):
        """Reset training state."""
        self._step_count = 0
        self._epoch_count = 0
        self._total_loss = 0.0
        self._loss_count = 0
        self._history.clear()
        self._state = ProgressiveState()
        self.layer_freezer._initial_freeze()
        self.resolver.set_active_fraction(self.config.initial_model_fraction)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_progressive_config(
    total_steps: int = 100000,
    **overrides,
) -> ProgressiveConfig:
    """Create a ProgressiveConfig with common defaults.

    Args:
        total_steps: Total training steps.
        **overrides: Override default values.

    Returns:
        Configured ProgressiveConfig.
    """
    defaults = {
        "total_steps": total_steps,
        "initial_batch_size": 8,
        "final_batch_size": 256,
        "initial_sequence_length": 32,
        "final_sequence_length": 4096,
        "initial_learning_rate": 1e-4,
        "final_learning_rate": 1e-6,
        "initial_model_fraction": 0.25,
        "final_model_fraction": 1.0,
        "initial_frozen_fraction": 0.75,
        "final_frozen_fraction": 0.0,
    }
    defaults.update(overrides)
    return ProgressiveConfig(**defaults)


def create_standard_stages(
    total_steps: int = 100000,
    warmup_fraction: float = 0.05,
    growth_fraction: float = 0.7,
) -> List[StageConfig]:
    """Create a standard 3-stage training schedule.

    Args:
        total_steps: Total training steps.
        warmup_fraction: Fraction for warmup stage.
        growth_fraction: Fraction for growth stage.

    Returns:
        List of StageConfig objects.
    """
    warmup_end = int(total_steps * warmup_fraction)
    growth_end = int(total_steps * (warmup_fraction + growth_fraction))

    return [
        StageConfig(
            name="warmup",
            stage_type=StageType.WARMUP,
            start_step=0,
            end_step=warmup_end,
            batch_size=8,
            learning_rate=1e-4,
            sequence_length=128,
            frozen_layers=0.75,
        ),
        StageConfig(
            name="growth",
            stage_type=StageType.GROWTH,
            start_step=warmup_end,
            end_step=growth_end,
            batch_size=64,
            learning_rate=3e-5,
            sequence_length=2048,
            frozen_layers=0.25,
        ),
        StageConfig(
            name="finetune",
            stage_type=StageType.FINE_TUNE,
            start_step=growth_end,
            end_step=total_steps,
            batch_size=128,
            learning_rate=1e-5,
            sequence_length=4096,
            frozen_layers=0.0,
        ),
    ]


def progressive_collate_fn(
    batch: List[Dict[str, Any]],
    max_seq_len: int = 4096,
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Collate function that respects progressive sequence length.

    Args:
        batch: List of sample dictionaries.
        max_seq_len: Maximum sequence length.
        pad_value: Padding value.

    Returns:
        Dictionary of collated tensors.
    """
    keys = set()
    for item in batch:
        keys.update(item.keys())

    result = {}
    for key in keys:
        values = [item.get(key) for item in batch]

        if isinstance(values[0], torch.Tensor):
            tensors = values
            max_len = min(max_seq_len, max(v.size(-1) for v in tensors))
            padded = []
            for v in tensors:
                if v.size(-1) > max_len:
                    padded.append(v[..., :max_len])
                elif v.size(-1) < max_len:
                    pad = torch.full(
                        (v.shape[0], max_len - v.size(-1)),
                        pad_value, dtype=v.dtype, device=v.device,
                    )
                    padded.append(torch.cat([v, pad], dim=-1))
                else:
                    padded.append(v)
            result[key] = torch.stack(padded)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values, dtype=torch.float32)
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = values

    return result


def export_progressive_report(
    trainer: ProgressiveTrainer,
    path: str,
):
    """Export a comprehensive progressive training report.

    Args:
        trainer: The ProgressiveTrainer instance.
        path: Output file path.
    """
    summary = trainer.get_summary()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Progressive training report exported to {path}")


def plot_progressive_schedule(
    config: ProgressiveConfig,
    num_points: int = 200,
) -> Dict[str, List[Tuple[int, float]]]:
    """Generate data for plotting progressive training schedules.

    Args:
        config: ProgressiveConfig.
        num_points: Number of data points.

    Returns:
        Dictionary mapping parameter name to list of (step, value) tuples.
    """
    steps = [
        int(i * config.total_steps / num_points)
        for i in range(num_points)
    ]

    schedules = {
        "batch_size": [],
        "sequence_length": [],
        "learning_rate": [],
        "frozen_fraction": [],
        "model_fraction": [],
    }

    for step in steps:
        progress = step / max(1, config.total_steps)

        bs = int(compute_growth(
            progress, config.initial_batch_size,
            config.final_batch_size, config.growth_function,
        ))
        schedules["batch_size"].append((step, float(bs)))

        sl = int(compute_growth(
            progress, config.initial_sequence_length,
            config.final_sequence_length, config.growth_function,
        ))
        schedules["sequence_length"].append((step, float(sl)))

        lr = compute_growth(
            progress, config.initial_learning_rate,
            config.final_learning_rate, GrowthFunction.LOGARITHMIC,
        )
        schedules["learning_rate"].append((step, lr))

        ff = config.initial_frozen_fraction + progress * (
            config.final_frozen_fraction - config.initial_frozen_fraction
        )
        schedules["frozen_fraction"].append((step, ff))

        mf = compute_growth(
            progress, config.initial_model_fraction,
            config.final_model_fraction, config.growth_function,
        )
        schedules["model_fraction"].append((step, mf))

    return schedules
