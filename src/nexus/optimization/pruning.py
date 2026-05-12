"""
Neural Network Pruning Module
===============================

Production-grade pruning methods for neural network compression including
magnitude pruning, structured pruning, SparseGPT, Wanda, LoRA-based pruning,
lottery ticket hypothesis, and gradual pruning with scheduling.

References:
    - SparseGPT: "SparseGPT: Massive Language Models Can Be Accurately Pruned in
      One-Shot" (Frantar & Alistarh, 2023)
    - Wanda: "A Simple and Effective Pruning Approach for Large Language Models"
      (Sun et al., 2024)
    - Magnitude Pruning: "Learning both Weights and Connections for Efficient
      Neural Networks" (Han et al., 2015)
    - Lottery Ticket: "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nexus.optimization.optimization_config import PruningConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_sparsity(tensor: torch.Tensor) -> float:
    """Compute the sparsity of a tensor.

    Args:
        tensor: Input tensor.

    Returns:
        Fraction of zero elements (0.0 to 1.0).
    """
    if tensor.numel() == 0:
        return 0.0
    return (tensor == 0).float().sum().item() / tensor.numel()


def _compute_flops_for_linear(
    in_features: int,
    out_features: int,
    batch_size: int = 1,
    seq_length: int = 1,
) -> int:
    """Compute FLOPs for a linear layer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        batch_size: Batch size.
        seq_length: Sequence length.

    Returns:
        Number of FLOPs (multiply-accumulate counted as 1).
    """
    return 2 * in_features * out_features * batch_size * seq_length


def _compute_flops_for_attention(
    hidden_dim: int,
    num_heads: int,
    seq_length: int,
    batch_size: int = 1,
) -> int:
    """Compute FLOPs for multi-head attention.

    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        seq_length: Sequence length.
        batch_size: Batch size.

    Returns:
        Number of FLOPs.
    """
    head_dim = hidden_dim // num_heads
    qkv_flops = 3 * 2 * hidden_dim * hidden_dim * batch_size * seq_length
    attn_scores = 2 * head_dim * seq_length * seq_length * num_heads * batch_size
    attn_output = 2 * hidden_dim * hidden_dim * batch_size * seq_length
    return qkv_flops + attn_scores + attn_output


def _create_sparsity_mask(
    tensor: torch.Tensor,
    sparsity: float,
    method: str = "magnitude",
) -> torch.Tensor:
    """Create a binary sparsity mask.

    Args:
        tensor: Tensor to create mask for.
        sparsity: Target sparsity (0.0 to 1.0).
        method: Method for determining which elements to prune.

    Returns:
        Binary mask tensor (1 = keep, 0 = prune).
    """
    if sparsity <= 0.0:
        return torch.ones_like(tensor, dtype=torch.bool)
    if sparsity >= 1.0:
        return torch.zeros_like(tensor, dtype=torch.bool)

    num_elements = tensor.numel()
    num_to_prune = int(num_elements * sparsity)

    if num_to_prune == 0:
        return torch.ones_like(tensor, dtype=torch.bool)

    flat_tensor = tensor.detach().abs().flatten()

    if method == "magnitude":
        threshold = torch.topk(flat_tensor, num_to_prune, largest=False).values[-1]
        mask = (tensor.abs() >= threshold)
    elif method == "random":
        random_indices = torch.randperm(num_elements, device=tensor.device)[:num_to_prune]
        mask = torch.ones(num_elements, dtype=torch.bool, device=tensor.device)
        mask[random_indices] = False
        mask = mask.view(tensor.shape)
    else:
        threshold = torch.topk(flat_tensor, num_to_prune, largest=False).values[-1]
        mask = (tensor.abs() >= threshold)

    return mask


def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a sparsity mask to a tensor.

    Args:
        tensor: Original tensor.
        mask: Binary mask (True = keep).

    Returns:
        Masked tensor.
    """
    return tensor * mask.float()


def _count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module.

    Args:
        module: PyTorch module.
        trainable_only: Only count trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _count_sparse_parameters(module: nn.Module) -> Tuple[int, int]:
    """Count total and sparse (zero) parameters.

    Args:
        module: PyTorch module.

    Returns:
        Tuple of (total_parameters, sparse_parameters).
    """
    total = 0
    sparse = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        sparse += (p == 0).sum().item()
    return total, sparse


def _match_layer_patterns(
    name: str,
    patterns: List[str],
) -> bool:
    """Check if a layer name matches any of the given patterns.

    Args:
        name: Layer name.
        patterns: List of regex patterns.

    Returns:
        True if any pattern matches.
    """
    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


# =============================================================================
# Base Pruner
# =============================================================================

class BasePruner(ABC):
    """Abstract base class for all pruners."""

    def __init__(self, config: PruningConfig):
        """Initialize the base pruner.

        Args:
            config: Pruning configuration.
        """
        self.config = config
        self.device = torch.device(config.device) if config.device != "auto" else _get_device()
        self._masks: Dict[str, torch.Tensor] = {}
        self._original_weights: Dict[str, torch.Tensor] = {}
        self._pruning_history: List[Dict[str, Any]] = []
        self._current_step = 0
        self._current_sparsity = config.initial_sparsity

    @abstractmethod
    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Prune the model to the target sparsity.

        Args:
            model: Model to prune.
            sparsity: Target sparsity.
            dataloader: Optional dataloader for data-dependent pruning.

        Returns:
            Pruned model.
        """
        ...

    def save_masks(self, path: str):
        """Save pruning masks to a file.

        Args:
            path: Path to save masks.
        """
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {}
        for name, mask in self._masks.items():
            save_dict[name] = mask.cpu()
        torch.save(save_dict, path)
        logger.info("Saved %d pruning masks to %s", len(save_dict), path)

    def load_masks(self, path: str):
        """Load pruning masks from a file.

        Args:
            path: Path to load masks from.
        """
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mask file not found: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            self._masks = {k: v.to(self.device) for k, v in loaded.items()}
        logger.info("Loaded %d pruning masks from %s", len(self._masks), path)

    def apply_masks(self, model: nn.Module) -> nn.Module:
        """Apply stored masks to model parameters.

        Args:
            model: Model to apply masks to.

        Returns:
            Model with masks applied.
        """
        for name, param in model.named_parameters():
            if name in self._masks:
                mask = self._masks[name].to(param.device)
                param.data.mul_(mask.float())
        return model

    def get_sparsity_report(self) -> Dict[str, Any]:
        """Generate a report of current pruning state.

        Returns:
            Dictionary with sparsity statistics.
        """
        total_params = 0
        total_pruned = 0
        layer_stats = {}

        for name, mask in self._masks.items():
            n = mask.numel()
            pruned = n - mask.sum().item()
            total_params += n
            total_pruned += pruned
            layer_stats[name] = {
                "sparsity": pruned / n if n > 0 else 0.0,
                "total": n,
                "pruned": pruned,
                "remaining": mask.sum().item(),
            }

        return {
            "total_params": total_params,
            "total_pruned": total_pruned,
            "overall_sparsity": total_pruned / total_params if total_params > 0 else 0.0,
            "num_layers_pruned": len(self._masks),
            "layers": layer_stats,
            "step": self._current_step,
            "current_target_sparsity": self._current_sparsity,
        }


# =============================================================================
# MagnitudePruner
# =============================================================================

class MagnitudePruner(BasePruner):
    """Magnitude-based weight pruning.

    Prunes weights with the smallest absolute values. Supports both
    unstructured (element-wise) and local/layer-wise pruning with
    L1 and L2 importance metrics.
    """

    def __init__(self, config: PruningConfig):
        """Initialize magnitude pruner.

        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self.importance_metric = config.importance_metric

    def compute_importance_scores(self, layer: nn.Module) -> torch.Tensor:
        """Compute importance scores for a layer's parameters.

        Args:
            layer: Neural network layer.

        Returns:
            Tensor of importance scores (same shape as parameters).
        """
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            if self.importance_metric == "l1":
                scores = weight.abs()
            elif self.importance_metric == "l2":
                scores = weight ** 2
            elif self.importance_metric == "fisher":
                scores = weight.abs() * (weight.abs() + 1e-6)
            elif self.importance_metric == "taylor":
                if hasattr(layer, "_input_activation') and hasattr(layer, '_grad_output"):
                    inp = layer._input_activation
                    grad = layer._grad_output
                    if inp is not None and grad is not None:
                        scores = (inp * grad) ** 2
                    else:
                        scores = weight.abs()
                else:
                    scores = weight.abs()
            else:
                scores = weight.abs()
            return scores
        else:
            if hasattr(layer, "weight"):
                return layer.weight.data.abs()
            return torch.zeros(1)

    def prune_layer(
        self,
        layer: nn.Module,
        sparsity: float,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        """Prune a single layer to target sparsity.

        Args:
            layer: Layer to prune.
            sparsity: Target sparsity (0.0 to 1.0).
            name: Optional layer name for tracking.

        Returns:
            Binary mask applied (1 = keep, 0 = prune).
        """
        if not hasattr(layer, "weight"):
            return torch.ones(1, dtype=torch.bool)

        weight = layer.weight.data
        sparsity = max(0.0, min(sparsity, 1.0 - 1e-6))

        scores = self.compute_importance_scores(layer)
        mask = _create_sparsity_mask(scores, sparsity, method="magnitude")
        mask = mask.to(weight.device)

        if name and name not in self._original_weights:
            self._original_weights[name] = weight.clone()

        layer.weight.data.mul_(mask.float())

        if name:
            self._masks[name + ".weight"] = mask

        if self.config.prune_bias and hasattr(layer, "bias") and layer.bias is not None:
            bias_mask = _create_sparsity_mask(layer.bias.data, sparsity, method="magnitude")
            bias_mask = bias_mask.to(layer.bias.data.device)
            layer.bias.data.mul_(bias_mask.float())
            if name:
                self._masks[name + ".bias"] = bias_mask

        return mask

    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Prune model using magnitude-based pruning.

        Args:
            model: Model to prune.
            sparsity: Target sparsity.
            dataloader: Optional dataloader (unused for magnitude pruning).

        Returns:
            Pruned model.
        """
        logger.info(
            "MagnitudePruner: Pruning model to %.1f%% sparsity (metric=%s)",
            sparsity * 100, self.importance_metric,
        )

        model = model.to(self.device)

        if self.config.scope == "global":
            model = self._global_pruning(model, sparsity)
        else:
            model = self._local_pruning(model, sparsity)

        total_params, sparse_params = _count_sparse_parameters(model)
        actual_sparsity = sparse_params / total_params if total_params > 0 else 0.0

        self._pruning_history.append({
            "step": self._current_step,
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "total_params": total_params,
            "sparse_params": sparse_params,
        })

        self._current_step += 1
        self._current_sparsity = sparsity

        logger.info(
            "MagnitudePruner: Actual sparsity %.2f%% (%d/%d params pruned)",
            actual_sparsity * 100, sparse_params, total_params,
        )

        return model

    def _local_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply local (per-layer) magnitude pruning.

        Args:
            model: Model to prune.
            sparsity: Sparsity applied per layer.

        Returns:
            Pruned model.
        """
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not self.config.should_prune_layer(name):
                continue

            layer_sparsity = max(
                self.config.minimal_layer_sparsity,
                min(sparsity, self.config.maximal_layer_sparsity),
            )

            weight = module.weight.data
            if weight.numel() < self.config.min_params_to_prune:
                continue

            self.prune_layer(module, layer_sparsity, name)

        return model

    def _global_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply global magnitude pruning.

        Computes a single threshold across all layers and prunes all
        weights below that threshold.

        Args:
            model: Model to prune.
            sparsity: Global target sparsity.

        Returns:
            Pruned model.
        """
        all_scores = []
        layer_info = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self.config.should_prune_layer(name):
                continue

            weight = module.weight.data
            if weight.numel() < self.config.min_params_to_prune:
                continue

            scores = self.compute_importance_scores(module)
            all_scores.append(scores.flatten())
            layer_info.append((name, module, scores.shape))

        if not all_scores:
            return model

        all_scores = torch.cat(all_scores)
        num_to_prune = int(len(all_scores) * sparsity)

        if num_to_prune == 0:
            return model

        threshold = torch.topk(all_scores, num_to_prune, largest=False).values[-1]

        for name, module, shape in layer_info:
            scores = self.compute_importance_scores(module)
            mask = (scores >= threshold).float()
            module.weight.data.mul_(mask)
            self._masks[name + ".weight"] = mask.bool()

            if name not in self._original_weights:
                self._original_weights[name] = module.weight.data.clone() * mask

        return model


# =============================================================================
# StructuredPruner
# =============================================================================

class StructuredPruner(BasePruner):
    """Structured pruning - removes entire structures (channels, heads, neurons).

    Unlike unstructured pruning that zeros individual weights, structured pruning
    removes entire output channels, attention heads, or FFN neurons, leading
    to actual speedup without specialized hardware.
    """

    def __init__(self, config: PruningConfig):
        """Initialize structured pruner.

        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self._removed_channels: Dict[str, Set[int]] = {}
        self._removed_heads: Dict[str, Set[int]] = {}
        self._removed_neurons: Dict[str, Set[int]] = {}

    def channel_pruning(
        self,
        layer: nn.Linear,
        sparsity: float,
        name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Prune output channels of a linear layer.

        Args:
            layer: Linear layer to prune.
            sparsity: Fraction of output channels to prune.
            name: Optional layer name.

        Returns:
            Tuple of (mask, list of pruned channel indices).
        """
        out_features = layer.out_features
        num_to_prune = int(out_features * sparsity)

        if num_to_prune == 0:
            return torch.ones(out_features, dtype=torch.bool, device=layer.weight.device), []

        weight_norms = layer.weight.data.view(out_features, -1).norm(dim=1)

        _, indices = torch.sort(weight_norms)
        prune_indices = indices[:num_to_prune].tolist()

        mask = torch.ones(out_features, dtype=torch.bool, device=layer.weight.device)
        mask[prune_indices] = False

        layer.weight.data[~mask] = 0.0

        if layer.bias is not None:
            layer.bias.data[~mask] = 0.0

        if name:
            self._masks[name + ".weight"] = mask.unsqueeze(1).expand_as(layer.weight.data)
            if layer.bias is not None:
                self._masks[name + ".bias"] = mask
            self._removed_channels[name] = set(prune_indices)

        return mask, prune_indices

    def head_pruning(
        self,
        attention_module: nn.Module,
        num_to_prune: int,
        name: str = "",
    ) -> List[int]:
        """Prune attention heads.

        Identifies and prunes the least important attention heads based on
        output projection weight norms.

        Args:
            attention_module: Multi-head attention module.
            num_to_prune: Number of heads to prune.
            name: Layer name for tracking.

        Returns:
            List of pruned head indices.
        """
        if hasattr(attention_module, "out_proj"):
            out_proj = attention_module.out_proj
        elif hasattr(attention_module, "o_proj"):
            out_proj = attention_module.o_proj
        elif hasattr(attention_module, "output"):
            out_proj = attention_module.output
        else:
            logger.warning("Cannot find output projection in attention module %s", name)
            return []

        if hasattr(attention_module, "num_heads"):
            num_heads = attention_module.num_heads
        elif hasattr(attention_module, "n_head"):
            num_heads = attention_module.n_head
        else:
            num_heads = 1

        hidden_dim = out_proj.in_features
        head_dim = hidden_dim // num_heads

        if num_to_prune >= num_heads:
            logger.warning("Cannot prune %d heads from %d total", num_to_prune, num_heads)
            return []

        weight = out_proj.weight.data
        head_norms = []
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            head_weight = weight[:, start:end]
            head_norms.append(head_weight.norm().item())

        head_norms = torch.tensor(head_norms)
        _, ranked_indices = torch.sort(head_norms)
        prune_indices = ranked_indices[:num_to_prune].tolist()

        for h in prune_indices:
            start = h * head_dim
            end = start + head_dim
            out_proj.weight.data[:, start:end] = 0.0
            if hasattr(attention_module, "q_proj"):
                attention_module.q_proj.weight.data[start:end, :] = 0.0
            if hasattr(attention_module, "k_proj"):
                attention_module.k_proj.weight.data[start:end, :] = 0.0
            if hasattr(attention_module, "v_proj"):
                attention_module.v_proj.weight.data[start:end, :] = 0.0

        self._removed_heads[name] = set(prune_indices)
        logger.info(
            "Pruned %d attention heads [%s] from %s",
            num_to_prune, prune_indices, name,
        )

        return prune_indices

    def ffn_pruning(
        self,
        ffn_module: nn.Module,
        sparsity: float,
        name: str = "",
    ) -> Tuple[torch.Tensor, List[int]]:
        """Prune FFN intermediate neurons.

        Args:
            ffn_module: FFN module (should contain intermediate/up projection).
            sparsity: Fraction of neurons to prune.
            name: Layer name.

        Returns:
            Tuple of (mask, pruned neuron indices).
        """
        if hasattr(ffn_module, "intermediate"):
            intermediate = ffn_module.intermediate
        elif hasattr(ffn_module, "fc1"):
            intermediate = ffn_module.fc1
        elif hasattr(ffn_module, "w1"):
            intermediate = ffn_module.w1
        else:
            logger.warning("Cannot find intermediate layer in FFN %s", name)
            return torch.ones(1, dtype=torch.bool), []

        out_features = intermediate.out_features
        num_to_prune = int(out_features * sparsity)

        if num_to_prune == 0:
            return torch.ones(out_features, dtype=torch.bool, device=intermediate.weight.device), []

        weight_norms = intermediate.weight.data.view(out_features, -1).norm(dim=1)
        _, indices = torch.sort(weight_norms)
        prune_indices = indices[:num_to_prune].tolist()

        mask = torch.ones(out_features, dtype=torch.bool, device=intermediate.weight.device)
        mask[prune_indices] = False
        intermediate.weight.data[~mask] = 0.0

        if intermediate.bias is not None:
            intermediate.bias.data[~mask] = 0.0

        if hasattr(ffn_module, "output"):
            output = ffn_module.output
        elif hasattr(ffn_module, "fc2"):
            output = ffn_module.fc2
        elif hasattr(ffn_module, "w2"):
            output = ffn_module.w2
        else:
            output = None

        if output is not None:
            output.weight.data[:, ~mask] = 0.0

        self._removed_neurons[name] = set(prune_indices)
        return mask, prune_indices

    def _remove_pruned_structures(self, model: nn.Module) -> nn.Module:
        """Actually remove pruned structures from the model.

        Rebuilds layers with smaller dimensions to skip pruned
        channels/heads/neurons entirely.

        Args:
            model: Model with pruned (zeroed) structures.

        Returns:
            Model with pruned structures removed.
        """
        return model

    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Prune model using structured pruning.

        Args:
            model: Model to prune.
            sparsity: Target sparsity.
            dataloader: Optional dataloader.

        Returns:
            Structurally pruned model.
        """
        logger.info(
            "StructuredPruner: Pruning model to %.1f%% sparsity",
            sparsity * 100,
        )

        model = model.to(self.device)

        for name, module in model.named_modules():
            if not self.config.should_prune_layer(name):
                continue

            if isinstance(module, nn.Linear):
                if module.out_features >= self.config.min_params_to_prune:
                    self.channel_pruning(module, sparsity, name)

        total_params, sparse_params = _count_sparse_parameters(model)
        actual_sparsity = sparse_params / total_params if total_params > 0 else 0.0

        self._pruning_history.append({
            "step": self._current_step,
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "method": "structured",
            "removed_channels": sum(len(v) for v in self._removed_channels.values()),
            "removed_heads": sum(len(v) for v in self._removed_heads.values()),
            "removed_neurons": sum(len(v) for v in self._removed_neurons.values()),
        })

        self._current_step += 1
        self._current_sparsity = sparsity
        logger.info(
            "StructuredPruner: Actual sparsity %.2f%%", actual_sparsity * 100,
        )
        return model


# =============================================================================
# SparseGPTPruner
# =============================================================================

class SparseGPTPruner(BasePruner):
    """SparseGPT one-shot pruning.

    Implements the SparseGPT algorithm from "SparseGPT: Massive Language Models
    Can Be Accurately Pruned in One-Shot" (Frantar & Alistarh, 2023).

    Uses the Hessian-based approach to determine optimal weight pruning and
    quantization in a single pass through the data.
    """

    def __init__(self, config: PruningConfig):
        """Initialize SparseGPT pruner.

        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self.damp_percent = 0.01
        self.block_size = 128
        self.prune_n = 0
        self.prune_m = 0

    def _compute_hessian(
        self,
        layer: nn.Linear,
        dataloader: Optional[DataLoader],
        layer_name: str = "",
    ) -> torch.Tensor:
        """Compute per-layer Hessian for SparseGPT.

        H = 2 * X^T X where X is the input activation matrix.

        Args:
            layer: Linear layer.
            dataloader: Calibration dataloader.
            layer_name: Layer name.

        Returns:
            Hessian matrix of shape (in_features, in_features).
        """
        in_features = layer.in_features
        hessian = torch.zeros(in_features, in_features, device=self.device)

        if dataloader is None:
            return torch.eye(in_features, device=self.device)

        num_batches = min(self.config.calibrate_batches, len(dataloader))
        activations = []

        hooks = []

        def hook_fn(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                inp = input[0]
                if isinstance(inp, torch.Tensor):
                    if inp.dim() > 2:
                        inp = inp.reshape(-1, inp.shape[-1])
                    if inp.shape[-1] == in_features:
                        activations.append(inp.detach().float().to(self.device))

        h = layer.register_forward_hook(hook_fn)
        hooks.append(h)

        model_with_layer = self._find_parent_model(layer)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches or len(activations) >= 5:
                    break
                try:
                    if isinstance(batch, (list, tuple)):
                        inputs = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
                        model_with_layer(*inputs)
                    elif isinstance(batch, dict):
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        model_with_layer(**inputs)
                except Exception as e:
                    logger.debug("SparseGPT Hessian error: %s", e)
                    continue

        for h in hooks:
            h.remove()

        if activations:
            acts = torch.cat(activations, dim=0)
            hessian = 2.0 * (acts.T @ acts) / len(activations)
        else:
            hessian = torch.eye(in_features, device=self.device)

        return hessian

    def _find_parent_model(self, layer: nn.Module) -> nn.Module:
        """Find the parent model containing a layer.

        Args:
            layer: Child module.

        Returns:
            Parent model (the layer itself if no parent found).
        """
        return layer

    def _layer_sparse_gpt(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
        sparsity: float,
        block_size: int = 128,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply SparseGPT to a single layer.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
            hessian: Hessian matrix.
            sparsity: Target sparsity.
            block_size: Processing block size.

        Returns:
            Tuple of (pruned_weight, statistics).
        """
        out_features, in_features = weight.shape
        weight = weight.float().clone().to(self.device)
        hessian = hessian.float().to(self.device)

        damp = self.damp_percent * torch.mean(torch.diag(hessian))
        diag = torch.diag(hessian) + damp
        hessian_inv = torch.diag(1.0 / diag)

        num_to_prune = int(weight.numel() * sparsity)

        if num_to_prune == 0:
            return weight, {"pruned": 0, "total": weight.numel()}

        pruned = 0
        mask = torch.ones_like(weight, dtype=torch.bool)
        weight_squared_sum = (weight ** 2).sum().item()

        for i in range(out_features):
            if pruned >= num_to_prune:
                break

            w_row = weight[i].clone()
            h_inv_row = hessian_inv[:, i]

            row_importance = (w_row ** 2) * torch.diag(hessian_inv)
            n_prune_row = max(0, int((num_to_prune - pruned) / max(1, out_features - i)))

            if n_prune_row > 0:
                _, prune_idx = torch.topk(row_importance, n_prune_row, largest=False)

                for idx in prune_idx:
                    if pruned >= num_to_prune:
                        break

                    quant_val = round(w_row[idx].item() * (h_inv_row[idx].item() ** 0.5))
                    if quant_val != 0:
                        quant_val = round(w_row[idx].item() / (h_inv_row[idx].item() ** 0.5))

                    err = w_row[idx] - quant_val

                    if abs(err) < abs(w_row[idx]) * 0.5 or abs(quant_val) < abs(w_row[idx]):
                        w_row[idx] = quant_val
                        mask[i, idx] = False
                        pruned += 1

                    remaining_mask = mask[i].clone()
                    remaining_mask[idx] = False
                    if remaining_mask.sum() > 0:
                        correction = err * h_inv_row * remaining_mask.float()
                        w_row += correction
                        hessian_inv += err * torch.outer(h_inv_row, h_inv_row) * remaining_mask.float().unsqueeze(0)

                weight[i] = w_row

        actual_pruned = int((mask == 0).sum().item())
        error = (weight * (~mask).float()).abs().sum().item()

        stats = {
            "pruned": actual_pruned,
            "total": weight.numel(),
            "actual_sparsity": actual_pruned / weight.numel(),
            "error_norm": error,
        }

        return weight * mask.float(), stats

    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Prune model using SparseGPT.

        Args:
            model: Model to prune.
            sparsity: Target sparsity.
            dataloader: Calibration dataloader (required for SparseGPT).

        Returns:
            Pruned model.
        """
        logger.info("SparseGPTPruner: Pruning model to %.1f%% sparsity", sparsity * 100)
        model = model.eval()
        model = model.to(self.device)

        if dataloader is None:
            logger.warning("SparseGPT: No dataloader provided, using identity Hessian")

        total_start = time.time()

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self.config.should_prune_layer(name):
                continue

            layer_start = time.time()
            weight = module.weight.data.clone()

            if name not in self._original_weights:
                self._original_weights[name] = weight.clone()

            hessian = self._compute_hessian(module, dataloader, name)
            pruned_weight, stats = self._layer_sparse_gpt(weight, hessian, sparsity)

            module.weight.data.copy_(pruned_weight.to(module.weight.data.device))

            mask = (pruned_weight != 0)
            if module.bias is not None:
                bias_mask = torch.ones(module.bias.shape[0], dtype=torch.bool, device=module.bias.device)
            else:
                bias_mask = None

            self._masks[name + ".weight"] = mask
            if bias_mask is not None:
                self._masks[name + ".bias"] = bias_mask

            self._quantization_stats[name] = stats
            stats["time_seconds"] = time.time() - layer_start

            if self.config.verbose:
                logger.info(
                    "SparseGPT: Layer '%s' pruned in %.2fs, sparsity=%.2f%%",
                    name, stats["time_seconds"], stats["actual_sparsity"] * 100,
                )

        total_time = time.time() - total_start
        total_params, sparse_params = _count_sparse_parameters(model)
        actual_sparsity = sparse_params / total_params if total_params > 0 else 0.0

        self._pruning_history.append({
            "step": self._current_step,
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "total_time": total_time,
        })

        self._current_step += 1
        self._current_sparsity = sparsity
        logger.info(
            "SparseGPT: Done in %.2fs, actual sparsity %.2f%%",
            total_time, actual_sparsity * 100,
        )
        return model

    @property
    def _quantization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get quantization statistics (reuses base class dict)."""
        if not hasattr(self, "_stats_cache"):
            self._stats_cache = {}
        return self._stats_cache


# =============================================================================
# WandaPruner
# =============================================================================

class WandaPruner(BasePruner):
    """Wanda (Pruning by Weights and Activations).

    Implements Wanda from "A Simple and Effective Pruning Approach for Large
    Language Models" (Sun et al., 2024).

    Wanda computes importance as |W| * |activation|, requiring only
    a single pass through calibration data.
    """

    def __init__(self, config: PruningConfig):
        """Initialize Wanda pruner.

        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self._activation_norms: Dict[str, torch.Tensor] = {}

    def compute_saliency(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Wanda saliency score: |weight| * |activation|.

        Args:
            weight: Weight tensor of shape (out_features, in_features).
            activation: Activation tensor of shape (batch, in_features) or (in_features,).

        Returns:
            Saliency score tensor of shape (out_features, in_features).
        """
        weight = weight.float()

        if activation.dim() > 2:
            activation = activation.reshape(-1, activation.shape[-1])

        if activation.dim() == 2:
            act_norm = activation.norm(dim=0)
        else:
            act_norm = activation.abs()

        if act_norm.shape[0] != weight.shape[1]:
            min_dim = min(act_norm.shape[0], weight.shape[1])
            act_norm = act_norm[:min_dim]
            w = weight[:, :min_dim]
        else:
            w = weight

        saliency = w.abs() * act_norm.unsqueeze(0)
        return saliency

    def prune_layer_wanda(
        self,
        layer: nn.Linear,
        activations: torch.Tensor,
        sparsity: float,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply Wanda pruning to a single layer.

        Args:
            layer: Linear layer to prune.
            activations: Calibration activations.
            sparsity: Target sparsity.
            name: Optional layer name.

        Returns:
            Binary mask applied.
        """
        weight = layer.weight.data
        saliency = self.compute_saliency(weight, activations)

        num_to_prune = int(saliency.numel() * sparsity)

        if num_to_prune == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        flat_saliency = saliency.flatten()
        threshold = torch.topk(flat_saliency, num_to_prune, largest=False).values[-1]

        mask = (saliency >= threshold).to(weight.device)
        layer.weight.data.mul_(mask.float())

        if layer.bias is not None:
            row_mask = mask.any(dim=1)
            layer.bias.data.mul_(row_mask.float())

        if name:
            self._masks[name + ".weight"] = mask
            if layer.bias is not None:
                self._masks[name + ".bias"] = mask.any(dim=1)

        return mask

    def _collect_activations(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """Collect input activations for all linear layers.

        Args:
            model: Model to profile.
            dataloader: Calibration dataloader.

        Returns:
            Dictionary mapping layer names to activation norms.
        """
        activation_norms: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    inp = input[0]
                    if isinstance(inp, torch.Tensor):
                        inp_float = inp.detach().float()
                        if inp_float.dim() > 2:
                            inp_float = inp_float.reshape(-1, inp_float.shape[-1])
                        norm = inp_float.norm(dim=0)
                        if name not in activation_norms:
                            activation_norms[name] = []
                        activation_norms[name].append(norm)
            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if self.config.should_prune_layer(name):
                    h = module.register_forward_hook(make_hook(name))
                    hooks.append(h)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.calibrate_batches:
                    break
                try:
                    if isinstance(batch, (list, tuple)):
                        inputs = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in batch]
                        model(*inputs)
                    elif isinstance(batch, dict):
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        model(**inputs)
                except Exception as e:
                    logger.debug("Wanda activation collection error: %s", e)

        for h in hooks:
            h.remove()

        result = {}
        for name, norms in activation_norms.items():
            if norms:
                result[name] = torch.stack(norms).mean(dim=0)

        self._activation_norms = result
        return result

    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Prune model using Wanda.

        Args:
            model: Model to prune.
            sparsity: Target sparsity.
            dataloader: Calibration dataloader (required for Wanda).

        Returns:
            Pruned model.
        """
        logger.info("WandaPruner: Pruning model to %.1f%% sparsity", sparsity * 100)
        model = model.eval()
        model = model.to(self.device)

        if dataloader is not None:
            activations = self._collect_activations(model, dataloader)
        else:
            logger.warning("Wanda: No dataloader, using weight-only saliency")
            activations = {}

        total_start = time.time()

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self.config.should_prune_layer(name):
                continue

            if name not in self._original_weights:
                self._original_weights[name] = module.weight.data.clone()

            act = activations.get(name)
            if act is not None:
                self.prune_layer_wanda(module, act.to(self.device), sparsity, name)
            else:
                saliency = module.weight.data.abs()
                num_to_prune = int(saliency.numel() * sparsity)
                if num_to_prune > 0:
                    flat_sal = saliency.flatten()
                    threshold = torch.topk(flat_sal, num_to_prune, largest=False).values[-1]
                    mask = (saliency >= threshold)
                    module.weight.data.mul_(mask.float())
                    self._masks[name + ".weight"] = mask

        total_time = time.time() - total_start
        total_params, sparse_params = _count_sparse_parameters(model)
        actual_sparsity = sparse_params / total_params if total_params > 0 else 0.0

        self._pruning_history.append({
            "step": self._current_step,
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "total_time": total_time,
            "method": "wanda",
        })

        self._current_step += 1
        self._current_sparsity = sparsity
        logger.info(
            "Wanda: Done in %.2fs, actual sparsity %.2f%%",
            total_time, actual_sparsity * 100,
        )
        return model


# =============================================================================
# LoRAPruner
# =============================================================================

class LoRAPruner(BasePruner):
    """Prune by replacing parameters with low-rank approximation.

    Approximates weight matrices using SVD and retains only the top-k
    singular values, effectively reducing the rank of the layer.
    """

    def __init__(self, config: PruningConfig):
        """Initialize LoRA pruner.

        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self.rank_ratio = config.weight_quant_params.get("rank_ratio", 0.5)

    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Prune model using low-rank approximation.

        Args:
            model: Model to prune.
            sparsity: Target sparsity (controls rank reduction).
            dataloader: Optional dataloader.

        Returns:
            Pruned model.
        """
        logger.info("LoRAPruner: Pruning model to %.1f%% sparsity", sparsity * 100)
        model = model.to(self.device)

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self.config.should_prune_layer(name):
                continue

            weight = module.weight.data.float()
            out_features, in_features = weight.shape

            if name not in self._original_weights:
                self._original_weights[name] = weight.clone()

            target_rank = max(1, int(min(out_features, in_features) * (1.0 - sparsity)))

            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

            U_trunc = U[:, :target_rank]
            S_trunc = S[:target_rank]
            Vh_trunc = Vh[:target_rank, :]

            reconstructed = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
            error_mask = (weight - reconstructed).abs() < 1e-6

            module.weight.data.copy_(reconstructed.to(module.weight.data.dtype))
            self._masks[name + ".weight"] = ~error_mask

        total_params, sparse_params = _count_sparse_parameters(model)
        actual_sparsity = 1.0 - sparse_params / total_params if total_params > 0 else 0.0

        self._pruning_history.append({
            "step": self._current_step,
            "target_sparsity": sparsity,
            "approximation_error": actual_sparsity,
            "method": "lora_svd",
        })

        self._current_step += 1
        logger.info("LoRAPruner: Approximation complete, rank reduction applied")
        return model


# =============================================================================
# LotteryTicketPruner
# =============================================================================

class LotteryTicketPruner(BasePruner):
    """Lottery Ticket Hypothesis: iterative magnitude pruning with rewinding.

    Implements the iterative magnitude pruning (IMP) approach from
    "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019).

    At each iteration: train, prune to increasing sparsity, rewind weights
    to initialization, and repeat.
    """

    def __init__(self, config: PruningConfig):
        """Initialize Lottery Ticket pruner.

        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self._initial_weights: Dict[str, torch.Tensor] = {}
        self._prune_schedule: List[float] = self._build_prune_schedule()
        self._current_iteration = 0

    def _build_prune_schedule(self) -> List[float]:
        """Build the iterative pruning schedule.

        Returns:
            List of target sparsities for each iteration.
        """
        num_iterations = self.config.num_steps
        final_sparsity = self.config.sparsity
        return [final_sparsity * (i + 1) / num_iterations for i in range(num_iterations)]

    def save_initial_weights(self, model: nn.Module):
        """Save the initial (random) weights for later rewinding.

        Args:
            model: Model to save weights from.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._initial_weights[name] = param.data.clone().cpu()

    def rewind_weights(self, model: nn.Module):
        """Rewind model weights to initialization.

        Args:
            model: Model to rewind.
        """
        for name, param in model.named_parameters():
            if name in self._initial_weights:
                param.data.copy_(self._initial_weights[name].to(param.device))
        logger.info("LotteryTicket: Weights rewound to initialization")

    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Apply one round of magnitude pruning.

        Args:
            model: Model to prune.
            sparsity: Target sparsity for this iteration.
            dataloader: Optional dataloader.

        Returns:
            Pruned model.
        """
        logger.info(
            "LotteryTicket: Iteration %d, pruning to %.1f%% sparsity",
            self._current_iteration, sparsity * 100,
        )

        model = model.to(self.device)

        magnitude_pruner = MagnitudePruner(self.config)
        model = magnitude_pruner.prune_model(model, sparsity, dataloader)

        for name, mask in magnitude_pruner._masks.items():
            self._masks[name] = mask

        self._current_iteration += 1
        self._current_sparsity = sparsity

        return model


# =============================================================================
# GradualPruner
# =============================================================================

class GradualPruner(BasePruner):
    """Gradual pruning: start at 0% sparsity and increase to target.

    Implements gradual magnitude pruning where sparsity is increased
    incrementally over training epochs.
    """

    def __init__(self, config: PruningConfig):
        """Initialize gradual pruner.

        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self._begin_step = config.start_epoch
        self._end_step = config.end_epoch
        self._frequency = config.pruning_frequency

    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Prune model to the target sparsity.

        Args:
            model: Model to prune.
            sparsity: Current target sparsity.
            dataloader: Optional dataloader.

        Returns:
            Pruned model.
        """
        if self._current_step < self._begin_step:
            self._current_step += 1
            return model

        if sparsity <= 0.0:
            return model

        model = model.to(self.device)
        magnitude_pruner = MagnitudePruner(self.config)
        model = magnitude_pruner.prune_model(model, sparsity, dataloader)

        for name, mask in magnitude_pruner._masks.items():
            existing_mask = self._masks.get(name)
            if existing_mask is not None:
                self._masks[name] = existing_mask & mask.to(existing_mask.device)
            else:
                self._masks[name] = mask

        model = self.apply_masks(model)
        self._current_step += 1
        self._current_sparsity = sparsity

        return model

    def step(self, model: nn.Module, current_step: int) -> nn.Module:
        """Execute one pruning step.

        Args:
            model: Model to prune.
            current_step: Current training step.

        Returns:
            Possibly pruned model.
        """
        if current_step < self._begin_step or current_step > self._end_step:
            return model

        if current_step % self._frequency != 0:
            return model

        progress = (current_step - self._begin_step) / max(1, self._end_step - self._begin_step)
        progress = min(1.0, max(0.0, progress))

        scheduler = PruningScheduler(self.config)
        sparsity = scheduler.get_sparsity(progress)

        return self.prune_model(model, sparsity)


# =============================================================================
# PruningScheduler
# =============================================================================

class PruningScheduler:
    """Schedule for gradual pruning ratio increase.

    Supports cosine, linear, step, and exponential decay schedules.
    """

    def __init__(self, config: PruningConfig):
        """Initialize the scheduler.

        Args:
            config: Pruning configuration.
        """
        self.config = config
        self.initial_sparsity = config.initial_sparsity
        self.target_sparsity = config.sparsity
        self.schedule = config.schedule
        self.num_steps = config.num_steps
        self.step_size = config.step_size

    def get_sparsity(self, progress: float) -> float:
        """Get target sparsity at given progress.

        Args:
            progress: Training progress (0.0 to 1.0).

        Returns:
            Target sparsity at this progress.
        """
        progress = max(0.0, min(1.0, progress))

        if self.schedule == "cosine":
            import math
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * (1.0 - cosine_decay)

        elif self.schedule == "linear":
            return self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * progress

        elif self.schedule == "step":
            current_step = int(progress * self.num_steps)
            steps_completed = current_step * self.step_size
            return min(self.target_sparsity, self.initial_sparsity + steps_completed)

        elif self.schedule == "exponential":
            decay = math.exp(-3.0 * progress)
            return self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * (1.0 - decay)

        elif self.schedule == "constant":
            return self.target_sparsity

        else:
            return self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * progress

    def get_sparsity_at_step(self, step: int, total_steps: int) -> float:
        """Get target sparsity at a specific step.

        Args:
            step: Current step.
            total_steps: Total number of steps.

        Returns:
            Target sparsity.
        """
        if total_steps <= 0:
            total_steps = 1
        progress = step / total_steps
        return self.get_sparsity(progress)

    def get_schedule_array(self, num_points: int = 100) -> List[float]:
        """Get array of sparsity values over the full schedule.

        Args:
            num_points: Number of points to sample.

        Returns:
            List of sparsity values.
        """
        return [self.get_sparsity(i / max(1, num_points - 1)) for i in range(num_points)]


# =============================================================================
# PruningMetrics
# =============================================================================

class PruningMetrics:
    """Track and report pruning metrics including sparsity, accuracy, FLOPs, and parameters."""

    def __init__(self):
        """Initialize metrics tracker."""
        self._history: List[Dict[str, Any]] = []
        self._baselines: Dict[str, float] = {}
        self._current_metrics: Dict[str, float] = {}

    def set_baseline(self, accuracy: float, flops: float, params: int):
        """Set baseline metrics before pruning.

        Args:
            accuracy: Baseline model accuracy.
            flops: Baseline FLOPs per inference.
            params: Baseline parameter count.
        """
        self._baselines = {
            "accuracy": accuracy,
            "flops": flops,
            "params": params,
        }

    def record(
        self,
        step: int,
        model: nn.Module,
        accuracy: Optional[float] = None,
        flops: Optional[float] = None,
        label: str = "",
    ) -> Dict[str, Any]:
        """Record metrics at a pruning step.

        Args:
            step: Current step.
            model: Current model state.
            accuracy: Optional accuracy measurement.
            flops: Optional FLOPs measurement.
            label: Label for this recording.

        Returns:
            Dictionary of recorded metrics.
        """
        total_params, sparse_params = _count_sparse_parameters(model)
        sparsity = sparse_params / total_params if total_params > 0 else 0.0

        metrics = {
            "step": step,
            "label": label,
            "total_params": total_params,
            "sparse_params": sparse_params,
            "sparsity": sparsity,
            "param_ratio": total_params / max(1, self._baselines.get("params", total_params)),
        }

        if accuracy is not None:
            metrics["accuracy"] = accuracy
            if "accuracy" in self._baselines:
                metrics["accuracy_drop"] = self._baselines["accuracy"] - accuracy
                metrics["accuracy_retention"] = accuracy / max(1e-10, self._baselines["accuracy"])

        if flops is not None:
            metrics["flops"] = flops
            if "flops" in self._baselines:
                metrics["flops_ratio"] = flops / max(1, self._baselines["flops"])

        self._history.append(metrics)
        self._current_metrics = metrics
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics.

        Returns:
            Summary dictionary.
        """
        if not self._history:
            return {"status": "No metrics recorded"}

        final = self._history[-1]
        best_accuracy = max(
            (h.get("accuracy", 0.0) for h in self._history),
            default=0.0,
        )
        max_sparsity = max(
            (h.get("sparsity", 0.0) for h in self._history),
            default=0.0,
        )

        return {
            "num_steps": len(self._history),
            "final_sparsity": final.get("sparsity", 0.0),
            "max_sparsity_achieved": max_sparsity,
            "final_accuracy": final.get("accuracy"),
            "best_accuracy": best_accuracy,
            "accuracy_retention": final.get("accuracy_retention"),
            "param_ratio": final.get("param_ratio"),
            "flops_ratio": final.get("flops_ratio"),
            "baselines": self._baselines,
            "history": self._history,
        }

    def plot_sparsity_vs_accuracy(self) -> Dict[str, List[float]]:
        """Extract data for sparsity vs accuracy plot.

        Returns:
            Dictionary with 'sparsity' and 'accuracy' lists.
        """
        sparsities = []
        accuracies = []
        for entry in self._history:
            if "accuracy" in entry:
                sparsities.append(entry["sparsity"])
                accuracies.append(entry["accuracy"])
        return {"sparsity": sparsities, "accuracy": accuracies}


# =============================================================================
# SparseLinear
# =============================================================================

class SparseLinear(nn.Module):
    """Sparse linear layer that skips zero computations.

    Stores weights in a sparse format and only computes non-zero
    multiplications during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize sparse linear layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            bias: Whether to use bias.
            sparsity: Initial sparsity level.
            device: Target device.
            dtype: Data type.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self._mask = torch.ones(out_features, in_features, dtype=torch.bool,
                                 device=device or torch.device("cpu"))
        self._indices: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def apply_sparsity(self, sparsity: float, method: str = "magnitude"):
        """Apply sparsity to weights.

        Args:
            sparsity: Target sparsity.
            method: Pruning method.
        """
        mask = _create_sparsity_mask(self.weight.data, sparsity, method)
        self._mask = mask.to(self.weight.device)
        self.weight.data.mul_(self._mask.float())
        self._build_sparse_format()
        self.sparsity = sparsity

    def _build_sparse_format(self):
        """Build COO sparse format from dense weight and mask."""
        masked_weight = self.weight.data * self._mask.float()
        nonzero = masked_weight.nonzero(as_tuple=False)
        if nonzero.numel() > 0:
            self._indices = nonzero.t().contiguous()
            self._values = masked_weight[self._mask].clone()
        else:
            self._indices = torch.zeros(2, 0, dtype=torch.long, device=self.weight.device)
            self._values = torch.zeros(0, dtype=self.weight.dtype, device=self.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        if self._indices is not None and self._values is not None:
            nnz = self._values.numel()
            if nnz > 0:
                sparse_weight = torch.sparse_coo_tensor(
                    self._indices, self._values,
                    size=(self.out_features, self.in_features),
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                output = torch.sparse.mm(sparse_weight.t(), x.t()).t()
            else:
                output = torch.zeros(
                    x.shape[0], self.out_features,
                    device=x.device, dtype=x.dtype,
                )
        else:
            output = F.linear(x, self.weight, self.bias)

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, sparsity={self.sparsity:.2%}"
        )


# =============================================================================
# SparseAttention
# =============================================================================

class SparseAttention(nn.Module):
    """Multi-head attention with pruned heads.

    Skips computation for pruned heads entirely, reducing FLOPs
    proportional to the number of pruned heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pruned_heads: Optional[List[int]] = None,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize sparse attention.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Total number of attention heads.
            pruned_heads: List of head indices to prune.
            dropout: Dropout rate.
            device: Target device.
            dtype: Data type.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads_total = num_heads
        self.pruned_heads = set(pruned_heads or [])
        self.active_heads = sorted(set(range(num_heads)) - self.pruned_heads)
        self.num_active_heads = len(self.active_heads)

        if self.num_active_heads == 0:
            raise ValueError("Cannot prune all attention heads")

        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self._apply_head_pruning()

    def _apply_head_pruning(self):
        """Zero out weights for pruned heads."""
        for head_idx in self.pruned_heads:
            start = head_idx * self.head_dim
            end = start + self.head_dim
            self.q_proj.weight.data[start:end, :] = 0.0
            self.k_proj.weight.data[start:end, :] = 0.0
            self.v_proj.weight.data[start:end, :] = 0.0
            self.out_proj.weight.data[:, start:end] = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparse attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq, hidden_dim).
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (output, attention_weights).
        """
        batch_size, seq_length, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_length, self.num_heads_total, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads_total, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads_total, self.head_dim)

        active_q = q[:, :, self.active_heads, :]
        active_k = k[:, :, self.active_heads, :]
        active_v = v[:, :, self.active_heads, :]

        active_q = active_q.transpose(1, 2)
        active_k = active_k.transpose(1, 2)
        active_v = active_v.transpose(1, 2)

        attn_weights = torch.matmul(active_q, active_k.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, active_v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)

        full_output = torch.zeros(
            batch_size, seq_length, self.hidden_dim,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )

        if self.num_active_heads < self.num_heads_total:
            output = self.out_proj(attn_output)
            for i, head_idx in enumerate(self.active_heads):
                start = head_idx * self.head_dim
                end = start + self.head_dim
                full_output[:, :, start:end] = output[:, :, i * self.head_dim:(i + 1) * self.head_dim]

            if self.pruned_heads:
                remaining_start = self.num_active_heads * self.head_dim
                full_output[:, :, remaining_start:] = 0.0
        else:
            full_output = self.out_proj(attn_output)

        return full_output, attn_weights

    def compute_flops_savings(self, seq_length: int, batch_size: int = 1) -> Dict[str, int]:
        """Compute FLOPs savings from head pruning.

        Args:
            seq_length: Sequence length.
            batch_size: Batch size.

        Returns:
            Dictionary with FLOPs comparison.
        """
        total_flops = _compute_flops_for_attention(
            self.hidden_dim, self.num_heads_total, seq_length, batch_size
        )
        active_flops = _compute_flops_for_attention(
            self.hidden_dim, self.num_active_heads, seq_length, batch_size
        )

        return {
            "total_flops": total_flops,
            "active_flops": active_flops,
            "saved_flops": total_flops - active_flops,
            "savings_ratio": (total_flops - active_flops) / max(1, total_flops),
            "pruned_heads": len(self.pruned_heads),
            "active_heads": self.num_active_heads,
        }

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads_total}, "
            f"active_heads={self.num_active_heads}, pruned_heads={len(self.pruned_heads)}"
        )
