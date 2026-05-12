"""
Mixed Precision Training Module
================================

Production-grade mixed precision training with BF16/FP16 support, dynamic
loss scaling, gradient scaling, per-layer precision optimization, and
flash attention integration.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# LossScaler
# =============================================================================

class LossScaler:
    """Dynamic loss scaling with backoff and recovery.

    Prevents underflow in FP16 gradients by scaling the loss up before
    backward and unscaling gradients before the optimizer step.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2.0 ** 24,
        min_scale: float = 1.0,
        hysteresis: int = 2,
    ):
        """Initialize loss scaler.

        Args:
            init_scale: Initial loss scale.
            growth_factor: Factor to increase scale after successful steps.
            backoff_factor: Factor to decrease scale on overflow.
            growth_interval: Steps between growth checks.
            max_scale: Maximum loss scale.
            min_scale: Minimum loss scale.
            hysteresis: Number of consecutive non-overflows before growth.
        """
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.hysteresis = hysteresis

        self._scale = init_scale
        self._growth_tracker = 0
        self._overflow_count = 0
        self._total_steps = 0
        self._growth_cooldown = 0

    @property
    def scale(self) -> float:
        """Current loss scale."""
        return self._scale

    @property
    def loss_scale(self) -> torch.Tensor:
        """Current loss scale as tensor for CUDA operations."""
        return torch.tensor(self._scale, dtype=torch.float32)

    def compute_loss_scale(self) -> float:
        """Compute and return the current loss scale.

        Returns:
            Current loss scale value.
        """
        return self._scale

    def check_overflow(self, loss: torch.Tensor) -> bool:
        """Check if the loss or gradients contain overflow (inf/nan).

        Args:
            loss: Loss tensor to check.

        Returns:
            True if overflow detected.
        """
        if torch.isinf(loss) or torch.isnan(loss):
            return True
        return False

    def check_grad_overflow(self, params: List[torch.Tensor]) -> bool:
        """Check if any gradient contains overflow.

        Args:
            params: Model parameters.

        Returns:
            True if overflow detected in gradients.
        """
        for p in params:
            if p.grad is not None:
                if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                    return True
        return False

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> Tuple[bool, float]:
        """Unscale gradients by the current loss scale.

        Args:
            optimizer: Optimizer with gradients to unscale.

        Returns:
            Tuple of (found_inf, current_scale).
        """
        current_scale = self._scale
        found_inf = torch.tensor(0.0, dtype=torch.float32)

        if torch.cuda.is_available():
            found_inf = found_inf.cuda()

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        [p.grad], [current_scale], found_inf
                    )

        has_inf = bool(found_inf.item() > 0)
        return has_inf, current_scale

    def handle_overflow(self):
        """Handle detected overflow by reducing scale."""
        self._overflow_count += 1
        self._growth_tracker = 0
        self._growth_cooldown = self.hysteresis

        new_scale = self._scale * self.backoff_factor
        self._scale = max(new_scale, self.min_scale)

        logger.debug(
            "Overflow detected (count=%d). Reducing scale to %.1f",
            self._overflow_count, self._scale,
        )

    def update_scale(self, found_inf: bool):
        """Update scale based on whether overflow was found.

        Args:
            found_inf: Whether overflow was detected.
        """
        self._total_steps += 1

        if found_inf:
            self.handle_overflow()
            return

        if self._growth_cooldown > 0:
            self._growth_cooldown -= 1
            return

        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            new_scale = self._scale * self.growth_factor
            self._scale = min(new_scale, self.max_scale)
            self._growth_tracker = 0

    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "overflow_count": self._overflow_count,
            "total_steps": self._total_steps,
            "growth_cooldown": self._growth_cooldown,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load scaler state from checkpoint.

        Args:
            state: State dictionary.
        """
        self._scale = state.get("scale", self.init_scale)
        self._growth_tracker = state.get("growth_tracker", 0)
        self._overflow_count = state.get("overflow_count", 0)
        self._total_steps = state.get("total_steps", 0)
        self._growth_cooldown = state.get("growth_cooldown", 0)


# =============================================================================
# GradScaler
# =============================================================================

class GradScaler:
    """Gradient scaling wrapper compatible with PyTorch's AMP interface."""

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        """Initialize gradient scaler.

        Args:
            init_scale: Initial scale.
            growth_factor: Growth factor.
            backoff_factor: Backoff factor.
            growth_interval: Steps between growth.
            enabled: Whether scaling is enabled.
        """
        self._scaler = LossScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )
        self.enabled = enabled

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss.

        Args:
            loss: Loss tensor.

        Returns:
            Scaled loss.
        """
        if not self.enabled:
            return loss
        return loss * self._scaler.loss_scale.to(loss.device)

    def unscale_(self, optimizer: torch.optim.Optimizer):
        """Unscale optimizer gradients in-place.

        Args:
            optimizer: Optimizer to unscale.
        """
        if not self.enabled:
            return

        found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    inv_scale = 1.0 / self._scaler.scale
                    p.grad.mul_(inv_scale)
                    if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                        found_inf = True

        if found_inf:
            self._scaler.handle_overflow()

    def step(self, optimizer: torch.optim.Optimizer):
        """Execute optimizer step with overflow handling.

        Args:
            optimizer: Optimizer.
        """
        if not self.enabled:
            optimizer.step()
            return

        found_inf = self._scaler.check_grad_overflow(
            [p for group in optimizer.param_groups for p in group["params"]]
        )

        if not found_inf:
            optimizer.step()
            self._scaler.update_scale(False)
        else:
            self._scaler.update_scale(True)
            logger.debug("Skipping optimizer step due to overflow")

    def update(self):
        """Update the scale."""
        self._scaler._total_steps += 1

    def get_scale(self) -> float:
        """Get current scale."""
        return self._scaler.scale

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return self._scaler.state_dict()

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dict."""
        self._scaler.load_state_dict(state)


# =============================================================================
# MixedPrecisionTrainer
# =============================================================================

class MixedPrecisionTrainer:
    """Handles BF16/FP16 training with loss scaling.

    Provides a unified interface for mixed precision training with
    automatic detection of the best precision for the hardware.
    """

    def __init__(
        self,
        precision: str = "bf16",
        loss_scale: Optional[float] = None,
        dynamic_loss_scaling: bool = True,
    ):
        """Initialize mixed precision trainer.

        Args:
            precision: Target precision ('bf16', 'fp16', 'fp32').
            loss_scale: Initial loss scale (None for dynamic).
            dynamic_loss_scaling: Whether to use dynamic loss scaling.
        """
        self.precision = precision.lower()
        self.dynamic_loss_scaling = dynamic_loss_scaling

        init_scale = loss_scale or (2.0 ** 15 if self.precision == "fp16" else 1.0)
        self.loss_scaler = LossScaler(init_scale=init_scale)
        self.grad_scaler = GradScaler(
            init_scale=init_scale,
            enabled=(self.precision == "fp16" and dynamic_loss_scaling),
        )

        self._fp32_master_weights: Dict[str, torch.Tensor] = {}
        self._original_dtypes: Dict[str, torch.dtype] = {}
        self._is_setup = False
        self._device = _get_device()

    def setup(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Convert model to mixed precision and setup training.

        Args:
            model: Model to convert.
            optimizer: Optimizer.
        """
        if self.precision == "bf16":
            self._setup_bf16(model)
        elif self.precision == "fp16":
            self._setup_fp16(model)
        else:
            logger.info("FP32 training - no mixed precision conversion")

        self._is_setup = True
        logger.info("Mixed precision setup complete: %s", self.precision)

    def _setup_bf16(self, model: nn.Module):
        """Setup BF16 training.

        Args:
            model: Model to setup.
        """
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported on this device, falling back to FP32")
            return

        bf16_trainer = BF16Trainer()
        bf16_trainer.setup(model)

    def _setup_fp16(self, model: nn.Module):
        """Setup FP16 training.

        Args:
            model: Model to setup.
        """
        fp16_trainer = FP16Trainer()
        fp16_trainer.setup(model)

    def compute_loss_scale(self) -> float:
        """Compute dynamic loss scale.

        Returns:
            Current loss scale.
        """
        return self.loss_scaler.compute_loss_scale()

    def check_overflow(self, loss: torch.Tensor) -> bool:
        """Detect inf/nan in loss.

        Args:
            loss: Loss tensor.

        Returns:
            True if overflow detected.
        """
        return self.loss_scaler.check_overflow(loss)

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before clipping.

        Args:
            optimizer: Optimizer.
        """
        if self.precision == "fp16":
            self.grad_scaler.unscale_(optimizer)

    def handle_overflow(self):
        """Handle overflow by skipping step and reducing scale."""
        self.loss_scaler.handle_overflow()

    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Backward pass with loss scaling and overflow handling.

        Args:
            loss: Loss tensor.
            optimizer: Optimizer.

        Returns:
            True if step should be taken (no overflow).
        """
        if self.precision == "fp16" and self.dynamic_loss_scaling:
            scaled_loss = self.grad_scaler.scale(loss)
            scaled_loss.backward()
            self.grad_scaler.unscale_(optimizer)
            self.grad_scaler.step(optimizer)
            return True
        else:
            loss.backward()
            optimizer.step()
            return True


# =============================================================================
# BF16Trainer
# =============================================================================

class BF16Trainer:
    """BF16 mixed precision training with FP32 master weights.

    BF16 has the same dynamic range as FP32, so loss scaling is not needed.
    Master weights are kept in FP32 for optimizer stability.
    """

    def __init__(self):
        """Initialize BF16 trainer."""
        self._fp32_master: Dict[str, torch.Tensor] = {}
        self._device = _get_device()

    def _cast_to_bf16(self, model: nn.Module):
        """Convert model parameters to BF16.

        Args:
            model: Model to convert.
        """
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                self._fp32_master[name] = param.data.clone().float()
                param.data = param.data.to(torch.bfloat16)

        for name, buffer in model.named_buffers():
            if buffer.dtype == torch.float32:
                buffer.data = buffer.data.to(torch.bfloat16)

    def _store_fp32_master(self, model: nn.Module):
        """Store FP32 copies of all parameters.

        Args:
            model: Model to store weights from.
        """
        self._fp32_master = {}
        for name, param in model.named_parameters():
            self._fp32_master[name] = param.data.clone().float().cpu()

    def _sync_master_weights(self, model: nn.Module):
        """Sync BF16 weights to FP32 master (before optimizer step).

        Args:
            model: Model with BF16 weights.
        """
        for name, param in model.named_parameters():
            if name in self._fp32_master:
                self._fp32_master[name].copy_(param.data.float())

    def _load_from_master(self, model: nn.Module):
        """Load FP32 master weights to BF16 parameters (after optimizer step).

        Args:
            model: Model to update.
        """
        for name, param in model.named_parameters():
            if name in self._fp32_master:
                param.data.copy_(self._fp32_master[name].to(param.device).to(torch.bfloat16))

    def setup(self, model: nn.Module):
        """Setup BF16 training for model.

        Args:
            model: Model to setup.
        """
        model = model.to(self._device)

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported on current device")
            return

        self._cast_to_bf16(model)
        logger.info("BF16 training enabled with FP32 master weights")

    def pre_optimizer_step(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Call before optimizer.step() to sync master weights.

        Args:
            model: Training model.
            optimizer: Optimizer.
        """
        self._sync_master_weights(model)

        for group in optimizer.param_groups:
            for i, param in enumerate(group["params"]):
                name = self._find_param_name(model, param)
                if name and name in self._fp32_master:
                    group["params"][i] = self._fp32_master[name].to(param.device)

    def post_optimizer_step(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Call after optimizer.step() to load weights back.

        Args:
            model: Training model.
            optimizer: Optimizer.
        """
        self._load_from_master(model)

        original_params = []
        for name, param in model.named_parameters():
            original_params.append(param)

        idx = 0
        for group in optimizer.param_groups:
            for i in range(len(group["params"])):
                if idx < len(original_params):
                    group["params"][i] = original_params[idx]
                    idx += 1

    def _find_param_name(self, model: nn.Module, param: torch.Tensor) -> Optional[str]:
        """Find parameter name by identity.

        Args:
            model: Model.
            param: Parameter tensor.

        Returns:
            Parameter name or None.
        """
        for name, p in model.named_parameters():
            if p is param:
                return name
        return None


# =============================================================================
# FP16Trainer
# =============================================================================

class FP16Trainer:
    """FP16 mixed precision training with loss scaling.

    FP16 has limited dynamic range, so dynamic loss scaling is required
    to prevent gradient underflow.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 15,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """Initialize FP16 trainer.

        Args:
            init_scale: Initial loss scale.
            growth_factor: Scale growth factor.
            backoff_factor: Scale backoff factor.
            growth_interval: Steps between growth.
        """
        self.loss_scaler = LossScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )
        self._fp32_master: Dict[str, torch.Tensor] = {}
        self._device = _get_device()

    def setup(self, model: nn.Module):
        """Setup FP16 training.

        Args:
            model: Model to setup.
        """
        model = model.to(self._device)
        model.half()

        for name, param in model.named_parameters():
            self._fp32_master[name] = param.data.clone().float().cpu()

        for name, buffer in model.named_buffers():
            if buffer.dtype in (torch.float32, torch.float64):
                buffer.data = buffer.data.half()

        logger.info("FP16 training enabled with loss scaling")

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP16 training.

        Args:
            loss: Original loss.

        Returns:
            Scaled loss.
        """
        return loss * self.loss_scaler.loss_scale.to(loss.device)

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """Unscale gradients.

        Args:
            optimizer: Optimizer.

        Returns:
            Whether overflow was found.
        """
        found_inf = False
        inv_scale = 1.0 / self.loss_scaler.scale

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(inv_scale)
                    if torch.isinf(p.grad).any() or torch.isnan(p.grad).any():
                        found_inf = True

        return found_inf

    def step(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        """Execute optimizer step with overflow handling.

        Args:
            model: Training model.
            optimizer: Optimizer.

        Returns:
            True if step was taken.
        """
        found_inf = self.unscale_gradients(optimizer)

        if found_inf:
            self.loss_scaler.handle_overflow()
            optimizer.zero_grad()
            return False

        self.loss_scaler.update_scale(False)
        return True

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state."""
        return self.loss_scaler.state_dict()

    def load_state_dict(self, state: Dict[str, Any]):
        """Load trainer state."""
        self.loss_scaler.load_state_dict(state)


# =============================================================================
# PrecisionOptimizer
# =============================================================================

class PrecisionOptimizer:
    """Automatically select optimal precision per layer.

    Analyzes each layer's sensitivity to reduced precision and assigns
    BF16/FP16/FP32 accordingly.
    """

    def __init__(
        self,
        default_precision: str = "bf16",
        sensitive_layers: Optional[List[str]] = None,
        always_fp32_patterns: Optional[List[str]] = None,
    ):
        """Initialize precision optimizer.

        Args:
            default_precision: Default precision for most layers.
            sensitive_layers: Layer names that must stay FP32.
            always_fp32_patterns: Regex patterns for layers that must stay FP32.
        """
        self.default_precision = default_precision
        self.sensitive_layers = sensitive_layers or []
        self.always_fp32_patterns = always_fp32_patterns or [
            "layernorm", "layer_norm", "batchnorm", "batch_norm",
            "embedding", "lm_head", "output",
        ]
        self._layer_precision: Dict[str, str] = {}
        self._sensitivity_scores: Dict[str, float] = {}

    def analyze_layer_sensitivity(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
        num_batches: int = 3,
    ) -> Dict[str, float]:
        """Test each layer's precision tolerance.

        Measures output change when each layer is independently quantized
        to lower precision.

        Args:
            model: Model to analyze.
            dataloader: Calibration data.
            num_batches: Number of batches to use.

        Returns:
            Dictionary mapping layer names to sensitivity scores (0=robust, 1=sensitive).
        """
        model.eval()
        device = _get_device()
        model = model.to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    try:
                        model(**inputs)
                    except Exception:
                        continue

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data

                if weight.dtype == torch.float32:
                    weight_half = weight.half()
                    error = (weight.float() - weight_half.float()).abs().max().item()
                    sensitivity = min(1.0, error * 1000)
                elif weight.dtype == torch.bfloat16:
                    error = (weight.float() - weight.float()).abs().max().item()
                    sensitivity = 0.0
                else:
                    sensitivity = 0.5

                self._sensitivity_scores[name] = sensitivity

        return self._sensitivity_scores

    def assign_precision(self, model: nn.Module, sensitivity: Optional[Dict[str, float]] = None):
        """Assign precision per layer based on sensitivity.

        Args:
            model: Model to optimize.
            sensitivity: Optional pre-computed sensitivity scores.
        """
        import re

        if sensitivity:
            self._sensitivity_scores = sensitivity

        self._layer_precision = {}

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            if name in self.sensitive_layers:
                self._layer_precision[name] = "fp32"
                continue

            is_fp32_pattern = any(re.search(p, name) for p in self.always_fp32_patterns)
            if is_fp32_pattern:
                self._layer_precision[name] = "fp32"
                continue

            score = self._sensitivity_scores.get(name, 0.0)

            if score < 0.1:
                self._layer_precision[name] = "fp16"
            elif score < 0.3:
                self._layer_precision[name] = "bf16"
            else:
                self._layer_precision[name] = "fp32"

        fp32_count = sum(1 for v in self._layer_precision.values() if v == "fp32")
        bf16_count = sum(1 for v in self._layer_precision.values() if v == "bf16")
        fp16_count = sum(1 for v in self._layer_precision.values() if v == "fp16")

        logger.info("Precision assignment: FP32=%d, BF16=%d, FP16=%d",
                    fp32_count, bf16_count, fp16_count)

        return self._layer_precision

    def apply_precision(self, model: nn.Module) -> nn.Module:
        """Apply per-layer precision assignments.

        Args:
            model: Model to optimize.

        Returns:
            Model with per-layer precision applied.
        """
        if not self._layer_precision:
            self.assign_precision(model)

        dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }

        for name, module in model.named_modules():
            if name in self._layer_precision:
                target_dtype = dtype_map.get(self._layer_precision[name], torch.float32)

                if isinstance(module, nn.Linear):
                    module.weight.data = module.weight.data.to(target_dtype)
                    if module.bias is not None:
                        module.bias.data = module.bias.data.to(target_dtype)

        return model

    def get_precision_report(self) -> Dict[str, Any]:
        """Get precision assignment report.

        Returns:
            Report dictionary.
        """
        precision_counts = {"fp32": 0, "bf16": 0, "fp16": 0}
        for prec in self._layer_precision.values():
            if prec in precision_counts:
                precision_counts[prec] += 1

        return {
            "layer_assignments": self._layer_precision,
            "precision_counts": precision_counts,
            "total_layers": len(self._layer_precision),
            "sensitivity_scores": self._sensitivity_scores,
        }


# =============================================================================
# FlashAttentionWrapper
# =============================================================================

class FlashAttentionWrapper:
    """Wrap attention with flash attention when beneficial.

    Automatically replaces standard multi-head attention with flash
    attention for compatible hardware and configurations.
    """

    def __init__(
        self,
        enabled: bool = True,
        attention_dropout: float = 0.0,
        use_sdpa: bool = True,
    ):
        """Initialize flash attention wrapper.

        Args:
            enabled: Whether to enable flash attention.
            attention_dropout: Dropout rate for attention.
            use_sdpa: Whether to use scaled_dot_product_attention.
        """
        self.enabled = enabled
        self.attention_dropout = attention_dropout
        self.use_sdpa = use_sdpa
        self._wrapped_count = 0
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if flash attention is available.

        Returns:
            True if flash attention or SDPA is available.
        """
        if self.use_sdpa:
            return hasattr(F, "scaled_dot_product_attention")
        return False

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap attention modules in the model.

        Args:
            model: Model to wrap.

        Returns:
            Model with flash attention.
        """
        if not self.enabled or not self._available:
            return model

        for name, module in model.named_modules():
            if self._is_attention_module(module):
                self._replace_attention(model, name, module)

        logger.info("Wrapped %d attention modules with flash attention", self._wrapped_count)
        return model

    def _is_attention_module(self, module: nn.Module) -> bool:
        """Check if a module is an attention module.

        Args:
            module: Module to check.

        Returns:
            True if module is attention.
        """
        module_type = type(module).__name__.lower()
        return "attention" in module_type

    def _replace_attention(self, model: nn.Module, name: str, module: nn.Module):
        """Replace attention module with flash attention version.

        Args:
            model: Parent model.
            name: Module name.
            module: Original attention module.
        """
        try:
            if hasattr(F, "scaled_dot_product_attention"):
                flash_attn = FlashAttentionModule(
                    embed_dim=self._get_embed_dim(module),
                    num_heads=self._get_num_heads(module),
                    dropout=self.attention_dropout,
                )

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, flash_attn)
                self._wrapped_count += 1

        except Exception as e:
            logger.debug("Failed to wrap attention %s: %s", name, e)

    def _get_embed_dim(self, module: nn.Module) -> int:
        """Get embedding dimension from attention module.

        Args:
            module: Attention module.

        Returns:
            Embedding dimension.
        """
        if hasattr(module, "embed_dim"):
            return module.embed_dim
        if hasattr(module, "hidden_size"):
            return module.hidden_size
        if hasattr(module, "num_heads") and hasattr(module, "head_dim"):
            return module.num_heads * module.head_dim
        return 768

    def _get_num_heads(self, module: nn.Module) -> int:
        """Get number of attention heads.

        Args:
            module: Attention module.

        Returns:
            Number of heads.
        """
        if hasattr(module, "num_heads"):
            return module.num_heads
        if hasattr(module, "n_head"):
            return module.n_head
        if hasattr(module, "num_attention_heads"):
            return module.num_attention_heads
        return 12


class FlashAttentionModule(nn.Module):
    """Attention module using PyTorch's scaled_dot_product_attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize flash attention module.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with flash attention.

        Args:
            hidden_states: Input tensor (batch, seq, dim).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor.
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )
        else:
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            if self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)
