"""Optimizer configs: AdamW, SGD, Adafactor, 8-bit Adam, group configs, param decay."""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer creation."""
    optimizer_type: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    momentum: float = 0.9
    sgd_nesterov: bool = False
    sgd_dampening: float = 0.0
    adafactor_clip_threshold: float = 1.0
    adafactor_scale_parameter: bool = True
    adafactor_relative_step: bool = False
    adafactor_warmup_init: bool = False
    max_grad_norm: float = 1.0
    no_decay_params: List[str] = field(default_factory=lambda: ["bias", "LayerNorm.weight", "layer_norm.weight"])
    separate_decay_groups: bool = True
    use_8bit: bool = False
    eps2: float = 1e-30


def _get_param_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_params: List[str],
    separate_decay_groups: bool = True,
    lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Create parameter groups with separate weight decay settings.

    Args:
        model: The model to get parameters from.
        weight_decay: Weight decay coefficient.
        no_decay_params: Parameter names that should not have weight decay.
        separate_decay_groups: Whether to separate params into decay/no-decay groups.
        lr: Override learning rate for all groups.

    Returns:
        List of parameter group dictionaries.
    """
    if not separate_decay_groups:
        params = [p for p in model.parameters() if p.requires_grad]
        group = {"params": params}
        if lr is not None:
            group["lr"] = lr
        return [group]

    decay_params = []
    no_decay_params_list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        should_decay = not any(nd in name for nd in no_decay_params)
        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params_list.append(param)

    groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params_list,
            "weight_decay": 0.0,
        },
    ]

    if lr is not None:
        for group in groups:
            group["lr"] = lr

    logger.info(
        f"Parameter groups: {len(decay_params)} with decay, "
        f"{len(no_decay_params_list)} without decay"
    )

    return groups


def build_optimizer(
    model: nn.Module,
    config: Optional[OptimizerConfig] = None,
    **kwargs,
) -> torch.optim.Optimizer:
    """Build an optimizer based on configuration.

    Args:
        model: The model to optimize.
        config: Optimizer configuration.
        **kwargs: Override configuration parameters.

    Returns:
        Configured optimizer instance.
    """
    if config is None:
        config = OptimizerConfig()

    # Apply kwargs overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    param_groups = _get_param_groups(
        model,
        weight_decay=config.weight_decay,
        no_decay_params=config.no_decay_params,
        separate_decay_groups=config.separate_decay_groups,
        lr=config.learning_rate,
    )

    optimizer_type = config.optimizer_type.lower()

    if optimizer_type == "adamw":
        return _build_adamw(param_groups, config)
    elif optimizer_type == "sgd":
        return _build_sgd(param_groups, config)
    elif optimizer_type == "adafactor":
        return _build_adafactor(model, config)
    elif optimizer_type in ("adam8bit", "adam_8bit", "8bit_adam"):
        return _build_8bit_adam(param_groups, config)
    elif optimizer_type == "adam":
        return _build_adam(param_groups, config)
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Supported: adamw, adam, sgd, adafactor, adam8bit"
        )


def _build_adamw(param_groups: List[Dict[str, Any]], config: OptimizerConfig) -> torch.optim.AdamW:
    """Build an AdamW optimizer."""
    return torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def _build_adam(param_groups: List[Dict[str, Any]], config: OptimizerConfig) -> torch.optim.Adam:
    """Build a standard Adam optimizer."""
    return torch.optim.Adam(
        param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def _build_sgd(param_groups: List[Dict[str, Any]], config: OptimizerConfig) -> torch.optim.SGD:
    """Build an SGD optimizer."""
    return torch.optim.SGD(
        param_groups,
        lr=config.learning_rate,
        momentum=config.momentum,
        nesterov=config.sgd_nesterov,
        dampening=config.sgd_dampening,
    )


def _build_adafactor(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build an Adafactor optimizer."""
    try:
        from transformers import Adafactor
        return Adafactor(
            model.parameters(),
            lr=config.learning_rate,
            eps=(config.adam_epsilon, config.eps2),
            clip_threshold=config.adafactor_clip_threshold,
            scale_parameter=config.adafactor_scale_parameter,
            relative_step=config.adafactor_relative_step,
            warmup_init=config.adafactor_warmup_init,
            weight_decay=config.weight_decay,
        )
    except ImportError:
        logger.warning("Transformers not available for Adafactor. Falling back to AdamW.")
        param_groups = _get_param_groups(
            model, config.weight_decay, config.no_decay_params,
            config.separate_decay_groups, config.learning_rate,
        )
        return _build_adamw(param_groups, config)


def _build_8bit_adam(param_groups: List[Dict[str, Any]], config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build an 8-bit Adam optimizer (bitsandbytes)."""
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
    except ImportError:
        logger.warning(
            "bitsandbytes not available for 8-bit Adam. Falling back to standard AdamW."
        )
        return _build_adamw(param_groups, config)


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get the current learning rate from an optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set the learning rate for all parameter groups in an optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def scale_optimizer_lr(optimizer: torch.optim.Optimizer, scale: float):
    """Scale the learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * scale


def get_parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(count: int) -> str:
    """Format parameter count in human-readable form."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)
