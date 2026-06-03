"""LR schedulers: linear, cosine, cosine_with_restarts, polynomial, constant, warmup."""

import math
import logging
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def _get_linear_warmup_lambda(warmup_steps: int, total_steps: int):
    """Create a linear warmup + linear decay lambda."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )
    return lr_lambda


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with a linear warmup and linear decay."""
    lr_lambda = _get_linear_warmup_lambda(num_warmup_steps, num_training_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_warmup_lambda(warmup_steps: int, total_steps: int, num_cycles: float = 0.5):
    """Create a cosine warmup + cosine decay lambda."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return lr_lambda


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with a linear warmup and cosine decay."""
    lr_lambda = _get_cosine_warmup_lambda(num_warmup_steps, num_training_steps, num_cycles)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_with_restarts_lambda(
    warmup_steps: int, total_steps: int, num_cycles: float = 1.0
):
    """Create a cosine warmup + cosine with restarts decay lambda."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
    return lr_lambda


def get_cosine_with_restarts_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with a linear warmup and cosine decay with restarts."""
    lr_lambda = _get_cosine_with_restarts_lambda(num_warmup_steps, num_training_steps, num_cycles)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_polynomial_decay_lambda(warmup_steps: int, total_steps: int, lr_end: float, power: float):
    """Create a polynomial decay lambda with warmup."""
    lr_start = 1.0
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step > total_steps:
            return lr_end
        lr_range = lr_start - lr_end
        decay_steps = total_steps - warmup_steps
        pct_remaining = 1.0 - (current_step - warmup_steps) / decay_steps
        decay = lr_range * pct_remaining ** power + lr_end
        return decay
    return lr_lambda


def get_polynomial_decay_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with a linear warmup and polynomial decay."""
    lr_lambda = _get_polynomial_decay_lambda(num_warmup_steps, num_training_steps, lr_end, power)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_constant_with_warmup_lambda(warmup_steps: int):
    """Create a constant schedule with warmup lambda."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with a constant learning rate preceded by a warmup."""
    lr_lambda = _get_constant_with_warmup_lambda(num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_inverse_sqrt_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a schedule with inverse square root decay after warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return float(max(1, num_warmup_steps)) ** 0.5 / float(current_step) ** 0.5
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_warmup_decay_per_param_group(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = "linear",
    **kwargs,
) -> LambdaLR:
    """Create a scheduler that can apply different settings per parameter group.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        scheduler_type: One of 'linear', 'cosine', 'cosine_with_restarts',
                       'polynomial', 'constant'.
        **kwargs: Additional scheduler-specific arguments.

    Returns:
        A LambdaLR scheduler.
    """
    schedulers = {
        "linear": lambda: _get_linear_warmup_lambda(num_warmup_steps, num_training_steps),
        "cosine": lambda: _get_cosine_warmup_lambda(
            num_warmup_steps, num_training_steps, kwargs.get("num_cycles", 0.5)
        ),
        "cosine_with_restarts": lambda: _get_cosine_with_restarts_lambda(
            num_warmup_steps, num_training_steps, kwargs.get("num_cycles", 1.0)
        ),
        "polynomial": lambda: _get_polynomial_decay_lambda(
            num_warmup_steps, num_training_steps,
            kwargs.get("lr_end", 1e-7), kwargs.get("power", 1.0)
        ),
        "constant": lambda: _get_constant_with_warmup_lambda(num_warmup_steps),
    }

    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Choose from {list(schedulers.keys())}"
        )

    lr_lambda = schedulers[scheduler_type]()
    return LambdaLR(optimizer, lr_lambda, kwargs.get("last_epoch", -1))
