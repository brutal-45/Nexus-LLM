"""
Learning Rate Schedulers
==========================
Custom learning rate schedulers for LLM training.

Cosine Annealing with Warmup:
    lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    
    where progress = (t - warmup_steps) / (total_steps - warmup_steps)
    and during warmup: lr(t) = max_lr * t / warmup_steps

This schedule is used by nearly all modern LLMs:
    - GPT-3: cosine decay with 375M warmup steps
    - LLaMA-2: cosine decay with 2000 warmup steps
    - Chinchilla: cosine decay with 2% warmup
    - T5: inverse square root decay
    - PaLM: WSD (Warmup-Stable-Decay)
    - ViT: cosine with warm restarts (SGDR)
    - BERT: linear warmup + linear decay
    - ResNet: OneCycleLR for super-convergence
"""

from __future__ import annotations
import math
from typing import List, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR, Optimizer


class CosineAnnealingWithWarmup(LambdaLR):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    The schedule has two phases:
        1. Warmup (0 to warmup_steps): linear increase from 0 to max_lr
        2. Cosine decay (warmup_steps to total_steps): cosine decay to min_lr
    
    Args:
        optimizer: The optimizer to adjust learning rate for.
        warmup_steps: Number of warmup steps (linear increase).
        total_steps: Total training steps.
        min_lr_ratio: Ratio of min_lr to initial lr (default: 0.1).
        num_cycles: Number of cosine cycles (default: 0.5 for half-cycle).
    
    Example:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_steps=2000, total_steps=5_000_000
        )
        for step in range(5_000_000):
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles

        def lr_lambda(current_step: int) -> float:
            # Phase 1: Linear warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # Phase 2: Cosine decay
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            
            # Cosine decay formula
            cosine_decay = 0.5 * (
                1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress)
            )
            
            # Scale between min_lr_ratio and 1.0
            return max(self.min_lr_ratio, cosine_decay * (1.0 - self.min_lr_ratio) + self.min_lr_ratio)

        super().__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


class LinearWarmupWithDecay(LambdaLR):
    """
    Linear warmup followed by linear decay.
    
    Used in some early GPT models.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        end_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_lr_ratio = end_lr_ratio

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            decay_progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return 1.0 - (1.0 - self.end_lr_ratio) * decay_progress

        super().__init__(optimizer, lr_lambda, last_epoch)


class WarmupStableDecay(LambdaLR):
    """
    Warmup, then stable lr, then decay.
    
    Three-phase schedule:
        1. Warmup: linear increase to max_lr
        2. Stable: constant max_lr
        3. Decay: cosine or linear decay to min_lr
    
    Used in PaLM and some other models that need a long stable period.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        stable_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        decay_type: str = "cosine",
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.decay_type = decay_type

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Phase 1: Warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            elif current_step < self.warmup_steps + self.stable_steps:
                # Phase 2: Stable
                return 1.0
            else:
                # Phase 3: Decay
                decay_start = self.warmup_steps + self.stable_steps
                progress = float(current_step - decay_start) / float(
                    max(1, self.total_steps - decay_start)
                )
                
                if self.decay_type == "cosine":
                    return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                        1.0 + math.cos(math.pi * progress)
                    )
                else:  # linear
                    return 1.0 - (1.0 - self.min_lr_ratio) * progress

        super().__init__(optimizer, lr_lambda, last_epoch)


class CosineAnnealingWithWarmRestarts(LambdaLR):
    """
    Cosine annealing with periodic warm restarts (SGDR).
    
    Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2017).
    
    The learning rate follows a cosine curve within each cycle. At the end of each
    cycle, the learning rate "restarts" to the maximum value. Optionally, the
    maximum learning rate at each restart decays by a multiplicative factor.
    
    Formula within cycle i:
        lr(t) = eta_min + 0.5 * (eta_max_i - eta_min) * (1 + cos(pi * T_cur / T_i))
    
    where:
        T_cur = current step within the cycle (0 to T_i)
        T_i = length of cycle i = T_0 * (gamma^i)
        eta_max_i = eta_max * (gamma^i) if decay_restart_lr else eta_max
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps before first cosine cycle.
        T_0: Number of steps in the first cycle.
        T_mult: Factor to multiply cycle length after each restart (default: 2.0).
            If 1.0, all cycles have the same length.
        gamma: Multiplicative decay factor for max LR after each restart (default: 1.0).
            If < 1.0, each restart starts at a lower LR.
        eta_min: Minimum learning rate (default: 0).
        decay_restart_lr: Whether to decay the restart LR by gamma (default: False).
            If True, the peak LR at each restart is eta_max * gamma^cycle_num.
    
    References:
        - SGDR: Stochastic Gradient Descent with Warm Restarts (ICLR 2017)
        - Used in Vision Transformers (ViT) training schedules
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        T_0: int = 50000,
        T_mult: int = 2,
        gamma: float = 1.0,
        eta_min_ratio: float = 0.0,
        decay_restart_lr: bool = False,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.gamma = gamma
        self.eta_min_ratio = eta_min_ratio
        self.decay_restart_lr = decay_restart_lr

        def lr_lambda(current_step: int) -> float:
            # Phase 0: Linear warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))

            # Adjusted step after warmup
            step = current_step - self.warmup_steps

            # Determine which cycle we're in and how far into it
            T_i = self.T_0
            cycle_num = 0
            while step >= T_i:
                step -= T_i
                cycle_num += 1
                T_i = int(T_i * self.T_mult) if self.T_mult > 1 else T_i

            # Compute cosine within current cycle
            progress = step / max(1, T_i)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            # Apply restart LR decay
            if self.decay_restart_lr:
                restart_scale = self.gamma ** cycle_num
            else:
                restart_scale = 1.0

            return self.eta_min_ratio + (1.0 - self.eta_min_ratio) * restart_scale * cosine_decay

        super().__init__(optimizer, lr_lambda, last_epoch)


class InverseSquareRootDecay(LambdaLR):
    """
    Inverse square root learning rate schedule with linear warmup.
    
    Used in the original Transformer paper (Vaswani et al., 2017) and T5.
    
    Formula:
        During warmup:  lr(t) = max_lr * t / warmup_steps
        After warmup:   lr(t) = max_lr * sqrt(warmup_steps) / sqrt(t)
    
    The key insight is that after warmup, the LR decays as 1/sqrt(t), which
    provides a smooth transition from the warmup phase. At t = warmup_steps,
    both formulas give lr = max_lr, ensuring continuity.
    
    This schedule is particularly effective for transformer-based models because:
        1. The warmup phase prevents early training instability
        2. The 1/sqrt(t) decay is gentler than linear, allowing longer effective training
        3. It naturally balances exploration (early) and exploitation (late)
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps (linear increase).
        total_steps: Total training steps (used for min_lr computation).
        min_lr_ratio: Ratio of min_lr to initial lr (default: 0.0).
        last_epoch: Last epoch index (default: -1).
    
    References:
        - "Attention Is All You Need" (Vaswani et al., 2017)
        - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 4000,
        total_steps: int = 1000000,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Linear warmup: lr ramps from 0 to max_lr
                return float(current_step) / float(max(1, self.warmup_steps))

            # Inverse square root decay: lr = lr_init * sqrt(d_model) / sqrt(t)
            # Equivalent to: lr(t) = lr_init * sqrt(warmup_steps / t)
            decay_factor = math.sqrt(self.warmup_steps) / math.sqrt(float(current_step))
            return max(self.min_lr_ratio, decay_factor)

        super().__init__(optimizer, lr_lambda, last_epoch)


class OneCycleLR(LambdaLR):
    """
    One Cycle learning rate policy for super-convergence.
    
    Proposed by Leslie Smith (2018) as a way to achieve "super-convergence" —
    reaching the same or better accuracy in fewer training iterations.
    
    The policy has three distinct phases:
        1. Warmup (0 to pct_start * total_steps):
           LR increases from initial_lr to max_lr following a smooth curve.
        
        2. Annealing (pct_start * total_steps to pct_warmup * total_steps):
           LR decreases from max_lr to final_lr following a cosine curve.
        
        3. Final annealing (last few percent):
           LR further decreases to min_lr.
    
    The total learning rate budget follows a smooth, single cycle that goes
    up then down, inspired by the observation that large learning rates can
    regularize training and help escape sharp minima.
    
    Weight decay can also be cycled (increased during the annealing phase).
    
    Args:
        optimizer: The optimizer.
        max_lr: Maximum learning rate (peak of the cycle).
        total_steps: Total number of training steps.
        pct_start: Percentage of total steps for the warmup phase (default: 0.3).
        pct_warmup: Deprecated, use pct_start. Fraction of cycle spent in warmup.
        anneal_strategy: Strategy for annealing: "cos" (cosine) or "linear" (default: "cos").
        div_factor: Initial LR = max_lr / div_factor (default: 25.0).
        final_div_factor: Final LR = max_lr / final_div_factor (default: 1e4).
        warmup_steps: If provided, overrides pct_start for the warmup phase.
    
    References:
        - "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (Smith & Topin, 2018)
        - "A disciplined approach to neural network hyper-parameters" (Smith, 2017)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float = 1e-3,
        total_steps: int = 1000000,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase

        def lr_lambda(current_step: int) -> float:
            if current_step >= self.total_steps:
                return 1.0 / self.final_div_factor

            # Phase boundaries
            warmup_steps = int(self.total_steps * self.pct_start)

            if current_step < warmup_steps:
                # Phase 1: Warmup from initial_lr to max_lr
                progress = current_step / max(1, warmup_steps)
                if self.anneal_strategy == "cos":
                    return (1.0 / self.div_factor) + (1.0 - 1.0 / self.div_factor) * (
                        0.5 * (1.0 - math.cos(math.pi * progress))
                    )
                else:
                    # Linear warmup
                    return 1.0 / self.div_factor + (1.0 - 1.0 / self.div_factor) * progress

            elif self.three_phase and current_step < 2 * warmup_steps:
                # Optional Phase 2: Stay near max_lr
                progress = (current_step - warmup_steps) / max(1, warmup_steps)
                if self.anneal_strategy == "cos":
                    return 1.0 - 0.5 * (1.0 - math.cos(math.pi * progress))
                else:
                    return 1.0 - progress * 0.5

            else:
                # Phase 3: Anneal from max_lr to final_lr
                if self.three_phase:
                    anneal_start = 2 * warmup_steps
                else:
                    anneal_start = warmup_steps

                progress = (current_step - anneal_start) / max(
                    1, self.total_steps - anneal_start
                )

                if self.anneal_strategy == "cos":
                    return (
                        1.0 / self.final_div_factor
                        + 0.5 * (1.0 - 1.0 / self.final_div_factor)
                        * (1.0 + math.cos(math.pi * progress))
                    )
                else:
                    # Linear annealing
                    return (
                        1.0 / self.final_div_factor
                        + (1.0 - 1.0 / self.final_div_factor) * (1.0 - progress)
                    )

        super().__init__(optimizer, lr_lambda, last_epoch)


class PolynomialDecay(LambdaLR):
    """
    Polynomial learning rate decay with linear warmup.
    
    The LR decays as a polynomial function of training progress:
        lr(t) = min_lr + (max_lr - min_lr) * (1 - progress)^power
    
    where progress = (t - warmup_steps) / (total_steps - warmup_steps)
    
    Special cases:
        power = 1.0: Linear decay (equivalent to LinearWarmupWithDecay)
        power = 0.5: Square root decay (slower initial decay)
        power = 2.0: Quadratic decay (faster initial decay, slower later)
    
    Polynomial decay is commonly used in:
        - BERT: linear warmup + polynomial decay
        - GPT-2: cosine (special case of polynomial)
        - T5: inverse square root (not polynomial, but related)
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        power: Exponent for the polynomial decay (default: 1.0).
        min_lr_ratio: Ratio of minimum LR to initial LR (default: 0.0).
        cycle: If True, restart the schedule from the beginning when total_steps
            is reached (default: False).
        last_epoch: Last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 0,
        total_steps: int = 1000000,
        power: float = 1.0,
        min_lr_ratio: float = 0.0,
        cycle: bool = False,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        self.cycle = cycle

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))

            if self.cycle:
                # Wrap around for cycling
                progress = float(current_step - self.warmup_steps) % float(
                    max(1, self.total_steps - self.warmup_steps)
                ) / float(max(1, self.total_steps - self.warmup_steps))
            else:
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.total_steps - self.warmup_steps)
                )

            # Polynomial decay
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * max(
                0.0, 1.0 - progress
            ) ** self.power

        super().__init__(optimizer, lr_lambda, last_epoch)


class MultiPhaseSchedule(LambdaLR):
    """
    Custom multi-phase learning rate schedule builder.
    
    Allows composing arbitrary sequences of warmup, stable, and decay phases
    with different decay strategies per phase. This is the most flexible
    scheduler in the Nexus toolkit.
    
    Each phase is defined as a tuple:
        (name, num_steps, **kwargs)
    
    Supported phase types:
        - "linear_warmup": Linear warmup from 0 to 1.0 (multiplied by base LR).
        - "stable": Constant LR at current level.
        - "cosine_decay": Cosine decay to min_lr_ratio.
        - "linear_decay": Linear decay to end_lr_ratio.
        - "inverse_sqrt": Inverse square root decay.
        - "polynomial_decay": Polynomial decay with given power.
        - "exponential_decay": Exponential decay with given rate.
        - "step_decay": Multiply LR by factor at each step boundary.
        - "plateau": Constant LR (same as "stable" but explicit).
    
    Args:
        optimizer: The optimizer.
        phases: List of (phase_type, num_steps, **kwargs) tuples.
        min_lr: Absolute minimum learning rate floor (default: 1e-7).
        last_epoch: Last epoch (default: -1).
    
    Example:
        # Replicate WSD: warmup(2000) + stable(100000) + cosine_decay(898000)
        schedule = MultiPhaseSchedule(optimizer, [
            ("linear_warmup", 2000),
            ("stable", 100000),
            ("cosine_decay", 898000, {"min_lr_ratio": 0.1}),
        ])
        
        # Custom: warmup + polynomial + cosine + linear
        schedule = MultiPhaseSchedule(optimizer, [
            ("linear_warmup", 1000),
            ("polynomial_decay", 50000, {"power": 0.5}),
            ("stable", 200000),
            ("cosine_decay", 500000, {"min_lr_ratio": 0.01}),
            ("linear_decay", 249000, {"end_lr_ratio": 0.0}),
        ])
    
    References:
        - Inspired by PyTorch's LRScheduler design philosophy
        - Supports reproducing any published LLM training schedule
    """

    def __init__(
        self,
        optimizer: Optimizer,
        phases: list,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.phases = phases
        self.min_lr = min_lr

        # Pre-compute phase boundaries
        self.phase_boundaries = []
        cumulative = 0
        for phase_type, num_steps, *rest in phases:
            kwargs = rest[0] if rest else {}
            self.phase_boundaries.append({
                "type": phase_type,
                "start": cumulative,
                "end": cumulative + num_steps,
                "steps": num_steps,
                "kwargs": kwargs,
            })
            cumulative += num_steps

        self.total_steps = cumulative

        def lr_lambda(current_step: int) -> float:
            # Find current phase
            phase = None
            for p in self.phase_boundaries:
                if p["start"] <= current_step < p["end"]:
                    phase = p
                    break

            if phase is None:
                # Past all phases — return minimum
                return self.min_lr / max(1e-7, 1.0)  # Will be multiplied by base_lr

            phase_type = phase["type"]
            kwargs = phase["kwargs"]
            steps_in_phase = phase["steps"]
            step_in_phase = current_step - phase["start"]
            progress = float(step_in_phase) / max(1, steps_in_phase)

            if phase_type == "linear_warmup":
                # Linear increase from 0 to 1.0
                return progress

            elif phase_type == "stable" or phase_type == "plateau":
                # Constant at 1.0
                return 1.0

            elif phase_type == "cosine_decay":
                # Cosine decay
                min_ratio = kwargs.get("min_lr_ratio", 0.0)
                return min_ratio + (1.0 - min_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

            elif phase_type == "linear_decay":
                # Linear decay
                end_ratio = kwargs.get("end_lr_ratio", 0.0)
                return 1.0 - (1.0 - end_ratio) * progress

            elif phase_type == "inverse_sqrt":
                # Inverse square root: starts at 1.0, decays as 1/sqrt(1 + progress * steps)
                decay_steps = kwargs.get("decay_steps", steps_in_phase)
                return 1.0 / math.sqrt(1.0 + progress * (decay_steps / max(1, steps_in_phase)))

            elif phase_type == "polynomial_decay":
                power = kwargs.get("power", 1.0)
                end_ratio = kwargs.get("end_lr_ratio", 0.0)
                return end_ratio + (1.0 - end_ratio) * max(0.0, 1.0 - progress) ** power

            elif phase_type == "exponential_decay":
                rate = kwargs.get("rate", 0.1)
                return math.exp(-rate * progress)

            elif phase_type == "step_decay":
                factor = kwargs.get("factor", 0.5)
                step_size = kwargs.get("step_size", steps_in_phase // max(1, int(1.0 / max(0.01, factor))))
                num_drops = int(step_in_phase // max(1, step_size))
                return factor ** num_drops

            else:
                raise ValueError(f"Unknown phase type: {phase_type}")

        super().__init__(optimizer, lr_lambda, last_epoch)

    def get_phase_info(self, current_step: int) -> dict:
        """Return information about the current phase."""
        for p in self.phase_boundaries:
            if p["start"] <= current_step < p["end"]:
                return {
                    "phase_type": p["type"],
                    "phase_start": p["start"],
                    "phase_end": p["end"],
                    "step_in_phase": current_step - p["start"],
                    "phase_progress": (current_step - p["start"]) / max(1, p["steps"]),
                    "phase_kwargs": p["kwargs"],
                }
        return {"phase_type": "completed", "total_steps": self.total_steps}


class ConstantWithWarmup(LambdaLR):
    """
    Constant learning rate with linear warmup phase.
    
    Simplest useful schedule: warm up to a constant LR and maintain it
    for the entire training run.
    
    Formula:
        lr(t) = max_lr * min(1.0, t / warmup_steps)
    
    This is useful when:
        - You want to isolate the effect of the optimizer from the schedule
        - Fine-tuning where the LR is already well-tuned
        - Testing and debugging
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of linear warmup steps.
        last_epoch: Last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0

        super().__init__(optimizer, lr_lambda, last_epoch)


class ExponentialDecayWithWarmup(LambdaLR):
    """
    Exponential learning rate decay with linear warmup.
    
    Formula:
        During warmup:  lr(t) = max_lr * t / warmup_steps
        After warmup:   lr(t) = max_lr * gamma^(t - warmup_steps)
    
    Exponential decay reduces LR by a constant factor gamma every step.
    This gives very fast initial decay that slows over time.
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        gamma: Decay factor per step (default: 0.9999).
            After N steps: lr = initial_lr * gamma^N
            After 10000 steps with gamma=0.9999: lr = initial_lr * 0.368
        min_lr: Floor for the learning rate (default: 1e-7).
        last_epoch: Last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        gamma: float = 0.9999,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.min_lr = min_lr

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            # Exponential decay
            return max(
                self.min_lr / max(1e-7, 1.0),
                self.gamma ** (current_step - self.warmup_steps)
            )

        super().__init__(optimizer, lr_lambda, last_epoch)


class WarmupHoldDecay(LambdaLR):
    """
    Warmup-Hold-Decay (WHD) schedule with fine-grained control.
    
    Three-phase schedule with smooth transitions:
        1. Warmup (0 to warmup_steps): smooth increase using cosine ramp
        2. Hold (warmup_steps to hold_steps): stable at max_lr
        3. Decay (hold_steps to total_steps): smooth decay using cosine or linear
    
    Unlike WarmupStableDecay, this uses smooth cosine transitions between
    phases (no abrupt changes).
    
    Formula:
        Warmup:  lr(t) = 0.5 * max_lr * (1 - cos(pi * t / warmup_steps))
        Hold:    lr(t) = max_lr
        Decay:   lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        hold_steps: Total steps for warmup + hold phase.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of initial (default: 0.0).
        decay_type: "cosine" or "linear" (default: "cosine").
        last_epoch: Last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        hold_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        decay_type: str = "cosine",
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.decay_type = decay_type

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Smooth cosine warmup (no abrupt start)
                progress = float(current_step) / float(max(1, self.warmup_steps))
                return 0.5 * (1.0 - math.cos(math.pi * progress))

            elif current_step < self.hold_steps:
                # Hold at maximum
                return 1.0

            else:
                # Decay phase
                decay_start = self.hold_steps
                progress = float(current_step - decay_start) / float(
                    max(1, self.total_steps - decay_start)
                )
                progress = min(1.0, progress)

                if self.decay_type == "cosine":
                    return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                        1.0 + math.cos(math.pi * progress)
                    )
                else:
                    return 1.0 - (1.0 - self.min_lr_ratio) * progress

        super().__init__(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    warmup_steps: int = 0,
    total_steps: int = 1000000,
    **kwargs,
) -> LambdaLR:
    """
    Factory function for learning rate schedulers.
    
    Args:
        name: Scheduler name. Supported values:
            "cosine"              - Cosine annealing with warmup (GPT-3, LLaMA)
            "linear"              - Linear warmup + linear decay (early GPT, BERT)
            "warmup_stable_decay" - WSD: warmup + stable + decay (PaLM)
            "cosine_warm_restarts"- SGDR: cosine with periodic restarts (ViT)
            "inv_sqrt"            - Inverse square root decay (Transformer, T5)
            "onecycle"            - One Cycle policy for super-convergence (ResNet)
            "polynomial"          - Polynomial decay with configurable power
            "multi_phase"         - Custom multi-phase schedule builder
            "constant"            - Constant LR with warmup
            "exponential"         - Exponential decay with warmup
            "warmup_hold_decay"   - WHD: smooth warmup + hold + decay
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        **kwargs: Additional scheduler-specific arguments.
    
    Returns:
        Learning rate scheduler instance.
    """
    if name == "cosine":
        return CosineAnnealingWithWarmup(
            optimizer, warmup_steps, total_steps,
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.1),
        )
    elif name == "linear":
        return LinearWarmupWithDecay(
            optimizer, warmup_steps, total_steps,
            end_lr_ratio=kwargs.get("end_lr_ratio", 0.0),
        )
    elif name == "warmup_stable_decay":
        return WarmupStableDecay(
            optimizer, warmup_steps, total_steps,
            stable_steps=kwargs.get("stable_steps", 0),
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.1),
            decay_type=kwargs.get("decay_type", "cosine"),
        )
    elif name == "cosine_warm_restarts":
        return CosineAnnealingWithWarmRestarts(
            optimizer,
            warmup_steps=warmup_steps,
            T_0=kwargs.get("T_0", 50000),
            T_mult=kwargs.get("T_mult", 2),
            gamma=kwargs.get("gamma", 1.0),
            eta_min_ratio=kwargs.get("eta_min_ratio", 0.0),
            decay_restart_lr=kwargs.get("decay_restart_lr", False),
        )
    elif name == "inv_sqrt":
        return InverseSquareRootDecay(
            optimizer, warmup_steps, total_steps,
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
        )
    elif name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-3),
            total_steps=total_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4),
            three_phase=kwargs.get("three_phase", False),
        )
    elif name == "polynomial":
        return PolynomialDecay(
            optimizer, warmup_steps, total_steps,
            power=kwargs.get("power", 1.0),
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
            cycle=kwargs.get("cycle", False),
        )
    elif name == "multi_phase":
        if "phases" not in kwargs:
            raise ValueError("multi_phase scheduler requires 'phases' kwarg")
        return MultiPhaseSchedule(
            optimizer,
            phases=kwargs["phases"],
            min_lr=kwargs.get("min_lr", 1e-7),
        )
    elif name == "constant":
        return ConstantWithWarmup(
            optimizer, warmup_steps=warmup_steps,
        )
    elif name == "exponential":
        return ExponentialDecayWithWarmup(
            optimizer, warmup_steps,
            gamma=kwargs.get("gamma", 0.9999),
            min_lr=kwargs.get("min_lr", 1e-7),
        )
    elif name == "warmup_hold_decay":
        return WarmupHoldDecay(
            optimizer, warmup_steps,
            hold_steps=kwargs.get("hold_steps", total_steps // 2),
            total_steps=total_steps,
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
            decay_type=kwargs.get("decay_type", "cosine"),
        )
    else:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Supported: cosine, linear, warmup_stable_decay, cosine_warm_restarts, "
            f"inv_sqrt, onecycle, polynomial, multi_phase, constant, exponential, "
            f"warmup_hold_decay"
        )
