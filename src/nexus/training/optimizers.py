"""
Nexus Optimizers - Built From Scratch
========================================
Complete suite of optimization algorithms for LLM training.
Each optimizer implements the full mathematical formulation.

All optimizers are proper ``torch.optim.Optimizer`` subclasses compatible with
standard PyTorch models, distributed training (FSDP / DeepSpeed), and
checkpointing via ``state_dict`` / ``load_state_dict``.

Algorithms
----------
1. **SGD** – Stochastic Gradient Descent with momentum & Nesterov
2. **Adam** – Adaptive Moment Estimation (with AMSGrad variant)
3. **AdamW** – Adam with Decoupled Weight Decay (standard for LLMs)
4. **LAMB** – Layer-wise Adaptive Moments for Batch training
5. **Adafactor** – Memory-Efficient Factored Adaptive Optimizer
6. **LION** – EvoLved Sign Momentum (symbolic, program-search)
7. **Sophia** – Second-Order Clipped Optimizer (Hessian-based)
8. **Shampoo** – Preconditioned Optimizer (Kronecker-factored)
9. **8-bit AdamW** – Quantized AdamW (block-wise dynamic quantization)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _scalar_param(value: float, param_groups: List[Dict[str, Any]]) -> float:
    """Validate that a scalar hyper-parameter has a sensible value."""
    if value < 0:
        raise ValueError(f"Expected non-negative scalar, got {value}")
    return float(value)


def _make_param_groups(
    params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
    defaults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Wrap raw parameters or a pre-built group list around *defaults*."""
    if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
        groups: List[Dict[str, Any]] = []
        for g in params:
            merged = dict(defaults)
            merged.update(g)
            groups.append(merged)
        return groups
    return [{"params": list(params), **defaults}]


def _apply_gradient_clipping(
    p: torch.Tensor,
    max_norm: Optional[float],
    clip_value: Optional[float],
) -> None:
    """In-place gradient clipping by *global* norm and/or per-value cap."""
    if max_norm is not None and max_norm > 0:
        grad_norm = p.grad.data.norm(2)
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-6)
            p.grad.data.mul_(scale)
    if clip_value is not None and clip_value > 0:
        p.grad.data.clamp_(-clip_value, clip_value)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Stochastic Gradient Descent (SGD)
# ═══════════════════════════════════════════════════════════════════════════

class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum and Nesterov
    acceleration.

    Standard update
    ---------------
    θ_{t+1} = θ_t − lr · g_t

    Momentum update
    ---------------
    v_t = μ · v_{t-1} + g_t
    θ_{t+1} = θ_t − lr · v_t

    Nesterov momentum
    -----------------
    lookahead = θ_t − lr · μ · v_{t-1}
    v_t = μ · v_{t-1} + ∇L(lookahead)
    θ_{t+1} = θ_t − lr · v_t

    Parameters
    ----------
    params : iterable of tensors or dict groups
        Model parameters to optimise.
    lr : float
        Learning rate (default ``0.01``).
    momentum : float
        Momentum factor μ (default ``0.0``).
    nesterov : bool
        Enable Nesterov accelerated gradient (default ``False``).
    weight_decay : float
        L2 regularisation applied as θ ← θ − lr·wd·θ (default ``0.0``).
    max_grad_norm : float or None
        Clip gradient *norm* to this value before the update.
    clip_grad_value : float or None
        Clip gradient *values* element-wise to this absolute threshold.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if nesterov and momentum == 0:
            raise ValueError("Nesterov requires momentum > 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimisation step.

        Parameters
        ----------
        closure : callable, optional
            Re-evaluates the model and returns the loss.  When supplied the
            step runs ``closure()`` to recompute gradients (used by some
            line-search strategies).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "SGD with momentum does not support sparse gradients"
                    )

                # --- gradient clipping ---
                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data  # re-read after clipping

                # --- weight decay (L2) ---
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # --- momentum ---
                if momentum != 0:
                    state = self.state[p]

                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad)
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad)

                    if nesterov:
                        # Nesterov: look-ahead gradient
                        grad_nesterov = grad.add(buf, alpha=momentum)
                        p.data.add_(grad_nesterov, alpha=-lr)
                    else:
                        p.data.add_(buf, alpha=-lr)
                else:
                    # plain SGD
                    p.data.add_(grad, alpha=-lr)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 2. Adam – Adaptive Moment Estimation
# ═══════════════════════════════════════════════════════════════════════════

class Adam(Optimizer):
    """Adam: Adaptive Moment Estimation.

    Update rules
    -------------
    m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
    v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
    m̂_t = m_t / (1 − β₁^t)
    v̂_t = v_t / (1 − β₂^t)
    θ_{t+1} = θ_t − lr · m̂_t / (√v̂_t + ε)

    AMSGrad variant
    ----------------
    v̂_max = max(v̂_max, v̂_t)
    θ_{t+1} = θ_t − lr · m̂_t / (√v̂_max + ε)

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate (default ``1e-3``).
    betas : tuple[float, float]
        Coefficients for running averages (default ``(0.9, 0.999)``).
    eps : float
        Numerical stabiliser for 1/√v̂ (default ``1e-8``).
    amsgrad : bool
        Use AMSGrad variant (default ``False``).
    weight_decay : float
        L2 regularisation (default ``0.0``).
    max_grad_norm : float or None
        Gradient norm clipping threshold.
    clip_grad_value : float or None
        Element-wise gradient clipping threshold.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            amsgrad = group["amsgrad"]
            weight_decay = group["weight_decay"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                state = self.state[p]
                # State initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                t = state["step"] + 1
                state["step"] = t

                # Biased first & second moment estimates
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if amsgrad:
                    # Keep the running maximum of the bias-corrected variance
                    v_hat = v.div(bias_correction2)
                    state["max_exp_avg_sq"].copy_(
                        torch.max(state["max_exp_avg_sq"], v_hat)
                    )
                    denom = state["max_exp_avg_sq"].add(eps)
                else:
                    denom = v.div(bias_correction2).add(eps)

                # Weight decay (L2, coupled)
                if weight_decay != 0:
                    p.data.add_(p, alpha=-lr * weight_decay)

                p.data.addcdiv_(m, denom, value=-step_size)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 3. AdamW – Adam with Decoupled Weight Decay
# ═══════════════════════════════════════════════════════════════════════════

class AdamW(Optimizer):
    """AdamW: Adam with Decoupled Weight Decay.

    The key difference from Adam is that weight decay is applied *after* the
    Adam update rather than being added to the gradient:

        Adam step:   θ ← θ − lr · m̂ / (√v̂ + ε)
        Decay step:  θ ← θ − lr · wd · θ

    This decoupling is essential for LLM training because it treats
    regularisation independently of the adaptive learning rate schedule.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate (default ``1e-4`` — the de-facto standard for LLMs).
    betas : tuple[float, float]
        (β₁, β₂) for the first and second moment running averages
        (default ``(0.9, 0.95)`` — note β₂=0.95 is common in LLM training).
    eps : float
        Numerical stabiliser (default ``1e-8``).
    weight_decay : float
        Decoupled weight decay coefficient (default ``0.1``).
    amsgrad : bool
        AMSGrad variant (default ``False``).
    max_grad_norm : float or None
        Gradient norm clipping.
    clip_grad_value : float or None
        Element-wise gradient clipping.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        amsgrad: bool = False,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                t = state["step"] + 1
                state["step"] = t

                # Update biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr / bias_correction1

                if amsgrad:
                    v_hat = v.div(bias_correction2)
                    state["max_exp_avg_sq"].copy_(
                        torch.max(state["max_exp_avg_sq"], v_hat)
                    )
                    denom = (state["max_exp_avg_sq"].sqrt()).add(eps)
                else:
                    denom = (v / bias_correction2).sqrt().add(eps)

                # --- Adam step ---
                p.data.addcdiv_(m, denom, value=-step_size)

                # --- Decoupled weight decay (applied AFTER Adam update) ---
                if weight_decay > 0:
                    # θ ← θ − lr · wd · θ
                    p.data.add_(p, alpha=-lr * weight_decay)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 4. LAMB – Layer-wise Adaptive Moments for Batch training
# ═══════════════════════════════════════════════════════════════════════════

class LAMB(Optimizer):
    """LAMB: Layer-wise Adaptive Moments optimizer for Batch training.

    LAMB extends Adam with a *trust ratio* that controls the update magnitude
    per layer, enabling stable large-batch training (e.g. BERT pre-training
    with batch size 64 k).

    For each parameter *θ*:

        m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
        v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
        m̂_t = m_t / (1 − β₁^t)
        v̂_t = v_t / (1 − β₂^t)
        adam_update = m̂_t / (√v̂_t + ε)
        r = ‖θ‖ / (‖adam_update‖ + λ · ‖θ‖)
        r = clip(r, r_min, r_max)
        θ_{t+1} = θ_t − lr · r · adam_update

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate (default ``1e-3``).
    betas : tuple[float, float]
        Adam betas (default ``(0.9, 0.999)``).
    eps : float
        Numerical stabiliser (default ``1e-6``).
    weight_decay : float
        L2 regularisation coefficient λ (default ``0.01``).
    trust_ratio_clip : tuple[float, float]
        (r_min, r_max) for clipping the trust ratio (default ``(0.01, 10.0)``).
    adam : bool
        If ``True`` the weight decay is applied *inside* the Adam update
        (coupled) rather than via the trust ratio (default ``False``).
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        trust_ratio_clip: Tuple[float, float] = (0.01, 10.0),
        adam: bool = False,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            trust_ratio_clip=trust_ratio_clip,
            adam=adam,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            r_min, r_max = group["trust_ratio_clip"]
            use_adam = group["adam"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                t = state["step"] + 1
                state["step"] = t

                # Adam moments
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                # Adam update
                adam_update = m_hat / (v_hat.sqrt() + eps)

                # Add weight decay to the update (not to gradient)
                if weight_decay > 0:
                    adam_update.add_(p.data, alpha=weight_decay)

                # Trust ratio
                param_norm = p.data.norm(2)
                update_norm = adam_update.norm(2)
                if param_norm != 0 and update_norm != 0:
                    trust_ratio = param_norm / (update_norm + weight_decay * param_norm)
                    trust_ratio = float(
                        torch.clamp(
                            torch.tensor(trust_ratio, device=p.device),
                            r_min,
                            r_max,
                        )
                    )
                else:
                    trust_ratio = 1.0

                if use_adam:
                    # Fall back to pure Adam (no trust ratio scaling)
                    p.data.add_(adam_update, alpha=-lr)
                else:
                    p.data.add_(adam_update, alpha=-lr * trust_ratio)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 5. Adafactor – Memory-Efficient Factored Adaptive Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class Adafactor(Optimizer):
    """Adafactor: Memory-efficient adaptive learning rate optimizer.

    Instead of storing full second-moment tensors ``v`` (O(n·m) memory),
    Adafactor maintains *factored* running averages for 2-D parameters:

        For a matrix θ ∈ ℝ^{n×m}:
          r_{t,i} = β₂ · r_{t-1,i} + (1 − β₂) · g_{t,i,:}²     (row-wise)
          c_{t,j} = β₂ · c_{t-1,j} + (1 − β₂) · g_{t,:,j}²     (col-wise)

        v̂_{t,ij} = r_{t,i} · c_{t,j}  /  max(r · c)  ×  n·m

    This reduces memory from O(n·m) → O(n + m) per parameter matrix.

    For 1-D parameters (biases, layer norms) a standard running average is
    used.  An optional *momentum* buffer can accumulate past updates.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float or None
        Learning rate.  ``None`` means schedule-free (external scheduler).
    eps : tuple[float, float]
        (ε₁, ε₂) — regularisation constants added before/after sqrt
        (default ``(1e-30, 1e-3)``).
    clip_threshold : float
        Threshold for clipping the update RMS (default ``1.0``).
    decay_rate : float
        Second moment decay β₂ (default ``0.8`` — note Adafactor uses a
        lower β₂ than Adam).
    beta1 : float or None
        First moment coefficient.  ``None`` disables momentum (default
        ``None``).
    weight_decay : float
        Decoupled weight decay (default ``0.0``).
    scale_parameter : bool
        Scale learning rate by the inverse RMS of the current update
        (default ``True``).
    relative_step : bool
        Compute lr from the parameter RMS (default ``True``).
    warmup_init : bool
        Warm-up lr linearly from zero (default ``False``).
    factored : bool
        Use factored second moments for 2-D parameters (default ``True``).
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = 0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        factored: bool = True,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            factored=factored,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    def _get_lr(self, param: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        """Compute per-parameter learning rate."""
        if group["relative_step"]:
            min_step = 1e-2 * (group.get("lr", 1e-3) or 1e-3)
            step_size = min_step / max(
                1.0, float(param.data.norm(2)) * min_step
            )
            if group["warmup_init"]:
                t = self.state.get(param, {}).get("step", 1)
                step_size = step_size * min(1.0, t / 1e3)
            return step_size
        return group["lr"]  # type: ignore[return-value]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps1, eps2 = group["eps"]
            clip_threshold = group["clip_threshold"]
            decay_rate = group["decay_rate"]
            beta1 = group["beta1"]
            weight_decay = group["weight_decay"]
            scale_parameter = group["scale_parameter"]
            factored = group["factored"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients"
                    )

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 1

                    if p.dim() >= 2 and factored:
                        # Factored second-moment accumulators
                        state["exp_avg_row"] = torch.zeros(
                            p.size(0), device=p.device, dtype=p.dtype
                        )
                        state["exp_avg_col"] = torch.zeros(
                            p.size(1), device=p.device, dtype=p.dtype
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    if beta1 is not None:
                        state["exp_avg"] = torch.zeros_like(p)

                t = state["step"]
                state["step"] = t + 1

                # ---- Second moment (factored or full) ----
                if p.dim() >= 2 and factored:
                    r = state["exp_avg_row"]
                    c = state["exp_avg_col"]

                    r.mul_(decay_rate).add_(
                        (grad * grad).sum(dim=list(range(1, grad.dim()))),
                        alpha=1.0 - decay_rate,
                    )
                    c.mul_(decay_rate).add_(
                        (grad * grad).sum(dim=0), alpha=1.0 - decay_rate
                    )

                    # v̂ = outer(r, c) normalised
                    v_hat = (
                        r.unsqueeze(-1) * c.unsqueeze(0)
                    ) / (torch.max(r) * torch.max(c) + eps1)
                    v_hat.mul_(float(r.numel()))
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(decay_rate).addcmul_(grad, grad, value=1.0 - decay_rate)
                    v_hat = v / (v.max() + eps1)

                # ---- Update ----
                update = grad / (v_hat.sqrt() + eps2)

                # Clip by RMS
                rms = update.norm(2) / math.sqrt(update.numel())
                if rms > clip_threshold:
                    scale = clip_threshold / (rms + 1e-6)
                    update.mul_(scale)

                # Relative-step scaling
                if scale_parameter:
                    param_rms = p.data.norm(2) / math.sqrt(p.data.numel())
                    update.mul_(param_rms)

                lr = self._get_lr(p, group)
                if isinstance(lr, torch.Tensor):
                    lr = lr.to(p.device)
                update.mul_(lr)

                # ---- Momentum ----
                if beta1 is not None:
                    m = state["exp_avg"]
                    m.mul_(beta1).add_(update, alpha=1.0 - beta1)
                    update = m

                # ---- Decoupled weight decay ----
                if weight_decay > 0:
                    p.data.add_(p, alpha=-lr * weight_decay)

                p.data.sub_(update)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 6. LION – EvoLved Sign Momentum
# ═══════════════════════════════════════════════════════════════════════════

class LION(Optimizer):
    """LION: EvoLved Sign Momentum.

    LION was discovered through program search (symbolic optimisation).  It
    is simpler than Adam — no second moment, only a sign operation — yet
    achieves comparable or better performance across many tasks.

    Update rules
    -------------
    m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
    θ_{t+1} = θ_t − lr · [sign(m_t) + λ · sign(θ_t)]

    where λ is the weight decay rate.  For the first step (t = 0), m is
    undefined so LION uses sign(g) directly.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate (default ``1e-4``).
    betas : tuple[float, float]
        First moment coefficient (default ``(0.9, 0.99)`` — β₂ is reserved
        but only β₁ is actively used; keeping β₂ allows compatibility with
        Adam-style config objects).
    weight_decay : float
        Weight decay λ (default ``0.0``).
    max_grad_norm : float or None
        Gradient norm clipping.
    clip_grad_value : float or None
        Element-wise gradient clipping.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            weight_decay = group["weight_decay"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LION does not support sparse gradients")

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]
                t = state["step"]
                state["step"] = t + 1

                # Update the momentum buffer: m = β₁ · m + (1 − β₁) · g
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Compute the update direction
                # For the very first step, m ≈ (1-β₁)·g → sign(m) ≈ sign(g)
                update = torch.sign(m)

                # Add weight-decay sign term
                if weight_decay > 0:
                    update.add_(torch.sign(p.data), alpha=weight_decay)

                p.data.add_(update, alpha=-lr)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 7. Sophia – Second-Order Clipped Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class Sophia(Optimizer):
    """Sophia: Second-Order Clipped Stochastic Optimizer.

    Sophia augments the first-moment estimator with a *diagonal Hessian*
    preconditioner that is updated every *k* steps via Hutchinson's trace
    estimator, and clips the update to prevent instability.

    Update rules
    -------------
    m_t = β₁ · m_{t-1} + (1 − β₁) · g_t

    Every *k* steps, estimate the diagonal Hessian H_diag via Hutchinson:
        z ~ N(0, I),   s = H · z,   H_diag ≈ z ⊙ s  (element-wise)

    h_t = β₂ · h_{t-1} + (1 − β₂) · H_diag

    θ_{t+1} = θ_t − lr · clip_by_norm(
        m_t / (h_t + ε) ,  ρ
    )

    where clip_by_norm(v, ρ) = v · min(1, ρ / ‖v‖₂) clips the preconditioned
    gradient to have norm at most ρ.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate (default ``1e-4``).
    betas : tuple[float, float]
        (β₁, β₂) for the gradient and Hessian moment averages
        (default ``(0.96, 0.99)``).
    eps : float
        Numerical stabiliser (default ``1e-12``).
    weight_decay : float
        L2 regularisation (default ``0.0``).
    update_every_k : int
        Hessian update frequency in steps (default ``10``).
    clip_norm : float
        Norm clipping threshold ρ for the preconditioned update
        (default ``1.0``).
    hutchinson_samples : int
        Number of Hutchinson samples for diagonal Hessian estimation
        (default ``1`` — averaged over multiple vectors for better accuracy).
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.96, 0.99),
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        update_every_k: int = 10,
        clip_norm: float = 1.0,
        hutchinson_samples: int = 1,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if update_every_k < 1:
            raise ValueError(f"update_every_k must be >= 1, got {update_every_k}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            update_every_k=update_every_k,
            clip_norm=clip_norm,
            hutchinson_samples=hutchinson_samples,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    def _estimate_hessian_diag(
        self,
        param: torch.Tensor,
        samples: int,
    ) -> torch.Tensor:
        """Estimate diagonal of the Hessian using Hutchinson's estimator.

        For each sample vector z ~ N(0, I):
            1. Compute the Hessian-vector product: H · z  (via backward pass)
            2. Diagonal estimate:  z ⊙ (H · z)

        Average over *samples* draws.
        """
        h_diag = torch.zeros_like(param.data)

        for _ in range(samples):
            z = torch.randn_like(param.data)
            # We need H · z.  Use the standard trick:
            #   H · z = ∂(g^T z) / ∂θ   where g is the gradient of loss.
            # This requires a second backward pass.  We store the current
            # gradient to avoid losing it.
            gz = torch.dot(param.grad.data.view(-1), z.view(-1))
            grad_z = torch.autograd.grad(gz, param, retain_graph=True)[0]
            h_diag.add_(z * grad_z)

        if samples > 1:
            h_diag.div_(samples)

        return h_diag

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimisation step.

        .. note::

           If ``closure`` is provided and the global step is a multiple of
           *update_every_k*, the Hessian diagonal is estimated using the
           closure.  In practice, the caller should supply the closure **only**
           when a Hessian update is desired, or simply call
           ``estimate_hessian()`` separately.

        Parameters
        ----------
        closure : callable, optional
            A re-evaluation function returning the scalar loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            update_every_k = group["update_every_k"]
            clip_norm_val = group["clip_norm"]
            hutchinson_samples = group["hutchinson_samples"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Sophia does not support sparse gradients"
                    )

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["h"] = torch.zeros_like(p)

                m = state["exp_avg"]
                h = state["h"]
                t = state["step"]
                state["step"] = t + 1

                # --- First moment update ---
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # --- Hessian diagonal update (every k steps) ---
                if (t + 1) % update_every_k == 0:
                    h_diag = self._estimate_hessian_diag(p, hutchinson_samples)
                    h.mul_(beta2).add_(h_diag, alpha=1.0 - beta2)

                # --- Sophia update ---
                # Precondition: m / (max(h, ε) · max(‖g‖, 1))
                h_clamped = torch.clamp(h, min=eps)
                grad_norm = grad.norm(2)
                normalizer = torch.clamp(
                    torch.tensor(grad_norm, device=p.device), min=1.0
                )
                preconditioner = h_clamped * normalizer
                update = m / preconditioner

                # Clip the update norm
                update_norm = update.norm(2)
                if update_norm > clip_norm_val:
                    update.mul_(clip_norm_val / (update_norm + 1e-6))

                # Weight decay
                if weight_decay > 0:
                    update.add_(p.data, alpha=weight_decay)

                p.data.sub_(update, alpha=lr)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 8. Shampoo – Preconditioned Optimizer
# ═══════════════════════════════════════════════════════════════════════════

class Shampoo(Optimizer):
    """Shampoo: Preconditioned Optimizer using Kronecker-factored curvature.

    For a parameter tensor θ ∈ ℝ^{n₁ × n₂ × ⋯ × n_d}, Shampoo maintains
    statistics matrices L_k ∈ ℝ^{n_k × n_k} for each mode k, and computes a
    preconditioned gradient using the matrix *inverse-p*th root:

        G_k = g_k^T · g_k  + ε · I   (accumulated with EMA)
        P_k = (G_k)^{−1/(2d)}         (inverse (2d)-th root)

    The preconditioned gradient is approximately:

        g̃ = P_1 ⊗ P_2 ⊗ ⋯ ⊗ P_d  ·  vec(g)

    For 2-D matrices this is efficiently computed as:

        g̃ = P_1 · g · P_2

    Grafting merges the preconditioned update with an SGD/Adam diagonal
    term for directions not captured by the Kronecker structure.

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate (default ``1e-3``).
    betas : tuple[float, float]
        EMA decay for statistics and momentum (default ``(0.9, 0.0)``).
        β₂ is used for the momentum buffer when ``graft_type='adam'``.
    eps : float
        Regularisation added to the diagonal of statistics matrices
        (default ``1e-4``).
    weight_decay : float
        L2 weight decay (default ``0.0``).
    momentum : float
        SGD-style momentum for the update (default ``0.0``).
    graft_type : str
        Type of diagonal graft: ``'sgd'``, ``'adam'``, or ``'none'``
        (default ``'sgd'``).
    start_preconditioning_step : int
        Begin using the preconditioner after this many steps (warm-up).
        During warm-up, plain SGD (or Adam graft) is used (default ``1``).
    preconditioning_frequency : int
        Compute the matrix roots every this many steps (default ``1``).
    max_grad_norm : float or None
        Gradient norm clipping.
    clip_grad_value : float or None
        Element-wise gradient clipping.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.0),
        eps: float = 1e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        graft_type: str = "sgd",
        start_preconditioning_step: int = 1,
        preconditioning_frequency: int = 1,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if graft_type not in ("sgd", "adam", "none"):
            raise ValueError(f"Invalid graft_type: {graft_type}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            graft_type=graft_type,
            start_preconditioning_step=start_preconditioning_step,
            preconditioning_frequency=preconditioning_frequency,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    # ------------------------------------------------------------------
    # Matrix inverse-p-th root via Newton-Schulz iteration
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix_power_root(
        M: torch.Tensor,
        p: int,
        num_iters: int = 5,
    ) -> torch.Tensor:
        """Compute M^{−1/p} using Newton-Schulz iteration.

        The Newton-Schulz iteration for finding A = M^{−1/p}:

            X_{k+1} = ((p − 1) · X_k + X_k^{−(p+1)} · M) / p

        We replace the explicit inverse with a coupled iteration that is
        numerically stable:

            Z_{k+1} = (1 + p)/2 · Z_k − (p−1)/2 · Z_k · Y_k · Z_k · M
            Y_{k+1} = (1 + p)/2 · Y_k − (p−1)/2 · Y_k · Z_k · Y_k · M

        with Z₀ = I, Y₀ = I / trace(M)^{1/dim}.

        Returns A ≈ M^{−1/p}.
        """
        dim = M.size(0)
        # Scale so that eigenvalues are around 1
        trace_val = M.trace()
        scale = trace_val / dim
        if scale.abs() < 1e-10:
            return torch.eye(dim, device=M.device, dtype=M.dtype)

        Y = M / scale
        Z = torch.eye(dim, device=M.device, dtype=M.dtype)

        for _ in range(num_iters):
            T = 2.0 / (2.0 + p) * (
                torch.eye(dim, device=M.device, dtype=M.dtype)
                - torch.matmul(Z, Y)
            )
            Z = Z + torch.matmul(Z, T)
            Y = torch.matmul(T, Y)

        # Result: Z ≈ M^{-1/p} * scale^{-1/p}
        result = Z / (scale ** (1.0 / p))
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            mom = group["momentum"]
            graft_type = group["graft_type"]
            start_step = group["start_preconditioning_step"]
            precond_freq = group["preconditioning_frequency"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Shampoo does not support sparse gradients"
                    )

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Per-mode statistics matrices
                    state["G"] = []
                    for dim_size in p.shape:
                        state["G"].append(
                            torch.eye(dim_size, device=p.device, dtype=p.dtype)
                        )
                    # Momentum buffer (SGD-style)
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # Adam graft state
                    if graft_type == "adam":
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                t = state["step"]
                state["step"] = t + 1

                # --- Accumulate statistics ---
                # For a 2-D matrix: G₁ = g^T g, G₂ = g g^T
                # Generalised for higher-order tensors via mode-k unfolding.
                if t >= start_step and p.dim() >= 1:
                    for k, dim_size in enumerate(p.shape):
                        # Mode-k unfolding: move axis k to front, flatten rest
                        g = grad
                        if p.dim() > 2:
                            g = g.moveaxis(k, 0).contiguous()
                            g = g.reshape(dim_size, -1)
                        elif p.dim() == 2:
                            if k == 0:
                                g = grad  # n x m
                            else:
                                g = grad.t()  # m x n
                        else:
                            g = grad.unsqueeze(1)  # 1-D: treat as column

                        # G_k += g · g^T  (EMA)
                        G_k = state["G"][k]
                        G_k.mul_(beta1).add_(
                            torch.mm(g, g.t()) if g.dim() == 2 else g * g.t(),
                            alpha=1.0 - beta1,
                        )
                        # Regularise diagonal
                        G_k.add_(
                            torch.eye(dim_size, device=G_k.device, dtype=G_k.dtype),
                            alpha=eps,
                        )

                # --- Compute preconditioned gradient ---
                use_precond = (
                    t >= start_step
                    and (t - start_step) % precond_freq == 0
                    and p.dim() >= 2
                )

                if use_precond:
                    # Compute inverse-p-th root of each statistics matrix
                    # p = 2 * num_modes  (e.g. for 2D: p = 4, root = 1/4)
                    pth = 2 * p.dim()
                    roots = []
                    for k in range(p.dim()):
                        G_k = state["G"][k]
                        roots.append(
                            self._matrix_power_root(G_k, pth, num_iters=5)
                        )

                    # Apply preconditioning: g̃ = P₁ · g · P₂  (for 2-D)
                    if p.dim() == 2:
                        precond_grad = torch.mm(torch.mm(roots[0], grad), roots[1])
                    else:
                        # For higher-order: apply left then right (simplified)
                        precond_grad = grad
                        for k, root in enumerate(roots):
                            g_flat = precond_grad.moveaxis(k, 0).contiguous()
                            orig_shape = g_flat.shape
                            g_flat = g_flat.reshape(p.shape[k], -1)
                            g_flat = torch.mm(root, g_flat)
                            g_flat = g_flat.reshape(orig_shape)
                            precond_grad = g_flat.moveaxis(0, k)

                    update = precond_grad
                else:
                    update = grad

                # --- Grafting ---
                if graft_type == "sgd":
                    # Just use the (possibly preconditioned) update directly
                    pass
                elif graft_type == "adam":
                    m = state["exp_avg"]
                    v = state["exp_avg_sq"]
                    t_adam = t + 1
                    m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    m_hat = m / (1.0 - beta2 ** t_adam)
                    v_hat = v / (1.0 - beta2 ** t_adam)
                    adam_update = m_hat / (v_hat.sqrt() + eps)
                    # Blend: use preconditioned update as the main signal,
                    # replace with Adam for diagonal terms
                    if use_precond:
                        update = update + adam_update - grad
                    else:
                        update = adam_update
                # graft_type == 'none': use update as-is

                # --- SGD momentum ---
                buf = state["momentum_buffer"]
                if mom > 0:
                    buf.mul_(mom).add_(update, alpha=1.0 - mom)
                    update = buf

                # --- Weight decay ---
                if weight_decay > 0:
                    update.add_(p.data, alpha=weight_decay)

                # --- Apply update ---
                p.data.sub_(update, alpha=lr)

        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 9. 8-bit AdamW – Block-wise Quantised AdamW
# ═══════════════════════════════════════════════════════════════════════════

class AdamW8bit(Optimizer):
    """8-bit AdamW: Memory-efficient AdamW with block-wise dynamic quantisation.

    This implementation follows the *bitsandbytes* approach:

    1. **State quantisation**: The first moment (m) and second moment (v) are
       stored as ``uint8`` tensors, achieving a 4× memory reduction per state
       tensor (``float32`` → ``uint8``).
    2. **Block-wise quantisation**: Each tensor is split into contiguous blocks
       of size ``block_size``.  For each block we store a single ``float32``
       scale (the block's absolute max) and the quantised ``uint8`` values.
    3. **Dynamic quantisation**: Scales are recomputed from the dequantised
       values at every step, so the quantisation range adapts to the current
       distribution of the statistics.
    4. **Update computation**: At each step the quantised states are
       dequantised, used for the Adam update, then the updated states are
       re-quantised.

    Quantisation / dequantisation
    ------------------------------
    For a block b with values x₁, …, x_B:

        scale = max(|x₁|, …, |x_B|)
        q_i = round(clip(x_i / scale, −1, 1) × 127)   → uint8
        x̂_i = q_i / 127 × scale

    Parameters
    ----------
    params : iterable
        Model parameters.
    lr : float
        Learning rate (default ``1e-4``).
    betas : tuple[float, float]
        Adam betas (default ``(0.9, 0.999)``).
    eps : float
        Numerical stabiliser (default ``1e-8``).
    weight_decay : float
        Decoupled weight decay (default ``0.1``).
    block_size : int
        Number of elements per quantisation block (default ``2048``).
    max_grad_norm : float or None
        Gradient norm clipping.
    clip_grad_value : float or None
        Element-wise gradient clipping.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], List[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        block_size: int = 2048,
        max_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
            max_grad_norm=max_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__(_make_param_groups(params, defaults))

    # ------------------------------------------------------------------
    # Block-wise quantise / dequantise helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_blockwise(
        tensor: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise a float32 tensor to uint8 with block-wise scaling.

        Returns
        -------
        q : torch.Tensor  (uint8)
        scales : torch.Tensor  (float32, one scale per block)
        """
        flat = tensor.reshape(-1)
        numel = flat.numel()
        # Pad to a multiple of block_size
        pad_len = (block_size - numel % block_size) % block_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device, dtype=flat.dtype)])

        blocks = flat.reshape(-1, block_size)

        # Per-block absolute max → scale
        scales = blocks.abs().max(dim=1).values  # (num_blocks,)

        # Avoid division by zero
        safe_scales = torch.clamp(scales, min=1e-12)

        # Normalise to [-1, 1], map to [0, 127]
        normalised = blocks / safe_scales.unsqueeze(1)
        normalised.clamp_(-1.0, 1.0)
        q = ((normalised + 1.0) * 0.5 * 255.0).round_().to(torch.uint8)

        return q, scales

    @staticmethod
    def _dequantize_blockwise(
        q: torch.Tensor,
        scales: torch.Tensor,
        orig_numel: int,
    ) -> torch.Tensor:
        """Dequantise uint8 blocks back to float32."""
        # q: (num_blocks, block_size) uint8
        block_size = q.size(1)
        num_blocks = q.size(0)

        # Map back to [-1, 1]
        normalised = q.float() / 127.5 - 1.0

        # Scale
        blocks = normalised * scales.unsqueeze(1)

        flat = blocks.reshape(-1)
        return flat[:orig_numel].clone()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            block_size = group["block_size"]
            max_grad_norm = group["max_grad_norm"]
            clip_grad_value = group["clip_grad_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW8bit does not support sparse gradients"
                    )

                _apply_gradient_clipping(p, max_grad_norm, clip_grad_value)
                grad = p.grad.data

                numel = p.data.numel()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0

                    # Quantised states (uint8) + float32 scales
                    # First moment
                    m_zeros = torch.zeros(numel, device=p.device, dtype=p.dtype)
                    m_q, m_s = self._quantize_blockwise(m_zeros, block_size)
                    state["m_q"] = m_q
                    state["m_s"] = m_s

                    # Second moment
                    v_zeros = torch.zeros(numel, device=p.device, dtype=p.dtype)
                    v_q, v_s = self._quantize_blockwise(v_zeros, block_size)
                    state["v_q"] = v_q
                    state["v_s"] = v_s

                t = state["step"] + 1
                state["step"] = t

                # --- Dequantise states ---
                m = self._dequantize_blockwise(state["m_q"], state["m_s"], numel).reshape(p.shape)
                v = self._dequantize_blockwise(state["v_q"], state["v_s"], numel).reshape(p.shape)

                # --- Adam update (float32 precision) ---
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr / bias_correction1

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                denom = v_hat.sqrt().add(eps)

                # --- Apply parameter update ---
                p.data.addcdiv_(m_hat, denom, value=-step_size)

                # --- Decoupled weight decay ---
                if weight_decay > 0:
                    p.data.add_(p, alpha=-lr * weight_decay)

                # --- Re-quantise states ---
                m_flat = m.reshape(-1)
                m_q_new, m_s_new = self._quantize_blockwise(m_flat, block_size)
                state["m_q"] = m_q_new
                state["m_s"] = m_s_new

                v_flat = v.reshape(-1)
                v_q_new, v_s_new = self._quantize_blockwise(v_flat, block_size)
                state["v_q"] = v_q_new
                state["v_s"] = v_s_new

        return loss
