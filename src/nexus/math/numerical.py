"""
Nexus Numerical Stability Utilities
========================================
Utilities for maintaining numerical stability during training.

Gradient Management:
    - Gradient clipping (by norm and by value)
    - Gradient accumulation
    - Gradient noise injection

Loss Scaling:
    - Static loss scaling (for FP16 training)
    - Dynamic loss scaling (automatically adjusts scale)
    
Numerical Guards:
    - Safe softmax (prevents overflow)
    - Safe log (prevents log(0))
    - Safe division (prevents division by zero)
    - NaN/Inf detection and handling
    - Check for numerical stability issues

These are critical for training large models at scale where
floating-point precision issues are common, especially in:
    - FP16/BF16 mixed precision training
    - Deep networks with 100+ layers
    - Attention mechanisms (large exponentials)
    - Cross-entropy loss (very small probabilities)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, Union

Array = np.ndarray


# ================================================================
# GRADIENT CLIPPING
# ================================================================

def gradient_clip_norm(
    parameters: Dict[str, Array],
    max_norm: float = 1.0,
) -> Tuple[Dict[str, Array], float, float]:
    """
    Clip gradients by global norm.
    
    If ||g|| > max_norm, scale all gradients: g <- g * (max_norm / ||g||)
    
    This prevents gradient explosions in deep networks. The total gradient
    norm is computed across ALL parameters:
        ||g|| = sqrt(sum_i ||g_i||^2)
    
    If ||g|| > max_norm, ALL gradients are scaled by the same factor,
    preserving their relative magnitudes.
    
    Algorithm:
        1. Compute global gradient norm
        2. If norm > max_norm:
           clip_coef = max_norm / norm
           g_i <- g_i * clip_coef  for all i
        3. Return clipped gradients
    
    This is the default clipping strategy used in most LLM training
    (GPT-3, LLaMA, etc.) with max_norm typically set to 1.0.
    
    Args:
        parameters: Dictionary of {name: gradient}.
        max_norm: Maximum allowed gradient norm.
    
    Returns:
        Tuple of (clipped_parameters, original_norm, clip_coef).
    """
    # Compute global norm
    total_norm_sq = 0.0
    for name, grad in parameters.items():
        grad = np.asarray(grad, dtype=np.float64)
        total_norm_sq += np.sum(grad ** 2)
    total_norm = float(np.sqrt(total_norm_sq))
    
    # Compute clipping coefficient
    clip_coef = max_norm / max(total_norm, 1e-6)
    clip_coef = min(clip_coef, 1.0)  # Don't scale UP
    
    # Clip gradients
    clipped = {}
    if clip_coef < 1.0:
        for name, grad in parameters.items():
            clipped[name] = np.asarray(grad, dtype=np.float64) * clip_coef
    else:
        clipped = {k: np.asarray(v, dtype=np.float64) for k, v in parameters.items()}
    
    return clipped, total_norm, float(clip_coef)


def gradient_clip_value(
    parameters: Dict[str, Array],
    clip_value: float = 1.0,
) -> Dict[str, Array]:
    """
    Clip gradients by value (element-wise).
    
    Each gradient element is clamped to [-clip_value, clip_value]:
        g_i = clamp(g_i, -clip_value, clip_value)
    
    Simpler than norm clipping but doesn't preserve gradient direction.
    Less commonly used in LLM training.
    
    Args:
        parameters: Dictionary of {name: gradient}.
        clip_value: Maximum absolute value for each gradient element.
    
    Returns:
        Dictionary of clipped gradients.
    """
    clipped = {}
    for name, grad in parameters.items():
        grad = np.asarray(grad, dtype=np.float64)
        clipped[name] = np.clip(grad, -clip_value, clip_value)
    return clipped


def gradient_noise_injection(
    parameters: Dict[str, Array],
    step: int,
    eta: float = 0.01,
    gamma: float = 0.55,
    noise_std: float = 0.01,
) -> Dict[str, Array]:
    """
    Inject noise into gradients for regularization.
    
    Noise scale: sigma_t = eta / (1 + step)^gamma
    
    This helps escape saddle points and improves generalization
    (Neelakantan et al., "Adding Gradient Noise Improves Learning
    for Very Deep Networks", 2015).
    """
    sigma_t = noise_std / (1 + step) ** gamma
    
    noisy = {}
    for name, grad in parameters.items():
        grad = np.asarray(grad, dtype=np.float64)
        noise = np.random.normal(0, sigma_t, size=grad.shape) * np.abs(grad)
        noisy[name] = grad + noise
    
    return noisy


# ================================================================
# LOSS SCALING (for mixed precision training)
# ================================================================

class StaticLossScaler:
    """
    Static loss scaling for FP16 training.
    
    Problem: In FP16, gradients smaller than 2^(-24) underflow to zero.
    Solution: Scale the loss UP before backward, then scale gradients DOWN.
    
    Algorithm:
        Forward: loss_scaled = loss * scale_factor
        Backward: grad = grad_scaled / scale_factor
    
    The scale factor must be chosen large enough to prevent underflow
    but not so large that it causes overflow (FP16 max ≈ 65504).
    
    Common scale factors: 128, 256, 1024, 65536
    
    Args:
        init_scale: Initial scale factor (power of 2 recommended).
        backoff_factor: Factor to reduce scale on overflow.
        growth_factor: Factor to increase scale on success.
        growth_interval: Steps between growth attempts.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 15,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
    ):
        self.scale = init_scale
        self.backoff_factor = backoff_factor
        self.growth_factor = growth_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False

    def scale_loss(self, loss: Array) -> Tuple[Array, float]:
        """Scale loss before backward pass."""
        scaled_loss = loss * self.scale
        return scaled_loss, self.scale

    def unscale_grad(self, grad: Array) -> Array:
        """Unscale gradient after backward pass."""
        return grad / self.scale

    def update(self, found_inf: bool):
        """Update scale factor based on whether overflow occurred."""
        if found_inf:
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0


class DynamicLossScaler:
    """
    Dynamic loss scaling (auto-adjusts scale factor).
    
    Automatically finds the optimal scale factor by:
        1. Starting with a large scale
        2. If overflow detected → halve the scale
        3. If N consecutive steps without overflow → double the scale
        4. Repeat
    
    This eliminates the need to manually tune the scale factor.
    Used in NVIDIA's AMP (Automatic Mixed Precision) and most
    modern FP16 training frameworks.
    
    Args:
        init_scale: Initial scale factor.
        growth_factor: Multiply scale by this after growth_interval successes.
        backoff_factor: Divide scale by this on overflow.
        growth_interval: Number of successful steps before growing.
        max_scale: Maximum scale factor to prevent overflow.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2.0 ** 24,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self._success_count = 0
        self._overflow_count = 0

    def scale_loss(self, loss: Array) -> Tuple[Array, float]:
        """Scale loss and return (scaled_loss, current_scale)."""
        return loss * self.scale, self.scale

    def unscale_grad(self, grad: Array) -> Array:
        """Unscale gradient."""
        return grad / self.scale

    def update(self, found_inf: bool) -> Dict[str, Union[float, bool]]:
        """
        Update scale based on overflow detection.
        
        Args:
            found_inf: Whether infinity/NaN was found in gradients.
        
        Returns:
            Dict with scale, found_inf, overflow_count, success_count.
        """
        if found_inf:
            # Overflow: reduce scale
            self.scale = max(self.scale * self.backoff_factor, 1.0)
            self._success_count = 0
            self._overflow_count += 1
        else:
            # Success: maybe increase scale
            self._success_count += 1
            if self._success_count >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self._success_count = 0
        
        return {
            "scale": self.scale,
            "found_inf": found_inf,
            "overflow_count": self._overflow_count,
            "success_count": self._success_count,
        }


def static_loss_scaling(
    loss: Array,
    scale_factor: float = 128.0,
) -> Tuple[Array, float]:
    """Simple static loss scaling (convenience function)."""
    return loss * scale_factor, scale_factor


def dynamic_loss_scaling(
    loss: Array,
    scaler: DynamicLossScaler,
) -> Tuple[Array, float]:
    """Dynamic loss scaling (convenience function)."""
    return scaler.scale_loss(loss)


# ================================================================
# NUMERICAL GUARDS
# ================================================================

def safe_softmax(logits: Array, axis: int = -1) -> Array:
    """
    Numerically stable softmax.
    
    Standard softmax can overflow/underflow:
        softmax(x) = exp(x) / sum(exp(x))  ← exp(large_x) → inf!
    
    Stable version (log-sum-exp trick):
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Subtracting max(x) ensures all exponentials are <= 1, preventing overflow.
    The result is identical (subtracting a constant doesn't change softmax).
    """
    logits = np.asarray(logits, dtype=np.float64)
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def safe_log(x: Array, epsilon: float = 1e-12) -> Array:
    """
    Numerically stable logarithm.
    
    Prevents log(0) → -inf by clamping:
        safe_log(x) = log(max(x, epsilon))
    
    Used in:
        - Cross-entropy loss: log(softmax(x))
        - KL divergence: p * log(p/q)
        - Entropy: -sum(p * log(p))
    """
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.maximum(x, epsilon))


def safe_log_softmax(logits: Array, axis: int = -1) -> Array:
    """
    Numerically stable log-softmax.
    
    log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    
    More stable than log(softmax(x)) because it avoids computing
    the intermediate softmax which can have very small values.
    """
    logits = np.asarray(logits, dtype=np.float64)
    x = logits - np.max(logits, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def safe_divide(numerator: Array, denominator: Array, epsilon: float = 1e-12) -> Array:
    """
    Safe division that prevents division by zero.
    
    a / b → a / max(|b|, epsilon) * sign(b)
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    return num / np.where(np.abs(den) > epsilon, den, epsilon)


def check_numerical_stability(
    loss: float,
    parameters: Dict[str, Array],
    step: int,
    max_grad_norm: float = 100.0,
    max_loss: float = 1e6,
) -> Dict[str, Union[bool, float, str]]:
    """
    Check for numerical issues in training.
    
    Monitors:
        - NaN loss
        - Inf loss
        - NaN gradients
        - Inf gradients
        - Excessively large gradients
        - Zero gradients
    
    Returns diagnostic information.
    """
    diagnostics = {
        "step": step,
        "loss": loss,
        "is_nan_loss": np.isnan(loss),
        "is_inf_loss": np.isinf(loss),
        "has_nan_grad": False,
        "has_inf_grad": False,
        "max_grad_norm": 0.0,
        "warnings": [],
    }
    
    if diagnostics["is_nan_loss"]:
        diagnostics["warnings"].append("NaN loss detected!")
    
    if diagnostics["is_inf_loss"]:
        diagnostics["warnings"].append("Inf loss detected!")
    
    total_norm_sq = 0.0
    for name, grad in parameters.items():
        grad = np.asarray(grad, dtype=np.float64)
        
        if np.any(np.isnan(grad)):
            diagnostics["has_nan_grad"] = True
            diagnostics["warnings"].append(f"NaN gradient in {name}!")
        
        if np.any(np.isinf(grad)):
            diagnostics["has_inf_grad"] = True
            diagnostics["warnings"].append(f"Inf gradient in {name}!")
        
        total_norm_sq += np.sum(grad ** 2)
    
    diagnostics["max_grad_norm"] = float(np.sqrt(total_norm_sq))
    
    if diagnostics["max_grad_norm"] > max_grad_norm:
        diagnostics["warnings"].append(
            f"Gradient norm {diagnostics['max_grad_norm']:.2f} exceeds "
            f"threshold {max_grad_norm}"
        )
    
    diagnostics["is_stable"] = len(diagnostics["warnings"]) == 0
    
    return diagnostics


def detect_overflow(tensor: Array) -> bool:
    """Check if tensor contains Inf or NaN values."""
    return bool(np.any(np.isinf(tensor)) or np.any(np.isnan(tensor)))


def replace_nan_inf(tensor: Array, replacement: float = 0.0) -> Array:
    """Replace NaN and Inf values with a replacement value."""
    result = np.where(np.isfinite(tensor), tensor, replacement)
    return result


# ================================================================
# CONVENIENCE FUNCTIONS
# ================================================================

dynamic_loss_scaling = dynamic_loss_scaling
static_loss_scaling = static_loss_scaling
