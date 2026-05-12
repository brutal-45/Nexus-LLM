"""
Normalization Layers for Nexus
==================================
Complete collection of normalization strategies for transformer models.

Normalization is critical for training stable, deep transformer networks.
Different architectures employ different normalization strategies:

- **LayerNorm** (Vaswani et al., 2017): Original transformer, BERT
- **RMSNorm** (Zhang & Sennrich, 2019): LLaMA, Mistral, Gemma — faster,
  avoids mean subtraction.
- **DeepNorm** (Wang et al., 2022): Enables stable training of 1000+ layer
  transformers by scaling residual connections and normalization.
- **QKNorm**: Normalizes queries and keys independently before attention,
  stabilizing attention logits in very deep or high-dimension models.

All normalizations are implemented as ``nn.Module`` for consistency and to
support custom fused kernels (e.g. ``torch.compile``, Triton, CUDA).

References:
    - LayerNorm: Ba et al., 2016  (https://arxiv.org/abs/1607.06450)
    - RMSNorm: Zhang & Sennrich, 2019  (https://arxiv.org/abs/1910.07467)
    - DeepNorm: Wang et al., 2022  (https://arxiv.org/abs/2203.00555)
    - QKNorm: Henry et al., 2020  (https://arxiv.org/abs/2002.08252)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activations import get_activation


# ---------------------------------------------------------------------------
# 1. LayerNorm (standard)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Standard Layer Normalization.

    Formula::

        y = γ * (x - μ) / σ + β

    where μ and σ are the mean and standard deviation computed over the
    last dimension, and γ (scale) and β (bias) are learnable parameters.

    This is the normalization used in the original Transformer (Vaswani et
    al., 2017) and in BERT, GPT-2, and many other models.

    Args:
        normalized_shape: Input shape from an expected input of size.
            If a single integer is used, it is treated as a singleton list.
            ``-1`` may be used to infer the shape from the input at runtime.
        eps: A value added to the denominator for numerical stability.
            Default: 1e-5.
        elementwise_affine: If True, adds learnable γ and β parameters.
            Default: True.
        bias: If True, adds a learnable bias (β). Only used when
            ``elementwise_affine=True``. Default: True.
        dtype: Desired data type for the parameters.
        device: Device on which the parameters should be initialized.

    Shape:
        - Input:  ``(*, N)``
        - Output: ``(*, N)``, same shape as input.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(normalized_shape, dtype=dtype, device=device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if self.elementwise_affine:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        return F.layer_norm(x, self.normalized_shape, None, None, self.eps)

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


# ---------------------------------------------------------------------------
# 2. RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Normalization.

    Formula::

        y = x / sqrt(mean(x²) + ε) * γ

    RMSNorm simplifies LayerNorm by removing the mean subtraction step,
    relying only on the root mean square of the inputs. This provides
    two key advantages:

    1. **Speed**: ~10-50% faster than LayerNorm because it avoids computing
       the mean and subtracting it.
    2. **Simplicity**: No bias parameter needed, reducing memory footprint.

    RMSNorm is used in LLaMA, Mistral, Mixtral, Gemma, Phi, Qwen, and
    most modern open-source LLMs. It has been shown to perform comparably
    to LayerNorm in practice while being computationally cheaper.

    The implementation is ``torch.compile`` friendly and can be further
    optimized with fused Triton kernels.

    Args:
        hidden_size: Dimension of the input tensor (last dimension).
        eps: Small constant for numerical stability. Default: 1e-6.
            LLaMA uses 1e-5, some models use 1e-6.
        elementwise_affine: If True, adds a learnable scale (γ).
            Default: True.
        dtype: Desired data type for parameters.
        device: Device on which parameters should be initialized.

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization without the scale."""
        # Compute the RMS (root mean square) over the last dimension
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms.type_as(x)

    def forward(self, x: Tensor) -> Tensor:
        x_normed = self._norm(x)
        if self.weight is not None:
            return x_normed * self.weight
        return x_normed

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


# ---------------------------------------------------------------------------
# 3. DeepNorm
# ---------------------------------------------------------------------------

class DeepNorm(nn.Module):
    """DeepNorm: Layer Normalization for very deep transformers.

    DeepNorm modifies the standard LayerNorm with depth-dependent scaling
    factors to enable stable training of transformers with 100+ layers.

    The scaling factors are computed as::

        α = (2 * N)^{1/4}      (sublayer input scaling)
        β = (8 * N)^{-1/4}     (sublayer output scaling)

    where N is the total number of transformer layers. The DeepNorm
    modification replaces the standard residual + sublayer pattern with::

        output = α * x + sublayer(β * LayerNorm(x))

    This re-parameterization provides:
    - Better gradient flow through very deep networks
    - Reduced need for careful learning rate tuning
    - Stability at depths where standard transformers diverge

    Args:
        hidden_size: Dimension of the input tensor.
        num_layers: Total number of transformer layers. Used to compute
            the depth-dependent scaling factors α and β.
        eps: Small constant for numerical stability. Default: 1e-5.
        elementwise_affine: If True, adds learnable γ and β to LayerNorm.
            Default: True.
        dtype: Desired data type for parameters.
        device: Device on which parameters should be initialized.

    References:
        Wang et al., "DeepNorm: Improving Deep Transformer Training with
        Layer Normalization for 1000+ Layer Language Models", 2022.

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.eps = eps

        # Compute depth-dependent scaling factors
        self.alpha = (2.0 * num_layers) ** 0.25
        self.beta = (8.0 * num_layers) ** -0.25

        self.layer_norm = LayerNorm(
            normalized_shape=hidden_size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply DeepNorm: scale input, normalize, then scale output.

        This should be applied *before* the sublayer (pre-norm style),
        and the caller must handle the residual connection using the
        alpha/beta scaling factors.
        """
        return self.beta * self.layer_norm(x)

    def get_alpha(self) -> float:
        """Return the input scaling factor α."""
        return self.alpha

    def get_beta(self) -> float:
        """Return the sublayer output scaling factor β."""
        return self.beta

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, "
            f"eps={self.eps}"
        )


# ---------------------------------------------------------------------------
# 4. QKNorm
# ---------------------------------------------------------------------------

class QKNorm(nn.Module):
    """Query-Key Normalization for stabilizing attention.

    Normalizes queries and keys independently before computing attention
    scores. This prevents the dot-product attention logits from growing
    too large (which would push softmax into saturated regions) and
    stabilizes training of very deep or high-dimension models.

    Formula::

        Q_norm = γ_Q * Q / RMS(Q)    (or LayerNorm(Q))
        K_norm = γ_K * K / RMS(K)    (or LayerNorm(K))
        attn   = softmax(Q_norm @ K_norm^T / sqrt(d))

    Args:
        hidden_size: Dimension of the query/key vectors.
        head_dim: Dimension of each attention head. If None, defaults to
            ``hidden_size`` (single-head case).
        norm_type: Type of normalization to apply. Either ``"rmsnorm"``
            (default, faster) or ``"layernorm"``.
        eps: Small constant for numerical stability.
        scale: Whether to apply learnable scale parameters. Default: True.
        dtype: Desired data type for parameters.
        device: Device on which parameters should be initialized.

    Shape:
        - Query Input:  ``(B, T, H)`` or ``(B, num_heads, T, head_dim)``
        - Key Input:    ``(B, S, H)`` or ``(B, num_heads, S, head_dim)``
        - Q Output:     Same shape as query input.
        - K Output:     Same shape as key input.

    References:
        Henry et al., "Query-Key Normalization for Transformers", 2020.
        (https://arxiv.org/abs/2002.08252)
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: Optional[int] = None,
        norm_type: str = "rmsnorm",
        eps: float = 1e-6,
        scale: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim or hidden_size
        self.norm_type = norm_type
        self.eps = eps

        norm_size = self.head_dim
        if norm_type == "rmsnorm":
            self.q_norm = RMSNorm(norm_size, eps=eps, dtype=dtype, device=device)
            self.k_norm = RMSNorm(norm_size, eps=eps, dtype=dtype, device=device)
        elif norm_type == "layernorm":
            self.q_norm = LayerNorm(norm_size, eps=eps, dtype=dtype, device=device)
            self.k_norm = LayerNorm(norm_size, eps=eps, dtype=dtype, device=device)
        else:
            raise ValueError(
                f"Unknown norm_type '{norm_type}'. Expected 'rmsnorm' or 'layernorm'."
            )

        # Optional learnable scaling after normalization
        if scale:
            self.q_scale = nn.Parameter(
                torch.ones(norm_size, dtype=dtype, device=device)
            )
            self.k_scale = nn.Parameter(
                torch.ones(norm_size, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("q_scale", None)
            self.register_parameter("k_scale", None)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Normalize queries and keys.

        Args:
            q: Query tensor of shape ``(B, T, H)`` or ``(B, nh, T, hd)``.
            k: Key tensor of shape ``(B, S, H)`` or ``(B, nh, S, hd)``.

        Returns:
            Tuple of ``(q_norm, k_norm)`` with the same shapes as inputs.
        """
        q_normed = self.q_norm(q)
        k_normed = self.k_norm(k)

        if self.q_scale is not None:
            q_normed = q_normed * self.q_scale
        if self.k_scale is not None:
            k_normed = k_normed * self.k_scale

        return q_normed, k_normed

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"head_dim={self.head_dim}, "
            f"norm_type='{self.norm_type}', "
            f"eps={self.eps}, "
            f"scale={self.q_scale is not None}"
        )


# ---------------------------------------------------------------------------
# 5. SubLayerNorm (Post-LayerNorm)
# ---------------------------------------------------------------------------

class SubLayerNorm(nn.Module):
    """Post-sublayer normalization (Post-LN).

    Applies LayerNorm *after* the residual connection::

        output = LayerNorm(x + sublayer(x))

    This is the normalization strategy used in the original Transformer
    (Vaswani et al., 2017), where normalization is applied after the
    sublayer output has been added to the residual stream. Post-LN is
    conceptually simpler but can be less stable during training for very
    deep networks compared to Pre-LN.

    This module provides a reusable Post-LN wrapper that combines the
    residual addition and normalization in a single module.

    Args:
        hidden_size: Dimension of the input tensor.
        sublayer: The sublayer module (e.g., attention or FFN).
        eps: Small constant for numerical stability. Default: 1e-5.
        elementwise_affine: If True, adds learnable parameters to LN.
            Default: True.
        dtype: Desired data type for parameters.
        device: Device on which parameters should be initialized.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        sublayer: nn.Module,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.sublayer = sublayer
        self.norm = LayerNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply post-LN: x + sublayer(x), then normalize."""
        return self.norm(x + self.sublayer(x))

    def extra_repr(self) -> str:
        sublayer_type = self.sublayer.__class__.__name__
        return (
            f"hidden_size={self.hidden_size}, "
            f"sublayer={sublayer_type}, "
            f"position='post'"
        )


# ---------------------------------------------------------------------------
# 6. Factory function
# ---------------------------------------------------------------------------

_NORM_REGISTRY: dict[str, type[nn.Module]] = {
    "layernorm": LayerNorm,
    "layer_norm": LayerNorm,
    "rmsnorm": RMSNorm,
    "rms_norm": RMSNorm,
    "deepnorm": DeepNorm,
    "deep_norm": DeepNorm,
    "qknorm": QKNorm,
    "qk_norm": QKNorm,
}


def get_norm(
    norm_type: str,
    hidden_size: int,
    **kwargs: Any,
) -> nn.Module:
    """Factory function to create a normalization layer by name.

    Args:
        norm_type: Name of the normalization layer (case-insensitive).
            Supported values:
            - ``"layernorm"`` / ``"layer_norm"`` → :class:`LayerNorm`
            - ``"rmsnorm"`` / ``"rms_norm"`` → :class:`RMSNorm`
            - ``"deepnorm"`` / ``"deep_norm"`` → :class:`DeepNorm`
              (requires ``num_layers`` kwarg)
            - ``"qknorm"`` / ``"qk_norm"`` → :class:`QKNorm`
        hidden_size: The hidden dimension (normalized shape).
        **kwargs: Additional keyword arguments forwarded to the
            normalization layer constructor.

    Returns:
        An ``nn.Module`` instance of the requested normalization.

    Raises:
        ValueError: If ``norm_type`` is not recognized or required
            arguments are missing.

    Examples::

        >>> norm = get_norm("rmsnorm", 768, eps=1e-5)
        >>> norm = get_norm("deepnorm", 768, num_layers=24)
        >>> norm = get_norm("layernorm", 512, elementwise_affine=False)
    """
    key = norm_type.lower().replace("-", "_")
    if key not in _NORM_REGISTRY:
        available = ", ".join(sorted(set(_NORM_REGISTRY.keys())))
        raise ValueError(
            f"Unknown normalization type '{norm_type}'. Available: {available}"
        )
    cls = _NORM_REGISTRY[key]
    try:
        return cls(hidden_size=hidden_size, **kwargs)
    except TypeError as e:
        raise ValueError(
            f"Failed to instantiate {cls.__name__} with "
            f"hidden_size={hidden_size}, kwargs={kwargs}. Error: {e}"
        ) from e
