"""
Residual Connection Strategies for Nexus
============================================
Different residual connection methods for stable deep network training.

Residual connections are a fundamental architectural component of deep
transformers. They enable gradient flow through very deep networks and
provide an identity mapping that prevents degradation with depth.

Different architectures use different residual strategies:

- **StandardResidual**: Simple ``x + f(x)`` — used in ResNet, early transformers
- **PreNormResidual**: ``x + f(LN(x))`` — modern standard (LLaMA, Mistral, GPT-2)
- **PostNormResidual**: ``LN(x + f(x))`` — original Transformer (Vaswani et al.)
- **ScaledResidual**: ``x + α * f(x)`` — learnable or fixed scaling
- **DeepNetResidual**: Depth-aware scaling for 1000+ layer stability
- **ParallelResidual**: ``x + attn(LN(x)) + ffn(LN2(x))`` — GPT-J, PaLM style

All residual classes are ``nn.Module`` wrappers that compose a sublayer
with appropriate normalization and scaling.

References:
    - Pre-LN: Xiong et al., 2020  (https://arxiv.org/abs/2002.04745)
    - DeepNet: Wang et al., 2022  (https://arxiv.org/abs/2203.00555)
    - Parallel: "GPT-J" (Wang & Komatsuzaki, 2021)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .normalization import DeepNorm, LayerNorm, RMSNorm, get_norm


# ---------------------------------------------------------------------------
# 1. StandardResidual
# ---------------------------------------------------------------------------

class StandardResidual(nn.Module):
    """Standard (vanilla) residual connection.

    Formula::

        output = x + sublayer(x)

    The simplest residual connection, which adds the sublayer output directly
    to the input. Used in ResNet and early transformer implementations. No
    normalization is applied within this module — the caller is responsible
    for any normalization.

    This style is suitable when normalization is handled externally or when
    a bare residual connection is desired (e.g., in combination with other
    custom normalization schemes).

    Args:
        sublayer: The sublayer module (e.g., attention block, FFN, etc.).

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(self, sublayer: nn.Module) -> None:
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x: Tensor) -> Tensor:
        return x + self.sublayer(x)

    def extra_repr(self) -> str:
        return f"sublayer={self.sublayer.__class__.__name__}"


# ---------------------------------------------------------------------------
# 2. ScaledResidual
# ---------------------------------------------------------------------------

class ScaledResidual(nn.Module):
    """Scaled residual connection with a learnable or fixed scaling factor.

    Formula::

        output = x + α * sublayer(x)

    where α is either a fixed scalar or a learnable parameter. Scaling the
    sublayer output can help stabilize training, particularly:

    - **Fixed α < 1**: Dampens the sublayer contribution, useful when the
      sublayer outputs are large (e.g., during early training).
    - **Learnable α**: Allows the model to adaptively control the residual
      vs. sublayer balance. Initialized to a small value (e.g., 0.0 or 1/√N).

    Args:
        sublayer: The sublayer module.
        scale: Fixed scale factor. If provided and ``learnable=False``,
            this value is used as the constant scaling factor.
        learnable: If True, α is a learnable parameter. Default: True.
        init_scale: Initial value for the learnable scale parameter.
            Default: 1.0.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        sublayer: nn.Module,
        scale: Optional[float] = None,
        learnable: bool = True,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.learnable = learnable

        if learnable:
            self.scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.register_buffer(
                "scale",
                torch.tensor(scale if scale is not None else init_scale),
            )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * self.sublayer(x)

    def extra_repr(self) -> str:
        val = self.scale.data.item() if isinstance(self.scale, nn.Parameter) else self.scale.item()
        return (
            f"sublayer={self.sublayer.__class__.__name__}, "
            f"scale={val:.4f}, "
            f"learnable={self.learnable}"
        )


# ---------------------------------------------------------------------------
# 3. PreNormResidual
# ---------------------------------------------------------------------------

class PreNormResidual(nn.Module):
    """Pre-normalization residual connection (modern standard).

    Formula::

        output = x + sublayer(norm(x))

    Normalization is applied *before* the sublayer, and the result is added
    to the (unnormalized) residual stream. This is the dominant residual
    strategy in modern LLMs, used by LLaMA, Mistral, GPT-2, GPT-NeoX,
    PaLM, Phi, and virtually all recent architectures.

    Pre-LN offers several advantages over Post-LN:
    - **Training stability**: The residual stream is always normalized before
      the sublayer, preventing explosion of activations.
    - **Simpler initialization**: No need for careful LR warmup or init tricks.
    - **Better gradient flow**: Gradients flow through the clean residual path.

    Args:
        hidden_size: Dimension of the input tensor.
        sublayer: The sublayer module (e.g., attention, FFN).
        norm_type: Type of normalization to use. Either ``"rmsnorm"`` (default)
            or ``"layernorm"``.
        norm_eps: Epsilon for the normalization layer. Default: 1e-6.
        norm_kwargs: Additional keyword arguments for the normalization layer.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        sublayer: nn.Module,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        **norm_kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.sublayer = sublayer
        self.norm = get_norm(norm_type, hidden_size, eps=norm_eps, **norm_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.sublayer(self.norm(x))

    def extra_repr(self) -> str:
        norm_type = self.norm.__class__.__name__
        return (
            f"hidden_size={self.hidden_size}, "
            f"sublayer={self.sublayer.__class__.__name__}, "
            f"norm={norm_type}, "
            f"position='pre'"
        )


# ---------------------------------------------------------------------------
# 4. PostNormResidual
# ---------------------------------------------------------------------------

class PostNormResidual(nn.Module):
    """Post-normalization residual connection (original Transformer).

    Formula::

        output = norm(x + sublayer(x))

    The sublayer output is first added to the residual stream, then the
    combined result is normalized. This is the strategy used in the original
    Transformer (Vaswani et al., 2017) and in BERT.

    Post-LN applies normalization to the *output* of the residual connection,
    which means every layer receives a normalized input (from the previous
    layer's Post-LN). While conceptually clean, Post-LN can be harder to
    train at very large depths due to gradient magnitude issues near the
    output layers.

    Args:
        hidden_size: Dimension of the input tensor.
        sublayer: The sublayer module.
        norm_type: Type of normalization. Default: ``"layernorm"``.
        norm_eps: Epsilon for the normalization layer. Default: 1e-5.
        norm_kwargs: Additional keyword arguments for the normalization layer.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        sublayer: nn.Module,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        **norm_kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.sublayer = sublayer
        self.norm = get_norm(norm_type, hidden_size, eps=norm_eps, **norm_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.sublayer(x))

    def extra_repr(self) -> str:
        norm_type = self.norm.__class__.__name__
        return (
            f"hidden_size={self.hidden_size}, "
            f"sublayer={self.sublayer.__class__.__name__}, "
            f"norm={norm_type}, "
            f"position='post'"
        )


# ---------------------------------------------------------------------------
# 5. DeepNetResidual
# ---------------------------------------------------------------------------

class DeepNetResidual(nn.Module):
    """Deep residual connection for 100+ layer transformers.

    Formula::

        output = x + sublayer(β * LayerNorm(x)) * γ

    where:
    - β = (8N)^{-1/4}  scales the normalized input (from DeepNorm)
    - γ = 1/√(2N)     scales the sublayer output (depth-dependent init)
    - N = total number of layers

    The sublayer output scaling (γ) ensures that the residual contribution
    is appropriately small at initialization, preventing the activations
    from growing too large in very deep networks. This is inspired by
    DeepNet (Wang et al., 2022) and the fixup initialization strategy.

    Combined with a pre-norm (or DeepNorm) strategy, this enables stable
    training of transformers with 1000+ layers.

    Args:
        hidden_size: Dimension of the input tensor.
        num_layers: Total number of transformer layers (used to compute γ).
        sublayer: The sublayer module.
        norm_type: Type of normalization. Default: ``"layernorm"``.
        norm_eps: Epsilon for the normalization layer.
        use_deepnorm_beta: If True, applies the DeepNorm β scaling to the
            normalized input. Default: True.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        sublayer: nn.Module,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
        use_deepnorm_beta: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sublayer = sublayer
        self.use_deepnorm_beta = use_deepnorm_beta

        # Compute the DeepNorm β scaling factor
        self.beta = (8.0 * num_layers) ** -0.25

        # Compute the sublayer output scaling factor
        self.gamma = 1.0 / math.sqrt(2.0 * num_layers)

        self.norm = get_norm(norm_type, hidden_size, eps=norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        # Apply β scaling if using DeepNorm
        if self.use_deepnorm_beta:
            x_normed = self.beta * self.norm(x)
        else:
            x_normed = self.norm(x)

        # Apply sublayer and scale output by γ
        return x + self.gamma * self.sublayer(x_normed)

    def extra_repr(self) -> str:
        norm_type = self.norm.__class__.__name__
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"sublayer={self.sublayer.__class__.__name__}, "
            f"norm={norm_type}, "
            f"beta={self.beta:.4f}, "
            f"gamma={self.gamma:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. ParallelResidual
# ---------------------------------------------------------------------------

class ParallelResidual(nn.Module):
    """Parallel attention + FFN residual connection (GPT-J / PaLM style).

    Formula::

        output = x + attn(norm1(x)) + ffn(norm2(x))

    Instead of sequential attention → FFN, both the attention and FFN are
    computed in parallel on the same normalized input. This allows the
    two sublayers to be computed simultaneously (potential for parallelism
    on hardware) and reduces the number of sequential operations per
    transformer block from 2 to 1 in the residual path.

    Used in GPT-J, PaLM, and some other architectures. Note that the
    parallel formulation requires careful normalization — typically each
    sublayer has its own independent normalization layer.

    Args:
        hidden_size: Dimension of the input tensor.
        attn: The attention sublayer module.
        ffn: The feed-forward network sublayer module.
        norm_type: Type of normalization for both sublayers.
            Default: ``"layernorm"``.
        norm_eps: Epsilon for the normalization layers.
        shared_norm: If True, both sublayers share the same normalization
            instance. Default: False (separate norms, as in GPT-J/PaLM).
        norm_kwargs: Additional keyword arguments for the normalization layers.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        attn: nn.Module,
        ffn: nn.Module,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        shared_norm: bool = False,
        **norm_kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = attn
        self.ffn = ffn

        if shared_norm:
            self.norm1 = get_norm(norm_type, hidden_size, eps=norm_eps, **norm_kwargs)
            self.norm2 = self.norm1  # Share the same norm
        else:
            self.norm1 = get_norm(norm_type, hidden_size, eps=norm_eps, **norm_kwargs)
            self.norm2 = get_norm(norm_type, hidden_size, eps=norm_eps, **norm_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        # Normalize once and feed to both sublayers
        x_normed = self.norm1(x)
        return x + self.attn(x_normed) + self.ffn(self.norm2(x))

    def extra_repr(self) -> str:
        norm_type = self.norm1.__class__.__name__
        shared = self.norm1 is self.norm2
        return (
            f"hidden_size={self.hidden_size}, "
            f"attn={self.attn.__class__.__name__}, "
            f"ffn={self.ffn.__class__.__name__}, "
            f"norm={norm_type}, "
            f"shared_norm={shared}"
        )


# ---------------------------------------------------------------------------
# 7. ResidualStream
# ---------------------------------------------------------------------------

class ResidualStream(nn.Module):
    """Residual stream manager with optional scaling and norm tracking.

    Wraps a sequence of residual blocks and manages the residual stream
    with optional per-layer scaling and norm monitoring for debugging.

    The residual stream is the central "information highway" of a
    transformer. Monitoring its magnitude across layers helps diagnose
    training issues like vanishing/exploding gradients or unstable
    attention patterns.

    Features:
    - **Per-layer scaling**: Optional learnable or fixed scale factor
      applied after each residual block.
    - **Norm tracking**: Records the L2 norm of the residual stream after
      each layer for monitoring during training.
    - **Final normalization**: Optional normalization applied at the end
      of the stream (standard in most modern architectures).

    Args:
        hidden_size: Dimension of the input tensor.
        blocks: List of residual block modules (e.g., PreNormResidual).
        final_norm: If True, applies normalization after all blocks.
            Default: True.
        norm_type: Type of normalization for the final norm.
            Default: ``"rmsnorm"``.
        norm_eps: Epsilon for the final normalization.
        track_norms: If True, tracks and stores residual stream norms
            during the forward pass (useful for debugging). Default: False.
        scale_init: If not None, adds a learnable per-layer scale parameter
            initialized to this value. Default: None (no scaling).

    Shape:
        - Input:  ``(B, T, H)``
        - Output: ``(B, T, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        blocks: list[nn.Module],
        final_norm: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        track_norms: bool = False,
        scale_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = len(blocks)
        self.track_norms = track_norms

        # Register the block layers
        self.blocks = nn.ModuleList(blocks)

        # Optional per-layer scaling
        if scale_init is not None:
            self.layer_scales = nn.ParameterList([
                nn.Parameter(torch.full((hidden_size,), scale_init))
                for _ in range(self.num_blocks)
            ])
        else:
            self.layer_scales = None

        # Final normalization
        if final_norm:
            self.final_norm = get_norm(norm_type, hidden_size, eps=norm_eps)
        else:
            self.final_norm = None

        # Storage for norm tracking (not a parameter)
        self._norms: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        """Process the input through all residual blocks.

        Args:
            x: Input tensor of shape ``(B, T, H)``.

        Returns:
            Output tensor of shape ``(B, T, H)``.
        """
        self._norms = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            # Apply per-layer scaling if configured
            if self.layer_scales is not None:
                x = x * self.layer_scales[i]

            # Track residual stream norm for debugging
            if self.track_norms:
                with torch.no_grad():
                    norm = x.float().norm(dim=-1).mean().item()
                    self._norms.append(torch.tensor(norm, device=x.device))

        # Apply final normalization
        if self.final_norm is not None:
            x = self.final_norm(x)

        return x

    def get_norms(self) -> list[float]:
        """Return the per-layer residual stream norms from the last forward pass.

        Returns:
            List of average L2 norms (floats), one per block.

        Note:
            Only populated if ``track_norms=True`` was set during
            construction and a forward pass has been executed.
        """
        return [n.item() if isinstance(n, Tensor) else n for n in self._norms]

    def get_norm_report(self) -> str:
        """Generate a human-readable report of residual stream norms.

        Returns:
            A formatted string with per-layer norm statistics.

        Note:
            Only meaningful after a forward pass with ``track_norms=True``.
        """
        if not self._norms:
            return "ResidualStream: No norms recorded (run a forward pass first with track_norms=True)."

        norms = self.get_norms()
        lines = [f"ResidualStream: {self.num_blocks} layers"]
        lines.append(f"  Mean norm:     {sum(norms) / len(norms):.4f}")
        lines.append(f"  Max norm:      {max(norms):.4f} (layer {norms.index(max(norms))})")
        lines.append(f"  Min norm:      {min(norms):.4f} (layer {norms.index(min(norms))})")
        lines.append(f"  Norm ratio:    {max(norms) / (min(norms) + 1e-8):.4f} (max/min)")
        lines.append(f"  Per-layer:")
        for i, n in enumerate(norms):
            bar = "█" * int(n * 10)
            lines.append(f"    [{i:3d}] {n:8.4f}  {bar}")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        parts = [
            f"hidden_size={self.hidden_size}",
            f"num_blocks={self.num_blocks}",
            f"final_norm={self.final_norm is not None}",
            f"track_norms={self.track_norms}",
        ]
        if self.layer_scales is not None:
            parts.append(f"per_layer_scaling=True")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

_RESIDUAL_REGISTRY: dict[str, type[nn.Module]] = {
    "standard": StandardResidual,
    "scaled": ScaledResidual,
    "prenorm": PreNormResidual,
    "pre_norm": PreNormResidual,
    "postnorm": PostNormResidual,
    "post_norm": PostNormResidual,
    "deepnet": DeepNetResidual,
    "deep_net": DeepNetResidual,
    "parallel": ParallelResidual,
}


def get_residual(
    residual_type: str,
    hidden_size: int,
    sublayer: Optional[nn.Module] = None,
    **kwargs: Any,
) -> nn.Module:
    """Factory function to create a residual connection by name.

    For simple residual wrappers that require a single sublayer:

        >>> residual = get_residual("prenorm", 768, sublayer=my_attn, norm_type="rmsnorm")

    Args:
        residual_type: Name of the residual strategy (case-insensitive).
            Supported: ``"standard"``, ``"scaled"``, ``"prenorm"``,
            ``"postnorm"``, ``"deepnet"``, ``"parallel"``.
        hidden_size: The hidden dimension.
        sublayer: The sublayer module. Required for all types except
            ``"parallel"`` (which needs ``attn`` and ``ffn`` kwargs).
        **kwargs: Additional keyword arguments forwarded to the residual
            constructor.

    Returns:
        An ``nn.Module`` instance of the requested residual connection.

    Raises:
        ValueError: If the type is unknown or required arguments are missing.
    """
    key = residual_type.lower().replace("-", "_")
    if key not in _RESIDUAL_REGISTRY:
        available = ", ".join(sorted(set(_RESIDUAL_REGISTRY.keys())))
        raise ValueError(
            f"Unknown residual type '{residual_type}'. Available: {available}"
        )
    cls = _RESIDUAL_REGISTRY[key]
    try:
        if sublayer is not None:
            return cls(hidden_size=hidden_size, sublayer=sublayer, **kwargs)
        return cls(hidden_size=hidden_size, **kwargs)
    except TypeError as e:
        raise ValueError(
            f"Failed to instantiate {cls.__name__} with "
            f"hidden_size={hidden_size}, kwargs={kwargs}. Error: {e}"
        ) from e
