"""
Activation Functions for Nexus
=================================
Complete collection of activation functions used in modern transformer models.
Each activation is implemented as an nn.Module for consistency and to support
custom fused CUDA kernels in the future.

References:
    - ReLU: Nair & Hinton, 2010  (http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf)
    - GELU: Hendrycks & Gimpel, 2016  (https://arxiv.org/abs/1606.08415)
    - SiLU/Swish: Elfwing et al., 2017 / Ramachandran et al., 2017
    - Mish: Misra, 2019  (https://arxiv.org/abs/1908.08681)
    - SquaredReLU: Chowdhery et al., 2022 (PaLM)  (https://arxiv.org/abs/2204.02311)
    - GeGLU: Shazeer, 2020  (https://arxiv.org/abs/2002.05202)
    - ReGLU: Shazeer, 2020  (https://arxiv.org/abs/2002.05202)
    - StarReLU: Chen et al., 2023  (https://arxiv.org/abs/2303.01892)
    - QuickGELU: Pham et al., 2021 (ViT-G)  (https://arxiv.org/abs/2110.06832)
    - HardSwish: Howard et al., 2019 (MobileNetV3)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReLU(nn.Module):
    """Rectified Linear Unit.

    Formula::

        f(x) = max(0, x)

    The simplest and most widely used activation function. Zeroes out all
    negative inputs while passing positive values through unchanged.

    Args:
        inplace: If True, performs the operation in-place to save memory.

    Shape:
        - Input:  ``(*, H)`` where * is any number of dimensions.
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class GELU(nn.Module):
    """Gaussian Error Linear Unit (exact version).

    Formula::

        f(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    where Φ is the cumulative distribution function of the standard Gaussian.
    This is the exact formulation used by BERT and many early transformers.

    Args:
        approximate: If ``'tanh'``, uses the tanh approximation which is
            faster on GPU. If ``'none'`` or ``False``, uses the exact
            erf-based formula.
        eps: Small constant for numerical stability (unused in exact mode,
            only present for API consistency).

    Note:
        The tanh approximation has a maximum absolute error of ~3.5e-4
        relative to the exact version, but is significantly faster on GPU.
    """

    def __init__(self, approximate: str = "none", eps: float = 1e-6) -> None:
        super().__init__()
        self.approximate = approximate
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate='{self.approximate}'"


class GELUTanh(nn.Module):
    """GELU with tanh approximation.

    Formula::

        f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    This is a faster approximation of GELU used in GPT-2, GPT-NeoX, and
    many other models. The approximation is accurate to within ~3.5e-4
    absolute error and is significantly faster on GPU due to the absence
    of the erf special function.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(self) -> None:
        super().__init__()
        # Pre-compute constants for the tanh approximation
        self._sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        self._coef = 0.044715

    def forward(self, x: Tensor) -> Tensor:
        # Use PyTorch's built-in gelu with tanh approximation
        return F.gelu(x, approximate="tanh")

    def extra_repr(self) -> str:
        return "approximation='tanh'"


class SiLU(nn.Module):
    """Sigmoid Linear Unit (also known as Swish).

    Formula::

        f(x) = x * σ(x) = x / (1 + e^{-x})

    where σ is the sigmoid function. Used in LLaMA, Mistral, Mixtral,
    Phi, and many modern LLMs. Provides smooth, non-monotonic behavior
    that can improve gradient flow compared to ReLU.

    Args:
        inplace: If True, performs the operation in-place to save memory.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class Mish(nn.Module):
    """Mish activation function.

    Formula::

        f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    A smooth, non-monotonic, self-regularizing activation function that
    tends to work well in vision transformers and some language models.
    It is unbounded above and bounded below, which helps avoid dead neurons.

    Mish consistently matches or outperforms ReLU and Swish on a variety
    of benchmarks, particularly in deeper networks.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.mish(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class SquaredReLU(nn.Module):
    """Squared Rectified Linear Unit (from PaLM).

    Formula::

        f(x) = (max(0, x))²

    Used in the PaLM family of models (Chowdhery et al., 2022). Squaring
    the ReLU output gives a non-linear response that is stronger for larger
    positive activations while still being exactly zero for negative inputs.
    This can help the model learn more expressive functions compared to
    standard ReLU, particularly in very large-scale models.

    Shape:
        - Input:  ``(*, H)``
        - Output: ``(*, H)``, same shape as input.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.square(F.relu(x, inplace=self.inplace))

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class GatedGELU(nn.Module):
    """Gated GELU activation (GeGLU).

    Formula::

        f(x) = GELU(x @ W_gate) ⊙ (x @ W_up)

    where W_gate and W_up are learned linear projections, and ⊙ is
    element-wise multiplication. The gating mechanism allows the network
    to dynamically control the flow of information through the activation.

    Used in PaLM and other models that employ gated linear units in
    their feed-forward networks. Compared to standard GELU, the gating
    provides a multiplicative interaction that can express more complex
    functions.

    Args:
        hidden_size: Input dimension size.
        intermediate_size: Intermediate (projected) dimension size.
            If None, defaults to ``4 * hidden_size``.
        bias: If True, adds a learnable bias to the projections.
            Default: False (following PaLM).

    Shape:
        - Input:  ``(B, T, H)``
        - Output:  ``(B, T, I)``
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.W_up = nn.Linear(hidden_size, intermediate_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.W_gate(x)) * self.W_up(x)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class GatedReLU(nn.Module):
    """Gated ReLU activation (ReGLU).

    Formula::

        f(x) = ReLU(x @ W_gate) ⊙ (x @ W_up)

    A simpler variant of gated activations that uses ReLU instead of GELU
    for the gating function. ReGLU is explored alongside GeGLU and SwiGLU
    in Shazeer (2020), where it provides a good trade-off between
    computational cost and performance.

    Args:
        hidden_size: Input dimension size.
        intermediate_size: Intermediate (projected) dimension size.
            If None, defaults to ``4 * hidden_size``.
        bias: If True, adds a learnable bias to the projections.

    Shape:
        - Input:  ``(B, T, H)``
        - Output:  ``(B, T, I)``
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.W_up = nn.Linear(hidden_size, intermediate_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.W_gate(x)) * self.W_up(x)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class StarReLU(nn.Module):
    """StarReLU activation.

    Formula::

        f(x) = s * (ReLU(x))²

    where s is a learnable scale parameter initialized to a small value.
    StarReLU introduces a learned multiplicative scale that allows the
    network to modulate the activation magnitude adaptively.

    The squared ReLU term provides non-linearity, while the learnable
    scale adds an extra degree of freedom that can improve optimization
    stability and model capacity.

    Args:
        hidden_size: Dimension of the input tensor (used for scale param).
        scale_init: Initial value for the learnable scale parameter.
            Default: 0.0 (following the paper recommendation).

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = nn.Parameter(torch.full((hidden_size,), scale_init))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * torch.square(F.relu(x))

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, scale_init={self.scale.data.mean().item():.4f}"


class QuickGELU(nn.Module):
    """Quick GELU activation.

    Formula::

        f(x) = x * σ(1.702 * x)

    A fast approximation of GELU that replaces the ``erf`` or ``tanh``
    function with a simple sigmoid. The constant 1.702 was chosen to
    closely match the standard GELU function. This activation is used
    in OpenAI's CLIP and some ViT-G models for its computational efficiency.

    Note:
        Maximum absolute error vs. exact GELU is ~0.01, which is larger
        than the tanh approximation (~3.5e-4), but the computation is
        simpler and may be faster on certain hardware.

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scale = 1.702

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self._scale * x)

    def extra_repr(self) -> str:
        return f"scale={self._scale}"


class LeakyReLU(nn.Module):
    """Leaky Rectified Linear Unit.

    Formula::

        f(x) = max(α * x, x)

    where α is a small positive constant (negative slope). Unlike standard
    ReLU, LeakyReLU does not completely zero out negative inputs, which helps
    mitigate the "dying ReLU" problem where neurons can get stuck during
    training by always outputting zero.

    Args:
        negative_slope: Controls the angle of the negative slope.
            Default: 0.01.
        inplace: If True, performs the operation in-place.

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input.
    """

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}, inplace={self.inplace}"


class HardSwish(nn.Module):
    """Hard Swish activation.

    Formula::

        f(x) = x * ReLU6(x + 3) / 6

    An integer-friendly approximation of the Swish/SiLU function designed
    for efficient inference on mobile and embedded hardware. Used in
    MobileNetV3 (Howard et al., 2019).

    The ReLU6 clamping to [0, 6] and the division by 6 make this activation
    amenable to quantization and efficient fixed-point arithmetic.

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.hardswish(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation.

    Formula::

        f(x) = ReLU6(x + 3) / 6

    A piecewise linear approximation of the sigmoid function used as a
    building block for HardSwish and MobileNet models. Clamps the output
    to the range [0, 1], similar to the standard sigmoid.

    The piecewise linear form makes it efficient for hardware that lacks
    native exponential or sigmoid instructions.

    Args:
        inplace: If True, performs the operation in-place.

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input. Values in [0, 1].
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.hardsigmoid(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class ELU(nn.Module):
    """Exponential Linear Unit.

    Formula::

        f(x) = x              if x > 0
        f(x) = α * (e^x - 1)  if x ≤ 0

    ELU is similar to ReLU for positive inputs but uses an exponential
    saturation for negative values. This pushes the mean activation
    closer to zero, which can improve learning dynamics. The exponential
    saturation also makes ELU less susceptible to the "dying neuron" problem.

    Args:
        alpha: Controls the saturation value for negative inputs.
            Default: 1.0.
        inplace: If True, performs the operation in-place.

    Shape:
        - Input:  ``(*, H)``
        - Output:  ``(*, H)``, same shape as input. Range: (``-α``, ∞).
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.elu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, inplace={self.inplace}"


class GLU_SiLU(nn.Module):
    """Gated Linear Unit with SiLU activation (SwiGLU).

    Formula::

        f(x) = SiLU(x @ W_gate) ⊙ (x @ W_up)

    The SiLU-gated variant of GLU, popularized by LLaMA, Mistral,
    and many modern LLM architectures. SwiGLU consistently outperforms
    ReLU-2 and GELU-2 FFN variants across model sizes and training
    regimes, making it the default choice for new model designs.

    Args:
        hidden_size: Input dimension size.
        intermediate_size: Intermediate (projected) dimension size.
            If None, defaults to ``4 * hidden_size * 2 // 3`` (following
            LLaMA convention for SwiGLU, which accounts for the gate).
        bias: If True, adds a learnable bias to the projections.
            Default: False (following LLaMA/Mistral convention).
        _pack_weights: If True, uses a single packed weight matrix that is
            split in half for the gate and up projections, saving memory.

    Shape:
        - Input:  ``(B, T, H)``
        - Output:  ``(B, T, I)``

    Note:
        When using SwiGLU in an FFN context, the effective intermediate
        size is typically ``4 * hidden_size * 2 // 3`` to keep the total
        parameter count comparable to a standard GELU FFN with
        ``4 * hidden_size`` intermediate size (since SwiGLU has two
        projection matrices).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if intermediate_size is None:
            # Default following LLaMA convention: 2/3 * 4H, rounded to
            # nearest multiple of a reasonable alignment (256)
            intermediate_size = (4 * hidden_size * 2) // 3
            intermediate_size = ((intermediate_size + 255) // 256) * 256

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.W_up = nn.Linear(hidden_size, intermediate_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.W_gate(x)) * self.W_up(x)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


# ---------------------------------------------------------------------------
# Convenience mapping: name -> class
# ---------------------------------------------------------------------------
ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": ReLU,
    "gelu": GELU,
    "gelu_tanh": GELUTanh,
    "silu": SiLU,
    "swish": SiLU,  # Alias: Swish == SiLU
    "mish": Mish,
    "squared_relu": SquaredReLU,
    "geglu": GatedGELU,
    "reglu": GatedReLU,
    "star_relu": StarReLU,
    "quick_gelu": QuickGELU,
    "leaky_relu": LeakyReLU,
    "hard_swish": HardSwish,
    "hard_sigmoid": HardSigmoid,
    "elu": ELU,
    "swiglu": GLU_SiLU,
}


def get_activation(
    name: str,
    hidden_size: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create an activation module by name.

    For simple pointwise activations (e.g. ``"relu"``, ``"silu"``), only the
    name is needed. For gated activations (e.g. ``"geglu"``, ``"swiglu"``),
    the ``hidden_size`` must be provided.

    Args:
        name: Case-insensitive name of the activation function. Supported
            values are keys of ``ACTIVATION_REGISTRY``.
        hidden_size: Required for gated activations (GeGLU, ReGLU, SwiGLU,
            StarReLU).
        intermediate_size: Optional intermediate size for gated activations.
        **kwargs: Additional keyword arguments passed to the activation
            constructor (e.g. ``inplace=True``, ``bias=True``).

    Returns:
        An ``nn.Module`` instance of the requested activation.

    Raises:
        ValueError: If ``name`` is not recognized or required arguments
            are missing.

    Examples::

        >>> act = get_activation("silu")
        >>> act = get_activation("swiglu", hidden_size=768)
        >>> act = get_activation("gelu", approximate="tanh")
    """
    key = name.lower().replace("-", "_")
    if key not in ACTIVATION_REGISTRY:
        available = ", ".join(sorted(ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown activation '{name}'. Available: {available}"
        )
    cls = ACTIVATION_REGISTRY[key]
    try:
        if hidden_size is not None:
            return cls(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                **kwargs,
            )
        return cls(**kwargs)
    except TypeError as e:
        raise ValueError(
            f"Failed to instantiate {cls.__name__} with kwargs={kwargs}. "
            f"Error: {e}"
        ) from e
