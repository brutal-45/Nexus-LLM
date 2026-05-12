"""
LoRA / QLoRA / DoRA Fine-Tuning Infrastructure
================================================

This module provides a complete implementation of Low-Rank Adaptation (LoRA) and its
variants for parameter-efficient fine-tuning of large language models.

Mathematical Background
-----------------------

**LoRA (Low-Rank Adaptation):**
    For a pre-trained weight matrix :math:`W_0 \\in \\mathbb{R}^{d \\times k}`, the
    update is decomposed into two low-rank matrices:

    .. math::
        W = W_0 + \\Delta W = W_0 + B A

    where :math:`B \\in \\mathbb{R}^{d \\times r}`, :math:`A \\in \\mathbb{R}^{r \\times k}`,
    and :math:`r \\ll \\min(d, k)`. The scaled output during forward pass is:

    .. math::
        h = W_0 x + \\frac{\\alpha}{r} B A x

    Only :math:`A` and :math:`B` are trainable, reducing parameters by a factor of
    :math:`\\frac{2dk}{rk + dk} \\approx \\frac{2}{r}` for large :math:`r`.

**DoRA (Weight-Decomposed Low-Rank Adaptation):**
    Decomposes the weight into magnitude and directional components:

    .. math::
        W' = m \\cdot \\frac{W_0 + BA}{\\|W_0 + BA\\|}

    where :math:`m = \\|W_0\\|` is the frozen magnitude vector. This provides
    learning behaviour closer to full fine-tuning while maintaining LoRA's
    parameter efficiency.

**QLoRA (Quantized LoRA):**
    Quantizes the base weights to 4-bit NormalFloat (NF4) representation:

    .. math::
        W_{\\text{quant}} = \\text{round}\\left(\\frac{W}{s}\\right), \\quad
        W_{\\text{dequant}} = W_{\\text{quant}} \\cdot s

    where :math:`s` is the per-group scale factor. The NF4 quantization levels are
    computed using information-theoretic optimal quantiles of an assumed normal
    weight distribution.

References
----------
    - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
    - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024.
    - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023.
"""

from __future__ import annotations

import copy
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Configuration for LoRA / QLoRA / DoRA fine-tuning.

    Attributes:
        rank: The rank :math:`r` of the low-rank decomposition. Higher rank
            increases capacity but also parameter count.
        alpha: The scaling factor :math:`\\alpha`. The effective scaling applied
            to the LoRA update is :math:`\\alpha / r`.
        dropout: Dropout probability applied to the input of the LoRA branch
            before the :math:`A` projection.
        bias: Whether to include a bias term in the LoRA output path.
        target_modules: List of module name substrings to target for adaptation.
            Common targets for LLMs include attention projections and MLP layers.
        use_dora: If True, use DoRA (Weight-Decomposed LoRA) instead of standard
            LoRA. DoRA decomposes updates into magnitude and direction components.
        quantize_base: If True, quantize the base model weights using the
            specified quantization type (enables QLoRA).
        quantization_type: The quantization format for base weights. Supported
            values: ``"nf4"`` (4-bit NormalFloat), ``"int8"``, ``"fp4"``.
        double_quantization: If True, apply a second round of quantization to
            the quantization constants themselves, further reducing memory usage.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    bias: bool = False
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_dora: bool = False
    quantize_base: bool = False
    quantization_type: str = "nf4"  # nf4, int8, fp4
    double_quantization: bool = False

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.quantization_type not in ("nf4", "int8", "fp4"):
            raise ValueError(
                f"quantization_type must be one of 'nf4', 'int8', 'fp4', "
                f"got '{self.quantization_type}'"
            )


# ---------------------------------------------------------------------------
# NF4 Quantizer
# ---------------------------------------------------------------------------

class NF4Quantizer:
    """4-bit NormalFloat quantizer for QLoRA.

    Implements the NF4 data type as described in Dettmers et al. (2023). The
    quantization levels are chosen as the optimal quantiles of a standard normal
    distribution :math:`\\mathcal{N}(0, 1)`.

    The 16 quantization levels :math:`\\{q_i\\}_{i=0}^{15}` are computed by:

    .. math::
        q_i = \\Phi^{-1}\\left(\\frac{i + 0.5}{16}\\right)

    where :math:`\\Phi^{-1}` is the inverse CDF (probit function) of the standard
    normal distribution. These levels are then normalized to :math:`[-1, 1]`.

    Weights are quantized using **absmax symmetric quantization** per group:

    .. math::
        s = \\frac{\\max|W_G|}{8}, \\quad
        W_{\\text{idx}} = \\text{round}\\left(\\frac{W_G}{s} + 7.5\\right), \\quad
        W_{\\text{dequant}} = q[W_{\\text{idx}}] \\cdot s

    where :math:`G` is a group of columns and :math:`s` is the group scale.

    Double quantization further compresses the scale factors by quantizing them
    to 8-bit integers with a shared global scale.
    """

    NUM_BITS: int = 4
    NUM_LEVELS: int = 16  # 2^4
    BLOCK_SIZE: int = 64

    def __init__(
        self,
        block_size: int = 64,
        double_quantization: bool = False,
    ) -> None:
        self.block_size = block_size
        self.double_quantization = double_quantization
        # Pre-compute the 16 NF4 quantization levels
        self.levels = self._compute_nf4_levels()

    @staticmethod
    def _compute_nf4_levels() -> torch.Tensor:
        """Compute the 16 NF4 quantization levels.

        The levels are the expected values of 16 equal-probability bins of
        :math:`\\mathcal{N}(0, 1)`, normalized so that the most extreme values
        are :math:`\\pm 1`.

        Returns:
            Tensor of shape (16,) containing the quantization levels.
        """
        # Use inverse normal CDF at bin centres (i + 0.5) / 16
        levels = []
        for i in range(16):
            # Prevent exact 0 or 1 which would give inf in erfinv
            p = (i + 0.5) / 16.0
            p = max(p, 1e-6)
            p = min(p, 1.0 - 1e-6)
            # Approximate Phi^{-1}(p) using torch's erfinv
            # Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
            level = math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)
            levels.append(level)
        levels = torch.tensor(levels, dtype=torch.float32)
        # Normalize to [-1, 1]
        levels = levels / levels.abs().max()
        return levels

    def quantize(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Quantize a weight matrix to NF4.

        Args:
            weight: Original FP32/FP16 weight tensor of shape
                ``(out_features, in_features)``.

        Returns:
            A tuple of:
            - **q_weight**: 4-bit packed quantized weights, shape
              ``(out_features, in_features // 2)`` with dtype ``torch.uint8``.
            - **scales**: Per-group scale factors of shape
              ``(out_features, num_groups)``.
            - **double_quant**: If double quantization is enabled, a tuple of
              ``(q_scales, dq_scale, dq_zero)`` for the nested quantization.
              Otherwise ``None``.
        """
        original_dtype = weight.dtype
        weight = weight.float()
        out_features, in_features = weight.shape
        num_groups = (in_features + self.block_size - 1) // self.block_size

        # Pad to make divisible by block_size if needed
        padded_in = num_groups * self.block_size
        if padded_in > in_features:
            weight = F.pad(weight, (0, padded_in - in_features))

        # Reshape into groups: (out_features, num_groups, block_size)
        reshaped = weight.view(out_features, num_groups, self.block_size)

        # Compute per-group absmax scales
        # s = max|W_G| / 8  (8 = 2^(bits-1) for signed 4-bit)
        absmax = reshaped.abs().amax(dim=-1)  # (out_features, num_groups)
        scales = absmax / 8.0
        # Avoid division by zero
        scales = scales.clamp(min=1e-8)

        # Normalize and map to discrete levels
        normalized = reshaped / scales.unsqueeze(-1)  # (out, groups, block)

        # Find nearest NF4 level by computing distances
        # normalized: (out, groups, block), levels: (16,)
        expanded_levels = self.levels.view(1, 1, 16).expand(
            out_features, num_groups, 16
        )
        distances = (normalized.unsqueeze(-1) - expanded_levels).abs()
        indices = distances.argmin(dim=-1)  # (out, groups, block)

        # Pack 4-bit values into uint8: two indices per byte
        # High nibble = indices[:, :, 0::2], Low nibble = indices[:, :, 1::2]
        indices_padded = indices  # (out, groups, block)
        # Reshape to (out, padded_in) for packing
        indices_flat = indices_padded.reshape(out_features, padded_in)
        high_nibbles = indices_flat[:, 0::2]  # (out, padded_in // 2)
        low_nibbles = indices_flat[:, 1::2]

        q_weight = (high_nibbles << 4) | low_nibbles  # pack two 4-bit into one 8-bit
        q_weight = q_weight.to(torch.uint8)

        # Double quantization of scales
        double_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.double_quantization:
            double_quant = self._double_quantize_scales(scales)

        # Move scales to original device
        scales = scales.to(original_dtype)

        return q_weight, scales, double_quant

    def _double_quantize_scales(
        self, scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 8-bit quantization to the scale factors.

        Quantizes per-group scales to FP8 using absmax quantization with a
        global scale, further reducing memory footprint.

        Args:
            scales: Scale tensor of shape ``(out_features, num_groups)``.

        Returns:
            Tuple of (quantized scales, global scale factor).
        """
        flat_scales = scales.float().flatten()
        global_scale = flat_scales.abs().max().clamp(min=1e-8)
        q_scales = torch.round(flat_scales / global_scale * 127.0).clamp(-128, 127)
        q_scales = q_scales.to(torch.int8)
        q_scales = q_scales.view(scales.shape)
        global_scale = global_scale.to(scales.dtype)
        return q_scales, global_scale

    def _double_dequantize_scales(
        self,
        q_scales: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize double-quantized scales.

        Args:
            q_scales: Quantized scales (int8).
            global_scale: Global scale factor.

        Returns:
            Dequantized scale tensor in FP32.
        """
        return q_scales.float() / 127.0 * global_scale.float()

    def dequantize(
        self,
        q_weight: torch.Tensor,
        scales: torch.Tensor,
        double_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        original_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """Dequantize NF4 packed weights back to floating point.

        Args:
            q_weight: Packed 4-bit weights of shape
                ``(out_features, in_features // 2)``, dtype ``uint8``.
            scales: Per-group scale factors of shape
                ``(out_features, num_groups)``.
            double_quant: Optional nested quantization info for scales.
            original_shape: If provided, the output is reshaped to this size
                (to strip any padding applied during quantization).

        Returns:
            Dequantized weight tensor in FP32.
        """
        if double_quant is not None:
            q_scales, dq_global_scale = double_quant
            scales = self._double_dequantize_scales(q_scales, dq_global_scale)

        device = q_weight.device
        out_features = q_weight.shape[0]
        packed_cols = q_weight.shape[1]
        in_features = packed_cols * 2

        # Unpack bytes into 4-bit indices
        high_nibbles = (q_weight >> 4) & 0xF  # (out, packed)
        low_nibbles = q_weight & 0xF
        # Interleave: index 0 from high, index 0 from low, index 1 from high, ...
        indices = torch.stack([high_nibbles, low_nibbles], dim=-1).reshape(
            out_features, in_features
        )

        # Look up NF4 levels
        # Ensure levels are on the same device
        levels = self.levels.to(device)
        weight = levels[indices]  # (out_features, in_features)

        # Apply per-group scaling
        num_groups = scales.shape[1]
        block_size = in_features // num_groups if num_groups > 0 else in_features
        if block_size == 0:
            block_size = self.block_size

        # Reshape for broadcasting: weight (out, in) / scales (out, groups)
        reshaped_weight = weight.view(out_features, num_groups, block_size)
        reshaped_scales = scales.float().unsqueeze(-1)
        weight = reshaped_weight * reshaped_scales
        weight = weight.reshape(out_features, in_features)

        if original_shape is not None:
            weight = weight[:, : original_shape[1]]

        return weight


# ---------------------------------------------------------------------------
# LoRA Linear Layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """LoRA-adapted linear layer.

    Wraps an existing ``nn.Linear`` layer with low-rank adapters:

    .. math::
        y = W_0 x + \\frac{\\alpha}{r} \\, \\text{drop}(x) A^T B

    where :math:`W_0` is the frozen pre-trained weight, :math:`A \\in
    \\mathbb{R}^{r \\times d_{\\text{in}}}`, :math:`B \\in \\mathbb{R}^{d_{\\text{out}} \\times r}`,
    and :math:`\\text{drop}` is an optional dropout.

    Initialization:
        - :math:`A` is initialized with Kaiming uniform (He init).
        - :math:`B` is initialized with zeros so the initial adaptation is
          the identity: :math:`\\Delta W = B A = 0`.

    Args:
        original_linear: The base ``nn.Linear`` layer to wrap.
        rank: Rank :math:`r` of the low-rank decomposition.
        alpha: Scaling factor :math:`\\alpha`.
        dropout: Dropout probability for the LoRA input path.
        bias: If True, add a learnable bias to the LoRA output.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear = original_linear
        out_features: int = original_linear.out_features
        in_features: int = original_linear.in_features
        self.scaling: float = alpha / rank

        # Freeze the base linear layer
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Optional bias on the LoRA path
        if bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("lora_bias", None)

        # Store config for serialization
        self.rank = rank
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + scaled LoRA output.

        .. math::
            y = W_0 x + \\frac{\\alpha}{r} B \\, A \\, \\text{drop}(x)

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        base_output = self.linear(x)
        lora_input = self.lora_dropout(x)
        lora_output = self.lora_B(self.lora_A(lora_input)) * self.scaling

        if self.lora_bias is not None:
            lora_output = lora_output + self.lora_bias

        return base_output + lora_output


# ---------------------------------------------------------------------------
# DoRA Linear Layer
# ---------------------------------------------------------------------------

class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) linear layer.

    DoRA decomposes the weight matrix into a learnable direction and a frozen
    magnitude:

    .. math::
        W' = m \\cdot \\frac{W_0 + BA}{\\|W_0 + BA\\|_c}

    where:
        - :math:`m = \\|W_0\\|_c` is the column-wise magnitude (frozen).
        - :math:`W_0 + BA` is the weight matrix with LoRA update.
        - :math:`\\|\\cdot\\|_c` denotes column-wise L2 norm.

    This decomposes the learning into two components:
        1. **Directional update**: controlled by LoRA matrices :math:`A` and :math:`B`.
        2. **Magnitude**: optionally learnable; defaults to frozen :math:`m`.

    DoRA has been shown to produce weight updates closer to full fine-tuning
    while retaining the parameter efficiency of LoRA.

    Args:
        original_linear: The base ``nn.Linear`` layer to wrap.
        rank: Rank :math:`r` of the low-rank decomposition.
        alpha: Scaling factor :math:`\\alpha`.
        dropout: Dropout probability for the LoRA input path.
        bias: If True, add a learnable bias to the output.
        learnable_magnitude: If True, make the magnitude vector trainable.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        bias: bool = False,
        learnable_magnitude: bool = False,
    ) -> None:
        super().__init__()
        self.linear = original_linear
        out_features: int = original_linear.out_features
        in_features: int = original_linear.in_features
        self.scaling: float = alpha / rank

        # Freeze the base linear layer
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        # LoRA matrices (same as standard LoRA)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Precompute frozen magnitude m = ||W_0||_c (column-wise L2 norm)
        # Shape: (out_features, 1)
        with torch.no_grad():
            self.register_buffer(
                "magnitude",
                self.linear.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8),
            )
            self.register_buffer(
                "weight_normalization_const",
                self.linear.weight.norm(dim=1, keepdim=True).clamp(min=1e-8),
            )

        # Optionally make magnitude learnable
        if learnable_magnitude:
            self.magnitude = nn.Parameter(
                self.linear.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            )

        # Optional bias
        if bias:
            self.output_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("output_bias", None)

        self.rank = rank
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight decomposition.

        .. math::
            y = m \\cdot \\text{norm}(W_0 + BA) \\cdot x

        where :math:`\\text{norm}(\\cdot)` normalizes each row and :math:`m` is
        the frozen (or learnable) column-wise magnitude.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        # Compute the updated weight: W_0 + (alpha/r) * B @ A
        lora_weight = (
            self.lora_B.weight @ self.lora_A.weight
        ) * self.scaling
        merged_weight = self.linear.weight + lora_weight  # (out, in)

        # Normalize rows of merged weight
        merged_norm = merged_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized_weight = merged_weight / merged_norm

        # Apply magnitude scaling
        scaled_weight = self.magnitude * normalized_weight

        # Compute output
        output = F.linear(x, scaled_weight, self.linear.bias)

        if self.output_bias is not None:
            output = output + self.output_bias

        return output


# ---------------------------------------------------------------------------
# QLoRA Linear Layer
# ---------------------------------------------------------------------------

class QLoRALinear(nn.Module):
    """Quantized-LoRA linear layer (QLoRA).

    Combines 4-bit NormalFloat quantization of base weights with LoRA adapters.
    The base weights are quantized offline and dequantized on-the-fly during
    the forward pass:

    .. math::
        y = \\text{dequant}(W_q, s) \\cdot x + \\frac{\\alpha}{r} B A x

    where :math:`W_q` is the NF4-quantized weight and :math:`s` is the
    per-group scale factor.

    Optional double quantization further compresses the scale factors:

    .. math::
        s = s_{\\text{global}} \\cdot \\frac{s_q}{127}

    Args:
        original_linear: The base ``nn.Linear`` layer (will be quantized).
        rank: Rank :math:`r` of the LoRA decomposition.
        alpha: Scaling factor :math:`\\alpha`.
        dropout: Dropout probability for the LoRA path.
        quantizer: NF4 quantizer instance.
        bias: If True, add a learnable bias to the LoRA output.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        quantizer: NF4Quantizer,
        bias: bool = False,
    ) -> None:
        super().__init__()
        out_features: int = original_linear.out_features
        in_features: int = original_linear.in_features
        self.scaling: float = alpha / rank
        self.in_features = in_features
        self.out_features = out_features

        # Quantize the base weight
        with torch.no_grad():
            original_shape = original_linear.weight.shape
            q_weight, scales, double_quant = quantizer.quantize(
                original_linear.weight.data
            )
            self.register_buffer("q_weight", q_weight)
            self.register_buffer("scales", scales)
            if double_quant is not None:
                q_scales, dq_global_scale = double_quant
                self.register_buffer("q_scales", q_scales)
                self.register_buffer("dq_global_scale", dq_global_scale)
            else:
                self.register_buffer("q_scales", None)
                self.register_buffer("dq_global_scale", None)
            self.original_shape = original_shape
            self.block_size = quantizer.block_size
            self.num_groups = scales.shape[1]

        # Store the original bias if present (kept in full precision)
        if original_linear.bias is not None:
            self.register_buffer("base_bias", original_linear.bias.data.clone())
        else:
            self.register_buffer("base_bias", None)

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Optional LoRA bias
        if bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("lora_bias", None)

        self.rank = rank
        self.alpha = alpha

        # Keep a reference to the quantizer for dequantization
        self._quantizer = quantizer

    def _dequantize_base_weight(self) -> torch.Tensor:
        """Dequantize the stored NF4 weights back to floating point.

        Returns:
            Dequantized weight tensor of shape ``(out_features, in_features)``.
        """
        double_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.q_scales is not None and self.dq_global_scale is not None:
            double_quant = (self.q_scales, self.dq_global_scale)

        return self._quantizer.dequantize(
            self.q_weight, self.scales, double_quant, self.original_shape
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized base + LoRA.

        .. math::
            y = \\text{dequant}(W_q, s) \\cdot x + \\frac{\\alpha}{r} B A \\, \\text{drop}(x)

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        # Dequantize base weight and compute base output
        base_weight = self._dequantize_base_weight()
        base_output = F.linear(x, base_weight, self.base_bias)

        # LoRA branch
        lora_input = self.lora_dropout(x)
        lora_output = self.lora_B(self.lora_A(lora_input)) * self.scaling

        if self.lora_bias is not None:
            lora_output = lora_output + self.lora_bias

        return base_output + lora_output


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def find_target_modules(
    model: nn.Module,
    target_names: Sequence[str],
) -> List[Tuple[str, nn.Linear]]:
    """Find all ``nn.Linear`` modules whose names contain any of the target strings.

    Walks the model's module tree and collects linear layers whose fully-qualified
    names contain at least one substring from ``target_names``.

    Args:
        model: The model to search through.
        target_names: List of name fragments to match against (e.g.,
            ``["q_proj", "v_proj"]``).

    Returns:
        List of ``(full_module_name, linear_module)`` tuples for all matches,
        sorted by name.
    """
    targets: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_names:
                if target in name:
                    targets.append((name, module))
                    break
    targets.sort(key=operator.itemgetter(0))
    return targets


def create_lora_config_from_dict(config_dict: Dict[str, Any]) -> LoRAConfig:
    """Create a :class:`LoRAConfig` from a dictionary (e.g., loaded JSON).

    Args:
        config_dict: Dictionary with keys matching :class:`LoRAConfig` fields.

    Returns:
        A fully constructed :class:`LoRAConfig` instance.

    Example::

        >>> cfg = create_lora_config_from_dict({
        ...     "rank": 32, "alpha": 64, "use_dora": True
        ... })
        >>> cfg.rank
        32
    """
    valid_fields = {
        f.name for f in LoRAConfig.__dataclass_fields__.values()
    }
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return LoRAConfig(**filtered)


def print_lora_info(model: nn.Module) -> str:
    """Compute and return a human-readable summary of LoRA-adapted modules.

    For each LoRA-adapted layer, reports the layer name, rank, shapes of
    ``lora_A`` and ``lora_B``, and the number of trainable parameters.

    Args:
        model: The (potentially LoRA-wrapped) model.

    Returns:
        A formatted multi-line string with LoRA statistics.
    """
    lines: List[str] = []
    total_lora_params = 0
    total_base_params = 0
    lora_layer_count = 0

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, DoRALinear, QLoRALinear)):
            lora_layer_count += 1
            a_params = sum(p.numel() for p in module.lora_A.parameters())
            b_params = sum(p.numel() for p in module.lora_B.parameters())
            lora_params = a_params + b_params
            total_lora_params += lora_params

            if isinstance(module, LoRALinear):
                base_params = module.linear.weight.numel()
                lines.append(
                    f"  [LoRA] {name}: "
                    f"rank={module.rank}, "
                    f"A=({module.lora_A.in_features}, {module.lora_A.out_features}), "
                    f"B=({module.lora_B.in_features}, {module.lora_B.out_features}), "
                    f"lora_params={lora_params:,}"
                )
            elif isinstance(module, DoRALinear):
                base_params = module.linear.weight.numel()
                lines.append(
                    f"  [DoRA] {name}: "
                    f"rank={module.rank}, "
                    f"A=({module.lora_A.in_features}, {module.lora_A.out_features}), "
                    f"B=({module.lora_B.in_features}, {module.lora_B.out_features}), "
                    f"lora_params={lora_params:,}"
                )
            elif isinstance(module, QLoRALinear):
                base_params = module.out_features * module.in_features
                lines.append(
                    f"  [QLoRA] {name}: "
                    f"rank={module.rank}, "
                    f"quantized_base={module.out_features}x{module.in_features}, "
                    f"lora_params={lora_params:,}"
                )
            total_base_params += base_params

    header = "=" * 72
    summary = [
        header,
        " LoRA Fine-Tuning Summary",
        header,
        f"  Total LoRA-adapted layers : {lora_layer_count}",
        f"  Total LoRA parameters     : {total_lora_params:,}",
        f"  Total base parameters     : {total_base_params:,}",
        f"  Parameter ratio (LoRA/Base): "
        f"{total_lora_params / max(total_base_params, 1) * 100:.4f}%",
    ]
    if lines:
        summary.append("  Adapted layers:")
        summary.extend(lines)
    summary.append(header)

    text = "\n".join(summary)
    print(text)
    return text


# ---------------------------------------------------------------------------
# apply_lora_to_model
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """Replace target ``nn.Linear`` layers with LoRA / DoRA / QLoRA variants.

    Walks the model's module tree, identifies linear layers whose names match
    the ``config.target_modules`` list, and replaces each with the appropriate
    LoRA-adapted layer. The base model weights are frozen; only LoRA parameters
    remain trainable.

    Args:
        model: The base model to adapt.
        config: :class:`LoRAConfig` controlling rank, alpha, quantization, etc.

    Returns:
        The modified model with LoRA adapters injected.

    Raises:
        ValueError: If ``config.quantize_base`` is True but ``config.quantization_type``
            is not ``"nf4"``.
    """
    model = copy.deepcopy(model)

    # Build a mapping of name -> parent, attr_name for replacement
    targets = find_target_modules(model, config.target_modules)

    quantizer: Optional[NF4Quantizer] = None
    if config.quantize_base:
        if config.quantization_type != "nf4":
            raise ValueError(
                f"Only 'nf4' quantization is supported, got '{config.quantization_type}'"
            )
        quantizer = NF4Quantizer(
            block_size=NF4Quantizer.BLOCK_SIZE,
            double_quantization=config.double_quantization,
        )

    for full_name, linear_module in targets:
        # Navigate to the parent module
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = parts[0]

        if config.use_dora:
            replacement = DoRALinear(
                original_linear=linear_module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                bias=config.bias,
            )
        elif config.quantize_base and quantizer is not None:
            replacement = QLoRALinear(
                original_linear=linear_module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                quantizer=quantizer,
                bias=config.bias,
            )
        else:
            replacement = LoRALinear(
                original_linear=linear_module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                bias=config.bias,
            )

        setattr(parent, attr_name, replacement)

    return model


# ---------------------------------------------------------------------------
# LoRA State Dict Utilities
# ---------------------------------------------------------------------------

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from a LoRA-wrapped model.

    Scans all modules and collects parameters from :class:`LoRALinear`,
    :class:`DoRALinear`, and :class:`QLoRALinear` instances.

    Args:
        model: The LoRA-wrapped model.

    Returns:
        A state dict mapping parameter names (prefixed with module path) to
        tensors, containing only LoRA-specific parameters.
    """
    sd: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            sd[name] = param.data
    return sd


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load LoRA weights into a LoRA-wrapped model.

    Args:
        model: The LoRA-wrapped model to load weights into.
        state_dict: State dict containing LoRA parameter mappings.
        strict: If True, raise an error if any keys in ``state_dict`` don't
            match the model's LoRA parameters.
    """
    model_state = lora_state_dict(model)
    if strict:
        missing = set(model_state.keys()) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(model_state.keys())
        if missing:
            raise KeyError(f"Missing keys in state dict: {missing}")
        if unexpected:
            raise KeyError(f"Unexpected keys in state dict: {unexpected}")

    for name, param in model.named_parameters():
        if param.requires_grad and name in state_dict:
            param.data.copy_(state_dict[name])


# ---------------------------------------------------------------------------
# merge_lora_weights
# ---------------------------------------------------------------------------

def merge_lora_weights(
    model: nn.Module,
    config: Optional[LoRAConfig] = None,
) -> nn.Module:
    """Merge LoRA adapters into the base model weights.

    Performs the in-place merge:

    .. math::
        W_{\\text{merged}} = W_0 + \\frac{\\alpha}{r} B A

    For DoRA layers, the merge reverses the weight decomposition before applying.
    For QLoRA layers, the base weights are first dequantized and then merged.

    After merging, all LoRA parameters are removed and the model reverts to
    a standard architecture with updated weights.

    Args:
        model: The LoRA-wrapped model.
        config: Optional :class:`LoRAConfig`. If provided, used to determine
            which layers to merge.

    Returns:
        The model with LoRA weights merged into base weights.
    """
    model = copy.deepcopy(model)

    # Collect modules to replace
    replacements: List[Tuple[str, nn.Module]] = []

    for full_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                merged_weight = (
                    module.linear.weight.data
                    + (module.lora_B.weight @ module.lora_A.weight) * module.scaling
                )
                new_linear = nn.Linear(
                    module.linear.in_features,
                    module.linear.out_features,
                    bias=module.linear.bias is not None,
                )
                new_linear.weight.data.copy_(merged_weight)
                if module.linear.bias is not None:
                    new_linear.bias.data.copy_(module.linear.bias.data)
                if module.lora_bias is not None:
                    if new_linear.bias is None:
                        new_linear.bias = nn.Parameter(
                            torch.zeros(module.linear.out_features)
                        )
                    new_linear.bias.data.add_(module.lora_bias.data)
            replacements.append((full_name, new_linear))

        elif isinstance(module, DoRALinear):
            with torch.no_grad():
                lora_weight = (
                    module.lora_B.weight @ module.lora_A.weight
                ) * module.scaling
                merged_weight = module.linear.weight + lora_weight
                # Apply the magnitude from DoRA
                merged_norm = merged_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
                final_weight = module.magnitude * (merged_weight / merged_norm)

                new_linear = nn.Linear(
                    module.linear.in_features,
                    module.linear.out_features,
                    bias=module.linear.bias is not None,
                )
                new_linear.weight.data.copy_(final_weight.squeeze())
                if module.linear.bias is not None:
                    new_linear.bias.data.copy_(module.linear.bias.data)
                if module.output_bias is not None:
                    if new_linear.bias is None:
                        new_linear.bias = nn.Parameter(
                            torch.zeros(module.linear.out_features)
                        )
                    new_linear.bias.data.add_(module.output_bias.data)
            replacements.append((full_name, new_linear))

        elif isinstance(module, QLoRALinear):
            with torch.no_grad():
                base_weight = module._dequantize_base_weight()
                lora_weight = (
                    module.lora_B.weight @ module.lora_A.weight
                ) * module.scaling
                merged_weight = base_weight + lora_weight

                new_linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.base_bias is not None,
                )
                new_linear.weight.data.copy_(merged_weight)
                if module.base_bias is not None:
                    new_linear.bias.data.copy_(module.base_bias)
                if module.lora_bias is not None:
                    if new_linear.bias is None:
                        new_linear.bias = nn.Parameter(
                            torch.zeros(module.out_features)
                        )
                    new_linear.bias.data.add_(module.lora_bias.data)
            replacements.append((full_name, new_linear))

    # Perform replacements
    for full_name, new_module in replacements:
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = parts[0]
        setattr(parent, attr_name, new_module)

    return model


# ---------------------------------------------------------------------------
# LoRAWrapper
# ---------------------------------------------------------------------------

class LoRAWrapper(nn.Module):
    """Training wrapper that manages LoRA-adapted models.

    This wrapper handles:
        - Freezing all base model parameters.
        - Exposing only LoRA parameters for gradient computation.
        - Saving and loading LoRA adapter weights.
        - Computing parameter statistics.

    Args:
        model: The base (pre-trained) model.
        config: :class:`LoRAConfig` controlling the LoRA adaptation.

    Example::

        >>> wrapper = LoRAWrapper(base_model, LoRAConfig(rank=8, alpha=16))
        >>> trainable = wrapper.get_trainable_parameters()
        >>> for name, param in trainable:
        ...     print(name, param.shape)
    """

    def __init__(self, model: nn.Module, config: LoRAConfig) -> None:
        super().__init__()
        self.config = config
        # Apply LoRA to the model
        self.model = apply_lora_to_model(model, config)

    @property
    def device(self) -> torch.device:
        """Return the device of the first LoRA parameter."""
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def get_trainable_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Return all LoRA parameters (requires_grad=True).

        Returns:
            List of ``(name, parameter)`` tuples for all trainable parameters.
        """
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]

    def get_frozen_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Return all frozen (base model) parameters.

        Returns:
            List of ``(name, parameter)`` tuples for frozen parameters.
        """
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if not param.requires_grad
        ]

    def save_lora_adapter(self, path: str) -> None:
        """Save only LoRA adapter weights to a file.

        Uses :func:`torch.save` to write the LoRA state dict along with
        the configuration.

        Args:
            path: File path for the saved adapter weights.
        """
        state = {
            "lora_state_dict": lora_state_dict(self.model),
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "bias": self.config.bias,
                "target_modules": self.config.target_modules,
                "use_dora": self.config.use_dora,
                "quantize_base": self.config.quantize_base,
                "quantization_type": self.config.quantization_type,
                "double_quantization": self.config.double_quantization,
            },
        }
        torch.save(state, path)

    def load_lora_adapter(self, path: str) -> None:
        """Load LoRA adapter weights from a file.

        Args:
            path: File path containing saved adapter weights.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        sd = state["lora_state_dict"]
        load_lora_state_dict(self.model, sd)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass delegated to the wrapped model."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model.

        This allows accessing model attributes like ``model.embed_tokens``
        through the wrapper.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ---------------------------------------------------------------------------
# LoRA Fine-Tuner
# ---------------------------------------------------------------------------

class LoRAFineTuner:
    """Complete fine-tuning orchestrator for LoRA-adapted models.

    Manages the training loop for LoRA fine-tuning, handling:
        - Optimizer creation with only LoRA parameters.
        - Single training step (forward, backward, optimizer step).
        - Memory and parameter statistics.
        - Checkpointing of LoRA weights.

    Args:
        model: The base model to fine-tune.
        lora_config: :class:`LoRAConfig` controlling the adaptation.
        optimizer_cls: Name of the optimizer class. Supported values:
            ``"adamw"``, ``"adam"``, ``"sgd"``.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        device: Device to place the model on. If ``None``, uses the model's
            current device.

    Example::

        >>> finetuner = LoRAFineTuner(model, LoRAConfig(rank=16), lr=1e-4)
        >>> loss = finetuner.train_step({"input_ids": ids, "labels": labels})
        >>> finetuner.save_checkpoint("adapter.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        optimizer_cls: str = "adamw",
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.config = lora_config
        self.lr = lr
        self.weight_decay = weight_decay

        # Create the LoRA wrapper
        self.wrapper = LoRAWrapper(model, lora_config)

        # Move to device
        if device is not None:
            self.wrapper = self.wrapper.to(device)

        # Create optimizer with only LoRA parameters
        trainable_params = self.wrapper.get_trainable_parameters()
        param_groups = [p for _, p in trainable_params]

        self.optimizer = self._create_optimizer(optimizer_cls, param_groups, lr, weight_decay)
        self._step_count: int = 0

    @staticmethod
    def _create_optimizer(
        name: str,
        params: List[nn.Parameter],
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Instantiate an optimizer by name.

        Args:
            name: Optimizer name (``"adamw"``, ``"adam"``, or ``"sgd"``).
            params: List of parameters to optimize.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            A PyTorch optimizer instance.

        Raises:
            ValueError: If the optimizer name is not recognized.
        """
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"Unknown optimizer '{name}'. Supported: 'adamw', 'adam', 'sgd'"
            )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single training step.

        The caller is responsible for defining the loss function. The model's
        forward method should accept the batch dict and return a loss tensor.

        .. math::
            \\mathcal{L} \\leftarrow \\text{model}(\\text{batch})

        The gradients are computed via backpropagation and the optimizer steps
        only the LoRA parameters.

        Args:
            batch: Dictionary of input tensors (e.g., ``input_ids``,
                ``attention_mask``, ``labels``).

        Returns:
            The scalar loss value for this step.
        """
        self.wrapper.train()
        self.optimizer.zero_grad()

        # Forward pass — model must return loss
        outputs = self.wrapper(**batch)
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, dict):
            loss = outputs["loss"]
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            raise TypeError(
                f"Model output must be a tensor, dict with 'loss' key, or "
                f"object with .loss attribute. Got {type(outputs)}"
            )

        # Backward pass — gradients flow only through LoRA parameters
        loss.backward()

        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in self.wrapper.get_trainable_parameters()],
            max_norm=1.0,
        )

        self.optimizer.step()
        self._step_count += 1

        return loss.detach()

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single evaluation step (no gradient computation).

        Args:
            batch: Dictionary of input tensors.

        Returns:
            The scalar loss value.
        """
        self.wrapper.eval()
        with torch.no_grad():
            outputs = self.wrapper(**batch)
            if isinstance(outputs, torch.Tensor):
                loss = outputs
            elif isinstance(outputs, dict):
                loss = outputs["loss"]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                raise TypeError(
                    f"Model output must be a tensor, dict with 'loss' key, or "
                    f"object with .loss attribute. Got {type(outputs)}"
                )
        return loss.detach()

    def compute_lora_parameter_count(self) -> Dict[str, int]:
        """Count LoRA vs. base model parameters.

        Returns:
            Dictionary with keys:
                - ``"trainable"``: Number of trainable (LoRA) parameters.
                - ``"frozen"``: Number of frozen (base model) parameters.
                - ``"total"``: Total parameter count.
                - ``"trainable_percentage"``: Percentage of total that is trainable.
        """
        trainable = sum(
            p.numel() for _, p in self.wrapper.get_trainable_parameters()
        )
        frozen = sum(
            p.numel() for _, p in self.wrapper.get_frozen_parameters()
        )
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_percentage": trainable / max(total, 1) * 100.0,
        }

    def compute_memory_savings(self) -> Dict[str, Union[int, float]]:
        """Estimate memory savings from LoRA vs. full fine-tuning.

        Computes the theoretical memory difference between storing all
        parameters in FP32 vs. only LoRA parameters in FP32 with the rest
        in the quantized/original format.

        Returns:
            Dictionary with memory estimates in bytes:
                - ``"full_ft_memory"``: Memory for full fine-tuning (all FP32).
                - ``"lora_memory"``: Memory for LoRA fine-tuning.
                - ``"savings_bytes"``: Absolute memory saved.
                - ``"savings_percentage"``: Percentage of memory saved.
        """
        counts = self.compute_lora_parameter_count()
        bytes_per_param = 4  # FP32

        full_ft_memory = counts["total"] * bytes_per_param

        if self.config.quantize_base:
            # Base weights stored as 4-bit NF4
            base_memory = counts["frozen"] // 2  # 4 bits = 0.5 bytes
        else:
            # Base weights frozen in original precision
            base_memory = counts["frozen"] * bytes_per_param

        lora_memory = counts["trainable"] * bytes_per_param + base_memory

        savings_bytes = full_ft_memory - lora_memory

        return {
            "full_ft_memory": full_ft_memory,
            "lora_memory": lora_memory,
            "savings_bytes": savings_bytes,
            "savings_percentage": savings_bytes / max(full_ft_memory, 1) * 100.0,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint including optimizer state.

        Args:
            path: File path for the checkpoint.
        """
        state = {
            "lora_state_dict": lora_state_dict(self.wrapper.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "bias": self.config.bias,
                "target_modules": self.config.target_modules,
                "use_dora": self.config.use_dora,
                "quantize_base": self.config.quantize_base,
                "quantization_type": self.config.quantization_type,
                "double_quantization": self.config.double_quantization,
            },
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint.

        Args:
            path: File path to the checkpoint.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        load_lora_state_dict(self.wrapper.model, state["lora_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._step_count = state.get("step_count", 0)

    def get_merged_model(self) -> nn.Module:
        """Return a model with LoRA weights merged into the base weights.

        This is useful for deployment, where the additional LoRA overhead is
        not desired.

        Returns:
            A copy of the model with LoRA weights merged in.
        """
        return merge_lora_weights(self.wrapper.model, self.config)

    @property
    def step_count(self) -> int:
        """Return the number of training steps completed."""
        return self._step_count
