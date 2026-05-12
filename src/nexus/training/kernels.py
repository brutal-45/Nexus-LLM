"""Production-quality GPU kernels for LLM training using OpenAI Triton.

This module provides high-performance fused kernels for critical LLM operations.
Each kernel has a Triton GPU implementation and a pure PyTorch fallback, selected
automatically based on Triton availability via the ``HAS_TRITON`` flag.

Kernels implemented:
    - fused_rms_norm: Fused Root Mean Square Layer Normalization
    - fused_swiglu: Fused SwiGLU activation (SiLU + element-wise multiply)
    - fused_rope: Fused Rotary Position Embedding (interleaved / half layout)
    - fused_cross_entropy: Fused Cross-Entropy loss with numerically stable softmax
    - quantize_int8 / dequantize_int8: INT8 per-token symmetric quantization
    - int8_matmul: INT8 quantized matrix multiplication
    - quantize_int4 / dequantize_int4: INT4 packed group-wise quantization
    - fused_adam: Fused Adam optimizer step
    - flash_attention_fwd: Simplified Flash Attention forward pass

Typical usage::

    from nexus.training.kernels import fused_rms_norm, fused_swiglu, fused_rope

    # RMSNorm with partial normalization over last 2 dims
    y = fused_rms_norm(x, weight, eps=1e-5, partial_norm_dims=2)

    # SwiGLU activation in feed-forward network
    y = fused_swiglu(x_gate, x_up)

    # Rotary position embedding (non-interleaved)
    y = fused_rope(x, cos_table, sin_table, interleaved=False)

    # INT8 quantized matmul (weight-only quantization)
    w_q, w_scale = quantize_int8(weight)
    out = int8_matmul(input_tensor, w_q, w_scale)

References:
    - Triton: https://github.com/openai/triton
    - RMSNorm: https://arxiv.org/abs/1910.07467
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - RoPE: https://arxiv.org/abs/2104.09864
    - Flash Attention: https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton availability check
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.debug("Triton not available; all kernels will use PyTorch fallbacks.")

__all__: List[str] = [
    "HAS_TRITON",
    "fused_rms_norm",
    "fused_swiglu",
    "fused_rope",
    "fused_cross_entropy",
    "quantize_int8",
    "dequantize_int8",
    "int8_matmul",
    "quantize_int4",
    "dequantize_int4",
    "fused_adam",
    "flash_attention_fwd",
]


# ===========================================================================
# 1. Fused RMSNorm
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_rms_norm_kernel(
        X, Y, W, stride, n_cols, eps, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMSNorm kernel.

        Computes in a single pass (no intermediate tensor materialisation):
            1. variance = mean(x^2)
            2. rms = sqrt(variance + eps)
            3. y = (x / rms) * weight

        Each program instance processes one row of X independently.
        """
        row_idx = tl.program_id(0)
        row_ptr = X + row_idx * stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load row and compute RMS
        x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / n_cols
        rrms = 1.0 / tl.sqrt(mean_sq + eps)

        # Load weight and compute output
        w = tl.load(W + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y_ptr = Y + row_idx * stride
        tl.store(y_ptr + col_offsets, x * rrms * w, mask=mask)


def _triton_fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Triton-accelerated RMSNorm. Requires 2-D input on CUDA."""
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _fused_rms_norm_kernel[(n_rows,)](
        x, y, weight, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.view(original_shape)


def _pytorch_fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """PyTorch reference implementation of RMSNorm.

    Computes RMS(x) = sqrt(mean(x^2) + eps) then returns (x / RMS(x)) * weight.
    This is the standard implementation used in LLaMA, Mistral, and other models.
    """
    dtype = x.dtype
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(dtype)


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    partial_norm_dims: Optional[int] = None,
) -> torch.Tensor:
    """Apply Root Mean Square Layer Normalization.

    RMSNorm computes the root-mean-square of the input and normalises by it,
    then applies a learned element-wise scale.  Unlike LayerNorm it does not
    centre the input (no mean subtraction), making it simpler and faster while
    achieving comparable accuracy in practice.

    Mathematical formulation::

        RMS(x) = sqrt(mean(x^2) + eps)
        y = (x / RMS(x)) * weight

    Args:
        x: Input tensor of shape ``(..., hidden_size)``.
        weight: Learned scale (gamma) parameter of shape ``(hidden_size,)``.
        eps: Small constant for numerical stability.  Default ``1e-5``.
        partial_norm_dims: If set to N > 1, normalise over the last N dimensions
            by reshaping them into a single axis.  Default ``None`` (last dim only).

    Returns:
        Normalised tensor with the same shape and dtype as *x*.
    """
    if partial_norm_dims is not None and partial_norm_dims > 1:
        original_shape = x.shape
        n_last = partial_norm_dims
        x = x.view(*original_shape[:-n_last], math.prod(original_shape[-n_last:]))
        weight = weight.view(1, -1)

    if HAS_TRITON and x.is_cuda and x.dim() == 2:
        out = _triton_fused_rms_norm(x, weight, eps)
    else:
        out = _pytorch_fused_rms_norm(x, weight, eps)

    if partial_norm_dims is not None and partial_norm_dims > 1:
        out = out.view(original_shape)
    return out


# ===========================================================================
# 2. Fused SwiGLU
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_swiglu_kernel(
        X_gate, X_up, Y, stride, n_cols, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SwiGLU kernel.

        Computes:
            gate = SiLU(X_gate) = X_gate * sigmoid(X_gate)
            Y = gate * X_up

        Fuses the SiLU activation and element-wise multiplication into a single
        kernel to reduce memory traffic (two loads + one store instead of three
        stores for intermediate tensors).
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        gate = tl.load(X_gate + row_idx * stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(X_up + row_idx * stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = gate * tl.sigmoid(gate) * up

        tl.store(Y + row_idx * stride + col_offsets, y, mask=mask)


def _triton_fused_swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated SwiGLU. Requires 2-D inputs on CUDA."""
    original_shape = x_gate.shape
    x_gate = x_gate.view(-1, x_gate.shape[-1])
    x_up = x_up.view(-1, x_up.shape[-1])
    n_rows, n_cols = x_gate.shape
    y = torch.empty_like(x_gate)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _fused_swiglu_kernel[(n_rows,)](
        x_gate, x_up, y, x_gate.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.view(original_shape)


def _pytorch_fused_swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: SwiGLU(x_gate, x_up) = SiLU(x_gate) * x_up."""
    return F.silu(x_gate) * x_up


def fused_swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    """Apply the SwiGLU activation function.

    SwiGLU is the gating activation used in modern LLMs (LLaMA, Mistral, PaLM)
    as part of the SwiGLU-FFN block.  It combines a SiLU-activated gate with an
    element-wise up projection::

        output = SiLU(x_gate) * x_up
        SiLU(x) = x * sigmoid(x)

    Args:
        x_gate: Gate projection tensor of shape ``(..., hidden_size)``.
        x_up: Up projection tensor of shape ``(..., hidden_size)``.

    Returns:
        Activated tensor of the same shape and dtype as *x_gate*.
    """
    if HAS_TRITON and x_gate.is_cuda and x_gate.dim() == 2:
        return _triton_fused_swiglu(x_gate, x_up)
    return _pytorch_fused_swiglu(x_gate, x_up)


# ===========================================================================
# 3. Fused Rotary Position Embedding (RoPE)
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_rope_interleaved_kernel(
        X, cos, sin, head_dim, stride_s, stride_h, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RoPE kernel — interleaved layout.

        Pairs consecutive elements (x_{2i}, x_{2i+1}) and applies a 2D rotation
        parameterised by pre-computed cos/sin values::

            x_{2i}   = x_{2i} * cos_i - x_{2i+1} * sin_i
            x_{2i+1} = x_{2i} * sin_i + x_{2i+1} * cos_i
        """
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        half_dim = head_dim // 2
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < half_dim

        base = seq_idx * stride_s + head_idx * stride_h
        cos_vals = tl.load(cos + seq_idx * head_dim + col_offsets, mask=mask, other=0.0)
        sin_vals = tl.load(sin + seq_idx * head_dim + col_offsets, mask=mask, other=0.0)

        idx_even = 2 * col_offsets
        idx_odd = 2 * col_offsets + 1
        mask_e = idx_even < head_dim
        mask_o = idx_odd < head_dim

        x_e = tl.load(X + base + idx_even, mask=mask_e, other=0.0).to(tl.float32)
        x_o = tl.load(X + base + idx_odd, mask=mask_o, other=0.0).to(tl.float32)

        out_e = x_e * cos_vals - x_o * sin_vals
        out_o = x_e * sin_vals + x_o * cos_vals

        tl.store(X + base + idx_even, out_e, mask=mask_e)
        tl.store(X + base + idx_odd, out_o, mask=mask_o)

    @triton.jit
    def _fused_rope_half_kernel(
        X, cos, sin, head_dim, stride_s, stride_h, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RoPE kernel — non-interleaved (half-rotated) layout.

        Splits the head dimension into two halves and rotates::

            x[..., :d//2]  = x[..., :d//2] * cos - x[..., d//2:] * sin
            x[..., d//2:]  = x[..., :d//2] * sin + x[..., d//2:] * cos
        """
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        half_dim = head_dim // 2
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < half_dim

        base = seq_idx * stride_s + head_idx * stride_h
        cos_vals = tl.load(cos + seq_idx * head_dim + col_offsets, mask=mask, other=0.0)
        sin_vals = tl.load(sin + seq_idx * head_dim + col_offsets, mask=mask, other=0.0)

        x_first = tl.load(X + base + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x_second = tl.load(X + base + half_dim + col_offsets, mask=mask, other=0.0).to(tl.float32)

        out_first = x_first * cos_vals - x_second * sin_vals
        out_second = x_first * sin_vals + x_second * cos_vals

        tl.store(X + base + col_offsets, out_first, mask=mask)
        tl.store(X + base + half_dim + col_offsets, out_second, mask=mask)


def _triton_fused_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """Triton-accelerated RoPE. Operates on a clone (does not mutate input)."""
    original_shape = x.shape
    if x.dim() == 4:
        x = x.view(-1, x.shape[-2], x.shape[-1])
    seq_len, num_heads, head_dim = x.shape
    y = x.clone()
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
    stride_s, stride_h = y.stride(0), y.stride(1)
    grid = (seq_len, num_heads)

    if interleaved:
        _fused_rope_interleaved_kernel[grid](
            y, cos, sin, head_dim, stride_s, stride_h, BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        _fused_rope_half_kernel[grid](
            y, cos, sin, head_dim, stride_s, stride_h, BLOCK_SIZE=BLOCK_SIZE,
        )
    return y.view(original_shape)


def _pytorch_fused_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """PyTorch reference implementation of RoPE.

    For interleaved layout, pairs (x_{2i}, x_{2i+1}) are rotated.
    For non-interleaved (half) layout, halves (x[:d/2], x[d/2:]) are rotated.
    """
    d = x.shape[-1]
    half_d = d // 2
    xf = x.float()

    if interleaved:
        x_pairs = xf.reshape(*x.shape[:-1], half_d, 2)
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).reshape(x.shape)
    else:
        x0, x1 = xf[..., :half_d], xf[..., half_d:]
        out = torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.to(x.dtype)


def fused_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True,
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to an input tensor.

    RoPE encodes position by applying a 2D rotation to pairs of features.
    It is the standard positional encoding in modern LLMs (LLaMA, GPT-NeoX,
    Mistral) and injects relative position information into self-attention.

    Non-interleaved formulation::

        x_1 = x[..., :d//2];  x_2 = x[..., d//2:]
        out[..., :d//2]  = x_1 * cos - x_2 * sin
        out[..., d//2:]  = x_1 * sin + x_2 * cos

    For interleaved layout the pairs are ``(x_{2i}, x_{2i+1})`` instead.

    Args:
        x: Input of shape ``(batch, seq_len, num_heads, head_dim)`` or
            ``(seq_len, num_heads, head_dim)``.
        cos: Cosine table ``(..., head_dim)``.
        sin: Sine table ``(..., head_dim)``.
        interleaved: Whether the input uses an interleaved pair layout.
            Default ``True``.

    Returns:
        Tensor with RoPE applied, same shape and dtype as *x*.
    """
    if HAS_TRITON and x.is_cuda and x.dim() in (3, 4):
        return _triton_fused_rope(x, cos, sin, interleaved=interleaved)
    return _pytorch_fused_rope(x, cos, sin, interleaved=interleaved)


# ===========================================================================
# 4. Fused Cross-Entropy Loss
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_cross_entropy_kernel(
        logits, targets, losses, stride, n_cols, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused cross-entropy kernel with numerically stable log-sum-exp.

        Per-sample computation (no materialisation of full softmax):
            1. max_val = max(logits)
            2. log_sum_exp = log(sum(exp(logits - max_val))) + max_val
            3. loss = -logits[target] + log_sum_exp

        This is equivalent to -log(softmax[target]) but avoids computing the
        full softmax vector.
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        logit_ptr = logits + row_idx * stride

        row_logits = tl.load(logit_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        max_val = tl.max(row_logits, axis=0)
        exp_shifted = tl.exp(row_logits - max_val)
        log_sum_exp = tl.log(tl.sum(exp_shifted, axis=0)) + max_val

        target_idx = tl.load(targets + row_idx).to(tl.int32)
        target_logit = tl.load(logit_ptr + target_idx).to(tl.float32)
        tl.store(losses + row_idx, -target_logit + log_sum_exp)


def _triton_fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated per-sample cross-entropy loss."""
    n_rows, n_cols = logits.shape
    losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _fused_cross_entropy_kernel[(n_rows,)](
        logits, targets, losses, logits.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE,
    )
    return losses


def _pytorch_fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: loss_i = -log(softmax(logits_i)[target_i])."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute per-sample cross-entropy loss.

    Uses a numerically stable log-sum-exp formulation that avoids materialising
    the full softmax distribution::

        loss_i = -logits_i[target_i] + log(sum_j exp(logits_i[j]))

    This is mathematically equivalent to ``-log(softmax(logits)[target])``.

    Args:
        logits: Unscaled log-probabilities of shape ``(N, C)``.
        targets: Ground-truth class indices of shape ``(N,)`` with values in
            ``[0, C)``.

    Returns:
        Per-sample losses of shape ``(N,)`` as a float32 tensor.
    """
    if HAS_TRITON and logits.is_cuda and logits.dim() == 2:
        return _triton_fused_cross_entropy(logits, targets)
    return _pytorch_fused_cross_entropy(logits, targets)


# ===========================================================================
# 5. INT8 Quantization / Dequantization / Matmul
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _quantize_int8_kernel(
        X, X_q, scale, stride, n_cols, BLOCK_SIZE: tl.constexpr,
    ):
        """Per-token symmetric INT8 quantization kernel.

        Computes:
            scale = max(|x|) / 127
            x_q = clamp(round(x / scale), -128, 127)

        Each row (token) gets its own scale factor.
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_ptr = X + row_idx * stride
        x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        abs_max = tl.max(tl.abs(x), axis=0)
        s = abs_max / 127.0
        x_q = tl.extra.cuda.libdevice.rint(x / s)
        x_q = tl.minimum(tl.maximum(x_q, -128.0), 127.0)

        tl.store(X_q + row_idx * stride + col_offsets, x_q, mask=mask)
        tl.store(scale + row_idx, s)


def _triton_quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated INT8 quantization."""
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    n_rows, n_cols = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _quantize_int8_kernel[(n_rows,)](
        x, x_q, scale, x.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE,
    )
    return x_q.view(original_shape), scale


def _pytorch_quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference INT8 symmetric quantization.

    scale = max(|x|) / 127  per token;  x_q = clamp(round(x / scale), -128, 127).
    """
    xf = x.float()
    abs_max = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = abs_max.squeeze(-1) / 127.0
    x_q = torch.clamp(torch.round(xf / abs_max * 127.0), -128, 127).to(torch.int8)
    return x_q, scale


def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to INT8 with per-token symmetric quantization.

    Each token (row) is scaled independently::

        scale_i = max_j(|x_{ij}|) / 127
        x_q = clamp(round(x / scale), -128, 127)

    Args:
        x: Input tensor of shape ``(..., hidden_size)``.

    Returns:
        A tuple ``(x_q, scale)`` where *x_q* is INT8 and *scale* is float32
        with one scale value per token.
    """
    if HAS_TRITON and x.is_cuda and x.dim() == 2:
        return _triton_quantize_int8(x)
    return _pytorch_quantize_int8(x)


def dequantize_int8(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an INT8 tensor back to float32.

    Simply multiplies by the per-token scale: ``x = x_q * scale``.

    Args:
        x_q: Quantized INT8 tensor ``(..., hidden_size)``.
        scale: Per-token scale ``(...)``.

    Returns:
        Dequantized float32 tensor.
    """
    if scale.dim() < x_q.dim():
        for _ in range(x_q.dim() - scale.dim()):
            scale = scale.unsqueeze(-1)
    return x_q.float() * scale


def int8_matmul(
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_scale: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """INT8 quantized matrix multiplication.

    Computes ``C = (A * scale_a) @ (B_q * scale_b)`` efficiently by performing
    the matmul in INT8 and rescaling the result.  This is commonly used for
    weight-only quantization in LLM inference.

    Implementation: dequantize B, then use standard matmul.  A production system
    would use cublas GEMM with INT8 inputs directly for maximum throughput.

    Args:
        a: Input tensor (not quantized) ``(..., K)``.
        b_q: Quantized weight ``(..., K, N)`` or ``(K, N)``.
        b_scale: Per-token/channel scale for B.
        a_scale: Optional scale for A (for symmetric quantization of both sides).

    Returns:
        Result tensor ``(..., N)`` in float32.
    """
    b_float = dequantize_int8(b_q, b_scale)
    result = torch.matmul(a.float(), b_float)
    if a_scale is not None:
        if a_scale.dim() < a.dim():
            for _ in range(a.dim() - a_scale.dim()):
                a_scale = a_scale.unsqueeze(-1)
        result = result * a_scale
    return result


# ===========================================================================
# 6. INT4 Quantization / Dequantization
# ===========================================================================

def _pytorch_quantize_int4(x: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """INT4 group-wise quantization with nibble packing.

    Quantization per group:
        scale_g = max(|x_g|) / 7
        x_q = clamp(round(x / scale), -8, 7)

    Packing: two int4 values per uint8 byte.
        byte = (x_q[even] + 8) | ((x_q[odd] + 8) << 4)
    """
    original_shape = x.shape
    x = x.float()
    n_cols = x.shape[-1]

    # Pad to multiple of group_size
    pad_len = (group_size - n_cols % group_size) % group_size
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    padded_cols = x.shape[-1]
    n_groups = padded_cols // group_size

    # Per-group scale and quantize
    x_grouped = x.view(*x.shape[:-1], n_groups, group_size)
    abs_max = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = abs_max.squeeze(-1) / 7.0
    x_q = torch.clamp(torch.round(x_grouped / abs_max * 7.0), -8, 7).to(torch.int8)

    # Flatten and pack pairs of int4 into uint8 bytes
    x_q_flat = x_q.reshape(*x_q.shape[:-2], -1)
    total = x_q_flat.shape[-1]
    if total % 2 != 0:
        x_q_flat = F.pad(x_q_flat, (0, 1))
    even_idx = x_q_flat[..., 0::2]
    odd_idx = x_q_flat[..., 1::2]
    packed = (even_idx + 8) | ((odd_idx + 8) << 4)
    packed_uint8 = packed.to(torch.uint8)[..., : padded_cols // 2]

    return packed_uint8, scale


def _pytorch_dequantize_int4(
    x_packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
    original_cols: Optional[int] = None,
) -> torch.Tensor:
    """Unpack int4 nibbles and dequantize.

    Each byte contains two 4-bit values: low nibble = x_q[even], high = x_q[odd].
    """
    n_groups = scale.shape[-1]
    low = (x_packed & 0x0F).to(torch.int8) - 8
    high = ((x_packed >> 4) & 0x0F).to(torch.int8) - 8

    total_elements = x_packed.shape[-1] * 2
    result = torch.empty(*x_packed.shape[:-1], total_elements, dtype=torch.int8,
                         device=x_packed.device)
    result[..., 0::2] = low
    result[..., 1::2] = high

    result = result.view(*result.shape[:-1], n_groups, group_size).float()
    result = result * scale.unsqueeze(-1)
    result = result.view(*result.shape[:-2], -1)

    if original_cols is not None:
        result = result[..., :original_cols]
    return result


def quantize_int4(x: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to packed INT4 with group-wise quantization.

    Two int4 values are packed into a single uint8 byte.  Groups of *group_size*
    consecutive elements share a scale factor::

        scale_g = max(|x_g|) / 7
        x_q = clamp(round(x / scale), -8, 7)
        byte = (x_q[even] + 8) | ((x_q[odd] + 8) << 4)

    Args:
        x: Input tensor of shape ``(..., hidden_size)``.
        group_size: Number of elements per quantization group.  Default ``32``.

    Returns:
        A tuple ``(x_packed, scale)`` where *x_packed* is a uint8 tensor
        with ``hidden_size // 2`` elements per token and *scale* has shape
        ``(..., hidden_size // group_size)``.
    """
    return _pytorch_quantize_int4(x, group_size=group_size)


def dequantize_int4(
    x_packed: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
    original_cols: Optional[int] = None,
) -> torch.Tensor:
    """Dequantize a packed INT4 tensor back to float32.

    Unpacks the two nibbles per byte and applies per-group rescaling.

    Args:
        x_packed: Packed uint8 tensor ``(..., hidden_size // 2)``.
        scale: Per-group scale ``(..., n_groups)``.
        group_size: Group size used during quantization.  Default ``32``.
        original_cols: If set, trim output to this many columns.

    Returns:
        Dequantized float32 tensor.
    """
    return _pytorch_dequantize_int4(x_packed, scale, group_size, original_cols)


# ===========================================================================
# 7. Fused Adam Optimizer
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_adam_kernel(
        param_ptr, grad_ptr, m_ptr, v_ptr,
        beta1, beta2, eps, lr, weight_decay,
        step, t, n_elements, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused Adam optimizer kernel.

        Implements the standard Adam update with bias correction in a single
        fused pass (no intermediate Python-level operations):

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad^2
            m_hat = m / (1 - beta1^t)
            v_hat = v / (1 - beta2^t)
            param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
        """
        block_idx = tl.program_id(0)
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - tl.pow(beta1, t))
        v_hat = v / (1.0 - tl.pow(beta2, t))
        param = param - lr * (m_hat / (tl.sqrt(v_hat) + eps) + weight_decay * param)

        tl.store(param_ptr + offsets, param, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)


def _triton_fused_adam(
    param: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.0, step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fused Adam step (in-place on param, m, v)."""
    n_elements = param.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fused_adam_kernel[grid](
        param, grad, m, v, beta1, beta2, eps, lr, weight_decay,
        step, step + 1, n_elements, BLOCK_SIZE=BLOCK_SIZE,
    )
    return param, m, v


def _pytorch_fused_adam(
    param: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.0, step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference Adam implementation (in-place updates)."""
    t = step + 1
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    param.add_(lr * (m_hat / (torch.sqrt(v_hat) + eps) + weight_decay * param), alpha=-1.0)
    return param, m, v


def fused_adam(
    param: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.0, step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform a single fused Adam optimizer step (in-place).

    Updates param, m, and v using the standard Adam algorithm with bias
    correction and optional weight decay::

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)

    Args:
        param: Model parameter tensor (updated in-place).
        grad: Gradient tensor, same shape as *param*.
        m: First moment estimate (updated in-place).
        v: Second moment estimate (updated in-place).
        lr: Learning rate.  Default ``1e-3``.
        beta1: Exponential decay rate for first moment.  Default ``0.9``.
        beta2: Exponential decay rate for second moment.  Default ``0.999``.
        eps: Numerical stability constant.  Default ``1e-8``.
        weight_decay: L2 regularisation coefficient.  Default ``0.0``.
        step: Current step index (0-based), used for bias correction.

    Returns:
        Tuple ``(param, m, v)`` with the updated tensors.
    """
    if HAS_TRITON and param.is_cuda:
        return _triton_fused_adam(
            param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step,
        )
    return _pytorch_fused_adam(
        param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step,
    )


# ===========================================================================
# 8. Simplified Flash Attention (Forward)
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX, head_dim,
        causal: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Simplified Flash Attention forward kernel.

        Uses tiled computation with online softmax (Flash Attention algorithm)
        to avoid materialising the full N_CTX × N_CTX attention matrix,
        reducing memory from O(N²) to O(N).

        Algorithm per query block:
            for each key block:
                S = Q @ K^T / sqrt(d)          # attention scores
                S = apply_causal_mask(S)         # mask future if causal
                m_new = max(m_old, max(S))       # online max
                alpha = exp(m_old - m_new)       # rescaling factor
                exp_S = exp(S - m_new)           # numerically stable exp
                l_new = alpha * l_old + rowsum(exp_S)  # online sum
                O = alpha * O + exp_S @ V        # update accumulator
            O /= l  # final normalisation
        """
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        m_block = tl.program_id(2)

        q_off_z = batch_idx * stride_qz + head_idx * stride_qh
        k_off_z = batch_idx * stride_kz + head_idx * stride_kh
        v_off_z = batch_idx * stride_vz + head_idx * stride_vh
        o_off_z = batch_idx * stride_oz + head_idx * stride_oh

        m_range = tl.arange(0, BLOCK_M)
        n_range = tl.arange(0, BLOCK_N)
        d_range = tl.arange(0, BLOCK_D)
        m_start = m_block * BLOCK_M
        m_idx = m_start + m_range
        m_mask = m_idx < N_CTX

        # Running statistics: row-wise max, sum of exp, output accumulator
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
        acc = tl.full([BLOCK_M, BLOCK_D], 0.0, dtype=tl.float32)

        scale = 1.0 / tl.sqrt(head_dim.to(tl.float32))

        # For causal: only iterate over key blocks that contain valid positions
        n_blocks = (
            ((m_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N) if causal
            else ((N_CTX + BLOCK_N - 1) // BLOCK_N)
        )

        for n_block in range(n_blocks):
            n_start = n_block * BLOCK_N
            n_idx = n_start + n_range
            n_mask = n_idx < N_CTX

            # Load K, V blocks (BLOCK_N × BLOCK_D)
            k_ptrs = K + k_off_z + n_idx[:, None] * stride_kn + d_range[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=n_mask[:, None] & (d_range[None, :] < head_dim), other=0.0)
            v_ptrs = V + v_off_z + n_idx[:, None] * stride_vn + d_range[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=n_mask[:, None] & (d_range[None, :] < head_dim), other=0.0)

            # Load Q block (BLOCK_M × BLOCK_D)
            q_ptrs = Q + q_off_z + m_idx[:, None] * stride_qm + d_range[None, :] * stride_qk
            q = tl.load(q_ptrs, mask=m_mask[:, None] & (d_range[None, :] < head_dim), other=0.0)

            # S = Q K^T / sqrt(d)  — shape (BLOCK_M, BLOCK_N)
            S = tl.dot(q, k.trans()) * scale

            # Apply causal mask: set future positions to -inf
            if causal:
                S = tl.where((m_idx[:, None] < n_idx[None, :]) & n_mask[None, :], float("-inf"), S)
            else:
                S = tl.where(~n_mask[None, :], float("-inf"), S)

            # Online softmax update
            m_block_max = tl.max(S, axis=1)
            m_new = tl.maximum(m_i, m_block_max)
            alpha = tl.exp(m_i - m_new)
            exp_S = tl.exp(S - m_new[:, None])

            # Zero out masked (future / padding) positions
            if causal:
                exp_S = tl.where((m_idx[:, None] < n_idx[None, :]) & n_mask[None, :], 0.0, exp_S)
            else:
                exp_S = tl.where(~n_mask[None, :], 0.0, exp_S)

            l_new = alpha * l_i + tl.sum(exp_S, axis=1)
            acc = alpha[:, None] * acc + tl.dot(exp_S.to(v.dtype), v)
            m_i = m_new
            l_i = l_new

        # Finalise: divide by running sum of exponentials
        acc = acc * (1.0 / l_i)[:, None]
        o_ptrs = Out + o_off_z + m_idx[:, None] * stride_om + d_range[None, :] * stride_ok
        tl.store(o_ptrs, acc, mask=m_mask[:, None] & (d_range[None, :] < head_dim))


def _triton_flash_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Triton Flash Attention forward with BLOCK_M=BLOCK_N=64 tiling."""
    batch, num_heads, n_ctx, head_dim = q.shape
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (batch, num_heads, triton.cdiv(n_ctx, BLOCK_M))

    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, num_heads, n_ctx, head_dim,
        causal=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o


def _pytorch_flash_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """PyTorch reference scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        seq_len = q.shape[-2]
        attn = attn.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
    return torch.matmul(F.softmax(attn, dim=-1), v)


def flash_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Compute scaled dot-product attention (simplified Flash Attention).

    The Triton path uses tiled QKV computation with an online softmax to
    reduce peak memory from O(N²) to O(N), matching the core insight of
    Flash Attention (Dao et al., 2022)::

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    With causal masking, future key positions are set to ``-inf`` before the
    softmax, ensuring autoregressive attention patterns.

    Note: This is a simplified implementation.  Production FlashAttention-2
    uses additional optimisations (kernel fusion with dropout, backwards pass,
    complex autotuning, SWA, etc.).

    Args:
        q: Query tensor ``(batch, num_heads, seq_len, head_dim)``.
        k: Key tensor   ``(batch, num_heads, seq_len, head_dim)``.
        v: Value tensor ``(batch, num_heads, seq_len, head_dim)``.
        causal: Apply causal (lower-triangular) masking.  Default ``True``.

    Returns:
        Output tensor ``(batch, num_heads, seq_len, head_dim)``.
    """
    if HAS_TRITON and q.is_cuda and q.dim() == 4:
        return _triton_flash_attention_fwd(q, k, v, causal=causal)
    return _pytorch_flash_attention_fwd(q, k, v, causal=causal)
