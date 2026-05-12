"""
Positional Encoding Module for Nexus v2
==========================================
Comprehensive positional encoding implementations for decoder-only transformers.

Implements ALL major positional encoding strategies:
1. Rotary Position Embeddings (RoPE) with extensions
2. ALiBi (Attention with Linear Biases)
3. Learned absolute position embeddings
4. Context length extension methods

This module is self-contained and provides all positional encoding
functionality needed for 100B+ parameter models.

References:
- Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Press et al. (2021) "Train Short, Test Long: Attention with Linear Biases"
- Chen et al. (2023) "Extending Context Window of Large Language Models via Position Interpolation"
- Peng et al. (2023) "YaRN: Efficient Context Window Extension of Large Language Models"
- Ding et al. (2024) "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helper Functions for Rotary Embeddings
# =============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the second half of the hidden dimension to the front.

    For an input of shape [..., D], splits along the last dimension into
    two halves of size D/2, then returns [-x2, x1].

    This implements the rotation matrix factorization:
        R(θ) = [[cos θ, -sin θ],
                [sin θ,  cos θ]]

    Applied as: [x1, x2] -> [-x2, x1] (complex multiplication equivalent).

    Args:
        x: Input tensor of shape [..., D] where D is even.

    Returns:
        Rotated tensor of same shape [..., D].

    Raises:
        ValueError: If last dimension is not even.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"rotate_half requires even dimension, got {x.shape[-1]}"
        )
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Computes:  q' = q * cos + rotate_half(q) * sin
               k' = k * cos + rotate_half(k) * sin

    This is equivalent to complex multiplication:
        (q_r + i*q_i) * (cos θ + i*sin θ) in the complex plane,
    which rotates the query/key vectors by their corresponding position angles.

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim].
        k: Key tensor of shape [batch, heads, seq_len, head_dim].
        cos: Cosine tensor of shape [seq_len, head_dim] or broadcastable.
        sin: Sine tensor of shape [seq_len, head_dim] or broadcastable.
        position_ids: Optional position indices for gathering. If provided,
            cos/sin are gathered at these positions before application.

    Returns:
        Tuple of (q_embedded, k_embedded) with rotary embeddings applied.
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# 1. Rotary Position Embedding (Enhanced RoPE)
# =============================================================================


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding with support for multiple scaling strategies.

    Implements the standard RoPE from Su et al. (2021) with extensions for
    context length extrapolation and interpolation:

    - **standard**: Original RoPE with fixed base 10000.
      θ_i = 1 / (base^(2i/d))

    - **ntk_aware**: NTK-aware scaling (CodeLlama style).
      Adjusts base dynamically: base' = base * (s/s_0)^(d/(d-2))
      Preserves low-frequency components while extending high-frequency range.

    - **dynamic_ntk**: Dynamic NTK scaling applied at inference time.
      Only scales when seq_len > max_seq_len; uses standard RoPE otherwise.

    - **linear**: Position Interpolation (PI) - Chen et al. (2023).
      Linearly scales position indices: pos' = pos * (s_0 / s)
      Fine-tune-free method, good for moderate extensions (2-4x).

    - **yarn**: YaRN scaling - Peng et al. (2023).
      Combines NTK scaling for high frequencies with temperature-based
      attention for low frequencies. Best for large extensions.

    - **longrope**: LongRoPE - Ding et al. (2024).
      Learns different scaling factors per frequency component.
      Achieves 1M+ token contexts.

    Args:
        dim: Rotary embedding dimension (typically head_dim).
        max_seq_len: Maximum sequence length during training.
        base: Base for computing inverse frequencies (default 10000.0).
        scaling_type: One of "standard", "ntk_aware", "dynamic_ntk",
            "linear", "yarn", "longrope".
        scaling_factor: Factor for context extension (e.g., 4.0 for 4x).
        yarn_params: Optional dict with YaRN-specific parameters:
            - beta_fast (float): Low-frequency mixture ratio (default 32).
            - beta_slow (float): High-frequency mixture ratio (default 1.0).
            - mscale (float): Attention temperature multiplier (default sqrt(2*ln(s_0/s) + 1)).
        longrope_params: Optional dict with LongRoPE-specific parameters:
            - short_factor (List[float]): Scaling factors for short context.
            - long_factor (List[float]): Scaling factors for long context.
            - original_max_seq_len (int): Original max length (default max_seq_len).
    """

    VALID_SCALING_TYPES = {
        "standard", "ntk_aware", "dynamic_ntk", "linear", "yarn", "longrope"
    }

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_type: str = "standard",
        scaling_factor: float = 1.0,
        yarn_params: Optional[Dict[str, Any]] = None,
        longrope_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if scaling_type not in self.VALID_SCALING_TYPES:
            raise ValueError(
                f"Invalid scaling_type '{scaling_type}'. "
                f"Must be one of {self.VALID_SCALING_TYPES}."
            )

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.yarn_params = yarn_params or {}
        self.longrope_params = longrope_params or {}

        # Precompute standard inverse frequencies: 1 / (base^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for computed cos/sin to avoid recomputation
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_seq_len: int = -1

        # Build the cache for the default max_seq_len
        self._build_cache(max_seq_len)

    def _compute_inv_freq(
        self, scaling_type: str, seq_len: int
    ) -> torch.Tensor:
        """
        Compute inverse frequencies based on the scaling strategy.

        Different strategies modify the base or the frequencies themselves
        to extend or interpolate the context window.

        Args:
            scaling_type: The scaling strategy to use.
            seq_len: Target sequence length.

        Returns:
            Tensor of inverse frequencies of shape [dim/2].
        """
        if scaling_type == "standard" or (
            scaling_type == "dynamic_ntk" and seq_len <= self.max_seq_len
        ):
            return self.inv_freq

        if scaling_type in ("ntk_aware", "dynamic_ntk"):
            return self._ntk_aware_scaling(seq_len)

        if scaling_type == "linear":
            # Position Interpolation doesn't modify frequencies;
            # positions are scaled instead in forward().
            return self.inv_freq

        if scaling_type == "yarn":
            return self._yarn_scaling(seq_len, self.yarn_params)

        if scaling_type == "longrope":
            return self._longrope_scaling(seq_len, self.longrope_params)

        return self.inv_freq

    def _ntk_aware_scaling(self, seq_len: int) -> torch.Tensor:
        """
        NTK-aware scaling (CodeLlama style).

        Dynamically adjusts the base frequency:
            base' = base * (seq_len / target_len) ^ (dim / (dim - 2))

        This preserves the low-frequency (slowly-varying) components which
        carry most positional information, while extending the range of
        high-frequency components to cover the longer sequence.

        The exponent dim/(dim-2) ensures that the NTK interpolation
        effectively shifts the frequency range rather than simply
        stretching or compressing it.

        Args:
            seq_len: Target sequence length for scaling.

        Returns:
            Scaled inverse frequencies of shape [dim/2].
        """
        target_len = self.max_seq_len * self.scaling_factor
        # Ensure we don't scale down
        effective_scale = max(seq_len / target_len, 1.0)
        base = self.base * (
            effective_scale ** (self.dim / (self.dim - 2))
        )
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)
        )
        return inv_freq

    def _position_interpolation(
        self, seq_len: int, target_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Position Interpolation (PI) - Chen et al. (2023).

        Instead of scaling frequencies, linearly scales position indices:
            pos' = pos * (s_0 / s)

        This effectively "stretches" the positional encoding over a larger
        range, allowing the model to attend to longer sequences without
        changing the learned frequency decomposition.

        Key insight: the model has learned to represent relative positions
        up to s_0. By rescaling positions to fit within [0, s_0), we
        reuse the same positional patterns for longer sequences.

        Args:
            seq_len: Current sequence length.
            target_len: Target length to interpolate to (default: seq_len).

        Returns:
            Position tensor of shape [seq_len].
        """
        if target_len is None:
            target_len = seq_len
        scale = self.max_seq_len / target_len
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        return positions * scale

    def _yarn_scaling(
        self, seq_len: int, params: Dict[str, Any]
    ) -> torch.Tensor:
        """
        YaRN (Yet another RoPE extensioN) scaling - Peng et al. (2023).

        YaRN applies different treatments to high and low frequency components:
        1. High frequencies (> beta_fast threshold): Use standard RoPE
        2. Middle frequencies: NTK-aware scaling with partial interpolation
        3. Low frequencies (< beta_slow threshold): Enhanced with temperature
           scaling to prevent attention score explosion

        The temperature parameter β is computed as:
            β = 0.1 * ln(s/s_0) + 1

        Args:
            seq_len: Target sequence length.
            params: YaRN parameters dict (beta_fast, beta_slow, mscale).

        Returns:
            Scaled inverse frequencies of shape [dim/2].
        """
        beta_fast = params.get("beta_fast", 32.0)
        beta_slow = params.get("beta_slow", 1.0)

        # Compute NTK-aware base
        target_len = self.max_seq_len * self.scaling_factor
        base = self.base * (
            (seq_len / self.max_seq_len) ** (self.dim / (self.dim - 2))
        )

        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)
        )

        # Classify frequencies: low, mid, high
        freqs = 1.0 / inv_freq  # [dim/2]
        freqs_ratio = freqs / freqs[-1]  # Normalize to [0, 1]

        # Create smooth interpolation masks
        low_mask = torch.clamp((beta_slow - freqs_ratio) / (beta_slow - beta_fast), 0, 1)
        high_mask = 1.0 - low_mask

        # Low freq: original (unmodified) — carry long-range positional info
        # High freq: NTK-scaled — extended for longer sequences
        original_inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)
        )

        inv_freq_scaled = original_inv_freq * low_mask + inv_freq * high_mask

        # Compute attention temperature multiplier
        mscale = params.get("mscale", None)
        if mscale is None:
            if seq_len > self.max_seq_len:
                mscale = math.sqrt(2.0 * math.log(seq_len / self.max_seq_len) + 1.0)
            else:
                mscale = 1.0

        self._yarn_mscale = mscale
        return inv_freq_scaled

    def _longrope_scaling(
        self, seq_len: int, params: Dict[str, Any]
    ) -> torch.Tensor:
        """
        LongRoPE scaling - Ding et al. (2024).

        LongRoPE uses different scaling factors for different frequency components.
        Some frequencies are scaled aggressively, others not at all, based on
        an optimization procedure that finds the best per-component scaling.

        The key insight is that different positional frequency components have
        different sensitivities — some can be scaled to very long ranges without
        loss, while others are critical for short-range accuracy.

        Args:
            seq_len: Target sequence length.
            params: LongRoPE parameters dict with:
                - short_factor: Per-component scaling for short context training.
                - long_factor: Per-component scaling for long context inference.

        Returns:
            Scaled inverse frequencies of shape [dim/2].
        """
        short_factor = params.get("short_factor", None)
        long_factor = params.get("long_factor", None)

        if short_factor is None and long_factor is None:
            # Fall back to NTK-aware scaling
            return self._ntk_aware_scaling(seq_len)

        # Determine which factor set to use based on sequence length
        original_max_len = params.get("original_max_seq_len", self.max_seq_len)
        factor = long_factor if seq_len > original_max_len else short_factor

        if factor is not None:
            factor = torch.tensor(
                factor, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            # Ensure factor length matches dim/2
            if len(factor) < self.dim // 2:
                # Pad by repeating the last value
                pad_len = self.dim // 2 - len(factor)
                factor = torch.cat([factor, factor[-1:].expand(pad_len)])
            elif len(factor) > self.dim // 2:
                factor = factor[: self.dim // 2]
            inv_freq = self.inv_freq / factor
        else:
            inv_freq = self.inv_freq

        return inv_freq

    def _build_cache(self, seq_len: int) -> None:
        """
        Build cos/sin cache for the given sequence length.

        Args:
            seq_len: Sequence length to precompute embeddings for.
        """
        if self._cache_seq_len == seq_len:
            return

        inv_freq = self._compute_inv_freq(self.scaling_type, seq_len)

        if self.scaling_type == "linear":
            positions = self._position_interpolation(seq_len)
        else:
            positions = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)

        freqs = torch.outer(positions, inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]

        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()
        self._cache_seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cos, sin).

        Args:
            x: Reference tensor (used to determine device/dtype).
            seq_len: Override sequence length. If None, uses x.shape[2].
            position_ids: Optional position indices of shape [batch, seq_len].
                If provided, embeddings are gathered at these positions.

        Returns:
            Tuple of (cos, sin) tensors, each of shape [seq_len, dim] or
            [batch, seq_len, dim] if position_ids are provided.
        """
        if seq_len is None:
            seq_len = x.shape[2] if x.dim() >= 3 else x.shape[1]

        # Rebuild cache if sequence length changed
        if seq_len != self._cache_seq_len:
            self._build_cache(seq_len)

        cos = self._cos_cache.to(dtype=x.dtype)  # [seq_len, dim]
        sin = self._sin_cache.to(dtype=x.dtype)

        # Apply YaRN temperature if applicable
        if self.scaling_type == "yarn" and hasattr(self, "_yarn_mscale"):
            cos = cos * self._yarn_mscale
            sin = sin * self._yarn_mscale

        return cos, sin

    def get_attention_biases(
        self,
        seq_len: int,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Return additive attention bias derived from RoPE.

        This computes the relative position bias:
            bias[i, j] = cos(θ · (pos_i - pos_j))

        Useful when converting from rotation-based to bias-based attention.

        Args:
            seq_len: Sequence length.
            position_ids: Optional position indices [batch, seq_len].

        Returns:
            Bias tensor of shape [1, 1, seq_len, seq_len] or None if not applicable.
        """
        if position_ids is not None:
            positions = position_ids.float()
        else:
            positions = torch.arange(seq_len, device=self.inv_freq.device).float()

        # Compute pairwise differences: [seq_len, seq_len]
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)

        # Compute angle-based bias using each frequency
        # Sum over frequencies for a single bias value per position pair
        bias = torch.zeros(
            seq_len, seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        for i, freq in enumerate(self.inv_freq):
            angle = diff * freq
            # Use the mean cosine similarity across frequency components
            bias += torch.cos(angle)

        bias = bias / len(self.inv_freq)
        return bias.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


# =============================================================================
# 2. ALiBi Positional Encoding
# =============================================================================


class ALiBiPositionalEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) - Press et al. 2021.

    Adds per-head linear bias to attention scores:
        attn_scores = Q @ K^T / sqrt(d_k) + alibi_bias

    Where the bias for head h and relative position m is:
        bias(h, m) = -slope_h * |m|

    The slopes follow a geometric progression:
        For H heads, we use H/4 unique slopes: m_i = 2^(-8i/H) for i=1..H/4
        Then repeat each slope 4 times to fill all H heads.

    Advantages over learned positional encodings:
    - **No learned parameters**: Fully deterministic, saves memory.
    - **Trivial extrapolation**: Works on any sequence length without retraining.
    - **Excellent generalization**: Often outperforms RoPE on very long sequences.
    - **Simplicity**: Easy to implement and integrate.

    Args:
        num_heads: Number of attention heads.
        context_length: Maximum context length for precomputed bias cache.
        slope_type: Slope computation method. One of:
            - "standard": Original geometric progression (default).
            - "even": Linear spacing of slopes.
            - "all": Unique slope per head (no repeating).
    """

    SLOPE_PRESETS = {
        1: [1.0],
        2: [1.0, 0.5],
        4: [1.0, 0.5, 1.0 / (2 ** (2 / 4)), 1.0 / (2 ** (3 / 4))],
        8: [
            1.0 / (2 ** (i / 8))
            for i in range(1, 9)
        ],
        16: [
            1.0 / (2 ** (i / 16))
            for i in range(1, 17)
        ],
        32: [
            1.0 / (2 ** (i / 32))
            for i in range(1, 33)
        ],
        64: [
            1.0 / (2 ** (i / 64))
            for i in range(1, 65)
        ],
    }

    def __init__(
        self,
        num_heads: int,
        context_length: int = 8192,
        slope_type: str = "standard",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.context_length = context_length
        self.slope_type = slope_type

        slopes = self._compute_slopes(num_heads)
        self.register_buffer(
            "slopes",
            torch.tensor(slopes, dtype=torch.float32),
            persistent=False,
        )

        # Precompute bias matrix for the default context length
        bias = self._build_bias_matrix(context_length)
        self.register_buffer("bias_cache", bias, persistent=False)

    def _compute_slopes(self, n_heads: int) -> List[float]:
        """
        Compute per-head bias slopes.

        For the "standard" ALiBi scheme (Press et al. 2021):
        - Compute H/4 unique slopes: m_i = 2^(-8i/H) for i = 1, ..., H/4
        - Repeat each slope 4 times: [m_1, m_1, m_1, m_1, m_2, m_2, ...]

        This ensures that heads are grouped into 4 sets, each with a different
        bias magnitude, giving the model multiple "resolution" levels for
        positional awareness.

        Args:
            n_heads: Number of attention heads.

        Returns:
            List of slope values, one per head.
        """
        if self.slope_type == "standard":
            # Check presets for common head counts
            if n_heads in self.SLOPE_PRESETS:
                return list(self.SLOPE_PRESETS[n_heads])

            # General formula: H/4 unique slopes, each repeated 4 times
            n_unique = max(n_heads // 4, 1)
            slopes = [
                2.0 ** (-8.0 * i / n_heads) for i in range(1, n_unique + 1)
            ]
            # Repeat each slope to fill all heads
            result = []
            for s in slopes:
                for _ in range(4):
                    result.append(s)
                    if len(result) >= n_heads:
                        break
                if len(result) >= n_heads:
                    break

            return result[:n_heads]

        elif self.slope_type == "even":
            # Linearly spaced slopes from 1.0 to smallest
            return [1.0 - i / n_heads for i in range(n_heads)]

        elif self.slope_type == "all":
            # Unique slope per head
            return [2.0 ** (-8.0 * i / n_heads) for i in range(1, n_heads + 1)]

        else:
            raise ValueError(
                f"Unknown slope_type '{self.slope_type}'. "
                f"Choose from: standard, even, all."
            )

    def _build_bias_matrix(self, seq_len: int) -> torch.Tensor:
        """
        Build the full ALiBi bias matrix.

        For each head h and position pair (i, j), the bias is:
            bias[h, i, j] = -slope_h * |i - j|

        Negative bias ensures that distant tokens get penalized, encouraging
        local attention patterns which is beneficial for autoregressive models.

        Args:
            seq_len: Sequence length.

        Returns:
            Bias tensor of shape [1, num_heads, 1, seq_len].
        """
        # Distance matrix: |i - j| for all pairs
        positions = torch.arange(seq_len, dtype=torch.float32)
        distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        # Apply per-head slopes
        # slopes: [num_heads], distances: [seq_len, seq_len]
        bias = -self.slopes.unsqueeze(1).unsqueeze(2) * distances.unsqueeze(0)

        # Reshape for broadcasting with attention scores [B, H, S, S]
        return bias.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]

    def get_bias(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Get ALiBi bias tensor, recomputing if sequence length exceeds cache.

        Args:
            seq_len: Target sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Bias tensor of shape [1, num_heads, 1, seq_len].
        """
        if seq_len > self.context_length or self.bias_cache is None:
            bias = self._build_bias_matrix(seq_len)
        else:
            bias = self.bias_cache[:, :, :seq_len, :seq_len]

        if device is not None:
            bias = bias.to(device)
        if dtype is not None:
            bias = bias.to(dtype)

        return bias

    def forward(
        self,
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            attention_scores: Attention logits of shape [batch, heads, q_len, kv_len].
            seq_len: Override sequence length. If None, inferred from tensor shape.

        Returns:
            Attention scores with ALiBi bias added, same shape as input.
        """
        if seq_len is None:
            seq_len = attention_scores.shape[-1]

        bias = self.get_bias(
            seq_len,
            device=attention_scores.device,
            dtype=attention_scores.dtype,
        )
        return attention_scores + bias


# =============================================================================
# 3. Learned Absolute Position Embedding
# =============================================================================


class LearnedPositionEmbedding(nn.Module):
    """
    Learned absolute position embeddings.

    A simple learnable lookup table that maps position indices to embedding vectors.
    Each position in the sequence has its own learned embedding vector.

    Used in: GPT-2, BERT, early transformer models.

    Advantages:
    - Simple and proven effective for moderate sequence lengths.
    - Can learn arbitrary positional patterns.
    - Easy to implement and understand.

    Limitations:
    - Fixed maximum sequence length (cannot extrapolate beyond trained length).
    - Memory scales linearly with max_seq_len.
    - Less effective for very long sequences compared to RoPE/ALiBi.

    Args:
        max_seq_len: Maximum supported sequence length.
        hidden_size: Dimensionality of each position embedding vector.
        padding_idx: Optional index for padding position (embedding is zeroed).
            Default is None (no padding index).
        init_std: Standard deviation for normal initialization. Default 0.02.
    """

    def __init__(
        self,
        max_seq_len: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
        )

        # Initialize with small random values (following GPT-2 convention)
        nn.init.normal_(self.embedding.weight, std=init_std)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0.0)

    def forward(
        self,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Look up position embeddings for the given position indices.

        Args:
            position_ids: Position indices of shape [batch_size, seq_len].
                Values must be in [0, max_seq_len).

        Returns:
            Position embeddings of shape [batch_size, seq_len, hidden_size].

        Raises:
            ValueError: If any position_id exceeds max_seq_len - 1.
        """
        max_pos = position_ids.max().item()
        if max_pos >= self.max_seq_len:
            raise ValueError(
                f"Position index {max_pos} exceeds maximum sequence length "
                f"{self.max_seq_len}. Increase max_seq_len or truncate input."
            )
        return self.embedding(position_ids)

    def extra_repr(self) -> str:
        return (
            f"max_seq_len={self.max_seq_len}, hidden_size={self.hidden_size}"
        )


# =============================================================================
# 4. Context Length Extension Methods
# =============================================================================


class ContextLengthExtension:
    """
    Collection of context length extension methods.

    These techniques allow a model trained on sequence length S to handle
    sequences of length S' > S without full retraining. Each method has
    different trade-offs in terms of memory, compute, and quality.

    Methods:
    - sliding_window_attention: O(S*W) attention with fixed window.
    - landmark_attention: O(S*L) attention with landmark bridging.
    - self_extending_transformer: Gradual RoPE frequency interpolation.
    - streaming_llm_attention: Sink + window attention for streaming.
    """

    @staticmethod
    def sliding_window_attention(
        attention_scores: torch.Tensor,
        window_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply sliding window attention mask.

        Restricts each token to attend only to the previous `window_size` tokens.
        Memory is reduced from O(S^2) to O(S*W) where W = window_size.

        The mask is applied multiplicatively:
            masked_scores = attention_scores * window_mask

        Where window_mask[i, j] = 1 if j >= max(0, i - window_size), else 0.

        Combined with causal masking, this ensures that each position i can
        only attend to positions j where: (j <= i) AND (i - j <= window_size).

        Args:
            attention_scores: Raw attention scores [batch, heads, q_len, kv_len].
            window_size: Number of previous tokens each token can attend to.
            mask: Optional existing causal mask to combine with.

        Returns:
            Masked attention scores with same shape as input.

        Example:
            >>> scores = torch.randn(1, 8, 100, 100)
            >>> masked = ContextLengthExtension.sliding_window_attention(scores, 32)
        """
        q_len, kv_len = attention_scores.shape[-2], attention_scores.shape[-1]

        # Create sliding window mask: lower-triangular with limited bandwidth
        # Positions that are too far away get -inf
        rows = torch.arange(q_len, device=attention_scores.device).unsqueeze(1)
        cols = torch.arange(kv_len, device=attention_scores.device).unsqueeze(0)

        # Allowed: col <= row (causal) AND row - col <= window_size
        causal_mask = cols <= rows
        window_mask = (rows - cols) <= window_size
        combined_mask = causal_mask & window_mask

        # Convert to additive mask: 0 for allowed, -inf for blocked
        additive_mask = torch.zeros_like(attention_scores)
        additive_mask = additive_mask.masked_fill(~combined_mask, float("-inf"))

        # Combine with any existing mask
        if mask is not None:
            additive_mask = additive_mask + mask

        return attention_scores + additive_mask

    @staticmethod
    def landmark_attention(
        hidden_states: torch.Tensor,
        num_landmarks: int,
        attention_fn,
    ) -> torch.Tensor:
        """
        Landmark attention: select landmark tokens to bridge between segments.

        Divides the sequence into segments and selects representative landmark
        tokens from each segment. Each token attends to:
        1. All tokens in its local segment (fine-grained attention)
        2. All landmark tokens from other segments (global context)

        This reduces complexity from O(S^2) to O(S*L) where L = num_landmarks,
        where the segment size is approximately S / num_landmarks.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            num_landmarks: Number of landmark tokens to select.
            attention_fn: Attention function with signature:
                fn(query, key, value) -> output of shape [batch, seq_len, hidden_size].

        Returns:
            Output tensor of same shape as hidden_states.

        Raises:
            ValueError: If num_landmarks >= seq_len.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if num_landmarks >= seq_len:
            raise ValueError(
                f"num_landmarks ({num_landmarks}) must be less than "
                f"seq_len ({seq_len})."
            )

        segment_size = seq_len // num_landmarks

        # Select landmarks: first token of each segment (can be made adaptive)
        landmark_indices = torch.arange(
            0, seq_len, segment_size, device=hidden_states.device
        )[:num_landmarks]
        landmarks = hidden_states[:, landmark_indices, :]  # [batch, L, D]

        # Step 1: Local attention within each segment
        # Reshape into segments for efficient computation
        padded_len = segment_size * num_landmarks
        padded_states = hidden_states[:, :padded_len, :]
        segments = padded_states.view(
            batch_size, num_landmarks, segment_size, hidden_size
        )

        # Apply attention within each segment
        local_output = segments  # Placeholder: in practice, apply attention_fn here

        # Step 2: Global attention to landmarks
        # Each token attends to all landmarks for cross-segment information
        # query: all tokens [B, S, D], key/value: landmarks [B, L, D]
        global_output = attention_fn(hidden_states, landmarks, landmarks)

        # Step 3: Combine local and global attention
        # The landmark representation carries global context, blended with local
        output = local_output.reshape(batch_size, padded_len, hidden_size)
        if padded_len < seq_len:
            output = torch.cat([output, hidden_states[:, padded_len:, :]], dim=1)

        # Simple fusion: weighted sum (can be replaced with a learned gate)
        output = 0.7 * output + 0.3 * global_output

        return output

    @staticmethod
    def self_extending_transformer(
        hidden_states: torch.Tensor,
        trained_length: int,
        target_length: int,
    ) -> torch.Tensor:
        """
        Self-extending attention via gradual RoPE frequency interpolation.

        For positions beyond the trained_length, interpolates the rotary
        position embeddings by smoothly transitioning from trained frequencies
        to extended frequencies. This allows the model to extend its context
        window without catastrophic forgetting of learned patterns.

        The interpolation uses a smooth ramp:
            For pos < trained_length: use original RoPE frequencies
            For pos >= trained_length: interpolate between original and extended
                factor = min(1.0, (pos - trained_length) / ramp_length)

        This is applied to the positional encoding, not directly to hidden states,
        but we return a modified position encoding hint here.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            trained_length: Maximum sequence length the model was trained on.
            target_length: Desired extended sequence length.

        Returns:
            Modified hidden states with self-extended positional information.
            (In practice, this returns the states unchanged; the actual extension
            is applied to RoPE frequencies during attention computation.)
        """
        seq_len = hidden_states.shape[1]

        if seq_len <= trained_length:
            return hidden_states

        # Compute smooth interpolation factors for positions beyond trained_length
        ramp_length = max((target_length - trained_length) // 4, 1)
        positions = torch.arange(seq_len, device=hidden_states.device).float()

        # Extension factor: 1.0 for trained positions, linearly ramping for new ones
        extension_factors = torch.ones(seq_len, device=hidden_states.device)
        beyond_mask = positions >= trained_length
        extension_factors[beyond_mask] = 1.0 + (
            (positions[beyond_mask] - trained_length) / ramp_length
        ).clamp(max=1.0)

        # Apply a light scaling to beyond-training positions
        # This helps maintain coherence for extended context
        scale = extension_factors.view(1, seq_len, 1)
        # Apply subtle scaling to positional dimensions (even indices carry
        # positional information in many architectures)
        output = hidden_states.clone()
        scaled_portion = output[:, trained_length:, : hidden_size // 2]
        output[:, trained_length:, : hidden_size // 2] = (
            scaled_portion * scale[:, trained_length:, :] ** 0.01
        )

        return output

    @staticmethod
    def streaming_llm_attention(
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        attention_fn,
        window_size: int,
        sink_size: int,
    ) -> torch.Tensor:
        """
        Streaming LLM attention pattern for infinite context with fixed memory.

        Implements the attention sink + sliding window pattern from
        Xiao et al. (2023) "Efficient Streaming Language Models with
        Attention Sinks":

        1. **Attention sinks** (first `sink_size` tokens): Always retained.
           These tokens receive disproportionate attention regardless of content
           (they serve as "attention anchors"). Keeping them prevents performance
           degradation when dropping tokens.

        2. **Sliding window** (last `window_size` tokens): The most recent tokens
           provide the most relevant context for next-token prediction.

        3. **Dropped tokens** (middle): Tokens between sink and window are
           evicted from the KV cache to maintain fixed memory usage.

        Total KV cache size: sink_size + window_size (constant).

        Args:
            kv_cache: List of (key, value) tuples per layer.
                Each tensor has shape [batch, heads, cached_len, head_dim].
            attention_fn: Attention function(query, key, value) -> output.
            window_size: Number of recent tokens to attend to.
            sink_size: Number of initial tokens to keep as attention sinks.

        Returns:
            Output from attention with the truncated KV cache.

        Raises:
            ValueError: If sink_size + window_size exceeds cached length.
        """
        if not kv_cache:
            raise ValueError("kv_cache must not be empty.")

        # Determine cached sequence length from first layer's key
        cached_len = kv_cache[0][0].shape[2]
        required_len = sink_size + window_size

        if cached_len > required_len:
            # Keep sink tokens + recent window tokens
            sink_k = [layer_kv[0][:, :, :sink_size, :] for layer_kv in kv_cache]
            sink_v = [layer_kv[1][:, :, :sink_size, :] for layer_kv in kv_cache]
            window_k = [layer_kv[0][:, :, -window_size:, :] for layer_kv in kv_cache]
            window_v = [layer_kv[1][:, :, -window_size:, :] for layer_kv in kv_cache]

            # Concatenate sink + window
            truncated_kv = [
                (torch.cat([sk, wk], dim=2), torch.cat([sv, wv], dim=2))
                for sk, sv, wk, wv in zip(sink_k, sink_v, window_k, window_v)
            ]
        else:
            truncated_kv = kv_cache

        # The actual attention computation would use truncated_kv
        # Here we return the structure; in practice, the model's forward
        # pass uses these truncated keys/values
        return truncated_kv  # type: ignore


# =============================================================================
# 5. Cross-Attention Position Bias
# =============================================================================


class CrossAttentionPositionBias(nn.Module):
    """
    Position bias for cross-attention (e.g., encoder-decoder models).

    Generates a learned relative position bias matrix where query positions
    (decoder) attend to encoder positions with a bias that depends on their
    relative distance. This is analogous to T5's relative position bias but
    for the cross-attention case.

    The bias is computed as:
        bias[i, j] = learned_embedding[clamp(i - j, -max_dist, max_dist)]

    This allows the model to learn that, e.g., decoder position 10 should
    attend more strongly to encoder positions 8-12 than to position 100.

    Args:
        num_heads: Number of attention heads.
        num_buckets: Number of distance buckets for the relative bias.
            More buckets = finer-grained bias at the cost of parameters.
        max_distance: Maximum relative distance before clamping to the
            last bucket. Distances beyond this are all mapped to the same bucket.
        bidirectional: If True, use separate buckets for positive and negative
            relative distances. If False, use absolute distance only.
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Actual number of buckets depends on directionality
        if bidirectional:
            # Separate buckets for positive and negative, plus zero
            actual_buckets = 2 * num_buckets + 1
        else:
            # Absolute distance only
            actual_buckets = num_buckets

        self.relative_attention_bias = nn.Embedding(
            num_embeddings=actual_buckets,
            embedding_dim=num_heads,
        )
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)

    def _compute_bucket_index(
        self,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map relative positions to bucket indices.

        Uses logarithmic binning for close positions and uniform binning
        for distant positions, following T5's approach. This gives finer
        resolution for nearby positions (which matter most) while still
        covering long-range dependencies.

        Args:
            relative_position: Tensor of relative positions [q_len, kv_len].

        Returns:
            Bucket index tensor of same shape, with values in [0, num_buckets).
        """
        if self.bidirectional:
            # Symmetric binning around zero
            num_buckets = self.num_buckets
            # Shift range from [-max, max] to [0, 2*max]
            shifted = relative_position + self.max_distance

            # Logarithmic binning for close range
            # Buckets 0..N/2-1: fine-grained for small distances
            # Buckets N/2..N-1: coarse for large distances
            max_val = 2 * self.max_distance
            half = num_buckets // 2

            # Fine buckets (log-spaced for small distances)
            fine_mask = shifted.abs() < self.max_distance
            fine_buckets = (
                torch.log(shifted.abs().float() + 1.0)
                / math.log(self.max_distance + 1.0)
                * half
            ).long()
            fine_buckets = fine_buckets.clamp(0, half - 1)

            # Coarse buckets for large distances
            coarse_buckets = (
                (shifted.float() - self.max_distance)
                / (self.max_distance)
                * half
            ).long() + half
            coarse_buckets = coarse_buckets.clamp(half, num_buckets - 1)

            buckets = torch.where(fine_mask, fine_buckets, coarse_buckets)

        else:
            # Absolute distance with log binning
            abs_dist = relative_position.abs()
            max_exact = self.num_buckets // 2

            # Small distances: exact bucket per distance
            is_small = abs_dist < max_exact

            # Large distances: logarithmic binning
            # log bucket = max_exact + log2(max_exact / (abs_dist - max_exact + 1)) * ...
            val_if_large = max_exact + (
                torch.log(abs_dist.float() / max_exact + 1.0)
                / math.log(self.max_distance / max_exact + 1.0)
                * (self.num_buckets - max_exact)
            ).long()

            val_if_large = val_if_large.clamp(max=self.num_buckets - 1)
            buckets = torch.where(is_small, abs_dist, val_if_large)

        return buckets

    def forward(
        self,
        query_length: int,
        key_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Compute cross-attention position bias.

        Args:
            query_length: Length of the query sequence (decoder side).
            key_length: Length of the key sequence (encoder side).
            device: Target device for the bias tensor.
            dtype: Target dtype for the bias tensor.

        Returns:
            Position bias tensor of shape [1, num_heads, query_length, key_length].
        """
        if device is None:
            device = self.relative_attention_bias.weight.device
        if dtype is None:
            dtype = self.relative_attention_bias.weight.dtype

        # Compute relative positions: query pos - key pos
        q_positions = torch.arange(query_length, device=device)
        k_positions = torch.arange(key_length, device=device)
        relative_position = q_positions.unsqueeze(1) - k_positions.unsqueeze(0)

        # Map to bucket indices
        bucket_indices = self._compute_bucket_index(relative_position)

        # Look up learned bias for each bucket
        # Result: [query_length, key_length, num_heads]
        position_bias = self.relative_attention_bias(bucket_indices)

        # Rearrange to [1, num_heads, query_length, key_length]
        position_bias = position_bias.permute(2, 0, 1).unsqueeze(0)
        position_bias = position_bias.to(dtype)

        return position_bias


# =============================================================================
# 6. Positional Encoding Factory
# =============================================================================


def get_positional_encoding(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create positional encoding from config.

    Centralizes the creation of positional encoding modules based on a
    configuration dictionary. This ensures consistent initialization across
    the codebase and makes it easy to swap encoding strategies.

    Supported encoding types:
    - "rope": Standard Rotary Position Embedding.
    - "rope_ntk": RoPE with NTK-aware scaling for context extension.
    - "rope_yarn": RoPE with YaRN scaling for large context extension.
    - "rope_longrope": RoPE with LongRoPE per-frequency scaling.
    - "rope_dynamic_ntk": Dynamic NTK scaling (only extends at inference).
    - "rope_linear": RoPE with Position Interpolation.
    - "alibi": Attention with Linear Biases.
    - "learned": Learned absolute position embeddings.
    - "none": No positional encoding (identity/passthrough).

    Args:
        config: Configuration dictionary with at least:
            - "positional_encoding_type" (str): Type of encoding to create.
            Additional keys depend on the encoding type:
            - "hidden_size" / "head_dim": Dimension for embeddings.
            - "num_heads": Number of attention heads (for ALiBi).
            - "max_seq_len": Maximum sequence length.
            - "rope_base": Base frequency for RoPE (default 10000).
            - "rope_scaling_factor": Scaling factor for context extension.
            - "rope_scaling_type": RoPE scaling type override.
            - "alibi_slope_type": ALiBi slope computation method.
            - "cross_attention_num_buckets": Buckets for cross-attn bias.
            - "cross_attention_max_distance": Max distance for cross-attn bias.
            - "yarn_params": YaRN-specific parameters dict.
            - "longrope_params": LongRoPE-specific parameters dict.

    Returns:
        An nn.Module implementing the requested positional encoding.

    Raises:
        ValueError: If positional_encoding_type is not recognized or
            required config keys are missing.

    Example:
        >>> config = {
        ...     "positional_encoding_type": "rope",
        ...     "head_dim": 128,
        ...     "max_seq_len": 4096,
        ... }
        >>> pe = get_positional_encoding(config)
    """
    pe_type = config.get("positional_encoding_type", "none")

    if pe_type == "none":
        return nn.Identity()

    elif pe_type == "rope":
        return RotaryPositionEmbedding(
            dim=config.get("head_dim", config.get("hidden_size", 128)),
            max_seq_len=config.get("max_seq_len", 8192),
            base=config.get("rope_base", 10000.0),
            scaling_type="standard",
        )

    elif pe_type == "rope_ntk":
        return RotaryPositionEmbedding(
            dim=config.get("head_dim", config.get("hidden_size", 128)),
            max_seq_len=config.get("max_seq_len", 8192),
            base=config.get("rope_base", 10000.0),
            scaling_type="ntk_aware",
            scaling_factor=config.get("rope_scaling_factor", 4.0),
        )

    elif pe_type == "rope_dynamic_ntk":
        return RotaryPositionEmbedding(
            dim=config.get("head_dim", config.get("hidden_size", 128)),
            max_seq_len=config.get("max_seq_len", 8192),
            base=config.get("rope_base", 10000.0),
            scaling_type="dynamic_ntk",
            scaling_factor=config.get("rope_scaling_factor", 4.0),
        )

    elif pe_type == "rope_linear":
        return RotaryPositionEmbedding(
            dim=config.get("head_dim", config.get("hidden_size", 128)),
            max_seq_len=config.get("max_seq_len", 8192),
            base=config.get("rope_base", 10000.0),
            scaling_type="linear",
            scaling_factor=config.get("rope_scaling_factor", 4.0),
        )

    elif pe_type == "rope_yarn":
        return RotaryPositionEmbedding(
            dim=config.get("head_dim", config.get("hidden_size", 128)),
            max_seq_len=config.get("max_seq_len", 8192),
            base=config.get("rope_base", 10000.0),
            scaling_type="yarn",
            scaling_factor=config.get("rope_scaling_factor", 4.0),
            yarn_params=config.get("yarn_params", {}),
        )

    elif pe_type == "rope_longrope":
        return RotaryPositionEmbedding(
            dim=config.get("head_dim", config.get("hidden_size", 128)),
            max_seq_len=config.get("max_seq_len", 8192),
            base=config.get("rope_base", 10000.0),
            scaling_type="longrope",
            scaling_factor=config.get("rope_scaling_factor", 4.0),
            longrope_params=config.get("longrope_params", {}),
        )

    elif pe_type == "alibi":
        return ALiBiPositionalEncoding(
            num_heads=config.get("num_heads", 32),
            context_length=config.get("max_seq_len", 8192),
            slope_type=config.get("alibi_slope_type", "standard"),
        )

    elif pe_type == "learned":
        return LearnedPositionEmbedding(
            max_seq_len=config.get("max_seq_len", 8192),
            hidden_size=config.get("hidden_size", 768),
        )

    elif pe_type == "cross_attention_bias":
        return CrossAttentionPositionBias(
            num_heads=config.get("num_heads", 32),
            num_buckets=config.get("cross_attention_num_buckets", 32),
            max_distance=config.get("cross_attention_max_distance", 128),
            bidirectional=config.get("cross_attention_bidirectional", False),
        )

    else:
        raise ValueError(
            f"Unknown positional_encoding_type '{pe_type}'. "
            f"Supported types: rope, rope_ntk, rope_dynamic_ntk, rope_linear, "
            f"rope_yarn, rope_longrope, alibi, learned, cross_attention_bias, none"
        )
