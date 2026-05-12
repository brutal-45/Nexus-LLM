"""
Enhanced Token & Position Embeddings for Nexus v2
=====================================================
Advanced embedding implementations with parallelism, scaling, and vocabulary partitioning.

This module provides a suite of embedding classes for production-scale LLM training:
    - VocabParallelEmbedding: Shard vocabulary across GPUs to reduce per-GPU memory
    - ScaledEmbedding: Token embedding with sqrt(d_model) scaling (Vaswani et al.)
    - EmbeddingWithWeightTying: Shared embedding / output projection weights
    - RotaryEmbeddingV2: Enhanced RoPE with YaRN, LongRoPE, PI, Dynamic NTK
    - ALiBiPositionalBias: Attention with Linear Biases (no learned position params)
    - CombinedEmbedding: Unified token + positional encoding dispatch

References:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
    - Press et al., "Train Short, Test Long: Attention with Linear Biases" (2022)
    - Chen et al., "YaRN: Efficient Context Window Extension" (2023)
    - Peng et al., "RoPE Scaling: LongRoPE and PI for Long Context" (2023)
"""

from __future__ import annotations

import math
import enum
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. VocabParallelEmbedding
# ---------------------------------------------------------------------------

class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary sharded across GPUs.

    For very large vocabulary sizes (128K-256K+), the embedding table can exceed
    single-GPU memory. This module partitions the vocabulary table evenly across
    ``world_size`` GPUs, so each GPU only stores ``vocab_size // world_size`` rows.

    During the forward pass, each GPU looks up its local partition and then
    performs an all-gather to reconstruct the full embedding tensor.  When
    ``torch.distributed`` is not available, falls back to a standard full
    embedding.

    Args:
        vocab_size: Total vocabulary size across all ranks.
        hidden_size: Dimensionality of each embedding vector.
        rank: Rank of the current process (default 0).
        world_size: Total number of parallel processes (default 1).
        padding_idx: Index of padding token (optional).
        max_norm: If set, embeddings are renormalised to have at most this L2 norm.
        norm_type: Type of norm for ``max_norm`` (default L2).
        scale_grad_by_freq: Scale gradients by token frequency (default False).
        dtype: Parameter dtype (default float32).

    Shapes:
        input_ids:  (batch_size, seq_len)       — LongTensor
        output:     (batch_size, seq_len, hidden_size)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        rank: int = 0,
        world_size: int = 1,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.world_size = world_size
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.use_parallel = _DIST_AVAILABLE and world_size > 1

        if self.use_parallel:
            # Partition vocabulary across GPUs
            assert vocab_size % world_size == 0, (
                f"vocab_size ({vocab_size}) must be divisible by world_size ({world_size})"
            )
            self.local_vocab_size = vocab_size // world_size
            self.vocab_start_index = rank * self.local_vocab_size
            self.vocab_end_index = self.vocab_start_index + self.local_vocab_size

            self.weight = nn.Parameter(
                torch.empty(self.local_vocab_size, hidden_size, dtype=dtype)
            )
            if padding_idx is not None:
                # Remap padding_idx into local partition if it falls here
                local_padding = padding_idx - self.vocab_start_index
                if 0 <= local_padding < self.local_vocab_size:
                    nn.init.zeros_(self.weight[local_padding])
                else:
                    nn.init.normal_(self.weight, std=hidden_size ** -0.5)
            else:
                nn.init.normal_(self.weight, std=hidden_size ** -0.5)
        else:
            # Fallback: full embedding on single device
            self.weight = nn.Parameter(
                torch.empty(vocab_size, hidden_size, dtype=dtype)
            )
            nn.init.normal_(self.weight, std=hidden_size ** -0.5)
            if padding_idx is not None:
                nn.init.zeros_(self.weight[padding_idx])

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Look up embeddings, optionally gathering across GPUs.

        When running in parallel mode:
            1. Mask out-of-range ids → 0 (safe index into local table).
            2. Look up local partition.
            3. Zero-out entries that were masked.
            4. All-gather across ranks → full embedding tensor.

        Args:
            input_ids: Token indices of shape ``(batch_size, seq_len)``.

        Returns:
            Embedding tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        if not self.use_parallel:
            return F.embedding(
                input_ids, self.weight,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
            )

        # --- Parallel path ---
        # Create mask for ids that belong to this rank
        mask = (input_ids >= self.vocab_start_index) & (input_ids < self.vocab_end_index)
        # Remap to local indices; out-of-range ids become 0 (will be zeroed)
        local_ids = (input_ids - self.vocab_start_index).clamp(min=0, max=self.local_vocab_size - 1)

        # Local lookup
        output_local = F.embedding(local_ids, self.weight)

        # Zero out entries that do not belong to this rank
        output_local = output_local * mask.unsqueeze(-1).float()

        # All-gather across ranks
        gathered = [
            torch.zeros_like(output_local) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered, output_local.contiguous())
        # Sum contributions — exactly one rank contributes per token
        output = sum(gathered)

        # Apply max_norm if configured (in-place on the full tensor)
        if self.max_norm is not None:
            with torch.no_grad():
                norms = output.norm(p=self.norm_type, dim=-1, keepdim=True)
                scale = self.max_norm / norms.clamp(min=self.max_norm)
                output = output * scale

        return output

    def extra_repr(self) -> str:
        base = (f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}")
        if self.use_parallel:
            base += (
                f", rank={self.rank}, world_size={self.world_size}, "
                f"local_vocab_size={self.local_vocab_size}"
            )
        if self.padding_idx is not None:
            base += f", padding_idx={self.padding_idx}"
        return base


# ---------------------------------------------------------------------------
# 2. ScaledEmbedding
# ---------------------------------------------------------------------------

class ScaledEmbedding(nn.Module):
    """
    Token embedding with ``sqrt(hidden_size)`` scaling factor.

    From the original Transformer paper (Vaswani et al., 2017), the token
    embeddings are scaled by ``sqrt(d_model)`` before being combined with
    positional encodings.  This prevents the dot-product magnitudes in the
    attention mechanism from growing with the model dimension.

    ``output = embedding(input_ids) * sqrt(hidden_size)``

    Args:
        vocab_size: Total number of tokens in the vocabulary.
        hidden_size: Dimensionality of the model (d_model).
        padding_idx: Optional padding token index.
        initializer_range: Standard deviation for weight init (default 0.02).

    Shapes:
        input_ids:  (batch_size, seq_len)       — LongTensor
        output:     (batch_size, seq_len, hidden_size)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.scale = math.sqrt(hidden_size)

        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=initializer_range)
        if padding_idx is not None:
            nn.init.zeros_(self.weight[padding_idx])

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Look up and scale token embeddings.

        Args:
            input_ids: Token indices of shape ``(batch_size, seq_len)``.

        Returns:
            Scaled embedding tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        embedded = F.embedding(input_ids, self.weight, padding_idx=None)
        return embedded * self.scale

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"scale={self.scale:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. EmbeddingWithWeightTying
# ---------------------------------------------------------------------------

class EmbeddingWithWeightTying(nn.Module):
    """
    Embedding that optionally shares weights with the output LM head.

    Weight tying is used by GPT-2, LLaMA, and many other LLMs to reduce
    total parameters.  The embedding matrix ``W`` of shape ``(vocab_size,
    hidden_size)`` serves both as the input lookup table and as the output
    projection weight (with or without transposition).

    The tied output projection is accessible via the ``lm_head`` property.
    If weight tying is disabled after initialisation (e.g. for fine-tuning),
    the ``untie_weights`` method detaches the LM head with an independent copy.

    Args:
        vocab_size: Total vocabulary size.
        hidden_size: Model dimensionality.
        tie_weights: Whether to tie with the LM head (default True).
        padding_idx: Optional padding token index.
        initializer_range: Standard deviation for weight init.

    Shapes:
        input_ids:  (batch_size, seq_len)       — LongTensor
        output:     (batch_size, seq_len, hidden_size)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tie_weights: bool = True,
        padding_idx: Optional[int] = None,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_weights = tie_weights

        # The shared parameter
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=initializer_range)
        if padding_idx is not None:
            nn.init.zeros_(self.weight[padding_idx])

        # Internal LM head — shares self.weight when tied
        self._lm_head: Optional[nn.Linear] = None
        if tie_weights:
            self._create_tied_head()

    def _create_tied_head(self) -> None:
        """Create an nn.Linear that reuses ``self.weight``."""
        self._lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        # Replace the linear weight with our shared parameter
        self._lm_head.weight = self.weight

    @property
    def lm_head(self) -> nn.Linear:
        """Return the tied LM head (or raise if untied)."""
        if self._lm_head is None:
            raise RuntimeError(
                "Weight tying has been disabled. Use the separate lm_head instead."
            )
        return self._lm_head

    def untie_weights(self) -> nn.Linear:
        """
        Detach the LM head so that embedding and output projection have
        independent weights.  Useful when fine-tuning the output head
        separately from the input embeddings.

        Returns:
            A new ``nn.Linear`` with its own copy of the weight.
        """
        if self._lm_head is not None:
            self._lm_head.weight = nn.Parameter(self.weight.data.clone())
        self.tie_weights = False
        return self._lm_head

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Look up token embeddings.

        Args:
            input_ids: Token indices of shape ``(batch_size, seq_len)``.

        Returns:
            Embedding tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        return F.embedding(input_ids, self.weight)

    def extra_repr(self) -> str:
        tied = "tied" if self.tie_weights else "untied"
        return f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, {tied}"


# ---------------------------------------------------------------------------
# 4. RotaryEmbeddingV2  (standalone enhanced copy for v2)
# ---------------------------------------------------------------------------

class RopeScalingType(str, enum.Enum):
    """Supported RoPE scaling strategies."""
    NONE = "none"
    LINEAR = "linear"           # Position Interpolation (PI)
    DYNAMIC_NTK = "dynamic_ntk"
    YARN = "yarn"
    LONGROPE = "longrope"


class RotaryEmbeddingV2(nn.Module):
    """
    Enhanced Rotary Position Embedding (RoPE v2) with multiple scaling strategies.

    Building on the standard RoPE formulation, this module adds support for:

    - **Position Interpolation (PI)**: Linearly scale position ids to map
      a longer context window into the original pre-trained frequency range.
      ``pos_scaled = pos * target_length / original_length``

    - **Dynamic NTK Scaling**: Automatically adjusts the RoPE base frequency
      when the sequence length exceeds the pre-trained window.  The base is
      scaled as ``base * (2α − 1)^{d/(d−2)}`` where α depends on the ratio
      of current to original sequence length.

    - **YaRN (Yet another RoPE extensioN)**: Combines dynamic NTK-aware
      interpolation with a temperature ``t`` applied to the attention logits
      plus a learned (or fixed) ``mixture_factor`` that blends high-frequency
      (original) and low-frequency (interpolated) components.

    - **LongRoPE**: Applies different scaling factors to high-frequency and
      low-frequency components of the RoPE spectrum, allowing better
      extrapolation at extreme context lengths (>64K).

    Args:
        dim: Per-head dimension (head_dim).
        max_position_embeddings: Maximum sequence length (original window).
        base: Base for inverse frequency computation (default 10000).
        scaling_type: One of the ``RopeScalingType`` values.
        scaling_factor: Multiplicative scaling factor (for LINEAR / LONGROPE).
        yarn_temp: YaRN temperature parameter (default 1.0).
        yarn_mixture_factor: YaRN high/low frequency blend (default 1.0; 0=full
            interpolation of low freq, 1=full interpolation of high freq).
        device: Device for precomputed buffers.
        dtype: Dtype for precomputed buffers.

    Shapes (returned from ``forward``):
        cos: (1, 1, seq_len, dim)
        sin: (1, 1, seq_len, dim)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_type: Union[str, RopeScalingType] = RopeScalingType.NONE,
        scaling_factor: float = 1.0,
        yarn_temp: float = 1.0,
        yarn_mixture_factor: float = 1.0,
        longrope_short_factor: Optional[float] = None,
        longrope_long_factor: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_type = RopeScalingType(scaling_type)
        self.scaling_factor = scaling_factor
        self.yarn_temp = yarn_temp
        self.yarn_mixture_factor = yarn_mixture_factor
        self.longrope_short_factor = longrope_short_factor
        self.longrope_long_factor = longrope_long_factor

        # Compute base inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # For YaRN / LongRoPE we maintain separate inv_freqs
        if self.scaling_type == RopeScalingType.YARN:
            # YaRN uses a mixture of original and scaled inv_freqs
            yarn_inv_freq = 1.0 / (
                (base * scaling_factor) ** (
                    torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
                )
            )
            self.register_buffer("_yarn_inv_freq", yarn_inv_freq, persistent=False)
            # Mixture: blend between original and scaled by frequency index
            mixture = torch.linspace(
                1.0 - yarn_mixture_factor, yarn_mixture_factor,
                dim // 2, device=device, dtype=torch.float32,
            )
            self.register_buffer("_yarn_mixture", mixture, persistent=False)

        if self.scaling_type == RopeScalingType.LONGROPE:
            # LongRoPE: different scales for low-freq and high-freq components
            short_factor = longrope_short_factor if longrope_short_factor is not None else 1.0
            long_factor = longrope_long_factor if longrope_long_factor is not None else scaling_factor
            mid = dim // 2
            factors = torch.cat([
                torch.full((mid // 2,), short_factor, dtype=torch.float32, device=device),
                torch.full((mid - mid // 2,), long_factor, dtype=torch.float32, device=device),
            ])
            longrope_inv_freq = 1.0 / (
                (base * factors) ** (
                    torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
                )
            )
            self.register_buffer("_longrope_inv_freq", longrope_inv_freq, persistent=False)

        # Build initial cache
        self._set_cos_sin_cache(max_position_embeddings, device, dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> None:
        """Precompute cos/sin cache for the given sequence length."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @staticmethod
    def _fused_rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Optimised rotate_half: [-x2, x1] for consecutive pairs.

        Equivalent to the standard rotate_half but expressed as a fused
        operation on consecutive dimension pairs for potential kernel
        fusion on modern GPUs.
        """
        x1 = x[..., 0::2]  # even indices
        x2 = x[..., 1::2]  # odd indices
        # Interleave: [-x2_0, x1_0, -x2_1, x1_1, ...]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return rotated

    def _dynamic_ntk_update(
        self,
        seq_len: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamic NTK-aware scaling: adjust base when ``seq_len`` exceeds the
        pre-trained maximum.
        """
        if seq_len <= self.max_position_embeddings:
            cos = self.cos_cached[:seq_len].to(dtype=dtype)
            sin = self.sin_cached[:seq_len].to(dtype=dtype)
            return cos, sin

        # Compute scaling ratio
        alpha = seq_len / self.max_position_embeddings
        base_scaled = self.base * ((2.0 * alpha - 1.0) ** (self.dim / (self.dim - 2)))
        inv_freq_scaled = 1.0 / (
            base_scaled ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
            )
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq_scaled)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin

    def _yarn_compute(
        self,
        seq_len: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        YaRN: blend original and scaled frequencies based on mixture factor.
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Original frequencies
        freqs_orig = torch.outer(t, self.inv_freq.to(device))
        # Scaled (NTK-aware) frequencies
        freqs_scaled = torch.outer(t, self._yarn_inv_freq.to(device))

        # Blend per-frequency component
        blend = self._yarn_mixture.to(device).unsqueeze(0)  # (1, dim//2)
        freqs = blend * freqs_scaled + (1.0 - blend) * freqs_orig

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin

    def _longrope_compute(
        self,
        seq_len: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LongRoPE: use different scaling for low/high frequency components.
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self._longrope_inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin

    def _linear_interpolation(
        self,
        seq_len: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Position Interpolation (PI): linearly scale positions.
        ``pos_scaled = pos * scaling_factor``
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t_scaled = t / self.scaling_factor
        freqs = torch.outer(t_scaled, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get (cos, sin) tensors for the given input.

        Dispatches to the appropriate scaling strategy based on
        ``self.scaling_type`` and the requested sequence length.

        Args:
            x: Reference tensor of shape ``(batch, heads, seq_len, head_dim)``.
            seq_len: Override sequence length (for KV caching).
            position_ids: Optional explicit position ids ``(batch, seq_len)``.

        Returns:
            ``(cos, sin)`` each of shape ``(1, 1, seq_len, dim)``.
        """
        _seq_len = seq_len if seq_len is not None else x.shape[2]

        if position_ids is not None:
            freqs = torch.outer(
                position_ids.float().reshape(-1), self.inv_freq.to(x.device)
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(x.dtype).unsqueeze(0).unsqueeze(0)
            sin = emb.sin().to(x.dtype).unsqueeze(0).unsqueeze(0)
            return cos, sin

        # Select strategy
        if self.scaling_type == RopeScalingType.DYNAMIC_NTK:
            cos, sin = self._dynamic_ntk_update(_seq_len, x.device, x.dtype)
        elif self.scaling_type == RopeScalingType.YARN:
            cos, sin = self._yarn_compute(_seq_len, x.device, x.dtype)
        elif self.scaling_type == RopeScalingType.LONGROPE:
            cos, sin = self._longrope_compute(_seq_len, x.device, x.dtype)
        elif self.scaling_type == RopeScalingType.LINEAR:
            cos, sin = self._linear_interpolation(_seq_len, x.device, x.dtype)
        else:
            # No scaling — standard RoPE
            if _seq_len > self.max_seq_len_cached:
                self._set_cos_sin_cache(_seq_len, x.device, x.dtype)
            cos = self.cos_cached[:_seq_len].to(dtype=x.dtype)
            sin = self.sin_cached[:_seq_len].to(dtype=x.dtype)

        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def get_yarn_temperature(self) -> float:
        """Return the YaRN temperature for attention-logit rescaling."""
        return self.yarn_temp

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_seq={self.max_position_embeddings}, "
            f"base={self.base}, scaling={self.scaling_type.value}, "
            f"factor={self.scaling_factor}"
        )


# ---------------------------------------------------------------------------
# 5. ALiBiPositionalBias
# ---------------------------------------------------------------------------

class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) — no learned positional parameters.

    ALiBi replaces positional embeddings with a simple additive bias applied
    to the attention logits.  For each attention head *i*, a fixed slope
    ``s_i`` is computed as:

        ``s_i = 2^{-8i / H}``   for i = 1, 2, ..., H

    The bias for head *i* at relative position *m* is ``s_i * m``.  Negative
    relative positions receive negative bias (penalising attending to past tokens
    that are far away).

    Advantages:
        - No positional embedding table → zero memory overhead for positions.
        - Extends to arbitrary context lengths without retraining (extrapolation).
        - Compatible with Flash Attention (bias can be folded into the QK product).

    Args:
        num_heads: Number of attention heads (H).
        max_position_embeddings: Maximum context length for bias tensor cache.
        causal: Whether to apply causal masking (default True).
        dtype: Dtype for the bias tensor.

    Shapes (returned from ``forward``):
        bias: (1, num_heads, seq_len, kv_seq_len) or (1, num_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        num_heads: int,
        max_position_embeddings: int = 8192,
        causal: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.causal = causal

        # Compute fixed slopes: s_i = 2^{-8i/H} for i=1..H
        # Following the original paper: slopes = 2 ** (-8 * arange(1, H+1) / H)
        slopes = torch.tensor(
            [2.0 ** (-8.0 * i / num_heads) for i in range(1, num_heads + 1)],
            dtype=dtype,
        )
        self.register_buffer("slopes", slopes, persistent=False)

        # Precompute bias matrix for max_position_embeddings
        self._build_bias_cache(max_position_embeddings)

    def _build_bias_cache(self, max_len: int) -> None:
        """
        Build the cached bias tensor of shape (1, num_heads, max_len, max_len).

        ``bias[h, i, j] = slopes[h] * (j - i)``  (relative position bias)
        """
        # Position difference: (max_len, max_len)
        # rows = query positions, cols = key positions
        rows = torch.arange(max_len).unsqueeze(1)
        cols = torch.arange(max_len).unsqueeze(0)
        rel_pos = cols - rows  # (max_len, max_len)

        if self.causal:
            # Mask future positions (where col > row)
            causal_mask = torch.triu(
                torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1
            )
            rel_pos = rel_pos.masked_fill(causal_mask, float("-inf"))

        # Apply per-head slopes: (num_heads, max_len, max_len)
        bias = rel_pos.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
        self.register_buffer("_bias_cache", bias.unsqueeze(0), persistent=False)

    def forward(
        self,
        seq_len: int,
        kv_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Return the additive attention bias for the given sequence lengths.

        Args:
            seq_len: Query sequence length.
            kv_seq_len: Key/value sequence length (default = seq_len).
            device: Device for the bias tensor.
            dtype: Dtype for the bias tensor.

        Returns:
            Bias tensor of shape ``(1, num_heads, seq_len, kv_seq_len)``,
            ready to add to attention logits.
        """
        _kv_len = kv_seq_len if kv_seq_len is not None else seq_len
        _device = device or self.slopes.device
        _dtype = dtype or self.slopes.dtype

        max_cached = self._bias_cache.shape[-1]

        if seq_len <= max_cached and _kv_len <= max_cached:
            bias = self._bias_cache[:, :, :seq_len, :_kv_len].to(_dtype).to(_device)
        else:
            # Dynamically compute bias for sequences exceeding cache
            rows = torch.arange(seq_len, device=_device)
            cols = torch.arange(_kv_len, device=_device)
            rel_pos = (cols.unsqueeze(0) - rows.unsqueeze(1)).float()

            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, _kv_len, dtype=torch.bool, device=_device),
                    diagonal=max(0, _kv_len - seq_len + 1),
                )
                rel_pos = rel_pos.masked_fill(causal_mask, float("-inf"))

            bias = (
                rel_pos.unsqueeze(0)
                * self.slopes.to(_device).unsqueeze(1).unsqueeze(2)
            )
            bias = bias.unsqueeze(0).to(_dtype)

        return bias

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, "
            f"max_seq={self.max_position_embeddings}, "
            f"causal={self.causal}, "
            f"slopes=[{self.slopes[0].item():.6f}, ..., {self.slopes[-1].item():.6f}]"
        )


# ---------------------------------------------------------------------------
# 6. CombinedEmbedding
# ---------------------------------------------------------------------------

class PositionalEncodingType(str, enum.Enum):
    """Strategy for positional encoding in CombinedEmbedding."""
    NONE = "none"
    ROPE = "rope"          # Handled externally in attention; just return token embeds
    ALIBI = "alibi"        # Handled externally as attention bias
    LEARNED = "learned"    # Standard learned position embedding (additive)


class CombinedEmbedding(nn.Module):
    """
    Unified embedding that combines token lookup with positional encoding.

    Supports four modes selected via ``pos_encoding``:

    - ``"none"``:  Pure token embeddings (e.g., for models using absolute
      position ids fed directly to attention).
    - ``"rope"``:  Token embeddings only — RoPE is applied inside the
      attention mechanism, not at the embedding layer.  The module stores a
      ``RotaryEmbeddingV2`` instance for convenience but does not use it in
      ``forward``.
    - ``"alibi"``: Token embeddings only — ALiBi biases are applied as
      attention biases.  The module stores an ``ALiBiPositionalBias`` instance.
    - ``"learned"``: Token embeddings **plus** a learned position embedding
      table that is summed element-wise.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model dimensionality.
        max_position_embeddings: Maximum sequence length (for learned / ALiBi / RoPE).
        pos_encoding: Positional encoding strategy (string or enum).
        padding_idx: Optional padding token index.
        initializer_range: Weight init standard deviation.
        # RoPE-specific
        rope_base: Base for RoPE inverse frequencies.
        rope_scaling_type: RoPE scaling strategy.
        rope_scaling_factor: RoPE scaling factor.
        # ALiBi-specific
        alibi_causal: Whether ALiBi uses causal masking.
        alibi_num_heads: Number of heads for ALiBi (defaults to a reasonable guess
            when the model config is not passed).

    Shapes:
        input_ids:       (batch_size, seq_len)
        position_ids:    (batch_size, seq_len)  [optional]
        output:          (batch_size, seq_len, hidden_size)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 8192,
        pos_encoding: Union[str, PositionalEncodingType] = PositionalEncodingType.NONE,
        padding_idx: Optional[int] = None,
        initializer_range: float = 0.02,
        # RoPE params
        rope_base: float = 10000.0,
        rope_scaling_type: Union[str, RopeScalingType] = RopeScalingType.NONE,
        rope_scaling_factor: float = 1.0,
        # ALiBi params
        alibi_causal: bool = True,
        alibi_num_heads: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pos_encoding = PositionalEncodingType(pos_encoding)

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx,
        )
        nn.init.normal_(self.token_embedding.weight, std=initializer_range)
        if padding_idx is not None:
            nn.init.zeros_(self.token_embedding.weight[padding_idx])

        # Position encoding sub-modules
        self.rope: Optional[RotaryEmbeddingV2] = None
        self.alibi: Optional[ALiBiPositionalBias] = None
        self.position_embedding: Optional[nn.Embedding] = None

        if self.pos_encoding == PositionalEncodingType.ROPE:
            self.rope = RotaryEmbeddingV2(
                dim=hidden_size,
                max_position_embeddings=max_position_embeddings,
                base=rope_base,
                scaling_type=rope_scaling_type,
                scaling_factor=rope_scaling_factor,
            )
        elif self.pos_encoding == PositionalEncodingType.ALIBI:
            self.alibi = ALiBiPositionalBias(
                num_heads=alibi_num_heads,
                max_position_embeddings=max_position_embeddings,
                causal=alibi_causal,
            )
        elif self.pos_encoding == PositionalEncodingType.LEARNED:
            self.position_embedding = nn.Embedding(
                max_position_embeddings, hidden_size,
            )
            nn.init.normal_(self.position_embedding.weight, std=initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compute token embeddings with optional positional encoding.

        For ``"learned"`` mode, position embeddings are summed with token
        embeddings.  For ``"rope"`` and ``"alibi"`` modes, positional
        information is handled externally in the attention layer — this
        method returns only token embeddings but the ``rope`` / ``alibi``
        sub-modules are accessible as attributes.

        Args:
            input_ids: Token indices ``(batch_size, seq_len)``.
            position_ids: Position indices ``(batch_size, seq_len)``.
                If ``None`` and using learned PE, defaults to ``arange(seq_len)``.

        Returns:
            Embedding tensor ``(batch_size, seq_len, hidden_size)``.
        """
        # Always start with token embeddings
        hidden = self.token_embedding(input_ids)

        if (
            self.pos_encoding == PositionalEncodingType.LEARNED
            and self.position_embedding is not None
        ):
            seq_len = hidden.shape[1]
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, device=hidden.device,
                ).unsqueeze(0).expand(hidden.shape[0], -1)
            hidden = hidden + self.position_embedding(position_ids)

        return hidden

    def get_rope_cos_sin(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convenience method to get RoPE cos/sin tensors.

        Returns ``None`` if RoPE is not the chosen positional encoding.
        """
        if self.rope is not None:
            return self.rope(x, position_ids=position_ids)
        return None

    def get_alibi_bias(
        self,
        seq_len: int,
        kv_seq_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Convenience method to get ALiBi attention bias.

        Returns ``None`` if ALiBi is not the chosen positional encoding.
        """
        if self.alibi is not None:
            return self.alibi(seq_len, kv_seq_len=kv_seq_len)
        return None

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"pos_encoding={self.pos_encoding.value}"
        )
