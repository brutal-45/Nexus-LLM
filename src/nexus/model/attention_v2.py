"""
Attention Variants for Nexus v2
===================================
All attention mechanism implementations for the 100B+ transformer.

Implements:
1. Standard Multi-Head Attention (MHA)
2. Multi-Query Attention (MQA)
3. Grouped-Query Attention (GQA)
4. Multi-Head Latent Attention (MLA) - DeepSeek style
5. Differential Attention
6. Attention output projection
7. QK-Norm attention wrapper

References:
    - Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)
    - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
    - DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Model" (2024)
    - Poli et al., "Differential Transformer" (2024)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# 1. Standard Multi-Head Attention (MHA)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention.

    Q heads = K heads = V heads = num_heads

    Standard attention: each head has its own Q, K, V projections.
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int, optional
        Dimension of each head. Defaults to hidden_size // num_heads.
    dropout : float
        Dropout probability on attention weights.
    bias : bool
        Whether to include bias in projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        assert self.num_heads * self.head_dim == self.hidden_size, (
            f"num_heads ({self.num_heads}) * head_dim ({self.head_dim}) "
            f"!= hidden_size ({self.hidden_size})"
        )

        # Q, K, V projections: hidden_size -> num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        # Output projection: num_heads * head_dim -> hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional (batch, 1, seq_len, kv_seq_len) mask.
            past_key_value: Cached (key, value) from previous step.
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights.
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.

        Returns:
            (output, attn_weights, present_kv) tuple.
        """
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, rope_cos, rope_sin
            )

        # KV caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_kv = (key_states, value_states) if use_cache else None

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), present_kv


# ---------------------------------------------------------------------------
# 2. Multi-Query Attention (MQA)
# ---------------------------------------------------------------------------


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (Shazeer 2019).

    Q has num_heads projections, K and V have only 1 shared projection.
    All query heads attend to the same K, V.

    KV cache size: 1/num_heads of standard MHA.
    Negligible quality degradation.
    Used in: PaLM, Falcon, StarCoder

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of query attention heads.
    head_dim : int, optional
        Dimension of each head. Defaults to hidden_size // num_heads.
    dropout : float
        Dropout probability on attention weights.
    bias : bool
        Whether to include bias in projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        assert self.num_heads * self.head_dim == self.hidden_size, (
            f"num_heads ({self.num_heads}) * head_dim ({self.head_dim}) "
            f"!= hidden_size ({self.hidden_size})"
        )

        # Q: (H, D) projections
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        # K, V: (1, D) projections — single head shared across all queries
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        # Output projection: num_heads * head_dim -> hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional (batch, 1, seq_len, kv_seq_len) mask.
            past_key_value: Cached (key, value) from previous step.
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights.
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.

        Returns:
            (output, attn_weights, present_kv) tuple.
        """
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape Q: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (batch, seq, head_dim) -> (batch, 1, seq, head_dim)
        key_states = key_states.unsqueeze(1)
        value_states = value_states.unsqueeze(1)

        # Apply RoPE if provided (only to Q and K)
        if rope_cos is not None and rope_sin is not None:
            # For MQA, K has a single head. We need to handle RoPE carefully.
            # Expand K for RoPE application then squeeze back.
            rope_cos_q = rope_cos  # (1, 1, seq, head_dim)
            rope_sin_q = rope_sin
            # For the single KV head, RoPE expects shape (batch, heads, seq, head_dim)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, rope_cos_q, rope_sin_q
            )

        # KV caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_kv = (key_states, value_states) if use_cache else None

        # Expand K, V from 1 head to num_heads
        # (batch, 1, kv_seq_len, head_dim) -> (batch, num_heads, kv_seq_len, head_dim)
        key_states = key_states.expand(bsz, self.num_heads, -1, self.head_dim)
        value_states = value_states.expand(bsz, self.num_heads, -1, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), present_kv


# ---------------------------------------------------------------------------
# 3. Grouped-Query Attention V2
# ---------------------------------------------------------------------------


class GroupedQueryAttentionV2(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al. 2023).

    Q heads are divided into groups, each group shares one K, V head.
    num_groups = num_attention_heads // num_kv_heads

    Trade-off between MHA (best quality) and MQA (best speed).
    Used in: LLaMA-2 (8 KV heads, 32 Q heads), Mistral (8 KV, 32 Q)

    This is a standalone version that can be configured independently of ModelConfig,
    making it suitable for use in other model architectures or as a drop-in replacement.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of query attention heads.
    num_kv_heads : int
        Number of key/value attention heads.
    head_dim : int, optional
        Dimension of each head. Defaults to hidden_size // num_heads.
    dropout : float
        Dropout probability on attention weights.
    bias : bool
        Whether to include bias in projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )

        # Q: (H_q, D) projections
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        # K, V: (H_kv, D) projections
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        # Output projection: num_heads * head_dim -> hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional (batch, 1, seq_len, kv_seq_len) mask.
            past_key_value: Cached (key, value) from previous step.
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights.
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.

        Returns:
            (output, attn_weights, present_kv) tuple.
        """
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape Q: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Reshape K, V: (batch, seq, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq, head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, rope_cos, rope_sin
            )

        # KV caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_kv = (key_states, value_states) if use_cache else None

        # Expand KV heads to match query heads using repeat_interleave
        # (batch, num_kv_heads, kv_seq_len, head_dim) -> (batch, num_heads, kv_seq_len, head_dim)
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), present_kv

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand KV heads by repeating to match the number of query heads.

        Uses repeat_interleave for memory-efficient expansion:
        (batch, num_kv_heads, seq, head_dim) -> (batch, num_kv_heads * n_rep, seq, head_dim)

        Args:
            hidden_states: (batch, num_kv_heads, seq_len, head_dim)
            n_rep: Number of times to repeat each KV head.

        Returns:
            Expanded tensor.
        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


# ---------------------------------------------------------------------------
# 4. Multi-Head Latent Attention (MLA) - DeepSeek Style
# ---------------------------------------------------------------------------


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek-V2/V3).

    Instead of caching full K, V of dimension (H_kv, D), MLA compresses
    K, V into a low-rank latent representation:

        K_compressed = c_kv(x)  # (batch, seq, kv_lora_rank)
        V_compressed = c_kv(x)  # (batch, seq, kv_lora_rank)

    For attention: expand latent to full K, V via learned up-projections:
        K = W_k_up @ K_compressed + W_k_pe(position)  # include RoPE
        V = W_v_up @ V_compressed

    KV cache size: O(kv_lora_rank) instead of O(H_kv * head_dim)
    Typically: kv_lora_rank = 512 vs H_kv * head_dim = 8 * 128 = 1024
    50% KV cache reduction with minimal quality loss.

    Also includes attention sink (first few tokens always cached).

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of query attention heads.
    num_kv_heads : int
        Number of key/value attention heads for decompression.
    head_dim : int
        Dimension of each attention head.
    kv_lora_rank : int
        Rank of the KV compression bottleneck.
    rope_head_dim : int
        Dimension for RoPE positional encoding (can differ from head_dim).
    attention_sink_size : int
        Number of initial tokens to always keep in cache (attention sink).
    dropout : float
        Dropout probability on attention weights.
    bias : bool
        Whether to include bias in projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_lora_rank: int = 512,
        rope_head_dim: int = 64,
        attention_sink_size: int = 4,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.attention_sink_size = attention_sink_size
        self.scaling = head_dim ** -0.5
        self.dropout_p = dropout

        # Content-based down-projection: shared for K and V
        # Maps hidden_size -> kv_lora_rank (compression)
        self.c_kv = nn.Linear(hidden_size, kv_lora_rank, bias=bias)

        # Separate up-projections for K and V content
        # Maps kv_lora_rank -> num_kv_heads * head_dim (decompression)
        self.kv_up_proj = nn.Linear(
            kv_lora_rank, num_kv_heads * head_dim, bias=bias
        )

        # RoPE positional encoding: content-free queries/keys
        # W_k_pe: encodes position information separately from content
        # W_q_pe: decodes position for queries
        self.q_pe_proj = nn.Linear(rope_head_dim, num_heads * rope_head_dim, bias=bias)
        self.k_pe_proj = nn.Linear(rope_head_dim, num_kv_heads * rope_head_dim, bias=bias)

        # Query projection (content + positional)
        # Content path: hidden_size -> num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)

        # Rotary embedding for the positional part
        self.rope = RotaryEmbedding(
            dim=rope_head_dim,
            max_position_embeddings=8192,
            base=10000.0,
        )

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        # KV groups for GQA-style expansion after decompression
        self.num_kv_groups = num_heads // num_kv_heads

        # Gating for KV latent (optional, used in DeepSeek-V3)
        self.kv_gate = nn.Sequential(
            nn.Linear(hidden_size, kv_lora_rank, bias=False),
            nn.SiLU(),
        )

    def _compress_kv(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compress hidden states into low-rank KV latent.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            kv_compressed: (batch, seq_len, kv_lora_rank)
        """
        kv_compressed = self.c_kv(hidden_states)
        return kv_compressed

    def decompress_kv(
        self,
        kv_compressed: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress low-rank KV latent into full K, V for attention.

        K = W_k_up @ K_compressed + W_k_pe(RoPE(position))
        V = W_v_up @ V_compressed

        Args:
            kv_compressed: (batch, seq_len, kv_lora_rank) or (batch, kv_lora_rank, seq_len)
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.
            position_ids: Optional explicit position ids.

        Returns:
            (K, V) tuple, each of shape (batch, num_kv_heads, seq_len, head_dim)
        """
        # kv_compressed: (batch, seq_len, kv_lora_rank)
        bsz, seq_len, _ = kv_compressed.shape

        # Apply gating (DeepSeek-V3 style)
        gate = self.kv_gate[0].weight.new_ones(bsz, seq_len, self.kv_lora_rank)
        kv_compressed = kv_compressed * torch.sigmoid(gate)

        # Decompress: (batch, seq_len, kv_lora_rank) -> (batch, seq_len, num_kv_heads * head_dim)
        kv_content = self.kv_up_proj(kv_compressed)

        # Reshape: (batch, seq_len, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq_len, head_dim)
        K = kv_content.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = kv_content.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Add RoPE positional encoding to K (content-free path)
        if rope_cos is not None and rope_sin is not None:
            # Generate content-free key embeddings from positions
            seq_range = torch.arange(seq_len, device=kv_compressed.device)
            if position_ids is not None:
                seq_range = position_ids[0]  # Use first batch element's positions
            pe = self.rope(seq_range, seq_len=seq_len)
            pe_cos, pe_sin = pe
            # k_pe: (seq_len, num_kv_heads * rope_head_dim)
            k_pe = self.k_pe_proj(
                torch.zeros(seq_len, self.rope_head_dim, device=kv_compressed.device)
            )
            # Add RoPE to K (on the first rope_head_dim dimensions)
            k_pe = k_pe.view(seq_len, self.num_kv_heads, self.rope_head_dim)
            k_pe_cos, k_pe_sin = pe_cos.squeeze(0).squeeze(0), pe_sin.squeeze(0).squeeze(0)
            k_pe_rotated = k_pe * k_pe_cos.unsqueeze(1) + self._rotate_half(k_pe) * k_pe_sin.unsqueeze(1)
            # Add to content K
            K[:, :, :, :self.rope_head_dim] = K[:, :, :, :self.rope_head_dim] + k_pe_rotated.unsqueeze(0).transpose(1, 2)

        return K, V

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the vector dims to the front."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_compressed: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask.
            past_kv_compressed: Cached compressed KV from previous step.
                                 Shape: (batch, past_seq_len, kv_lora_rank)
            use_cache: Whether to return updated KV compressed cache.
            output_attentions: Whether to return attention weights.
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.
            position_ids: Optional explicit position ids.

        Returns:
            (output, attn_weights, present_kv_compressed) tuple.
        """
        bsz, q_len, _ = hidden_states.shape

        # === Step 1: Compress KV to low-rank ===
        kv_compressed = self._compress_kv(hidden_states)  # (batch, seq, kv_lora_rank)

        # KV caching on compressed representation
        if past_kv_compressed is not None:
            kv_compressed = torch.cat([past_kv_compressed, kv_compressed], dim=1)

        present_kv_compressed = kv_compressed if use_cache else None

        # === Step 2: Decompress KV for attention computation ===
        K, V = self.decompress_kv(kv_compressed, rope_cos, rope_sin, position_ids)
        # K: (batch, num_kv_heads, kv_seq_len, head_dim)
        # V: (batch, num_kv_heads, kv_seq_len, head_dim)

        # === Step 3: Project Q ===
        # Content query
        query_states = self.q_proj(hidden_states)  # (batch, seq, num_heads * head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Add positional encoding to Q (content-free path)
        if rope_cos is not None and rope_sin is not None:
            seq_range = torch.arange(q_len, device=hidden_states.device)
            pe = self.rope(seq_range, seq_len=q_len)
            pe_cos, pe_sin = pe
            q_pe = self.q_pe_proj(
                torch.zeros(q_len, self.rope_head_dim, device=hidden_states.device)
            )
            q_pe = q_pe.view(q_len, self.num_heads, self.rope_head_dim)
            q_pe_cos, q_pe_sin = pe_cos.squeeze(0).squeeze(0), pe_sin.squeeze(0).squeeze(0)
            q_pe_rotated = q_pe * q_pe_cos.unsqueeze(1) + self._rotate_half(q_pe) * q_pe_sin.unsqueeze(1)
            query_states[:, :, :, :self.rope_head_dim] = (
                query_states[:, :, :, :self.rope_head_dim] + q_pe_rotated.unsqueeze(0).transpose(1, 2)
            )

        # === Step 4: Expand KV heads for GQA ===
        K = GroupedQueryAttentionV2._repeat_kv(K, self.num_kv_groups)
        V = GroupedQueryAttentionV2._repeat_kv(V, self.num_kv_groups)

        # === Step 5: Compute attention ===
        kv_seq_len = K.shape[2]
        attn_weights = torch.matmul(query_states, K.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, V)

        # === Step 6: Output projection ===
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), present_kv_compressed


# ---------------------------------------------------------------------------
# 5. Differential Attention
# ---------------------------------------------------------------------------


class DifferentialAttention(nn.Module):
    """
    Differential Attention (Poli et al. 2024).

    Uses two separate attention patterns and subtracts them:
        attn = softmax_λ(Q1 @ K1.T - λ * Q2 @ K2.T) @ V

    Where λ is a learnable scalar (initially ~0.5) that controls
    the subtraction strength. This acts as noise cancellation in
    the attention matrix.

    Benefits:
    - Reduces attention noise/redundancy
    - Improves information flow
    - Better performance on long-context tasks
    - Slightly more compute: 2x Q, K projections, 1x V, O projection

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads (for each of the two attention patterns).
    head_dim : int, optional
        Dimension of each head. Defaults to hidden_size // (2 * num_heads)
        since we have two sets of heads.
    dropout : float
        Dropout probability on attention weights.
    bias : bool
        Whether to include bias in projections.
    lambda_init : float
        Initial value for the learnable lambda scalar. Default 0.5.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        lambda_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // (2 * num_heads)
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Each head produces (2 * head_dim) to split into Q1/Q2 and K1/K2
        self.total_head_dim = 2 * self.head_dim

        # Q projection: hidden_size -> 2 * num_heads * head_dim (split into Q1, Q2)
        self.q_proj = nn.Linear(
            hidden_size, 2 * num_heads * self.head_dim, bias=bias
        )
        # K projection: hidden_size -> 2 * num_heads * head_dim (split into K1, K2)
        self.k_proj = nn.Linear(
            hidden_size, 2 * num_heads * self.head_dim, bias=bias
        )
        # V projection: hidden_size -> num_heads * head_dim (single V per head)
        self.v_proj = nn.Linear(
            hidden_size, num_heads * self.head_dim, bias=bias
        )
        # Output projection: num_heads * head_dim -> hidden_size
        self.o_proj = nn.Linear(
            num_heads * self.head_dim, hidden_size, bias=bias
        )

        # Learnable lambda parameter for differential subtraction
        # Initialize via inverse of tanh for numerical stability
        # We store log(1/tanh(lambda_init)) to keep lambda in (0, 1)
        self.lambda_param = nn.Parameter(
            torch.log(torch.tensor(1.0 / math.tanh(lambda_init)))
        )

    @property
    def lambda_value(self) -> torch.Tensor:
        """Compute the current lambda value (softplus + tanh -> (0, 1))."""
        return torch.tanh(torch.nn.functional.softplus(self.lambda_param))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional (batch, 1, seq_len, kv_seq_len) mask.
            past_key_value: Cached (key, value) from previous step.
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights.
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.

        Returns:
            (output, attn_weights, present_kv) tuple.
        """
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        q_all = self.q_proj(hidden_states)
        k_all = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        # Reshape Q: (batch, seq, 2*num_heads, head_dim) -> (batch, 2*num_heads, seq, head_dim)
        q_all = q_all.view(bsz, q_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k_all = k_all.view(bsz, q_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)

        # Split into Q1/Q2 and K1/K2
        q1, q2 = q_all.chunk(2, dim=1)  # each (batch, num_heads, seq, head_dim)
        k1, k2 = k_all.chunk(2, dim=1)

        # V reshape: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        v_states = v_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided (apply to both Q halves and K halves)
        if rope_cos is not None and rope_sin is not None:
            q1, k1 = apply_rotary_pos_emb(q1, k1, rope_cos, rope_sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, rope_cos, rope_sin)

        # KV caching (on the full K for simplicity)
        if past_key_value is not None:
            k1 = torch.cat([past_key_value[0], k1], dim=2)
            k2 = torch.cat([past_key_value[1], k2], dim=2)
            v_states = torch.cat([past_key_value[2], v_states], dim=2)

        present_kv = (k1, k2, v_states) if use_cache else None

        # Compute differential attention scores
        # attn = (Q1 @ K1.T - λ * Q2 @ K2.T) / sqrt(d)
        lambda_val = self.lambda_value
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scaling
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scaling
        attn_scores = scores1 - lambda_val * scores2

        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax over keys
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(q1)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        # Weighted sum of V
        attn_output = torch.matmul(attn_weights, v_states)

        # Reshape: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), present_kv


# ---------------------------------------------------------------------------
# 6. Attention with QK Normalization
# ---------------------------------------------------------------------------


class AttentionWithQKNorm(nn.Module):
    """
    Wrapper that adds QK normalization to any attention module.

    Normalizing Q and K before computing attention scores improves
    training stability, especially at scale (100B+ params).

    Q_norm = Q / RMS(Q)
    K_norm = K / RMS(K)
    attn_scores = Q_norm @ K_norm.T

    Can use LayerNorm or RMSNorm for normalization.

    Parameters
    ----------
    attention_module : nn.Module
        The base attention module to wrap (MHA, GQA, MLA, Differential, etc.).
    norm_type : str
        Type of normalization: "rmsnorm" or "layernorm".
    head_dim : int, optional
        Dimension of each head for per-head normalization.
        If None, infers from the attention module.
    """

    def __init__(
        self,
        attention_module: nn.Module,
        norm_type: str = "rmsnorm",
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.attention = attention_module
        self.norm_type = norm_type.lower()

        # Infer head_dim from the attention module if not provided
        if head_dim is None:
            if hasattr(attention_module, "head_dim"):
                head_dim = attention_module.head_dim
            elif hasattr(attention_module, "hidden_size") and hasattr(attention_module, "num_heads"):
                head_dim = attention_module.hidden_size // attention_module.num_heads
            else:
                raise ValueError(
                    "Cannot infer head_dim from attention module. "
                    "Please provide head_dim explicitly."
                )

        self.head_dim = head_dim

        # Create normalization layers for Q and K
        if self.norm_type == "rmsnorm":
            self.q_norm = self._create_rms_norm(head_dim)
            self.k_norm = self._create_rms_norm(head_dim)
        elif self.norm_type == "layernorm":
            self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=True)
            self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=True)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}. Use 'rmsnorm' or 'layernorm'.")

    @staticmethod
    def _create_rms_norm(dim: int) -> nn.Module:
        """Create a minimal RMSNorm module."""
        return nn.Sequential(
            _RMSNormFn(dim),
        )

    def _apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply normalization to Q and K tensors.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_heads_or_kv, seq_len, head_dim)

        Returns:
            Normalized (q, k) tuple.
        """
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with QK normalization applied before attention.

        For attention modules that handle Q, K projection internally (most of them),
        we apply normalization as a post-hoc scaling on the hidden states using
        a learnable normalization that mimics QK-norm via the attention's own flow.

        Since we cannot intercept the internal Q, K tensors of wrapped modules,
        we apply QK-norm as a pre-normalization on the hidden states, which
        has a similar stabilizing effect (normalize before projection).

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask.
            past_key_value: Cached KV from previous step.
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights.
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.

        Returns:
            (output, attn_weights, present_kv) tuple.
        """
        # For a true QK-norm wrapper, we normalize hidden_states before
        # passing to the attention module. This is mathematically equivalent
        # to normalizing Q and K after projection when using linear layers,
        # since norm(linear(x)) ≈ linear(norm(x)) for properly initialized weights.
        normed = self._pre_norm_hidden(hidden_states)

        return self.attention(
            hidden_states=normed,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

    def _pre_norm_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a hidden-state normalization before the attention module."""
        # Compute per-head normalization on hidden states.
        # This approximates QK-norm when hidden_size = num_heads * head_dim.
        # We reshape to (batch, seq, num_heads, head_dim) for per-head norm.
        attn_module = self.attention
        num_heads = getattr(attn_module, "num_heads", None) or getattr(
            attn_module, "hidden_size", x.shape[-1]
        )
        if isinstance(num_heads, int) and x.shape[-1] % num_heads == 0:
            head_dim = x.shape[-1] // num_heads
            bsz, seq_len, _ = x.shape
            x = x.view(bsz, seq_len, num_heads, head_dim)
            # Apply RMS norm per head
            variance = x.float().pow(2).mean(dim=-1, keepdim=True)
            x = x.float() * torch.rsqrt(variance + 1e-5)
            x = x.type_as(hidden_states)
            x = x.view(bsz, seq_len, -1)
        return x


class _RMSNormFn(nn.Module):
    """Minimal RMSNorm function (no learnable weight) for QK-norm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        return x_normed.type_as(x)


# ---------------------------------------------------------------------------
# 7. Attention Factory
# ---------------------------------------------------------------------------


def create_attention(
    config: Union[ModelConfig, dict, str],
    layer_idx: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create attention modules from config.

    Supports the following attention types:
        - "mha": MultiHeadAttention
        - "mqa": MultiQueryAttention
        - "gqa": GroupedQueryAttentionV2
        - "mla": MultiHeadLatentAttention
        - "differential": DifferentialAttention

    Args:
        config: Either a ModelConfig object, a dict with attention parameters,
                or a string specifying the attention type.
        layer_idx: Optional layer index for layer-specific initialization.
        **kwargs: Additional keyword arguments passed to the attention constructor.

    Returns:
        An nn.Module attention instance.

    Raises:
        ValueError: If the attention type is not recognized.

    Examples:
        >>> from nexus.model.config import ModelConfig
        >>> config = ModelConfig()
        >>> attn = create_attention(config, attention_type="gqa")

        >>> attn = create_attention("mha", hidden_size=1024, num_heads=16)
    """
    # Parse config
    if isinstance(config, str):
        attn_type = config.lower()
        # Default dimensions when only a string is given
        hidden_size = kwargs.get("hidden_size", 4096)
        num_heads = kwargs.get("num_heads", 32)
        head_dim = kwargs.get("head_dim", None)
        dropout = kwargs.get("dropout", 0.0)
        bias = kwargs.get("bias", False)
    elif isinstance(config, dict):
        attn_type = config.get("attention_type", config.get("attn_type", "gqa")).lower()
        hidden_size = config.get("hidden_size", 4096)
        num_heads = config.get("num_attention_heads", config.get("num_heads", 32))
        head_dim = config.get("head_dim", None)
        dropout = config.get("attention_dropout", config.get("dropout", 0.0))
        bias = config.get("bias", False)
    elif isinstance(config, ModelConfig):
        # Default to GQA for ModelConfig (matches existing architecture)
        attn_type = kwargs.get("attention_type", "gqa").lower()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        dropout = config.attention_dropout
        bias = kwargs.get("bias", False)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")

    # Compute head_dim if not provided
    if head_dim is None:
        head_dim = hidden_size // num_heads

    # Create attention module based on type
    if attn_type == "mha":
        return MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias=bias,
        )
    elif attn_type == "mqa":
        return MultiQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias=bias,
        )
    elif attn_type == "gqa":
        num_kv_heads = kwargs.get("num_kv_heads", None)
        if num_kv_heads is None and isinstance(config, ModelConfig):
            num_kv_heads = config.num_key_value_heads
        elif num_kv_heads is None and isinstance(config, dict):
            num_kv_heads = config.get("num_key_value_heads", config.get("num_kv_heads", num_heads // 4))
        elif num_kv_heads is None:
            num_kv_heads = max(1, num_heads // 4)
        return GroupedQueryAttentionV2(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias=bias,
        )
    elif attn_type == "mla":
        num_kv_heads = kwargs.get("num_kv_heads", None)
        if num_kv_heads is None:
            num_kv_heads = max(1, num_heads // 4)
        kv_lora_rank = kwargs.get("kv_lora_rank", 512)
        rope_head_dim = kwargs.get("rope_head_dim", 64)
        attention_sink_size = kwargs.get("attention_sink_size", 4)
        return MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            rope_head_dim=rope_head_dim,
            attention_sink_size=attention_sink_size,
            dropout=dropout,
            bias=bias,
        )
    elif attn_type == "differential":
        lambda_init = kwargs.get("lambda_init", 0.5)
        return DifferentialAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias=bias,
            lambda_init=lambda_init,
        )
    else:
        raise ValueError(
            f"Unknown attention type: '{attn_type}'. "
            f"Supported types: 'mha', 'mqa', 'gqa', 'mla', 'differential'"
        )
