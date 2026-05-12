"""
Grouped Query Attention (GQA) with Flash Attention
====================================================
Implements multi-head attention with grouped queries for memory efficiency.

Standard Multi-Head Attention (MHA): Each head has its own K, V projections.
    Q heads = K heads = V heads = num_heads

Grouped Query Attention (GQA): Multiple query heads share the same K, V heads.
    Q heads = num_attention_heads
    K heads = V heads = num_key_value_heads
    num_kv_groups = num_attention_heads / num_key_value_heads

For a 100B model with 96 Q heads and 8 KV heads:
    - 12 query heads share each KV head
    - KV cache size reduced by 12x compared to full MHA
    - Negligible quality impact (proven by LLaMA-2, Mistral)

This module supports:
    - Flash Attention 2 (via flash-attn package) for memory-efficient exact attention
    - Fallback to PyTorch SDPA (scaled_dot_product_attention)
    - KV caching for autoregressive generation
    - Sliding window attention (optional)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) layer with optional Flash Attention.
    
    Supports KV caching for efficient autoregressive generation.
    Position encoding is handled externally via RoPE.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx

        # Q, K, V projection layers
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # Attention scaling factor
        self.scaling = self.head_dim ** -0.5

        # Dropout
        self.attn_dropout_p = config.attention_dropout if hasattr(config, 'attention_dropout') else 0.0

        # Try to use Flash Attention 2
        self._use_flash_attn = self._check_flash_attn()

    def _check_flash_attn(self) -> bool:
        """Check if flash-attn is available."""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of grouped query attention.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size).
            attention_mask: Optional mask (batch, 1, seq_len, kv_seq_len).
            position_ids: Position indices (batch, seq_len).
            past_key_value: Cached (key, value) from previous forward pass.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated (key, value) for KV cache.
            rope_cos: Precomputed RoPE cosine tensor.
            rope_sin: Precomputed RoPE sine tensor.
        
        Returns:
            Tuple of:
                - output: Attention output (batch, seq_len, hidden_size)
                - attn_weights: Attention weights (optional)
                - present_kv: Updated KV cache (optional)
        """
        bsz, q_len, _ = hidden_states.shape

        # === Project Q, K, V ===
        # Q: (batch, seq_len, num_heads * head_dim)
        # K, V: (batch, seq_len, num_kv_heads * head_dim)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # === Reshape for multi-head format ===
        # Q: (batch, num_heads, seq_len, head_dim)
        # K: (batch, num_kv_heads, seq_len, head_dim)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # === Apply RoPE ===
        if rope_cos is not None and rope_sin is not None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, rope_cos, rope_sin
            )

        # === KV Caching ===
        if past_key_value is not None:
            # Append new K, V to cached K, V
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value_out = (key_states, value_states) if use_cache else None

        # === Expand KV heads for GQA ===
        # Repeat KV heads to match query heads
        # (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # === Compute Attention ===
        if self._use_flash_attn:
            attn_output = self._flash_attention(
                query_states, key_states, value_states,
                attention_mask, q_len
            )
        else:
            attn_output, attn_weights = self._sdpa_attention(
                query_states, key_states, value_states,
                attention_mask, bsz, q_len
            )

        # === Reshape and project output ===
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value_out

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
    ) -> torch.Tensor:
        """
        Compute attention using Flash Attention 2.
        
        Flash Attention provides:
            - Exact attention (not approximate)
            - O(N) memory instead of O(N^2) for attention matrix
            - 2-4x speedup via IO-aware tiling
            - Supports causal masking natively
        
        Args:
            query: (batch, num_heads, query_len, head_dim)
            key: (batch, num_heads, kv_len, head_dim)
            value: (batch, num_heads, kv_len, head_dim)
            attention_mask: Optional causal mask.
            query_length: Length of the query sequence.
        
        Returns:
            Attention output (batch, num_heads, query_len, head_dim).
        """
        from flash_attn import flash_attn_func

        # Flash attention expects (batch, seq_len, num_heads, head_dim)
        # We need to transpose from (batch, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Flash attention handles causal masking internally
        output = flash_attn_func(
            query, key, value,
            causal=True,
            softmax_scale=self.scaling,
        )

        # Transpose back: (batch, num_heads, seq_len, head_dim)
        return output.transpose(1, 2)

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        query_length: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention using PyTorch's Scaled Dot Product Attention (SDPA).
        
        Fallback when flash-attn is not available. Uses torch.nn.functional.scaled_dot_product_attention
        which automatically selects the best backend (memory-efficient, flash, or math).
        """
        # Use PyTorch 2.0's SDPA for automatic backend selection
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        # SDPA doesn't return attention weights, set to None
        return attn_output, None

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand KV heads by repeating them to match the number of query heads.
        
        This is the core operation for GQA. If n_rep=1, no repetition is needed
        (equivalent to standard MHA or MQA with same number of heads).
        
        Args:
            hidden_states: (batch, num_kv_heads, seq_len, head_dim)
            n_rep: Number of times to repeat each KV head.
        
        Returns:
            Expanded tensor (batch, num_kv_heads * n_rep, seq_len, head_dim).
        """
        if n_rep == 1:
            return hidden_states

        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(
            batch, num_kv_heads * n_rep, seq_len, head_dim
        )
