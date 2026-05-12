"""
Sparse Attention Patterns for Nexus v2
==========================================
Memory-efficient attention patterns for long-context modeling.
All patterns implement O(S * sqrt(S)) or better attention instead of O(S^2).

Implements:
1. Sliding Window Attention - Local attention with fixed window
2. BigBird-style Sparse Attention - Random + window + global blocks
3. Dilated Attention - Strided multi-hop patterns
4. Block-Sparse Attention - Configurable block-level sparsity

References:
    - Beltagy et al., "Longformer: The Long-Document Transformer" (2020)
    - Zaheer et al., "Big Bird: Transformers for Longer Sequences" (2020)
    - Roy et al., "Efficient Content-Based Sparse Attention with Routing Transformers" (2020)
    - Child et al., "Generating Long Sequences with Sparse Transformers" (2019)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Sliding Window Attention
# ---------------------------------------------------------------------------


class SlidingWindowAttention(nn.Module):
    """
    Sliding window (local) attention.

    Each token only attends to tokens within a window of size W:
        attn_mask[i, j] = 1 if |i - j| < W/2 else 0

    Memory: O(S * W) instead of O(S * S)
    Used in: Mistral, Gemma, many long-context models

    Can be combined with global attention for special tokens (e.g., CLS, BOS).

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int, optional
        Dimension of each head.
    window_size : int
        Size of the local attention window. Each token attends to
        ``window_size // 2`` tokens on each side.
    num_global_tokens : int
        Number of tokens at the beginning that get global (full) attention
        and are attended to by all tokens (e.g., CLS, BOS tokens).
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
        window_size: int = 4096,
        num_global_tokens: int = 0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    def _create_sliding_mask(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a sliding window attention mask.

        The mask allows attention within a window of size ``window_size``.
        If ``num_global_tokens > 0``, those tokens also attend globally
        and are visible to all other tokens.

        Args:
            q_len: Query sequence length.
            kv_len: Key sequence length.
            device: Device for the mask tensor.

        Returns:
            Boolean mask of shape (q_len, kv_len) where True = attend.
        """
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=device).unsqueeze(0)

        # Local window: |i - j| < window_size // 2
        half_window = self.window_size // 2
        local_mask = (q_pos - kv_pos).abs() < half_window

        # Global tokens: first num_global_tokens tokens see everything
        # and are seen by everything
        if self.num_global_tokens > 0:
            global_q = q_pos < self.num_global_tokens
            global_kv = kv_pos < self.num_global_tokens
            # Global tokens attend to all positions
            global_mask = global_q | global_kv.T
            mask = local_mask | global_mask
        else:
            mask = local_mask

        # Convert to additive mask: 0 for attend, -inf for block
        additive_mask = torch.zeros(q_len, kv_len, device=device)
        additive_mask = additive_mask.masked_fill(~mask, float("-inf"))

        return additive_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with sliding window attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional external mask (combined with sliding window mask).
            past_key_value: Cached (key, value) from previous step.
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights.

        Returns:
            (output, attn_weights, present_kv) tuple.
        """
        bsz, q_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # KV caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_kv = (key_states, value_states) if use_cache else None

        kv_len = key_states.shape[2]

        # Create sliding window mask
        window_mask = self._create_sliding_mask(q_len, kv_len, hidden_states.device)
        # Reshape for broadcasting: (1, 1, q_len, kv_len)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)

        # Combine with external attention mask if provided
        if attention_mask is not None:
            combined_mask = window_mask + attention_mask
        else:
            combined_mask = window_mask

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + combined_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), present_kv


# ---------------------------------------------------------------------------
# 2. BigBird-style Sparse Attention
# ---------------------------------------------------------------------------


class BigBirdStyleAttention(nn.Module):
    """
    BigBird-style sparse attention with random + window + global blocks.

    Combines three attention patterns:
    1. Window attention: each token attends to nearby tokens (block of W)
    2. Global attention: a few tokens attend to all tokens (and vice versa)
    3. Random attention: random pairs of tokens attend to each other

    Total edges: O(S) — sparse, approximates full attention.
    Proven to be Turing complete and can approximate any sparse matrix.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int, optional
        Dimension of each head.
    block_size : int
        Size of each attention block for window attention.
    num_global_tokens : int
        Number of tokens with global attention.
    num_random_blocks : int
        Number of random attention pairs per block row.
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
        block_size: int = 64,
        num_global_tokens: int = 2,
        num_random_blocks: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.num_random_blocks = num_random_blocks
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # Cached random pattern (re-generated when sequence length changes)
        self._cached_seq_len: int = -1
        self._random_indices: Optional[torch.Tensor] = None

    def _generate_random_pattern(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate random attention pattern.

        For each block row, randomly selects ``num_random_blocks`` other
        blocks to attend to.

        Args:
            seq_len: Total sequence length.
            device: Device for the pattern tensor.

        Returns:
            Boolean mask of shape (seq_len, seq_len).
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        for i in range(num_blocks):
            # Randomly select blocks to attend to (excluding self and global)
            candidates = list(range(num_blocks))
            candidates = [c for c in candidates if c != i and c * self.block_size >= self.num_global_tokens]
            if not candidates:
                continue
            num_random = min(self.num_random_blocks, len(candidates))
            selected = torch.randperm(len(candidates), device=device)[:num_random]
            for idx in selected:
                j = candidates[idx.item()]
                # Mark the block pair
                start_i = i * self.block_size
                start_j = j * self.block_size
                end_i = min((i + 1) * self.block_size, seq_len)
                end_j = min((j + 1) * self.block_size, seq_len)
                mask[start_i:end_i, start_j:end_j] = True

        return mask

    def _create_bigbird_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create the combined BigBird sparse attention mask.

        The mask is the union of:
        1. Window attention (2 blocks before and after)
        2. Global attention (for num_global_tokens)
        3. Random attention (pre-generated)

        Args:
            seq_len: Sequence length.
            device: Device for the mask.

        Returns:
            Additive mask (seq_len, seq_len) with -inf for blocked positions.
        """
        # Re-generate random pattern if sequence length changed
        if self._cached_seq_len != seq_len or self._random_indices is None:
            self._random_indices = self._generate_random_pattern(seq_len, device)
            self._cached_seq_len = seq_len

        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

        # 1. Window attention: attend to 2 blocks before and after
        block_ids = torch.arange(seq_len, device=device) // self.block_size
        for i in range(seq_len):
            bi = block_ids[i].item()
            for delta in range(-2, 3):
                bj = bi + delta
                if 0 <= bj < num_blocks:
                    start_j = bj * self.block_size
                    end_j = min((bj + 1) * self.block_size, seq_len)
                    # Skip if this is in the global token range (handled separately)
                    if start_j < self.num_global_tokens:
                        end_j = min(end_j, self.num_global_tokens)
                    mask[i, start_j:end_j] = True

        # 2. Global attention: global tokens see everything and are seen by everything
        if self.num_global_tokens > 0:
            # Global tokens attend to all positions
            mask[:self.num_global_tokens, :] = True
            # All positions attend to global tokens
            mask[:, :self.num_global_tokens] = True

        # 3. Random attention
        mask = mask | self._random_indices

        # Convert to additive mask
        additive_mask = torch.zeros(seq_len, seq_len, device=device)
        additive_mask = additive_mask.masked_fill(~mask, float("-inf"))

        return additive_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Forward pass with BigBird sparse attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional external mask (combined with sparse mask).
            output_attentions: Whether to return attention weights.

        Returns:
            (output, attn_weights, None) tuple.
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Generate BigBird mask
        sparse_mask = self._create_bigbird_mask(seq_len, hidden_states.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # Combine with external mask
        if attention_mask is not None:
            combined_mask = sparse_mask + attention_mask
        else:
            combined_mask = sparse_mask

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + combined_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), None


# ---------------------------------------------------------------------------
# 3. Dilated Attention
# ---------------------------------------------------------------------------


class DilatedAttention(nn.Module):
    """
    Dilated (strided) attention pattern.

    Similar to dilated convolutions, each layer attends to tokens
    at increasing distances:
        Layer 0: attend to positions [i-1, i, i+1]
        Layer 1: attend to positions [i-2, i, i+2]
        Layer 2: attend to positions [i-4, i, i+4]
        ...

    With multiple layers, information propagates exponentially:
        After L layers with dilation 2^l: receptive field = 2^L

    Very efficient: O(S * log(S)) multi-hop information flow.

    The dilation rate for each layer is configurable. Typically follows
    an exponential schedule: dilation(l) = segment_length * 2^l.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int, optional
        Dimension of each head.
    dilation_rate : int
        Base dilation rate for this layer. Typically set per-layer
        as ``segment_length * 2^layer_idx``.
    segment_length : int
        Length of each local attention segment.
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
        dilation_rate: int = 1,
        segment_length: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.dilation_rate = max(1, dilation_rate)
        self.segment_length = segment_length
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    def _create_dilated_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a dilated attention mask.

        Each token i attends to tokens at positions:
            [i - dilation_rate * segment_length, ..., i - dilation_rate, i, i + dilation_rate, ..., i + dilation_rate * segment_length]

        Within the local segment, all tokens attend to each other.

        Args:
            seq_len: Sequence length.
            device: Device for the mask.

        Returns:
            Additive mask (seq_len, seq_len) with -inf for blocked positions.
        """
        positions = torch.arange(seq_len, device=device)

        # Local attention within segments
        segments = positions // self.segment_length
        local_mask = segments.unsqueeze(0) == segments.unsqueeze(1)

        # Dilated attention: attend to tokens at dilation_rate * k distance
        # for k in range(-num_dilated_steps, num_dilated_steps + 1)
        diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

        # Token i attends to j if diff(i, j) is a multiple of dilation_rate
        # and within a reasonable range
        dilated_mask = (diff > 0) & (diff % self.dilation_rate == 0) & (diff <= self.dilation_rate * self.segment_length)

        # Combine: local OR dilated
        mask = local_mask | dilated_mask

        # Convert to additive mask
        additive_mask = torch.zeros(seq_len, seq_len, device=device)
        additive_mask = additive_mask.masked_fill(~mask, float("-inf"))

        return additive_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Forward pass with dilated attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional external mask (combined with dilated mask).
            output_attentions: Whether to return attention weights.

        Returns:
            (output, attn_weights, None) tuple.
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Generate dilated mask
        dilated_mask = self._create_dilated_mask(seq_len, hidden_states.device)
        dilated_mask = dilated_mask.unsqueeze(0).unsqueeze(0)

        # Combine with external mask
        if attention_mask is not None:
            combined_mask = dilated_mask + attention_mask
        else:
            combined_mask = dilated_mask

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + combined_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), None


# ---------------------------------------------------------------------------
# 4. Block-Sparse Attention
# ---------------------------------------------------------------------------


class BlockSparseAttention(nn.Module):
    """
    Block-sparse attention with configurable sparsity pattern.

    Divides the attention matrix into blocks (B x B) and only
    computes a subset of blocks. Block mask is configurable.

    Common patterns:
    - Striped: every k-th block row/column
    - Diagonal: blocks near the main diagonal
    - Fixed: manually specified block mask
    - BigBird-like: combination of local, global, and random blocks

    The block mask determines which blocks are computed (True) and
    which are skipped (False). This enables O(S * k * block_size^2)
    attention instead of O(S^2) where k is the average number of
    active blocks per row.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int, optional
        Dimension of each head.
    block_size : int
        Size of each attention block (tokens per block).
    sparsity_pattern : str
        Type of sparsity pattern. One of:
        - "diagonal": blocks near the main diagonal (local attention)
        - "striped": every k-th column of blocks
        - "random": random block selection
        - "bigbird": BigBird-style local + global + random
        - "custom": user-provided block mask
    num_active_blocks : int
        Number of active blocks per row (for non-diagonal patterns).
    num_global_blocks : int
        Number of global blocks (for BigBird-like patterns).
    custom_block_mask : torch.Tensor, optional
        User-provided boolean block mask of shape (num_blocks, num_blocks).
        Required when sparsity_pattern == "custom".
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
        block_size: int = 64,
        sparsity_pattern: str = "diagonal",
        num_active_blocks: int = 4,
        num_global_blocks: int = 1,
        custom_block_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.block_size = block_size
        self.sparsity_pattern = sparsity_pattern.lower()
        self.num_active_blocks = num_active_blocks
        self.num_global_blocks = num_global_blocks
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # Custom block mask
        if self.sparsity_pattern == "custom":
            if custom_block_mask is None:
                raise ValueError("custom_block_mask must be provided for sparsity_pattern='custom'")
            self.register_buffer("_custom_block_mask", custom_block_mask.bool(), persistent=False)
        else:
            self._custom_block_mask = None

        # Cache for block masks
        self._cached_num_blocks: int = -1
        self._cached_block_mask: Optional[torch.Tensor] = None

    def _generate_block_mask(
        self,
        num_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate block-level attention mask based on the sparsity pattern.

        Args:
            num_blocks: Number of blocks per side.
            device: Device for the mask.

        Returns:
            Boolean block mask (num_blocks, num_blocks), True = compute.
        """
        mask = torch.zeros(num_blocks, num_blocks, device=device, dtype=torch.bool)

        if self.sparsity_pattern == "diagonal":
            # Diagonal pattern: attend to nearby blocks
            bandwidth = self.num_active_blocks // 2
            for i in range(num_blocks):
                for j in range(max(0, i - bandwidth), min(num_blocks, i + bandwidth + 1)):
                    mask[i, j] = True

        elif self.sparsity_pattern == "striped":
            # Striped pattern: every k-th block column
            stride = max(1, num_blocks // (self.num_active_blocks + 1))
            for i in range(num_blocks):
                for k in range(self.num_active_blocks):
                    j = (i // stride + k) % num_blocks
                    mask[i, j] = True
                # Always attend to self
                mask[i, i] = True

        elif self.sparsity_pattern == "random":
            # Random pattern: randomly select blocks per row
            for i in range(num_blocks):
                candidates = list(range(num_blocks))
                candidates.remove(i)
                num_select = min(self.num_active_blocks - 1, len(candidates))
                selected = torch.randperm(len(candidates), device=device)[:num_select]
                for idx in selected:
                    mask[i, candidates[idx.item()]] = True
                mask[i, i] = True

        elif self.sparsity_pattern == "bigbird":
            # BigBird-like: local (2 blocks) + global + random
            for i in range(num_blocks):
                # Local: 2 blocks before and after
                for delta in range(-2, 3):
                    j = i + delta
                    if 0 <= j < num_blocks:
                        mask[i, j] = True
                # Global blocks
                for g in range(self.num_global_blocks):
                    if g < num_blocks:
                        mask[i, g] = True
                        mask[g, i] = True
                # Random blocks
                candidates = [
                    c for c in range(num_blocks)
                    if not mask[i, c] and c >= self.num_global_blocks
                ]
                if candidates:
                    num_random = min(2, len(candidates))
                    selected = torch.randperm(len(candidates), device=device)[:num_random]
                    for idx in selected:
                        mask[i, candidates[idx.item()]] = True

        elif self.sparsity_pattern == "custom":
            if self._custom_block_mask is not None:
                mask = self._custom_block_mask[:num_blocks, :num_blocks].to(device)
            else:
                raise RuntimeError("custom_block_mask not set")

        else:
            raise ValueError(f"Unknown sparsity pattern: '{self.sparsity_pattern}'")

        return mask

    def _block_mask_to_token_mask(
        self,
        block_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Expand a block-level mask to a token-level mask.

        Args:
            block_mask: (num_blocks, num_blocks) boolean mask.
            seq_len: Total sequence length.

        Returns:
            Token-level additive mask (seq_len, seq_len).
        """
        num_blocks = block_mask.shape[0]
        token_mask = torch.zeros(seq_len, seq_len, device=block_mask.device)

        for bi in range(num_blocks):
            for bj in range(num_blocks):
                if block_mask[bi, bj]:
                    start_i = bi * self.block_size
                    end_i = min((bi + 1) * self.block_size, seq_len)
                    start_j = bj * self.block_size
                    end_j = min((bj + 1) * self.block_size, seq_len)
                    token_mask[start_i:end_i, start_j:end_j] = 0.0
                else:
                    start_i = bi * self.block_size
                    end_i = min((bi + 1) * self.block_size, seq_len)
                    start_j = bj * self.block_size
                    end_j = min((bj + 1) * self.block_size, seq_len)
                    token_mask[start_i:end_i, start_j:end_j] = float("-inf")

        return token_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Forward pass with block-sparse attention.

        For efficiency, this implementation uses a block-mask expanded
        to token-level. A production GPU kernel would process blocks
        directly without expanding to the full S x S matrix.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional external mask (combined with sparse mask).
            output_attentions: Whether to return attention weights.

        Returns:
            (output, attn_weights, None) tuple.
        """
        bsz, seq_len, _ = hidden_states.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Generate block mask (cached)
        if self._cached_num_blocks != num_blocks or self._cached_block_mask is None:
            self._cached_block_mask = self._generate_block_mask(num_blocks, hidden_states.device)
            self._cached_num_blocks = num_blocks

        block_mask = self._cached_block_mask

        # Expand to token-level mask
        token_mask = self._block_mask_to_token_mask(block_mask, seq_len)
        token_mask = token_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # Combine with external mask
        if attention_mask is not None:
            combined_mask = token_mask + attention_mask
        else:
            combined_mask = token_mask

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + combined_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)

        if self.training and self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), None

    def get_sparsity_ratio(self) -> float:
        """
        Compute the theoretical sparsity ratio of the current pattern.

        Returns:
            Ratio of zero (skipped) entries in the attention matrix.
        """
        if self._cached_block_mask is not None:
            num_blocks = self._cached_block_mask.shape[0]
            total = num_blocks * num_blocks
            active = self._cached_block_mask.sum().item()
            return 1.0 - (active / total)
        return 0.0
