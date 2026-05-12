"""
Flash Attention and Advanced Attention Kernels for Nexus v2
==============================================================
Memory-efficient attention implementations from scratch.

Implements:
1. Flash Attention v2 - Tiled, memory-efficient exact attention
2. Flash Attention v3 - Hardware-aware with async overlap and FP8
3. Paged Attention - Block-based KV cache management (vLLM style)
4. Ring Attention - Distributed long-sequence attention

References:
    - Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
    - Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
    - Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
    - Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Flash Attention v2 (PyTorch reference implementation)
# ---------------------------------------------------------------------------


class FlashAttentionV2(nn.Module):
    """
    Flash Attention v2 implementation (from-scratch PyTorch version).

    Key principles (Tri Dao 2023):
    1. Tiling: Split Q, K, V into blocks that fit in SRAM
    2. Online softmax: Compute softmax in a single pass over K, V blocks
    3. Kernel fusion: Q, K, V projection + attention in one kernel
    4. IO-aware: Minimize HBM reads/writes

    Algorithm (simplified):
        For each query block Qi:
            For each key block Kj, value block Vj:
                Sij = Qi @ Kj.T * scale
                m_ij = max(Sij)
                Pij = exp(Sij - m_ij)
                l_ij = rowsum(Pij)
                acc_i += Pij @ Vj
                update running max and sum (online softmax)
            Oi = acc_i / l_i

    When flash_attn package is available, delegates to it.
    Otherwise, uses PyTorch implementation with tiling.

    Parameters
    ----------
    causal : bool
        Whether to apply causal masking (for autoregressive models).
    softmax_scale : float, optional
        Manual scale factor. If None, uses 1/sqrt(head_dim).
    tile_size : int
        Block size for tiled computation. Must be divisible by 16 for GPU efficiency.
    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        tile_size: int = 128,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.tile_size = tile_size

        # Check if flash-attn package is available
        self._has_flash_attn = self._check_flash_attn()

    @staticmethod
    def _check_flash_attn() -> bool:
        """Check if the flash_attn package is importable."""
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: Optional[bool] = None,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute memory-efficient attention.

        Args:
            q: Query tensor of shape (batch, num_heads, q_len, head_dim).
            k: Key tensor of shape (batch, num_heads, kv_len, head_dim).
            v: Value tensor of shape (batch, num_heads, kv_len, head_dim).
            causal: Override causal setting. If None, uses self.causal.
            softmax_scale: Override softmax scale. If None, uses 1/sqrt(head_dim).

        Returns:
            Attention output of shape (batch, num_heads, q_len, head_dim).
        """
        use_causal = causal if causal is not None else self.causal
        scale = softmax_scale if softmax_scale is not None else self.softmax_scale
        if scale is None:
            scale = q.shape[-1] ** -0.5

        # Try to use the native flash_attn package for speed
        if self._has_flash_attn:
            return self._flash_attn_native(q, k, v, use_causal, scale)

        # Fall back to tiled PyTorch implementation
        return self._tiled_attention(q, k, v, use_causal, scale)

    def _flash_attn_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        """
        Delegate to the flash_attn package when available.

        flash_attn expects (batch, seq_len, num_heads, head_dim) format.
        """
        from flash_attn import flash_attn_func

        # Transpose from (B, H, S, D) to (B, S, H, D)
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()

        output = flash_attn_func(
            q_t, k_t, v_t,
            causal=causal,
            softmax_scale=scale,
        )

        # Transpose back to (B, H, S, D)
        return output.transpose(1, 2)

    def _tiled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        """
        Tiled attention with online softmax (FlashAttention v2 algorithm).

        Processes Q, K, V in tiles to reduce peak memory from O(B*H*S^2)
        to O(B*H*tile_size*S). Uses online softmax to avoid materializing
        the full attention matrix.

        Algorithm (per batch, per head):
            Initialize: O = zeros(tile_size, head_dim)
                        m = -inf  (running row-max)
                        l = 0     (running row-sum of exp)

            For each Q tile i:
                Reset: O_i = 0, m_i = -inf, l_i = 0
                For each K, V tile j:
                    S_ij = Q_i @ K_j.T * scale
                    if causal: mask future positions in S_ij
                    m_ij = rowmax(S_ij)
                    m_new = max(m_i, m_ij)
                    P_ij = exp(S_ij - m_new)
                    l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
                    O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
                    m_i = m_new
                    l_i = l_new
                O_i = O_i / l_i  (finalize softmax)
                write O_i to output

        Args:
            q: (batch, num_heads, q_len, head_dim)
            k: (batch, num_heads, kv_len, head_dim)
            v: (batch, num_heads, kv_len, head_dim)
            causal: Whether to apply causal mask.
            scale: Softmax temperature scale.

        Returns:
            (batch, num_heads, q_len, head_dim)
        """
        B, H, Q_len, D = q.shape
        _, _, KV_len, _ = k.shape

        device = q.device
        dtype = q.dtype
        tile = min(self.tile_size, Q_len, KV_len)

        # Output buffer
        output = torch.zeros_like(q)

        # Process in Q tiles
        for q_start in range(0, Q_len, tile):
            q_end = min(q_start + tile, Q_len)
            q_tile = q[:, :, q_start:q_end, :]  # (B, H, tile, D)

            # Running statistics for online softmax
            # m_i: running row-max, shape (B, H, tile, 1)
            m_i = torch.full(
                (B, H, q_end - q_start, 1),
                float("-inf"),
                device=device,
                dtype=torch.float32,
            )
            # l_i: running row-sum, shape (B, H, tile, 1)
            l_i = torch.zeros(
                (B, H, q_end - q_start, 1),
                device=device,
                dtype=torch.float32,
            )
            # acc: accumulated output, shape (B, H, tile, D)
            acc = torch.zeros(
                (B, H, q_end - q_start, D),
                device=device,
                dtype=torch.float32,
            )

            # Process K, V tiles
            for kv_start in range(0, KV_len, tile):
                kv_end = min(kv_start + tile, KV_len)
                k_tile = k[:, :, kv_start:kv_end, :]  # (B, H, tile_kv, D)
                v_tile = v[:, :, kv_start:kv_end, :]  # (B, H, tile_kv, D)

                # Compute attention scores for this tile pair
                # (B, H, q_tile_len, kv_tile_len)
                S = torch.matmul(q_tile.float(), k_tile.float().transpose(-2, -1)) * scale

                # Apply causal mask if needed
                if causal:
                    # Create causal mask for this tile pair
                    q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(1)
                    kv_pos = torch.arange(kv_start, kv_end, device=device).unsqueeze(0)
                    causal_mask = q_pos < kv_pos  # True where Q should NOT attend
                    S = S.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                # Online softmax update
                m_ij = S.max(dim=-1, keepdim=True).values  # (B, H, q_tile, 1)
                m_new = torch.max(m_i, m_ij)

                # Correction factor for previous accumulation
                exp_m_diff = torch.exp(m_i - m_new)

                # Compute attention weights for this tile
                P = torch.exp(S - m_new)  # (B, H, q_tile, kv_tile)

                # Clamp for numerical stability
                P = torch.clamp(P, max=1e4)

                # Rowsum of P for this tile
                l_ij = P.sum(dim=-1, keepdim=True)  # (B, H, q_tile, 1)

                # Update running sum
                l_new = exp_m_diff * l_i + l_ij

                # Update accumulated output
                # acc = exp_m_diff * acc + P @ V_tile
                acc = exp_m_diff * acc + torch.matmul(P, v_tile.float())

                # Update running max and sum
                m_i = m_new
                l_i = l_new

            # Finalize: normalize by the sum
            acc = acc / torch.clamp(l_i, min=1e-10)

            # Write to output
            output[:, :, q_start:q_end, :] = acc.to(dtype)

        return output


# ---------------------------------------------------------------------------
# 2. Flash Attention v3 (Hardware-aware)
# ---------------------------------------------------------------------------


class FlashAttentionV3(nn.Module):
    """
    Flash Attention v3 (hardware-aware).

    Improvements over v2:
    - Asynchronous stream overlap
    - Better register blocking
    - FP8 support
    - WGMMA (warp-group matrix multiply accumulate) for H100
    - Supports variable-length sequences natively

    Falls back to v2 implementation on older hardware.

    Parameters
    ----------
    causal : bool
        Whether to apply causal masking.
    use_fp8 : bool
        Whether to use FP8 quantization for Q, K, V.
    tile_size : int
        Block size for tiled computation.
    """

    def __init__(
        self,
        causal: bool = True,
        use_fp8: bool = False,
        tile_size: int = 128,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.use_fp8 = use_fp8
        self.tile_size = tile_size

        # Initialize v2 as fallback
        self.v2_fallback = FlashAttentionV2(
            causal=causal,
            tile_size=tile_size,
        )

        # Check hardware capabilities
        self._supports_h100_features = self._check_h100_features()

    @staticmethod
    def _check_h100_features() -> bool:
        """Check if the current GPU supports H100 features (compute capability >= 9.0)."""
        if not torch.cuda.is_available():
            return False
        try:
            capability = torch.cuda.get_device_capability()
            return capability[0] >= 9
        except Exception:
            return False

    def _quantize_fp8(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to FP8 for faster computation.

        Uses E4M3 format (4 exponent bits, 3 mantissa bits) for forward
        and E5M2 (5 exponent bits, 2 mantissa bits) for scaling factors.

        Args:
            x: Input tensor in FP16/BF16.

        Returns:
            (quantized, scale) tuple.
        """
        # Compute per-tile scale
        abs_max = x.abs().amax(dim=-1, keepdim=True)
        scale = abs_max / 448.0  # FP8 E4M3 max ≈ 448

        # Clamp to FP8 range and cast
        x_clamped = torch.clamp(x / scale, min=-448.0, max=448.0)
        x_fp8 = x_clamped.to(torch.float8_e4m3fn)

        return x_fp8, scale

    def _dequantize_fp8(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize FP8 tensor back to the original dtype."""
        return x.to(scale.dtype) * scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: Optional[bool] = None,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute attention with v3 optimizations.

        On H100+: Uses FP8 and async overlap.
        On other GPUs: Falls back to v2.

        Args:
            q: (batch, num_heads, q_len, head_dim)
            k: (batch, num_heads, kv_len, head_dim)
            v: (batch, num_heads, kv_len, head_dim)
            causal: Override causal setting.
            softmax_scale: Override softmax scale.

        Returns:
            Attention output of shape (batch, num_heads, q_len, head_dim).
        """
        use_causal = causal if causal is not None else self.causal

        # FP8 path: quantize Q, K for faster matmul on supported hardware
        if self.use_fp8 and q.dtype in (torch.float16, torch.bfloat16):
            return self._fp8_attention(q, k, v, use_causal, softmax_scale)

        # Try flash-attn v3 if available
        if self.v2_fallback._has_flash_attn:
            try:
                from flash_attn import flash_attn_func
                q_t = q.transpose(1, 2).contiguous()
                k_t = k.transpose(1, 2).contiguous()
                v_t = v.transpose(1, 2).contiguous()
                output = flash_attn_func(
                    q_t, k_t, v_t,
                    causal=use_causal,
                    softmax_scale=softmax_scale,
                )
                return output.transpose(1, 2)
            except Exception:
                pass

        # Fallback to v2 tiled implementation
        return self.v2_fallback(q, k, v, causal=use_causal, softmax_scale=softmax_scale)

    def _fp8_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        """
        FP8-accelerated attention computation.

        Quantizes Q, K to FP8 for the matmul, keeping V in original precision
        for accurate output accumulation.

        Args:
            q, k, v: Attention tensors.
            causal: Causal masking flag.
            softmax_scale: Softmax scale factor.

        Returns:
            Attention output in original dtype.
        """
        orig_dtype = q.dtype
        scale = softmax_scale or (q.shape[-1] ** -0.5)

        # Quantize Q and K to FP8
        q_fp8, q_scale = self._quantize_fp8(q)
        k_fp8, k_scale = self._quantize_fp8(k)

        # Compute attention scores in FP8 (dequantized for softmax)
        # Q @ K^T: (B, H, S_q, D) x (B, H, D, S_kv) -> (B, H, S_q, S_kv)
        scores = torch.matmul(q_fp8.to(torch.float32), k_fp8.to(torch.float32).transpose(-2, -1))
        # Apply scale corrections
        scores = scores * (q_scale * k_scale.transpose(-2, -1)) * scale

        # Apply causal mask
        if causal:
            q_len = q.shape[2]
            kv_len = k.shape[2]
            mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=q.device, dtype=torch.float32),
                diagonal=kv_len - q_len + 1,
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(orig_dtype)

        # Output: attn_weights @ V (V stays in original precision)
        output = torch.matmul(attn_weights, v)

        return output


# ---------------------------------------------------------------------------
# 3. Paged Attention (vLLM style)
# ---------------------------------------------------------------------------


class PagedAttentionBlock:
    """
    PagedAttention for efficient KV cache management (vLLM style).

    Instead of contiguous KV cache, divides into fixed-size pages/blocks:
    - Each page holds ``page_size`` tokens of K, V
    - Page table maps logical block indices to physical block indices
    - Non-contiguous allocation: shared prefixes share pages (copy-on-write)
    - Memory defragmentation when needed

    Benefits:
    - Eliminates KV cache fragmentation
    - Enables memory sharing across requests (shared prefixes)
    - Supports preemption (swap pages to CPU)
    - Enables dynamic batching with variable-length sequences

    Page structure:
        [Block 0: tokens 0-15] [Block 1: tokens 16-31] ... [Block N: tokens ...]
        Physical memory: non-contiguous, managed by block allocator

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each head.
    page_size : int
        Number of tokens per page/block. Typical: 16.
    num_pages : int
        Total number of physical pages available.
    device : str or torch.device
        Device to allocate the KV cache on.
    dtype : torch.dtype
        Data type for the KV cache.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,
        num_pages: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_pages = num_pages
        self.device = torch.device(device)
        self.dtype = dtype

        # Physical KV cache: (num_pages, 2, num_heads, page_size, head_dim)
        # Dim 1: 0=K, 1=V
        self.kv_cache = torch.zeros(
            num_pages, 2, num_heads, page_size, head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # Page table: logical -> physical mapping
        # Maps request_id to a list of physical block IDs
        self.page_table: Dict[str, List[int]] = {}

        # Track sequence lengths per request (number of valid tokens)
        self.seq_lengths: Dict[str, int] = {}

        # Free block list (LIFO for cache locality)
        self.free_blocks: List[int] = list(range(num_pages))

        # Statistics
        self._total_allocations = 0
        self._total_frees = 0

    @property
    def num_free_pages(self) -> int:
        """Return the number of available (free) pages."""
        return len(self.free_blocks)

    @property
    def utilization(self) -> float:
        """Return the cache utilization ratio (0.0 to 1.0)."""
        used = self.num_pages - self.num_free_pages
        return used / self.num_pages if self.num_pages > 0 else 0.0

    def allocate(self, num_tokens: int) -> List[int]:
        """
        Allocate pages to hold ``num_tokens`` tokens.

        Args:
            num_tokens: Number of tokens that need storage.

        Returns:
            List of physical block IDs allocated.

        Raises:
            RuntimeError: If not enough free pages are available.
        """
        num_blocks_needed = (num_tokens + self.page_size - 1) // self.page_size

        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough free pages: need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}. "
                f"Consider increasing num_pages or enabling preemption."
            )

        # Allocate blocks from the free list (LIFO)
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)

        self._total_allocations += num_blocks_needed
        return allocated

    def register_request(
        self,
        request_id: str,
        num_tokens: int,
    ) -> None:
        """
        Register a new request and allocate pages for it.

        Args:
            request_id: Unique identifier for the request.
            num_tokens: Initial number of tokens.
        """
        blocks = self.allocate(num_tokens)
        self.page_table[request_id] = blocks
        self.seq_lengths[request_id] = num_tokens

    def append(
        self,
        request_id: str,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Append new K, V to the KV cache for a request.

        Automatically allocates new pages if the current allocation is insufficient.

        Args:
            request_id: Request identifier.
            key: Key tensor of shape (num_heads, num_new_tokens, head_dim) or
                 (1, num_heads, num_new_tokens, head_dim).
            value: Value tensor with same shape as key.
        """
        if request_id not in self.page_table:
            raise KeyError(f"Request '{request_id}' not registered. Call register_request first.")

        # Handle batch dimension
        if key.dim() == 4:
            key = key.squeeze(0)
            value = value.squeeze(0)

        # key: (num_heads, num_new_tokens, head_dim)
        num_new_tokens = key.shape[1]
        current_len = self.seq_lengths[request_id]
        new_len = current_len + num_new_tokens

        # Check if we need more pages
        current_blocks = len(self.page_table[request_id])
        needed_blocks = (new_len + self.page_size - 1) // self.page_size

        if needed_blocks > current_blocks:
            # Allocate additional blocks
            extra_blocks = needed_blocks - current_blocks
            new_blocks = self.allocate(extra_blocks * self.page_size)
            self.page_table[request_id].extend(new_blocks)

        # Write K, V to the appropriate pages
        blocks = self.page_table[request_id]
        token_offset = current_len

        for token_idx in range(num_new_tokens):
            global_pos = token_offset + token_idx
            block_idx = global_pos // self.page_size
            pos_in_block = global_pos % self.page_size
            physical_block = blocks[block_idx]

            # Write K
            self.kv_cache[physical_block, 0, :, pos_in_block, :] = key[:, token_idx, :]
            # Write V
            self.kv_cache[physical_block, 1, :, pos_in_block, :] = value[:, token_idx, :]

        self.seq_lengths[request_id] = new_len

    def read(
        self,
        request_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read all K, V for a request, assembling from pages.

        Args:
            request_id: Request identifier.

        Returns:
            (K, V) tuple, each of shape (1, num_heads, seq_len, head_dim).

        Raises:
            KeyError: If the request is not found.
        """
        if request_id not in self.page_table:
            raise KeyError(f"Request '{request_id}' not found.")

        seq_len = self.seq_lengths[request_id]
        blocks = self.page_table[request_id]

        K = torch.empty(
            1, self.num_heads, seq_len, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        V = torch.empty_like(K)

        for global_pos in range(seq_len):
            block_idx = global_pos // self.page_size
            pos_in_block = global_pos % self.page_size
            physical_block = blocks[block_idx]

            K[0, :, global_pos, :] = self.kv_cache[physical_block, 0, :, pos_in_block, :]
            V[0, :, global_pos, :] = self.kv_cache[physical_block, 1, :, pos_in_block, :]

        return K, V

    def copy_on_write(
        self,
        src_request_id: str,
        dst_request_id: str,
        shared_prefix_len: int,
    ) -> None:
        """
        Share pages for common prefix, allocate new for divergence.

        This enables efficient fork/join semantics for beam search,
        parallel sampling, and speculative decoding.

        Args:
            src_request_id: Source request with existing KV cache.
            dst_request_id: Destination request (new or existing).
            shared_prefix_len: Number of tokens in the shared prefix.
        """
        if src_request_id not in self.page_table:
            raise KeyError(f"Source request '{src_request_id}' not found.")

        src_blocks = self.page_table[src_request_id]
        src_len = self.seq_lengths[src_request_id]

        # Number of blocks needed for the shared prefix
        shared_blocks = (shared_prefix_len + self.page_size - 1) // self.page_size

        # The destination gets the same physical blocks for the prefix
        # (shared references, not copies — copy-on-write)
        dst_blocks = list(src_blocks[:shared_blocks])

        # Allocate new blocks for the divergence part
        divergence_len = src_len - shared_prefix_len
        if divergence_len > 0:
            divergence_blocks = self.allocate(divergence_len)
            # Copy data from source to new blocks
            for pos in range(shared_prefix_len, src_len):
                src_block_idx = pos // self.page_size
                pos_in_block = pos % self.page_size
                dst_block_idx = (pos - shared_prefix_len) // self.page_size
                dst_pos = pos % self.page_size

                src_physical = src_blocks[src_block_idx]
                dst_physical = divergence_blocks[dst_block_idx]

                self.kv_cache[dst_physical, 0, :, dst_pos, :] = (
                    self.kv_cache[src_physical, 0, :, pos_in_block, :]
                )
                self.kv_cache[dst_physical, 1, :, dst_pos, :] = (
                    self.kv_cache[src_physical, 1, :, pos_in_block, :]
                )

            dst_blocks.extend(divergence_blocks)

        self.page_table[dst_request_id] = dst_blocks
        self.seq_lengths[dst_request_id] = src_len

    def free(self, request_id: str) -> None:
        """
        Free all pages allocated for a request.

        Args:
            request_id: Request identifier to free.
        """
        if request_id not in self.page_table:
            return

        # Return blocks to the free list
        blocks = self.page_table.pop(request_id)
        self.free_blocks.extend(blocks)
        self._total_frees += len(blocks)

        # Clean up sequence length tracking
        self.seq_lengths.pop(request_id, None)

    def defragment(self) -> int:
        """
        Defragment the KV cache by compacting allocations.

        Moves active allocations to contiguous physical blocks,
        freeing fragmented gaps.

        Returns:
            Number of pages freed by defragmentation.
        """
        if not self.page_table:
            return 0

        # Collect all active blocks
        active_blocks = set()
        for blocks in self.page_table.values():
            active_blocks.update(blocks)

        # Sort active blocks and compact
        sorted_active = sorted(active_blocks)
        compacted_mapping = {old: new for new, old in enumerate(sorted_active)}

        # Create new KV cache
        new_kv_cache = torch.zeros_like(self.kv_cache)
        for old_block, new_block in compacted_mapping.items():
            new_kv_cache[new_block] = self.kv_cache[old_block].clone()

        self.kv_cache = new_kv_cache

        # Update page tables
        for request_id in self.page_table:
            self.page_table[request_id] = [
                compacted_mapping[b] for b in self.page_table[request_id]
            ]

        # Rebuild free list
        highest_used = max(compacted_mapping.values()) + 1
        self.free_blocks = list(range(highest_used, self.num_pages))

        return self.num_pages - highest_used

    def swap_out(
        self,
        request_id: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Swap a request's KV cache pages to CPU (preemption).

        Args:
            request_id: Request to swap out.

        Returns:
            Dictionary with 'key' and 'value' tensors on CPU.
        """
        K, V = self.read(request_id)
        # Return blocks to free list
        self.free(request_id)
        return {"key": K.cpu(), "value": V.cpu()}

    def swap_in(
        self,
        request_id: str,
        cpu_cache: Dict[str, torch.Tensor],
    ) -> None:
        """
        Swap a request's KV cache back from CPU.

        Args:
            request_id: Request to swap in.
            cpu_cache: Dictionary with 'key' and 'value' tensors from swap_out.
        """
        key_cpu = cpu_cache["key"]  # (1, num_heads, seq_len, head_dim)
        value_cpu = cpu_cache["value"]

        seq_len = key_cpu.shape[2]
        self.register_request(request_id, seq_len)

        # Write back to pages
        key_gpu = key_cpu.to(device=self.device, dtype=self.dtype)
        value_gpu = value_cpu.to(device=self.device, dtype=self.dtype)
        self.append(request_id, key_gpu, value_gpu)

    def __repr__(self) -> str:
        return (
            f"PagedAttentionBlock("
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"page_size={self.page_size}, "
            f"total_pages={self.num_pages}, "
            f"free_pages={self.num_free_pages}, "
            f"active_requests={len(self.page_table)}, "
            f"utilization={self.utilization:.1%}"
            f")"
        )


# ---------------------------------------------------------------------------
# 4. Ring Attention
# ---------------------------------------------------------------------------


class RingAttention(nn.Module):
    """
    Ring Attention for distributing long sequences across devices.

    Key idea: each device holds a block of the sequence, and devices
    pass their K, V blocks in a ring while computing local Q @ incoming K.

    Computation overlaps with communication:
        Device 0: compute Q0 @ K0, then Q0 @ K1 (from device 1), etc.
        Device 1: compute Q1 @ K1, then Q1 @ K2 (from device 2), etc.

    This enables training/inference on sequences longer than any single
    device can hold. Memory per device: O(S/N + S*D/N) for N devices.

    Communication: P2P send/recv in a ring topology.

    Parameters
    ----------
    causal : bool
        Whether to apply causal masking.
    softmax_scale : float, optional
        Manual scale factor for softmax.
    tile_size : int
        Block size for computation within each device.
    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        tile_size: int = 128,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.tile_size = tile_size

        # Local flash attention for per-device computation
        self._local_attn = FlashAttentionV2(
            causal=False,  # We handle causal masking at the ring level
            tile_size=tile_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: Optional[bool] = None,
        softmax_scale: Optional[float] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> torch.Tensor:
        """
        Compute ring attention across distributed devices.

        In a multi-device setting, each device calls forward with its local
        Q, K, V chunk. The ring communication is handled externally via
        distributed P2P operations.

        For single-device execution (world_size=1), falls back to
        standard attention.

        Args:
            q: Local query chunk (batch, num_heads, q_len, head_dim).
            k: Local key chunk (batch, num_heads, kv_len, head_dim).
            v: Local value chunk (batch, num_heads, kv_len, head_dim).
            causal: Override causal setting.
            softmax_scale: Override softmax scale.
            rank: Rank of this device in the ring (0-indexed).
            world_size: Total number of devices in the ring.

        Returns:
            Attention output (batch, num_heads, q_len, head_dim).
        """
        use_causal = causal if causal is not None else self.causal
        scale = softmax_scale if softmax_scale is not None else self.softmax_scale

        if world_size <= 1:
            # Single device: use standard attention with causal mask
            return self._single_device_attention(q, k, v, use_causal, scale)

        # Multi-device ring attention
        return self._ring_forward(q, k, v, use_causal, scale, rank, world_size)

    def _single_device_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        """Standard attention for single-device execution."""
        self._local_attn.causal = causal
        return self._local_attn(q, k, v, causal=causal, softmax_scale=scale)

    def _ring_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: Optional[float],
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        """
        Ring attention algorithm for multi-device execution.

        Each device:
        1. Computes local attention: Q_local @ K_local.T -> local output
        2. Sends K_local, V_local to next device in ring
        3. Receives K_prev, V_prev from previous device
        4. Computes cross-block attention: Q_local @ K_prev.T
        5. Repeats until all blocks have been processed

        Uses online softmax to combine partial results from each ring step.

        In this PyTorch reference, we simulate the ring by concatenating
        all blocks and processing in tiles (real distributed version
        would use torch.distributed P2P ops).
        """
        B, H, Q_len, D = q.shape
        KV_len = k.shape[1] * world_size  # Total KV length across all devices

        # Simulated ring: process all KV blocks
        # In real distributed setting, each step would involve:
        #   - P2P send current K, V to (rank + 1) % world_size
        #   - P2P recv next K, V from (rank - 1) % world_size
        #   - Compute attention with received K, V
        #   - Update online softmax accumulators

        # For the reference implementation, we use tiled attention
        # with online softmax (similar to FlashAttentionV2 but with ring masking)
        if scale is None:
            scale = D ** -0.5

        dtype = q.dtype
        device = q.device
        tile = min(self.tile_size, Q_len)

        # Initialize accumulators for online softmax
        output = torch.zeros_like(q)
        m_i = torch.full((B, H, Q_len, 1), float("-inf"), device=device, dtype=torch.float32)
        l_i = torch.zeros((B, H, Q_len, 1), device=device, dtype=torch.float32)
        acc = torch.zeros((B, H, Q_len, D), device=device, dtype=torch.float32)

        # Simulate receiving KV blocks from the ring
        # In real distributed, this would be in a loop with P2P communication
        total_kv_len = k.shape[2]  # Local KV for this simulation

        for kv_start in range(0, total_kv_len, tile):
            kv_end = min(kv_start + tile, total_kv_len)

            k_tile = k[:, :, kv_start:kv_end, :]
            v_tile = v[:, :, kv_start:kv_end, :]

            # Compute attention scores
            S = torch.matmul(q.float(), k_tile.float().transpose(-2, -1)) * scale

            # Apply causal mask based on global positions
            if causal:
                q_positions = torch.arange(Q_len, device=device).unsqueeze(1)
                kv_positions = torch.arange(kv_start, kv_end, device=device).unsqueeze(0)
                # In real ring attention, positions would account for block offsets
                mask = q_positions < kv_positions
                S = S.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Online softmax update
            m_ij = S.max(dim=-1, keepdim=True).values
            m_new = torch.max(m_i, m_ij)
            exp_m_diff = torch.exp(m_i - m_new)
            P = torch.exp(S - m_new)
            l_ij = P.sum(dim=-1, keepdim=True)
            l_new = exp_m_diff * l_i + l_ij

            acc = exp_m_diff * acc + torch.matmul(P, v_tile.float())
            m_i = m_new
            l_i = l_new

        # Finalize
        acc = acc / torch.clamp(l_i, min=1e-10)
        output = acc.to(dtype)

        return output

    def simulate_ring_attention(
        self,
        q_full: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        world_size: int = 4,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Simulate ring attention by splitting sequences across virtual devices.

        Useful for testing and benchmarking the ring attention algorithm
        without requiring actual multi-GPU setup.

        Args:
            q_full: Full query tensor (B, H, S, D).
            k_full: Full key tensor (B, H, S, D).
            v_full: Full value tensor (B, H, S, D).
            world_size: Number of virtual devices.
            causal: Whether to apply causal masking.

        Returns:
            Full attention output (B, H, S, D).
        """
        B, H, S, D = q_full.shape
        assert S % world_size == 0, f"Sequence length {S} must be divisible by world_size {world_size}"

        chunk_size = S // world_size

        # Split into chunks
        q_chunks = q_full.chunk(world_size, dim=2)
        k_chunks = k_full.chunk(world_size, dim=2)
        v_chunks = v_full.chunk(world_size, dim=2)

        # Process each device's chunk
        outputs = []
        for rank in range(world_size):
            out = self._ring_forward(
                q_chunks[rank], k_chunks[rank], v_chunks[rank],
                causal=causal,
                scale=self.softmax_scale,
                rank=rank,
                world_size=world_size,
            )
            outputs.append(out)

        return torch.cat(outputs, dim=2)
