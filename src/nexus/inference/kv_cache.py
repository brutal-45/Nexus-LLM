"""
KV Cache Management for Nexus
================================
Complete key-value cache system for efficient autoregressive inference.

Implements:
1. StandardKVCache - basic KV cache with pre-allocation
2. QuantizedKVCache - FP8/INT8/INT4 quantized cache for memory reduction
3. SlidingWindowKVCache - fixed-window cache for long context
4. CrossLayerKVCache - shared KV across layers to reduce memory
5. MultiTokenPredictionCache - cache for multi-token prediction heads
6. CacheManager - unified interface for managing caches across requests
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. StandardKVCache
# ---------------------------------------------------------------------------

class StandardKVCache:
    """
    Standard pre-allocated KV cache for autoregressive generation.

    Stores K, V tensors of shape
    ``(num_layers, 2, batch_size, num_heads, max_seq_len, head_dim)``.
    Uses a slot index to track the current write position.

    Memory for 100B model, BF16, seq_len=8192, 80 layers, 96 heads, 128 head_dim::

        K: 80 * 96 * 8192 * 128 * 2 bytes ≈ 19.3 GB
        V: same ≈ 19.3 GB
        Total: ≈ 38.6 GB per request

    Optimizations:
    - Pre-allocation avoids repeated malloc
    - Pinned memory for faster CPU-GPU transfer
    - Supports copy-on-write for beam search
    - Supports sequence concatenation (prefix caching)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        pin_memory: bool = False,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.pin_memory = pin_memory

        # Pre-allocate: (num_layers, 2[K,V], batch, heads, max_seq, head_dim)
        cache_opts: Dict[str, Any] = {
            "dtype": dtype,
            "device": self.device,
        }
        if pin_memory:
            cache_opts["pin_memory"] = True

        self.kv_cache = torch.zeros(
            num_layers, 2, batch_size, num_heads, max_seq_len, head_dim, **cache_opts
        )
        self.slot_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # Tracks how many tokens have been cached per batch entry
        self._seq_lens = torch.zeros(batch_size, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ core

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        slot_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert *new_k* and *new_v* into the cache.

        Parameters
        ----------
        layer_idx:
            Transformer layer index.
        new_k, new_v:
            Tensors of shape ``(batch, num_heads, num_new_tokens, head_dim)``.
        slot_indices:
            Optional per-batch write positions.  When ``None`` the current
            ``self.slot_indices`` are used and advanced automatically.
        """
        if slot_indices is None:
            slot_indices = self.slot_indices

        num_new = new_k.shape[2]  # number of new tokens

        for b in range(new_k.shape[0]):
            start = slot_indices[b].item() if slot_indices.dim() > 0 else slot_indices.item()
            end = start + num_new

            if end > self.max_seq_len:
                raise ValueError(
                    f"Cache overflow: attempting to write to slot {end} "
                    f"but max_seq_len is {self.max_seq_len}"
                )

            self.kv_cache[layer_idx, 0, b, :, start:end, :] = new_k[b]
            self.kv_cache[layer_idx, 1, b, :, start:end, :] = new_v[b]

        # Advance slot pointer when using default path
        if slot_indices is self.slot_indices:
            self.slot_indices.add_(num_new)
            self._seq_lens.add_(num_new)

    def get(
        self,
        layer_idx: int,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K, V from cache up to *seq_len*.

        Returns
        -------
        k, v : ``torch.Tensor``
            Shape ``(batch, num_heads, seq_len, head_dim)``.
        """
        if seq_len is None:
            seq_len = int(self.current_seq_len)
        k = self.kv_cache[layer_idx, 0, :, :, :seq_len, :]
        v = self.kv_cache[layer_idx, 1, :, :, :seq_len, :]
        return k, v

    def get_attn_mask(
        self,
        seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate a causal attention mask for the current cache state.

        Returns a boolean mask of shape ``(1, 1, seq_len, seq_len)`` where
        ``True`` means *allowed to attend*.
        """
        # For a fully-causal sequence, the upper triangle is masked out.
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
        # Expand to (1, 1, seq_len, seq_len) for broadcasting with
        # (batch, heads, q_len, kv_len)
        return mask.unsqueeze(0).unsqueeze(0)

    # -------------------------------------------------------- beam search helpers

    def copy(self) -> "StandardKVCache":
        """Deep copy for beam search (copy-on-write)."""
        new_cache = StandardKVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            batch_size=self.batch_size,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        new_cache.kv_cache.copy_(self.kv_cache)
        new_cache.slot_indices.copy_(self.slot_indices)
        new_cache._seq_lens.copy_(self._seq_lens)
        return new_cache

    def copy_from(self, src: "StandardKVCache", beam_indices: torch.Tensor) -> None:
        """Copy selected beam rows from *src* into *self*.

        Parameters
        ----------
        src:
            Source cache (e.g. from previous beam step).
        beam_indices:
            1-D tensor of length ``batch_size`` mapping each beam to a source row.
        """
        self.kv_cache = src.kv_cache[:, :, beam_indices].clone()
        self.slot_indices.copy_(src.slot_indices[beam_indices])
        self._seq_lens.copy_(src._seq_lens[beam_indices])

    # ------------------------------------------------------------ concatenation

    def concat_prefix(self, prefix_k: torch.Tensor, prefix_v: torch.Tensor) -> None:
        """Copy prefix K/V into the beginning of every layer's cache.

        Parameters
        ----------
        prefix_k, prefix_v:
            Shape ``(num_layers, batch, heads, prefix_len, head_dim)``.
        """
        prefix_len = prefix_k.shape[3]
        self.kv_cache[:, 0, :, :, :prefix_len, :] = prefix_k
        self.kv_cache[:, 1, :, :, :prefix_len, :] = prefix_v
        self.slot_indices.fill_(prefix_len)
        self._seq_lens.fill_(prefix_len)

    # ------------------------------------------------------------------ utils

    def clear(self) -> None:
        """Reset cache to initial state."""
        self.kv_cache.zero_()
        self.slot_indices.zero_()
        self._seq_lens.zero_()

    def memory_usage_bytes(self) -> int:
        """Return memory usage in bytes."""
        return (
            self.num_layers
            * 2
            * self.batch_size
            * self.num_heads
            * self.max_seq_len
            * self.head_dim
            * self.dtype.itemsize
        )

    @property
    def current_seq_len(self) -> int:
        """Current number of cached tokens (max across batch)."""
        return int(self._seq_lens.max().item()) if self._seq_lens.numel() > 0 else 0

    def __repr__(self) -> str:
        return (
            f"StandardKVCache(layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, max_seq={self.max_seq_len}, "
            f"batch={self.batch_size}, dtype={self.dtype}, "
            f"mem={self.memory_usage_bytes() / 1e9:.1f}GB)"
        )


# ---------------------------------------------------------------------------
# 2. QuantizedKVCache
# ---------------------------------------------------------------------------

class KVCacheQuantizer(nn.Module):
    """
    Quantizes KV cache entries to reduce memory usage.

    Supports:
    - FP8 (E4M3 / E5M2): 2x compression, minimal quality loss
    - INT8: 2x compression, symmetric per-tensor or per-channel
    - INT4: 4x compression, group-wise quantization

    Quantization scheme::

        FP8:   scale = max(|x|) / fp8_max;  x_q = round(x / scale)
        INT8:  scale = max(|x|) / 127;       x_q = clamp(round(x / scale), -128, 127)
        INT4:  group_size=32; per-group scale; x_q = clamp(round(x / scale), -8, 7)

    For a 100B model with INT4 quantization::

        Standard KV cache: ≈ 38.6 GB
        INT4 KV cache:     ≈  9.65 GB  (75 % reduction)

    Trade-offs:
        FP8:  Best quality, 2x compression, requires hardware support
        INT8: Good quality, 2x compression, works everywhere
        INT4: Acceptable quality with group-wise, 4x compression
    """

    FP8_E4M3_MAX = 448.0  # max representable in float8_e4m3fn
    FP8_E5M2_MAX = 57344.0

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        quant_type: str = "int8",
        group_size: int = 32,
    ):
        super().__init__()
        self.original_dtype = dtype
        self.quant_type = quant_type  # 'fp8_e4m3', 'fp8_e5m2', 'int8', 'int4'
        self.group_size = group_size

    def _target_dtype(self) -> torch.dtype:
        """Return the storage dtype for the quantized tensor."""
        if self.quant_type == "int8":
            return torch.int8
        if self.quant_type == "int4":
            return torch.int8  # stored as int8, 2 values packed per byte
        if self.quant_type in ("fp8_e4m3", "fp8_e5m2"):
            return torch.float8_e4m3fn if "e4m3" in self.quant_type else torch.float8_e5m2
        raise ValueError(f"Unsupported quant_type: {self.quant_type}")

    def _fp8_max_value(self) -> float:
        if self.quant_type == "fp8_e5m2":
            return self.FP8_E5M2_MAX
        return self.FP8_E4M3_MAX

    # ----- quantize / dequantize for scalar types (INT8, INT4) -----

    def _quantize_int(self, x: torch.Tensor, max_val: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric per-tensor / per-group quantization to integer type.

        Returns ``(quantized_values, scale)``.
        """
        x_flat = x.reshape(-1, self.group_size) if self.quant_type == "int4" else x

        # Per-group scale for INT4, per-tensor for INT8
        if self.quant_type == "int4":
            x_group = x.reshape(*x.shape[:-1], -1, self.group_size)
            scale = x_group.abs().amax(dim=-1, keepdim=True)
            scale = scale.clamp(min=1e-8) / max_val
            x_q = (x_group / scale).round().clamp(-max_val - 1, max_val)
            # Pack 2 int4 values into 1 int8: (low_nibble << 4) | high_nibble
            # For simplicity we store as int8 with the low value.
            # In production you would pack 2x more efficiently.
            x_q = x_q.to(torch.int8)
            return x_q, scale.squeeze(-1)
        else:
            # INT8 – per-tensor or per-head
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max_val
            x_q = (x / scale).round().clamp(-max_val - 1, max_val).to(torch.int8)
            return x_q, scale

    def _dequantize_int(
        self, x_q: torch.Tensor, scale: torch.Tensor, max_val: int
    ) -> torch.Tensor:
        """Dequantize integer tensor back to the original dtype."""
        if self.quant_type == "int4" and scale.ndim > x_q.ndim:
            # Scale was squeezed during quantize; expand back
            scale = scale.unsqueeze(-1)
        return (x_q.float() * scale).to(self.original_dtype)

    # ----- public API -----

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize *x*.  Returns ``(quantized_values, scale)``."""
        if self.quant_type in ("fp8_e4m3", "fp8_e5m2"):
            fp8_max = self._fp8_max_value()
            scale = x.abs().amax().clamp(min=1e-8) / fp8_max
            x_q = (x / scale).to(self._target_dtype())
            return x_q, scale
        elif self.quant_type == "int8":
            return self._quantize_int(x, max_val=127)
        elif self.quant_type == "int4":
            return self._quantize_int(x, max_val=7)
        else:
            raise ValueError(f"Unsupported quant_type: {self.quant_type}")

    def dequantize(self, x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor back to the original dtype."""
        if self.quant_type in ("fp8_e4m3", "fp8_e5m2"):
            return (x_q.float() * scale).to(self.original_dtype)
        elif self.quant_type == "int8":
            return self._dequantize_int(x_q, scale, max_val=127)
        elif self.quant_type == "int4":
            return self._dequantize_int(x_q, scale, max_val=7)
        else:
            raise ValueError(f"Unsupported quant_type: {self.quant_type}")

    def quantize_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Quantize both K and V.

        Returns ``((k_q, k_scale), (v_q, v_scale))``.
        """
        return self.quantize(k), self.quantize(v)

    def dequantize_kv(
        self,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v_q: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize K and V for attention computation."""
        return self.dequantize(k_q, k_scale), self.dequantize(v_q, v_scale)


class QuantizedKVCache:
    """
    KV cache with on-the-fly quantization.

    Stores quantized K, V in lower precision and dequantizes on-demand
    during attention computation.  This trades compute for memory.

    For attention: dequantize the relevant K, V slice, compute attention,
    then discard the dequantized copy (no persistent FP16/BF16 storage).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        batch_size: int = 1,
        quant_type: str = "int8",
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.quant_type = quant_type

        self.quantizer = KVCacheQuantizer(dtype=dtype, quant_type=quant_type)

        # Determine quantized element size for memory calculation
        if quant_type in ("fp8_e4m3", "fp8_e5m2"):
            self._elem_bytes = 1
        elif quant_type == "int8":
            self._elem_bytes = 1
        elif quant_type == "int4":
            self._elem_bytes = 1  # packed 2 per byte in practice
        else:
            self._elem_bytes = dtype.itemsize

        # Storage for quantized values  (num_layers, batch, heads, max_seq, head_dim)
        self.k_cache_q = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
            dtype=self.quantizer._target_dtype(), device=self.device,
        )
        self.v_cache_q = self.k_cache_q.new_zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim,
        )

        # Per-head/per-group scales (num_layers, batch, heads, max_seq, groups)
        if quant_type == "int4":
            groups_per_seq = head_dim // self.quantizer.group_size
            self.k_scale = torch.ones(
                num_layers, batch_size, num_heads, max_seq_len, groups_per_seq,
                dtype=dtype, device=self.device,
            )
            self.v_scale = self.k_scale.clone()
        elif quant_type == "int8":
            # Per-head scale: (num_layers, batch, heads, max_seq, 1)
            self.k_scale = torch.ones(
                num_layers, batch_size, num_heads, max_seq_len, 1,
                dtype=dtype, device=self.device,
            )
            self.v_scale = self.k_scale.clone()
        else:
            # FP8 – single scalar scale per layer-batch-head-seq is wasteful;
            # use per-head: (num_layers, batch, heads, max_seq, 1)
            self.k_scale = torch.ones(
                num_layers, batch_size, num_heads, max_seq_len, 1,
                dtype=dtype, device=self.device,
            )
            self.v_scale = self.k_scale.clone()

        self.slot_idx = 0

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Quantize and store new K, V."""
        num_new = new_k.shape[2]  # (batch, heads, seq, dim) or (batch, heads, dim) for 1 token
        if new_k.dim() == 3:
            # Single token: (batch, heads, dim) -> add seq dim
            new_k = new_k.unsqueeze(2)
            new_v = new_v.unsqueeze(2)
            num_new = 1

        end = self.slot_idx + num_new
        if end > self.max_seq_len:
            raise ValueError(
                f"QuantizedKVCache overflow: slot {end} > max_seq_len {self.max_seq_len}"
            )

        k_q, k_scale = self.quantizer.quantize(new_k)
        v_q, v_scale = self.quantizer.quantize(new_v)

        self.k_cache_q[layer_idx, :, :, self.slot_idx:end, :] = k_q
        self.v_cache_q[layer_idx, :, :, self.slot_idx:end, :] = v_q

        # Store scales – broadcast over the seq dimension if needed
        if self.quant_type in ("int8", "fp8_e4m3", "fp8_e5m2"):
            # k_scale shape: (batch, heads, seq, 1) or (batch, heads, 1)
            if k_scale.dim() == 2:
                k_scale = k_scale.unsqueeze(-1).unsqueeze(-1)  # (batch, heads, 1, 1)
            self.k_scale[layer_idx, :, :, self.slot_idx:end, :] = k_scale.expand(-1, -1, num_new, -1)
            if v_scale.dim() == 2:
                v_scale = v_scale.unsqueeze(-1).unsqueeze(-1)
            self.v_scale[layer_idx, :, :, self.slot_idx:end, :] = v_scale.expand(-1, -1, num_new, -1)
        else:
            # INT4: k_scale shape is (batch, heads, seq, groups)
            self.k_scale[layer_idx, :, :, self.slot_idx:end, :] = k_scale
            self.v_scale[layer_idx, :, :, self.slot_idx:end, :] = v_scale

        self.slot_idx = end

    def get(
        self, layer_idx: int, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return K, V up to *seq_len*."""
        if seq_len is None:
            seq_len = self.slot_idx

        k_q = self.k_cache_q[layer_idx, :, :, :seq_len, :]
        v_q = self.v_cache_q[layer_idx, :, :, :seq_len, :]
        k_s = self.k_scale[layer_idx, :, :, :seq_len, :]
        v_s = self.v_scale[layer_idx, :, :, :seq_len, :]

        return self.quantizer.dequantize(k_q, k_s), self.quantizer.dequantize(v_q, v_s)

    def memory_usage_bytes(self) -> int:
        """Memory usage considering quantization."""
        quant_bytes = (
            self.num_layers
            * self.batch_size
            * self.num_heads
            * self.max_seq_len
            * self.head_dim
            * self._elem_bytes
        )
        scale_bytes = self.k_scale.numel() * self.k_scale.element_size()
        return (quant_bytes + scale_bytes) * 2  # K + V

    def compression_ratio(self) -> float:
        """Return compression ratio vs FP16 (higher = more savings)."""
        original_bytes = (
            self.num_layers
            * 2
            * self.batch_size
            * self.num_heads
            * self.max_seq_len
            * self.head_dim
            * 2  # 2 bytes for FP16
        )
        return original_bytes / max(self.memory_usage_bytes(), 1)

    def clear(self) -> None:
        """Reset the cache."""
        self.k_cache_q.zero_()
        self.v_cache_q.zero_()
        self.k_scale.fill_(1.0)
        self.v_scale.fill_(1.0)
        self.slot_idx = 0

    def __repr__(self) -> str:
        return (
            f"QuantizedKVCache(layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, max_seq={self.max_seq_len}, "
            f"quant={self.quant_type}, "
            f"mem={self.memory_usage_bytes() / 1e9:.1f}GB, "
            f"compression={self.compression_ratio():.1f}x)"
        )


# ---------------------------------------------------------------------------
# 3. SlidingWindowKVCache
# ---------------------------------------------------------------------------

class SlidingWindowKVCache:
    """
    KV cache with a sliding window of fixed size.

    Only retains the most recent ``window_size`` tokens in the cache.
    Older tokens are evicted.  This enables inference on infinite-length
    sequences with fixed memory.

    Two modes:
    1. **Strict sliding**: only *window_size* tokens; older completely dropped.
    2. **Sink + sliding** (StreamingLLM): always keep the first ``sink_size``
       tokens plus the recent window.  Attention sinks prevent quality
       degradation on long sequences.

    Memory savings::

        Standard cache:  O(S)  where S = full sequence length
        Sliding window:  O(W)  where W = window_size (e.g. 4096)
        For S = 1M, W = 4096: 256x memory reduction

    Implementation:
        Uses a circular buffer.  When full, new tokens overwrite the oldest.
        A position-mapping table tracks logical → physical slot mapping.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int = 4096,
        sink_size: int = 4,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.sink_size = sink_size
        self.effective_size = window_size + sink_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device)

        # Circular buffer storage (num_layers, batch, heads, effective_size, head_dim)
        self.k_cache = torch.zeros(
            num_layers, batch_size, num_heads, self.effective_size, head_dim,
            dtype=dtype, device=self.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # Circular buffer pointers (per batch entry)
        self.write_pos = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.total_seen = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._filled = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Position ID tracking for RoPE / ALiBi
        # Stores logical position IDs for each slot in the circular buffer
        self.position_ids = torch.zeros(
            batch_size, self.effective_size, dtype=torch.long, device=self.device,
        )

    def _circular_idx(self, pos: torch.Tensor, capacity: int) -> torch.Tensor:
        """Map logical write positions to circular buffer indices."""
        return pos % capacity

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """Insert new K, V using circular buffer.  Overwrites oldest when full."""
        num_new = new_k.shape[2] if new_k.dim() == 4 else 1

        for b in range(new_k.shape[0]):
            wp = self.write_pos[b].item()
            seen = self.total_seen[b].item()

            for t in range(num_new):
                if seen < self.sink_size:
                    # Fill sink slots first
                    phys = t
                elif seen < self.effective_size:
                    # Fill sliding window slots
                    phys = self.sink_size + (seen - self.sink_size)
                else:
                    # Circular overwrite in the sliding window portion
                    window_pos = self.sink_size + ((seen - self.sink_size) % self.window_size)
                    phys = window_pos
                    self._filled[b] = True

                # Handle multi-token input
                if new_k.dim() == 4:
                    self.k_cache[layer_idx, b, :, phys, :] = new_k[b, :, t, :]
                    self.v_cache[layer_idx, b, :, phys, :] = new_v[b, :, t, :]
                else:
                    self.k_cache[layer_idx, b, :, phys, :] = new_k[b]
                    self.v_cache[layer_idx, b, :, phys, :] = new_v[b]

                self.position_ids[b, phys] = seen
                self.write_pos[b] = phys + 1
                seen += 1

            self.total_seen[b] = seen

    def get(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K, V in correct temporal order (sink tokens first, then window).

        Returns tensors of shape ``(batch, num_heads, effective_size, head_dim)``
        (or less if buffer not yet full).
        """
        b = 0  # Support first batch entry; extend as needed
        seen = self.total_seen[b].item()

        if seen <= self.effective_size:
            # Buffer not yet wrapped – return [0, seen)
            k = self.k_cache[layer_idx, b, :, :seen, :]
            v = self.v_cache[layer_idx, b, :, :seen, :]
        else:
            # Buffer has wrapped – return sink + recent window in order
            sink_k = self.k_cache[layer_idx, b, :, :self.sink_size, :]
            sink_v = self.v_cache[layer_idx, b, :, :self.sink_size, :]

            # Determine window order: the oldest unwritten slot is where
            # the next write will go; everything after it (wrapping) is newer.
            window_start_logical = self.total_seen[b].item() - self.window_size
            window_start_phys = self.sink_size + ((window_start_logical - self.sink_size) % self.window_size)

            # Collect window in order
            idx_a = torch.arange(window_start_phys, self.effective_size, device=self.device)
            idx_b = torch.arange(self.sink_size, window_start_phys, device=self.device)
            window_indices = torch.cat([idx_a, idx_b])

            window_k = self.k_cache[layer_idx, b, :, window_indices, :]
            window_v = self.v_cache[layer_idx, b, :, window_indices, :]

            k = torch.cat([sink_k, window_k], dim=2)
            v = torch.cat([sink_v, window_v], dim=2)

        return k, v

    def get_position_mapping(self, total_seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Map logical positions to cache slots.

        Returns ``position_ids`` of shape ``(1, effective_size_or_actual_len)``
        that the model should use for RoPE / ALiBi positional encodings.
        """
        seen = int(self.total_seen[0].item()) if total_seq_len is None else total_seq_len

        if seen <= self.effective_size:
            return self.position_ids[0, :seen].unsqueeze(0)
        else:
            sink_pos = self.position_ids[0, :self.sink_size]
            window_start_logical = seen - self.window_size
            window_start_phys = self.sink_size + ((window_start_logical - self.sink_size) % self.window_size)

            idx_a = torch.arange(window_start_phys, self.effective_size, device=self.device)
            idx_b = torch.arange(self.sink_size, window_start_phys, device=self.device)
            window_indices = torch.cat([idx_a, idx_b])

            window_pos = self.position_ids[0, window_indices]
            return torch.cat([sink_pos, window_pos]).unsqueeze(0)

    def get_attn_mask(self) -> torch.Tensor:
        """Return a causal attention mask suitable for the sliding window layout.

        Returns shape ``(1, 1, cache_len, cache_len)``.
        """
        seen = int(self.total_seen[0].item())
        if seen <= self.effective_size:
            length = seen
        else:
            length = self.effective_size

        # Causal mask: position i can attend to positions 0..i
        mask = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=self.device)
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def clear(self) -> None:
        """Reset circular buffer."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.write_pos.zero_()
        self.total_seen.zero_()
        self._filled.zero_()
        self.position_ids.zero_()

    @property
    def current_seq_len(self) -> int:
        """Total tokens processed (may exceed window_size)."""
        return int(self.total_seen.max().item()) if self.total_seen.numel() > 0 else 0

    def memory_usage_bytes(self) -> int:
        """Memory usage for the circular buffer."""
        return (
            self.num_layers
            * 2
            * self.batch_size
            * self.num_heads
            * self.effective_size
            * self.head_dim
            * self.dtype.itemsize
        )

    def __repr__(self) -> str:
        return (
            f"SlidingWindowKVCache(layers={self.num_layers}, heads={self.num_heads}, "
            f"window={self.window_size}, sink={self.sink_size}, "
            f"effective={self.effective_size}, "
            f"mem={self.memory_usage_bytes() / 1e9:.1f}GB)"
        )


# ---------------------------------------------------------------------------
# 4. CrossLayerKVCache
# ---------------------------------------------------------------------------

class CrossLayerKVCache:
    """
    KV cache that shares key-value pairs across layers.

    Observation: KV representations are often similar across adjacent layers.
    This cache stores KV for a subset of layers and interpolates / shares
    for the remaining layers.

    Strategies:
    1. **Uniform sharing**: store KV every N layers; share for in-between.
       Example: 80 layers, store every 4th → 20 layers (75 % reduction).
    2. **Early-late split**: store KV for first L/2 layers, share for the rest.
    3. **Custom list**: user-specified layer indices.

    Quality impact: minimal for well-trained models (~1-2 % perplexity increase
    for 4x sharing).  Significant memory savings for very deep models.

    For a 100B model with 80 layers, 4x sharing::

        Standard: 80 layers × ~0.48 GB/layer = ~38.6 GB
        Shared:   20 layers × ~0.48 GB/layer = ~9.6 GB (75 % reduction)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        batch_size: int = 1,
        sharing_strategy: str = "uniform",
        share_ratio: int = 4,
        custom_layers: Optional[List[int]] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.sharing_strategy = sharing_strategy
        self.share_ratio = share_ratio

        # Determine which layers get their own KV cache
        self.stored_layers = self._compute_stored_layers(
            sharing_strategy, share_ratio, custom_layers
        )

        # Map from any layer index -> stored layer index
        self._layer_map = self._build_layer_map()

        # Allocate cache only for stored layers
        num_stored = len(self.stored_layers)
        self.k_cache = torch.zeros(
            num_stored, 2, batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=self.device,
        )
        self.slot_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Optional: learned blending weights for shared layers
        if sharing_strategy == "learned":
            self.blend_weights = nn.Parameter(
                torch.ones(num_layers, dtype=dtype, device=device)
            )
        else:
            self.blend_weights = None

    def _compute_stored_layers(
        self,
        strategy: str,
        ratio: int,
        custom: Optional[List[int]],
    ) -> List[int]:
        """Determine which layer indices get their own KV cache."""
        if custom is not None:
            return sorted(set(custom))

        if strategy == "uniform":
            return list(range(0, self.num_layers, ratio))
        elif strategy == "early_late":
            half = self.num_layers // 2
            return list(range(0, half, max(1, ratio // 2)))
        elif strategy == "first_half":
            return list(range(self.num_layers // 2))
        elif strategy == "learned":
            # Learned strategy starts with uniform and can adapt
            return list(range(0, self.num_layers, ratio))
        else:
            return list(range(0, self.num_layers, ratio))

    def _build_layer_map(self) -> Dict[int, int]:
        """Map any layer index → the stored-layer index that serves it."""
        mapping: Dict[int, int] = {}
        for i in range(self.num_layers):
            # Find closest stored layer (prefer the one at or before i)
            best = min(self.stored_layers, key=lambda s: (abs(s - i), -s))
            mapping[i] = self.stored_layers.index(best)
        return mapping

    def get_source_layer(self, layer_idx: int) -> int:
        """Return the stored-layer cache index that serves *layer_idx*."""
        return self._layer_map[layer_idx]

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """Update the appropriate cache entry (shared or own).

        Only layers in ``self.stored_layers`` write to the cache.
        Other layers' KV is derived from their source layer.
        """
        stored_idx = self._layer_map[layer_idx]
        actual_stored_layer = self.stored_layers[stored_idx]

        if layer_idx != actual_stored_layer:
            # This layer shares a cache – optionally blend
            if self.blend_weights is not None:
                # Weighted blend: new_kv = α * new + (1-α) * stored
                w = torch.sigmoid(self.blend_weights[layer_idx])
                existing_k, existing_v = self._get_stored(stored_idx, self.slot_idx[0].item())
                new_k = w * new_k + (1 - w) * existing_k[:, :, -new_k.shape[2]:, :]
                new_v = w * new_v + (1 - w) * existing_v[:, :, -new_v.shape[2]:, :]
            else:
                # No blending – just use the source layer's KV (skip write)
                return

        num_new = new_k.shape[2] if new_k.dim() == 4 else 1
        start = int(self.slot_idx[0])
        end = start + num_new

        if end > self.max_seq_len:
            raise ValueError(
                f"CrossLayerKVCache overflow: slot {end} > max_seq_len {self.max_seq_len}"
            )

        for b in range(new_k.shape[0]):
            if new_k.dim() == 4:
                self.k_cache[stored_idx, 0, b, :, start:end, :] = new_k[b]
                self.k_cache[stored_idx, 1, b, :, start:end, :] = new_v[b]
            else:
                self.k_cache[stored_idx, 0, b, :, start:end, :] = new_k[b]
                self.k_cache[stored_idx, 1, b, :, start:end, :] = new_v[b]

        self.slot_idx.add_(num_new)

    def _get_stored(
        self, stored_idx: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.k_cache[stored_idx, 0, :, :, :seq_len, :]
        v = self.k_cache[stored_idx, 1, :, :, :seq_len, :]
        return k, v

    def get(
        self, layer_idx: int, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve KV for *layer_idx* (may be shared from another layer)."""
        stored_idx = self._layer_map[layer_idx]
        if seq_len is None:
            seq_len = int(self.slot_idx.max().item())
        return self._get_stored(stored_idx, seq_len)

    def memory_savings_ratio(self) -> float:
        """Ratio of memory saved vs a full (non-shared) cache."""
        full = self.num_layers
        stored = len(self.stored_layers)
        return 1.0 - (stored / full)

    def memory_usage_bytes(self) -> int:
        """Memory usage for the shared cache."""
        return (
            len(self.stored_layers)
            * 2
            * self.batch_size
            * self.num_heads
            * self.max_seq_len
            * self.head_dim
            * self.dtype.itemsize
        )

    def clear(self) -> None:
        """Reset cache."""
        self.k_cache.zero_()
        self.slot_idx.zero_()

    def __repr__(self) -> str:
        return (
            f"CrossLayerKVCache(layers={self.num_layers}, "
            f"stored={len(self.stored_layers)}, strategy={self.sharing_strategy}, "
            f"savings={self.memory_savings_ratio() * 100:.0f}%, "
            f"mem={self.memory_usage_bytes() / 1e9:.1f}GB)"
        )


# ---------------------------------------------------------------------------
# 5. MultiTokenPredictionCache
# ---------------------------------------------------------------------------

class MultiTokenPredictionCache:
    """
    Extended KV cache for multi-token prediction (MTP / speculative decoding).

    When the model predicts *N* future tokens at once (via extra prediction
    heads or a draft model), the cache must track:

    - **Verified tokens**: accepted into the main sequence (always in cache).
    - **Draft tokens**: speculative (may be rejected).
    - Draft tokens need separate KV entries that can be rolled back.

    State per request:
    - ``verified_len``: number of verified tokens.
    - ``draft_len``: number of draft tokens beyond the verified point.
    - ``draft_k/v_cache``: separate KV storage for draft tokens.
    """

    def __init__(
        self,
        base_cache: Any,
        max_draft_tokens: int = 16,
    ):
        """
        Parameters
        ----------
        base_cache:
            The underlying KV cache (e.g. ``StandardKVCache``) that stores
            verified tokens.
        max_draft_tokens:
            Maximum number of speculative draft tokens per step.
        """
        self.base_cache = base_cache
        self.max_draft_tokens = max_draft_tokens
        self.verified_len = 0
        self.draft_len = 0

        # Derive dimensions from the base cache
        self._num_layers = getattr(base_cache, "num_layers", 0)
        self._num_heads = getattr(base_cache, "num_heads", 0)
        self._head_dim = getattr(base_cache, "head_dim", 0)
        self._dtype = getattr(base_cache, "dtype", torch.bfloat16)
        self._device = getattr(base_cache, "device", torch.device("cuda"))
        self._batch_size = getattr(base_cache, "batch_size", 1)

        # Separate storage for draft token KV entries
        self.draft_k_cache = torch.zeros(
            self._num_layers, 2, self._batch_size,
            self._num_heads, max_draft_tokens, self._head_dim,
            dtype=self._dtype, device=self._device,
        )
        self.draft_v_cache = self.draft_k_cache.clone()

    def update_verified(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """Add verified tokens to the main (base) cache."""
        self.base_cache.update(layer_idx, new_k, new_v)
        num_new = new_k.shape[2] if new_k.dim() == 4 else 1
        self.verified_len += num_new

    def update_draft(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """Add speculative draft tokens (separate storage)."""
        num_new = new_k.shape[2] if new_k.dim() == 4 else 1

        if self.draft_len + num_new > self.max_draft_tokens:
            raise ValueError(
                f"Draft overflow: {self.draft_len + num_new} > max {self.max_draft_tokens}"
            )

        start = self.draft_len
        end = start + num_new

        for b in range(new_k.shape[0]):
            if new_k.dim() == 4:
                self.draft_k_cache[layer_idx, 0, b, :, start:end, :] = new_k[b]
                self.draft_v_cache[layer_idx, 0, b, :, start:end, :] = new_v[b]
                self.draft_k_cache[layer_idx, 1, b, :, start:end, :] = new_k[b]
                self.draft_v_cache[layer_idx, 1, b, :, start:end, :] = new_v[b]
            else:
                self.draft_k_cache[layer_idx, 0, b, :, start:end, :] = new_k[b]
                self.draft_v_cache[layer_idx, 0, b, :, start:end, :] = new_v[b]
                self.draft_k_cache[layer_idx, 1, b, :, start:end, :] = new_k[b]
                self.draft_v_cache[layer_idx, 1, b, :, start:end, :] = new_v[b]

        self.draft_len += num_new

    def accept_drafts(self, num_accepted: int) -> None:
        """Move *num_accepted* draft tokens into the verified (base) cache."""
        if num_accepted <= 0:
            return

        if num_accepted > self.draft_len:
            raise ValueError(
                f"Cannot accept {num_accepted} drafts, only {self.draft_len} available"
            )

        # Copy accepted draft KV into the base cache for each layer
        for layer_idx in range(self._num_layers):
            accepted_k = self.draft_k_cache[layer_idx, :, :, :, :num_accepted, :]
            accepted_v = self.draft_v_cache[layer_idx, :, :, :, :num_accepted, :]
            self.base_cache.update(layer_idx, accepted_k, accepted_v)

        self.verified_len += num_accepted

        # Shift remaining drafts to the front
        remaining = self.draft_len - num_accepted
        if remaining > 0:
            for layer_idx in range(self._num_layers):
                self.draft_k_cache[layer_idx, :, :, :, :remaining, :] = (
                    self.draft_k_cache[layer_idx, :, :, :, num_accepted:num_accepted + remaining, :]
                )
                self.draft_v_cache[layer_idx, :, :, :, :remaining, :] = (
                    self.draft_v_cache[layer_idx, :, :, :, num_accepted:num_accepted + remaining, :]
                )
            # Zero out the old positions
            self.draft_k_cache[:, :, :, :, remaining:self.draft_len, :].zero_()
            self.draft_v_cache[:, :, :, :, remaining:self.draft_len, :].zero_()

        self.draft_len = remaining

    def reject_drafts(self) -> None:
        """Discard all draft tokens and reset the draft cache."""
        self.draft_k_cache.zero_()
        self.draft_v_cache.zero_()
        self.draft_len = 0

    def get(
        self,
        layer_idx: int,
        include_drafts: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV including both verified and (optionally) draft tokens.

        Returns concatenated K, V of shape
        ``(batch, num_heads, verified_len + draft_len, head_dim)``.
        """
        k, v = self.base_cache.get(layer_idx, self.verified_len)

        if include_drafts and self.draft_len > 0:
            draft_k = self.draft_k_cache[layer_idx, 0, :, :, :self.draft_len, :]
            draft_v = self.draft_v_cache[layer_idx, 1, :, :, :self.draft_len, :]
            k = torch.cat([k, draft_k], dim=2)
            v = torch.cat([v, draft_v], dim=2)

        return k, v

    @property
    def total_seq_len(self) -> int:
        """Total length including draft tokens."""
        return self.verified_len + self.draft_len

    def clear(self) -> None:
        """Reset both verified and draft caches."""
        self.base_cache.clear()
        self.draft_k_cache.zero_()
        self.draft_v_cache.zero_()
        self.verified_len = 0
        self.draft_len = 0

    def __repr__(self) -> str:
        return (
            f"MultiTokenPredictionCache(verified={self.verified_len}, "
            f"draft={self.draft_len}/{self.max_draft_tokens}, "
            f"base={self.base_cache!r})"
        )


# ---------------------------------------------------------------------------
# 6. CacheManager
# ---------------------------------------------------------------------------

class CacheManager:
    """
    Unified manager for KV caches across multiple concurrent requests.

    Handles:
    - Cache allocation / deallocation for concurrent requests.
    - Memory budget enforcement.
    - Priority-based eviction (LRU).
    - Prefix caching (shared prefixes between requests).
    - Automatic cache type selection based on sequence length.

    Used by the inference server for efficient batched serving.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        memory_budget_gb: float = 80.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)
        self.memory_budget = int(memory_budget_gb * 1024 ** 3)

        # Active caches keyed by request_id
        self.active_caches: OrderedDict[str, Any] = OrderedDict()
        # Track per-request metadata
        self.cache_meta: Dict[str, Dict[str, Any]] = {}
        # Prefix hash -> (request_id, prefix_len) for sharing
        self.prefix_index: Dict[int, Tuple[str, int]] = {}

    # ---------------------------------------------------------- allocation

    def allocate(
        self,
        request_id: str,
        cache_type: str = "standard",
        **kwargs: Any,
    ) -> Any:
        """Allocate a new KV cache for *request_id*.

        Parameters
        ----------
        request_id:
            Unique identifier for the request.
        cache_type:
            One of ``'standard'``, ``'quantized'``, ``'sliding_window'``,
            ``'cross_layer'``.
        **kwargs:
            Forwarded to the cache constructor (e.g. ``quant_type``,
            ``window_size``).

        Returns
        -------
        The allocated cache object.

        Raises
        ------
        RuntimeError
            If the allocation would exceed the memory budget and no caches
            can be evicted.
        """
        # Check if allocation fits in budget
        estimated_mem = self._estimate_cache_size(cache_type, **kwargs)
        if estimated_mem > self.available_memory():
            # Try eviction first
            freed = self.evict_lru()
            if freed + self.available_memory() < estimated_mem:
                raise RuntimeError(
                    f"Cannot allocate {estimated_mem / 1e9:.1f} GB for request "
                    f"'{request_id}': budget {self.memory_budget / 1e9:.1f} GB, "
                    f"available {self.available_memory() / 1e9:.1f} GB after eviction"
                )

        cache = self._create_cache(cache_type, **kwargs)

        self.active_caches[request_id] = cache
        self.cache_meta[request_id] = {
            "type": cache_type,
            "created_at": self._timestamp(),
            "last_access": self._timestamp(),
            "memory_bytes": cache.memory_usage_bytes() if hasattr(cache, "memory_usage_bytes") else estimated_mem,
        }

        return cache

    def _create_cache(self, cache_type: str, **kwargs: Any) -> Any:
        """Instantiate a cache of the given type."""
        if cache_type == "standard":
            return StandardKVCache(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=kwargs.get("max_seq_len", self.max_seq_len),
                batch_size=kwargs.get("batch_size", 1),
                dtype=self.dtype,
                device=self.device,
            )
        elif cache_type == "quantized":
            return QuantizedKVCache(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=kwargs.get("max_seq_len", self.max_seq_len),
                batch_size=kwargs.get("batch_size", 1),
                quant_type=kwargs.get("quant_type", "int8"),
                dtype=self.dtype,
                device=self.device,
            )
        elif cache_type == "sliding_window":
            return SlidingWindowKVCache(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                window_size=kwargs.get("window_size", 4096),
                sink_size=kwargs.get("sink_size", 4),
                batch_size=kwargs.get("batch_size", 1),
                dtype=self.dtype,
                device=self.device,
            )
        elif cache_type == "cross_layer":
            return CrossLayerKVCache(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq_len=kwargs.get("max_seq_len", self.max_seq_len),
                batch_size=kwargs.get("batch_size", 1),
                sharing_strategy=kwargs.get("sharing_strategy", "uniform"),
                share_ratio=kwargs.get("share_ratio", 4),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown cache_type: {cache_type}")

    def _estimate_cache_size(self, cache_type: str, **kwargs: Any) -> int:
        """Rough estimate of a cache's memory usage in bytes."""
        max_seq = kwargs.get("max_seq_len", self.max_seq_len)
        batch = kwargs.get("batch_size", 1)
        per_layer = 2 * batch * self.num_heads * max_seq * self.head_dim * self.dtype.itemsize

        if cache_type == "quantized":
            qt = kwargs.get("quant_type", "int8")
            if qt == "int4":
                per_layer //= 4
            elif qt in ("int8", "fp8_e4m3", "fp8_e5m2"):
                per_layer //= 2
        elif cache_type == "sliding_window":
            window = kwargs.get("window_size", 4096)
            sink = kwargs.get("sink_size", 4)
            per_layer = 2 * batch * self.num_heads * (window + sink) * self.head_dim * self.dtype.itemsize
        elif cache_type == "cross_layer":
            ratio = kwargs.get("share_ratio", 4)
            per_layer = per_layer // ratio

        return per_layer * self.num_layers

    # ---------------------------------------------------------- deallocation

    def deallocate(self, request_id: str) -> None:
        """Free cache for a completed request."""
        if request_id in self.active_caches:
            del self.active_caches[request_id]
            self.cache_meta.pop(request_id, None)

    def get_cache(self, request_id: str) -> Optional[Any]:
        """Retrieve the cache for *request_id*.  Touches LRU order."""
        cache = self.active_caches.get(request_id)
        if cache is not None and request_id in self.cache_meta:
            # Move to end (most recently used)
            self.active_caches.move_to_end(request_id)
            self.cache_meta[request_id]["last_access"] = self._timestamp()
        return cache

    # ---------------------------------------------------------- prefix sharing

    def share_prefix(
        self,
        src_request_id: str,
        dst_request_id: str,
        prefix_len: int,
    ) -> Optional[Any]:
        """Share prefix KV cache between two requests (copy-on-write).

        If the source request's cache supports prefix extraction, copies the
        first *prefix_len* tokens' KV into the destination's cache.

        Returns the destination cache, or ``None`` if sharing is not possible.
        """
        src_cache = self.active_caches.get(src_request_id)
        if src_cache is None:
            return None

        dst_cache = self.active_caches.get(dst_request_id)
        if dst_cache is not None:
            # Destination already allocated – copy prefix into it
            if isinstance(src_cache, StandardKVCache) and isinstance(dst_cache, StandardKVCache):
                for layer_idx in range(self.num_layers):
                    k, v = src_cache.get(layer_idx, prefix_len)
                    dst_cache.concat_prefix(
                        k.unsqueeze(0).expand(self.num_layers, -1, -1, -1, -1),
                        v.unsqueeze(0).expand(self.num_layers, -1, -1, -1, -1),
                    )
                # More precise per-layer copy
                dst_cache.slot_idx = torch.tensor(prefix_len, dtype=torch.long, device=self.device)
            return dst_cache
        return None

    def register_prefix(self, request_id: str, prefix_hash: int, prefix_len: int) -> None:
        """Register a prefix for potential sharing with future requests."""
        self.prefix_index[prefix_hash] = (request_id, prefix_len)

    def lookup_prefix(self, prefix_hash: int) -> Optional[Tuple[str, int]]:
        """Look up a previously registered prefix."""
        return self.prefix_index.get(prefix_hash)

    # ----------------------------------------------------------- auto-select

    def auto_select_cache_type(
        self,
        seq_len: int,
        num_requests: int,
    ) -> str:
        """Select the optimal cache type based on current conditions.

        Heuristic:
        - seq_len < 4096 and low concurrency → ``'standard'``
        - seq_len > 4096 and high concurrency → ``'quantized'`` (INT8)
        - seq_len > 32768 → ``'sliding_window'``
        - Very deep model with low memory → ``'cross_layer'``
        """
        mem_per_request_std = (
            2 * self.num_layers * self.num_heads * seq_len
            * self.head_dim * self.dtype.itemsize
        )
        total_demand = mem_per_request_std * num_requests

        if total_demand > self.memory_budget * 0.8:
            # Under extreme pressure: use most aggressive compression
            if seq_len > 32768:
                return "sliding_window"
            return "quantized"  # INT8

        if seq_len > 32768:
            return "sliding_window"

        if total_demand > self.memory_budget * 0.5:
            # Moderate pressure
            if self.num_layers >= 64:
                return "cross_layer"  # Deep model benefits from layer sharing
            return "quantized"

        if total_demand > self.memory_budget * 0.25:
            return "quantized"

        return "standard"

    # --------------------------------------------------------------- memory

    def total_memory_usage(self) -> int:
        """Total memory used by all active caches."""
        total = 0
        for meta in self.cache_meta.values():
            total += meta["memory_bytes"]
        return total

    def available_memory(self) -> int:
        """Remaining memory budget in bytes."""
        return max(0, self.memory_budget - self.total_memory_usage())

    def evict_lru(self) -> int:
        """Evict the least-recently-used cache to free memory.

        Returns the number of bytes freed.
        """
        if not self.active_caches:
            return 0

        # Pop the first (oldest) entry from the OrderedDict
        request_id, _ = self.active_caches.popitem(last=False)
        meta = self.cache_meta.pop(request_id, {})
        freed = meta.get("memory_bytes", 0)
        return freed

    # --------------------------------------------------------------- stats

    def get_stats(self) -> Dict[str, Any]:
        """Return cache utilization statistics."""
        type_counts: Dict[str, int] = {}
        for meta in self.cache_meta.values():
            ct = meta.get("type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        return {
            "num_active_requests": len(self.active_caches),
            "total_memory_bytes": self.total_memory_usage(),
            "available_memory_bytes": self.available_memory(),
            "memory_budget_bytes": self.memory_budget,
            "utilization_pct": (
                self.total_memory_usage() / max(self.memory_budget, 1) * 100
            ),
            "cache_types": type_counts,
            "num_prefixes_registered": len(self.prefix_index),
        }

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _timestamp() -> float:
        """Simple monotonic timestamp for LRU ordering."""
        import time
        return time.monotonic()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"CacheManager(requests={stats['num_active_requests']}, "
            f"mem={stats['total_memory_bytes'] / 1e9:.1f}/"
            f"{self.memory_budget / 1e9:.1f} GB, "
            f"util={stats['utilization_pct']:.1f}%)"
        )
