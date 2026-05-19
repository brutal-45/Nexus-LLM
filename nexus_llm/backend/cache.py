"""KV cache management for Nexus-LLM backend.

Implements cache allocation, eviction, memory tracking, and PagedAttention-style
management for efficient attention key-value cache handling.
"""

import torch
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import math
import logging

logger = logging.getLogger(__name__)


class CacheStatus(Enum):
    """Status of a cache block."""
    FREE = "free"
    OCCUPIED = "occupied"
    EVICTABLE = "evictable"
    SWAPPED = "swapped"


@dataclass
class CacheBlock:
    """A single block in the paged KV cache."""
    block_id: int
    status: CacheStatus = CacheStatus.FREE
    ref_count: int = 0
    last_access_step: int = 0
    device: str = "gpu"
    key_tensor: Optional[torch.Tensor] = None
    value_tensor: Optional[torch.Tensor] = None

    def mark_accessed(self, step: int) -> None:
        self.last_access_step = step

    def increment_ref(self) -> None:
        self.ref_count += 1

    def decrement_ref(self) -> int:
        self.ref_count = max(0, self.ref_count - 1)
        return self.ref_count


@dataclass
class CacheEntry:
    """Metadata for a cached sequence."""
    sequence_id: str
    block_ids: List[int]
    num_tokens: int
    created_step: int
    last_access_step: int
    priority: int = 0


class PagedKVCache:
    """PagedAttention-style KV cache management.

    Divides the KV cache into fixed-size blocks (pages) and manages them
    with copy-on-write semantics, enabling efficient memory sharing between
    sequences with common prefixes.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        num_layers: int = 32,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        self._blocks: Dict[int, CacheBlock] = {
            i: CacheBlock(block_id=i, device=device) for i in range(num_blocks)
        }
        self._entries: Dict[str, CacheEntry] = {}
        self._free_blocks: List[int] = list(range(num_blocks))
        self._step_counter = 0
        self._lock = threading.RLock()

        self._key_cache: Optional[List[torch.Tensor]] = None
        self._value_cache: Optional[List[torch.Tensor]] = None
        self._initialize_cache_tensors()

    def _initialize_cache_tensors(self) -> None:
        """Pre-allocate the physical cache tensors for all blocks."""
        try:
            target_device = torch.device(self.device) if torch.cuda.is_available() else torch.device("cpu")
            self._key_cache = [
                torch.zeros(
                    (self.num_blocks, self.block_size, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=target_device,
                )
                for _ in range(self.num_layers)
            ]
            self._value_cache = [
                torch.zeros(
                    (self.num_blocks, self.block_size, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device=target_device,
                )
                for _ in range(self.num_layers)
            ]
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"Failed to allocate cache on {self.device}, falling back to CPU: {e}")
            self._key_cache = [
                torch.zeros(
                    (self.num_blocks, self.block_size, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                )
                for _ in range(self.num_layers)
            ]
            self._value_cache = [
                torch.zeros(
                    (self.num_blocks, self.block_size, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                )
                for _ in range(self.num_layers)
            ]

    @property
    def num_free_blocks(self) -> int:
        """Number of free blocks available."""
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        """Number of occupied blocks."""
        return self.num_blocks - self.num_free_blocks

    @property
    def utilization(self) -> float:
        """Cache utilization as a fraction (0 to 1)."""
        return self.num_used_blocks / max(1, self.num_blocks)

    def get_memory_usage_mb(self) -> float:
        """Get total memory used by the cache in MB."""
        if self._key_cache is None:
            return 0.0
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        per_block_bytes = self.block_size * self.num_heads * self.head_dim * element_size
        total_bytes = per_block_bytes * self.num_used_blocks * self.num_layers * 2  # key + value
        return total_bytes / (1024 * 1024)

    def get_total_memory_mb(self) -> float:
        """Get total allocated memory for the cache in MB."""
        if self._key_cache is None:
            return 0.0
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        per_block_bytes = self.block_size * self.num_heads * self.head_dim * element_size
        total_bytes = per_block_bytes * self.num_blocks * self.num_layers * 2
        return total_bytes / (1024 * 1024)

    def allocate_blocks(self, sequence_id: str, num_tokens: int, priority: int = 0) -> List[int]:
        """Allocate cache blocks for a sequence.

        Args:
            sequence_id: Unique identifier for the sequence.
            num_tokens: Number of tokens to allocate space for.
            priority: Priority for eviction (higher = less likely evicted).

        Returns:
            List of allocated block IDs.
        """
        with self._lock:
            num_needed = math.ceil(num_tokens / self.block_size)

            if num_needed > self.num_free_blocks:
                freed = self._evict_blocks(num_needed - self.num_free_blocks, exclude_seq=sequence_id)
                if freed < num_needed - self.num_free_blocks:
                    raise RuntimeError(
                        f"Cannot allocate {num_needed} blocks for '{sequence_id}'. "
                        f"Only {self.num_free_blocks} free blocks available after eviction."
                    )

            allocated = []
            for _ in range(num_needed):
                if not self._free_blocks:
                    break
                block_id = self._free_blocks.pop(0)
                block = self._blocks[block_id]
                block.status = CacheStatus.OCCUPIED
                block.ref_count = 1
                block.mark_accessed(self._step_counter)
                allocated.append(block_id)

            self._step_counter += 1
            self._entries[sequence_id] = CacheEntry(
                sequence_id=sequence_id,
                block_ids=allocated,
                num_tokens=num_tokens,
                created_step=self._step_counter,
                last_access_step=self._step_counter,
                priority=priority,
            )

            logger.debug(f"Allocated {len(allocated)} blocks for sequence '{sequence_id}'")
            return allocated

    def free_blocks(self, sequence_id: str) -> int:
        """Free all blocks associated with a sequence.

        Returns the number of blocks freed.
        """
        with self._lock:
            entry = self._entries.pop(sequence_id, None)
            if entry is None:
                return 0

            freed_count = 0
            for block_id in entry.block_ids:
                block = self._blocks[block_id]
                block.decrement_ref()
                if block.ref_count == 0:
                    block.status = CacheStatus.FREE
                    block.key_tensor = None
                    block.value_tensor = None
                    self._free_blocks.append(block_id)
                    freed_count += 1
                else:
                    block.status = CacheStatus.EVICTABLE

            logger.debug(f"Freed {freed_count} blocks for sequence '{sequence_id}'")
            return freed_count

    def _evict_blocks(self, num_needed: int, exclude_seq: Optional[str] = None) -> int:
        """Evict blocks using LRU policy, respecting priorities.

        Returns the number of blocks freed.
        """
        evictable_entries = []
        for seq_id, entry in self._entries.items():
            if seq_id == exclude_seq:
                continue
            evictable_entries.append((entry.priority, entry.last_access_step, entry.sequence_id))

        evictable_entries.sort(key=lambda x: (x[0], x[1]))

        freed = 0
        for priority, access_step, seq_id in evictable_entries:
            if freed >= num_needed:
                break
            freed += self.free_blocks(seq_id)

        return freed

    def update_access(self, sequence_id: str, new_tokens: int = 0) -> None:
        """Update access time and optionally allocate more blocks for a sequence."""
        with self._lock:
            entry = self._entries.get(sequence_id)
            if entry is None:
                return

            self._step_counter += 1
            entry.last_access_step = self._step_counter

            for block_id in entry.block_ids:
                self._blocks[block_id].mark_accessed(self._step_counter)

            if new_tokens > 0:
                current_capacity = len(entry.block_ids) * self.block_size
                new_total = entry.num_tokens + new_tokens
                if new_total > current_capacity:
                    additional_needed = math.ceil((new_total - current_capacity) / self.block_size)
                    for _ in range(additional_needed):
                        if self._free_blocks:
                            block_id = self._free_blocks.pop(0)
                            block = self._blocks[block_id]
                            block.status = CacheStatus.OCCUPIED
                            block.ref_count = 1
                            block.mark_accessed(self._step_counter)
                            entry.block_ids.append(block_id)

                entry.num_tokens = new_total

    def get_cache_tensors(self, sequence_id: str) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Get the key and value cache tensors for a specific sequence."""
        entry = self._entries.get(sequence_id)
        if entry is None:
            return None

        if self._key_cache is None or self._value_cache is None:
            return None

        key_slices = []
        value_slices = []
        for layer_idx in range(self.num_layers):
            key_layer = self._key_cache[layer_idx][entry.block_ids]
            value_layer = self._value_cache[layer_idx][entry.block_ids]
            key_slices.append(key_layer)
            value_slices.append(value_layer)

        return key_slices, value_slices

    def write_to_cache(
        self,
        sequence_id: str,
        layer_idx: int,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        slot_offset: int = 0,
    ) -> None:
        """Write key/value tensors into the cache for a specific layer."""
        entry = self._entries.get(sequence_id)
        if entry is None:
            return

        block_idx = slot_offset // self.block_size
        offset_in_block = slot_offset % self.block_size

        if block_idx < len(entry.block_ids):
            physical_block = entry.block_ids[block_idx]
            if self._key_cache is not None and self._value_cache is not None:
                write_len = min(key_tensor.shape[0], self.block_size - offset_in_block)
                self._key_cache[layer_idx][physical_block, offset_in_block:offset_in_block + write_len] = key_tensor[:write_len]
                self._value_cache[layer_idx][physical_block, offset_in_block:offset_in_block + write_len] = value_tensor[:write_len]

    def clear(self) -> None:
        """Clear all cache entries and free all blocks."""
        with self._lock:
            for block_id in range(self.num_blocks):
                self._blocks[block_id].status = CacheStatus.FREE
                self._blocks[block_id].ref_count = 0
                self._blocks[block_id].key_tensor = None
                self._blocks[block_id].value_tensor = None

            self._entries.clear()
            self._free_blocks = list(range(self.num_blocks))
            self._step_counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.num_used_blocks,
            "utilization": self.utilization,
            "num_sequences": len(self._entries),
            "memory_used_mb": self.get_memory_usage_mb(),
            "memory_total_mb": self.get_total_memory_mb(),
            "block_size": self.block_size,
        }

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if there are enough free blocks for the given number of tokens."""
        needed = math.ceil(num_tokens / self.block_size)
        return needed <= self.num_free_blocks

    def resize(self, new_num_blocks: int) -> None:
        """Resize the cache to a new number of blocks. Frees all existing entries."""
        self.clear()
        self.num_blocks = new_num_blocks
        self._blocks = {i: CacheBlock(block_id=i, device=self.device) for i in range(new_num_blocks)}
        self._free_blocks = list(range(new_num_blocks))
        self._initialize_cache_tensors()


def estimate_cache_size(
    max_seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Estimate KV cache size in bytes for given model parameters."""
    bytes_per_element = dtype_bytes
    per_token_per_layer = 2 * num_heads * head_dim * bytes_per_element  # key + value
    total = max_seq_len * num_layers * per_token_per_layer * batch_size
    return total


def compute_optimal_num_blocks(
    available_memory_bytes: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Compute the maximum number of cache blocks that fit in available memory."""
    per_block_bytes = 2 * block_size * num_heads * head_dim * dtype_bytes * num_layers
    if per_block_bytes == 0:
        return 0
    return available_memory_bytes // per_block_bytes
