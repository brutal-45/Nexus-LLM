# Copyright (c) 2024 Nexus LLM Contributors
# SPDX-License-Identifier: Apache-2.0
"""Distributed collective operations module.

Provides a comprehensive wrapper around torch.distributed collective operations
with automatic fallbacks, gradient-supporting custom collectives, communication
overlap with computation, and communication optimization including tensor
compression and batched operations.
"""

from __future__ import annotations

import abc
import logging
import math
import os
import struct
import threading
import time
import warnings
import weakref
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.autograd import Function
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)


# ==============================================================================
# Enums and Types
# ==============================================================================


class CommunicationCompressionType(Enum):
    """Types of communication compression."""
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"
    POW2 = "power_of_two"
    TOPK = "topk"


class ReduceOpType(Enum):
    """Reduce operations for collective ops."""
    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    BAND = "band"
    BOR = "bor"
    BXOR = "bxor"
    AVG = "avg"


@dataclass
class CollectiveOpStats:
    """Statistics for collective operations."""
    op_name: str = ""
    call_count: int = 0
    total_time_ms: float = 0.0
    total_bytes: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    errors: int = 0


@dataclass
class CommOverlapConfig:
    """Configuration for communication-computation overlap."""
    enabled: bool = True
    num_streams: int = 2
    chunk_size: int = 8192
    use_cuda_events: bool = True
    pipeline_depth: int = 2


# ==============================================================================
# Collective Ops
# ==============================================================================


class CollectiveOps:
    """Wrapper around torch.distributed collective operations with fallbacks.

    Provides a unified interface for all collective operations including
    broadcast, all_reduce, reduce, all_gather, reduce_scatter, scatter,
    gather, barrier, and point-to-point operations. Supports GPU (NCCL)
    and CPU (Gloo) backends with automatic fallback.

    Attributes:
        group: The process group for collective operations.
        rank: Rank of this process.
        world_size: World size of the process group.
        device: Device for tensor operations.
        _stats: Per-operation statistics.
        _lock: Thread lock.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        enable_profiling: bool = False,
        default_timeout: float = 300.0,
    ):
        self._group = group
        self._rank = dist.get_rank(group) if dist.is_initialized() and group is not None else 0
        self._world_size = dist.get_world_size(group) if dist.is_initialized() and group is not None else 1
        self._device = device or torch.device("cpu")
        self._stats: Dict[str, CollectiveOpStats] = defaultdict(CollectiveOpStats)
        self._lock = threading.Lock()
        self._enable_profiling = enable_profiling
        self._default_timeout = default_timeout
        self._logger = logging.getLogger(f"{__name__}.CollectiveOps")

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def group(self) -> Optional[dist.ProcessGroup]:
        return self._group

    @staticmethod
    def _get_reduce_op(op: Union[str, ReduceOp, ReduceOpType]) -> ReduceOp:
        """Convert string or enum to torch ReduceOp."""
        if isinstance(op, ReduceOp):
            return op
        if isinstance(op, ReduceOpType):
            mapping = {
                ReduceOpType.SUM: ReduceOp.SUM,
                ReduceOpType.PRODUCT: ReduceOp.PRODUCT,
                ReduceOpType.MIN: ReduceOp.MIN,
                ReduceOpType.MAX: ReduceOp.MAX,
                ReduceOpType.BAND: ReduceOp.BAND,
                ReduceOpType.BOR: ReduceOp.BOR,
                ReduceOpType.BXOR: ReduceOp.BXOR,
                ReduceOpType.AVG: ReduceOp.SUM,
            }
            if op not in mapping:
                raise ValueError(f"Unsupported reduce operation: {op}")
            return mapping[op]
        if isinstance(op, str):
            op_lower = op.lower()
            mapping = {
                "sum": ReduceOp.SUM,
                "product": ReduceOp.PRODUCT,
                "prod": ReduceOp.PRODUCT,
                "min": ReduceOp.MIN,
                "max": ReduceOp.MAX,
                "avg": ReduceOp.SUM,
                "mean": ReduceOp.SUM,
                "band": ReduceOp.BAND,
                "bor": ReduceOp.BOR,
                "bxor": ReduceOp.BXOR,
            }
            if op_lower not in mapping:
                raise ValueError(f"Unsupported reduce operation: {op}")
            return mapping[op_lower]
        raise TypeError(f"Invalid reduce op type: {type(op)}")

    def _record_op(
        self, op_name: str, elapsed_ms: float, bytes_transferred: float = 0.0
    ) -> None:
        """Record operation statistics."""
        if not self._enable_profiling:
            return
        with self._lock:
            stats = self._stats[op_name]
            stats.op_name = op_name
            stats.call_count += 1
            stats.total_time_ms += elapsed_ms
            stats.total_bytes += bytes_transferred
            stats.avg_time_ms = stats.total_time_ms / stats.call_count
            stats.min_time_ms = min(stats.min_time_ms, elapsed_ms)
            stats.max_time_ms = max(stats.max_time_ms, elapsed_ms)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all operation statistics."""
        with self._lock:
            return {
                name: {
                    "op_name": s.op_name,
                    "call_count": s.call_count,
                    "total_time_ms": round(s.total_time_ms, 3),
                    "total_bytes": round(s.total_bytes, 1),
                    "avg_time_ms": round(s.avg_time_ms, 3),
                    "min_time_ms": round(s.min_time_ms, 3) if s.min_time_ms != float("inf") else 0,
                    "max_time_ms": round(s.max_time_ms, 3),
                    "errors": s.errors,
                }
                for name, s in self._stats.items()
            }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._stats.clear()

    # ==========================================================================
    # Collective Operations
    # ==========================================================================

    def broadcast(
        self,
        tensor: Tensor,
        src: int,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Broadcast a tensor from source rank to all ranks.

        Args:
            tensor: The tensor to broadcast. Modified in-place.
            src: Source rank for the broadcast.
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            work = dist.broadcast(
                tensor, src=src, group=self._group, async_op=async_op
            )
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("broadcast", elapsed, tensor.element_size() * tensor.numel())
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("broadcast", elapsed)
            with self._lock:
                self._stats["broadcast"].errors += 1
            self._logger.error(f"broadcast failed: {e}")
            raise

    def all_reduce(
        self,
        tensor: Tensor,
        op: Union[str, ReduceOp, ReduceOpType] = ReduceOp.SUM,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """All-reduce operation across all ranks.

        Reduces the tensor across all ranks using the specified operation,
        and makes the result available on all ranks.

        Args:
            tensor: The tensor to all-reduce. Modified in-place.
            op: Reduce operation (sum, min, max, prod, avg).
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            reduce_op = self._get_reduce_op(op)
            work = dist.all_reduce(
                tensor, op=reduce_op, group=self._group, async_op=async_op
            )
            if isinstance(op, (str, ReduceOpType)):
                op_str = op.value if isinstance(op, ReduceOpType) else str(op).lower()
                if op_str in ("avg", "mean"):
                    tensor.div_(self._world_size)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("all_reduce", elapsed, tensor.element_size() * tensor.numel())
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("all_reduce", elapsed)
            with self._lock:
                self._stats["all_reduce"].errors += 1
            self._logger.error(f"all_reduce failed: {e}")
            raise

    def reduce(
        self,
        tensor: Tensor,
        dst: int,
        op: Union[str, ReduceOp, ReduceOpType] = ReduceOp.SUM,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Reduce operation to a destination rank.

        Args:
            tensor: The tensor to reduce. Result is only valid on dst.
            dst: Destination rank.
            op: Reduce operation.
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            reduce_op = self._get_reduce_op(op)
            work = dist.reduce(
                tensor, dst=dst, op=reduce_op, group=self._group, async_op=async_op
            )
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("reduce", elapsed, tensor.element_size() * tensor.numel())
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("reduce", elapsed)
            with self._lock:
                self._stats["reduce"].errors += 1
            self._logger.error(f"reduce failed: {e}")
            raise

    def all_gather(
        self,
        tensor_list: List[Tensor],
        tensor: Tensor,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """All-gather tensors from all ranks.

        Each rank contributes one tensor, and after all_gather, each rank
        has a list of all tensors.

        Args:
            tensor_list: Output list of tensors (one per rank).
            tensor: Input tensor from this rank.
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            work = dist.all_gather(
                tensor_list, tensor, group=self._group, async_op=async_op
            )
            elapsed = (time.perf_counter() - start) * 1000
            total_bytes = tensor.element_size() * tensor.numel() * self._world_size
            self._record_op("all_gather", elapsed, total_bytes)
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("all_gather", elapsed)
            with self._lock:
                self._stats["all_gather"].errors += 1
            self._logger.error(f"all_gather failed: {e}")
            raise

    def reduce_scatter(
        self,
        tensor: Tensor,
        tensor_list: List[Tensor],
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Reduce-scatter tensors across ranks.

        Reduces elements across ranks and scatters the result such that
        each rank gets a different chunk of the reduced result.

        Args:
            tensor: Output tensor (this rank's chunk of the reduced result).
            tensor_list: Input list of tensors to reduce and scatter.
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            work = dist.reduce_scatter(
                tensor, tensor_list, group=self._group, async_op=async_op
            )
            elapsed = (time.perf_counter() - start) * 1000
            total_bytes = sum(
                t.element_size() * t.numel() for t in tensor_list
            )
            self._record_op("reduce_scatter", elapsed, total_bytes)
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("reduce_scatter", elapsed)
            with self._lock:
                self._stats["reduce_scatter"].errors += 1
            self._logger.error(f"reduce_scatter failed: {e}")
            raise

    def scatter(
        self,
        tensor: Tensor,
        scatter_list: Optional[List[Tensor]],
        src: int,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Scatter tensors from source rank to all ranks.

        Args:
            tensor: Output tensor (receives this rank's chunk).
            scatter_list: List of tensors to scatter (only used on src rank).
            src: Source rank.
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            work = dist.scatter(
                tensor, scatter_list=scatter_list, src=src,
                group=self._group, async_op=async_op,
            )
            elapsed = (time.perf_counter() - start) * 1000
            bytes_xfer = tensor.element_size() * tensor.numel()
            self._record_op("scatter", elapsed, bytes_xfer)
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("scatter", elapsed)
            with self._lock:
                self._stats["scatter"].errors += 1
            self._logger.error(f"scatter failed: {e}")
            raise

    def gather(
        self,
        tensor: Tensor,
        gather_list: Optional[List[Tensor]],
        dst: int,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Gather tensors from all ranks to destination rank.

        Args:
            tensor: Input tensor from this rank.
            gather_list: Output list (only valid on dst rank).
            dst: Destination rank.
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            work = dist.gather(
                tensor, gather_list=gather_list, dst=dst,
                group=self._group, async_op=async_op,
            )
            elapsed = (time.perf_counter() - start) * 1000
            bytes_xfer = tensor.element_size() * tensor.numel() * self._world_size
            self._record_op("gather", elapsed, bytes_xfer)
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("gather", elapsed)
            with self._lock:
                self._stats["gather"].errors += 1
            self._logger.error(f"gather failed: {e}")
            raise

    def barrier(self, async_op: bool = False) -> Optional[dist.Work]:
        """Synchronization barrier.

        Blocks until all ranks have reached this point.

        Args:
            async_op: Whether to return an asynchronous work handle.

        Returns:
            Optional async work handle if async_op=True.
        """
        start = time.perf_counter()
        try:
            work = dist.barrier(group=self._group, async_op=async_op)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("barrier", elapsed)
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("barrier", elapsed)
            self._logger.error(f"barrier failed: {e}")
            raise

    # ==========================================================================
    # Point-to-Point Operations
    # ==========================================================================

    def send(self, tensor: Tensor, dst: int, tag: int = 0) -> None:
        """Send a tensor to destination rank (blocking).

        Args:
            tensor: Tensor to send.
            dst: Destination rank.
            tag: Communication tag.
        """
        start = time.perf_counter()
        try:
            dist.send(tensor, dst=dst, group=self._group, tag=tag)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("send", elapsed, tensor.element_size() * tensor.numel())
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("send", elapsed)
            self._logger.error(f"send to rank {dst} failed: {e}")
            raise

    def recv(self, tensor: Tensor, src: int, tag: int = 0) -> None:
        """Receive a tensor from source rank (blocking).

        Args:
            tensor: Tensor to receive into (modified in-place).
            src: Source rank.
            tag: Communication tag.
        """
        start = time.perf_counter()
        try:
            dist.recv(tensor, src=src, group=self._group, tag=tag)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("recv", elapsed, tensor.element_size() * tensor.numel())
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("recv", elapsed)
            self._logger.error(f"recv from rank {src} failed: {e}")
            raise

    def isend(self, tensor: Tensor, dst: int, tag: int = 0) -> dist.Work:
        """Non-blocking send.

        Args:
            tensor: Tensor to send.
            dst: Destination rank.
            tag: Communication tag.

        Returns:
            Work handle for the operation.
        """
        start = time.perf_counter()
        try:
            work = dist.isend(tensor, dst=dst, group=self._group, tag=tag)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("isend", elapsed, tensor.element_size() * tensor.numel())
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("isend", elapsed)
            self._logger.error(f"isend to rank {dst} failed: {e}")
            raise

    def irecv(self, tensor: Tensor, src: int, tag: int = 0) -> dist.Work:
        """Non-blocking receive.

        Args:
            tensor: Tensor to receive into.
            src: Source rank.
            tag: Communication tag.

        Returns:
            Work handle for the operation.
        """
        start = time.perf_counter()
        try:
            work = dist.irecv(tensor, src=src, group=self._group, tag=tag)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("irecv", elapsed, tensor.element_size() * tensor.numel())
            return work
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("irecv", elapsed)
            self._logger.error(f"irecv from rank {src} failed: {e}")
            raise

    # ==========================================================================
    # Higher-Level Operations
    # ==========================================================================

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast a Python object from source rank.

        Uses pickle for serialization.

        Args:
            obj: Object to broadcast (only meaningful on src rank).
            src: Source rank.

        Returns:
            The broadcasted object.
        """
        if not dist.is_initialized():
            return obj
        start = time.perf_counter()
        try:
            result = [obj]
            dist.broadcast_object_list(result, src=src, group=self._group)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("broadcast_object", elapsed)
            return result[0]
        except Exception as e:
            self._logger.error(f"broadcast_object failed: {e}")
            raise

    def all_gather_object(self, obj: Any) -> List[Any]:
        """Gather Python objects from all ranks.

        Args:
            obj: Object from this rank.

        Returns:
            List of objects from all ranks.
        """
        if not dist.is_initialized():
            return [obj]
        start = time.perf_counter()
        try:
            result = [None] * self._world_size
            dist.all_gather_object(result, obj, group=self._group)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("all_gather_object", elapsed)
            return result
        except Exception as e:
            self._logger.error(f"all_gather_object failed: {e}")
            raise

    def scatter_object(self, scatter_list: Optional[List[Any]], src: int = 0) -> Any:
        """Scatter Python objects from source rank.

        Args:
            scatter_list: List of objects to scatter (only on src rank).
            src: Source rank.

        Returns:
            The object for this rank.
        """
        if not dist.is_initialized():
            return scatter_list[0] if scatter_list else None
        start = time.perf_counter()
        try:
            result = [None]
            dist.scatter_object_list(result, scatter_list, src=src, group=self._group)
            elapsed = (time.perf_counter() - start) * 1000
            self._record_op("scatter_object", elapsed)
            return result[0]
        except Exception as e:
            self._logger.error(f"scatter_object failed: {e}")
            raise

    def all_reduce_list(
        self,
        tensor_list: List[Tensor],
        op: Union[str, ReduceOp, ReduceOpType] = ReduceOp.SUM,
    ) -> List[Tensor]:
        """Perform all-reduce on a list of tensors.

        Args:
            tensor_list: List of tensors to reduce.
            op: Reduce operation.

        Returns:
            The list of all-reduced tensors (modified in-place).
        """
        works = []
        for tensor in tensor_list:
            work = self.all_reduce(tensor, op=op, async_op=True)
            if work is not None:
                works.append(work)
        for work in works:
            work.wait()
        return tensor_list

    def reduce_scatter_tensor(
        self,
        output: Tensor,
        input_tensor: Tensor,
        op: Union[str, ReduceOp, ReduceOpType] = ReduceOp.SUM,
    ) -> Tensor:
        """Reduce-scatter on a single flat tensor.

        Splits the input into chunks, one per rank, reduces each chunk,
        and gives each rank its chunk of the result.

        Args:
            output: Output tensor (this rank's chunk).
            input_tensor: Full input tensor.
            op: Reduce operation.

        Returns:
            The output tensor.
        """
        chunks = list(input_tensor.chunk(self._world_size, dim=0))
        chunks_copied = [c.clone().contiguous() for c in chunks]
        return self.reduce_scatter(output, chunks_copied)

    def all_gather_into_tensor(
        self,
        output: Tensor,
        input_tensor: Tensor,
    ) -> Tensor:
        """All-gather into a single concatenated tensor.

        Args:
            output: Output tensor (will be overwritten with gathered data).
            input_tensor: Input tensor from this rank.

        Returns:
            The output tensor with data from all ranks concatenated.
        """
        tensor_list = list(output.chunk(self._world_size, dim=0))
        tensor_list_copied = [c.clone().contiguous() for c in tensor_list]
        self.all_gather(tensor_list_copied, input_tensor)
        for i, chunk in enumerate(tensor_list_copied):
            output[i * input_tensor.shape[0]:(i + 1) * input_tensor.shape[0]].copy_(chunk)
        return output


# ==============================================================================
# AllGatherWithGradient
# ==============================================================================


class _AllGatherWithGradientFn(Function):
    """Autograd function for all-gather with gradient support.

    Forward: all-gather tensors from all ranks.
    Backward: reduce-scatter gradients back to their source ranks.
    """

    @staticmethod
    def forward(
        ctx, tensor: Tensor, group: Optional[dist.ProcessGroup]
    ) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)

        tensor_list = [
            torch.zeros_like(tensor) for _ in range(ctx.world_size)
        ]
        dist.all_gather(tensor_list, tensor, group=group)

        gathered = torch.cat(tensor_list, dim=0).contiguous()
        return gathered

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        grad_input = torch.zeros_like(grad_output)
        chunk_size = grad_output.shape[0] // ctx.world_size

        chunks = list(grad_output.chunk(ctx.world_size, dim=0))
        chunks = [c.contiguous() for c in chunks]
        dist.reduce_scatter(grad_input, chunks, group=ctx.group)

        return grad_input, None


def all_gather_with_gradient(
    tensor: Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """All-gather with gradient flow.

    In the forward pass, tensors are gathered from all ranks and concatenated.
    In the backward pass, gradients are reduce-scattered back to their source.

    Args:
        tensor: Input tensor from this rank.
        group: Process group for the operation.

    Returns:
        Gathered tensor from all ranks (concatenated along dim 0).
    """
    if not dist.is_initialized():
        return tensor
    return _AllGatherWithGradientFn.apply(tensor, group)


class AllGatherWithGradient(nn.Module):
    """Module wrapper for all-gather with gradient support.

    Args:
        group: Process group for the operation.
        dim: Dimension along which to gather.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        dim: int = 0,
    ):
        super().__init__()
        self.group = group
        self.dim = dim

    def forward(self, tensor: Tensor) -> Tensor:
        """All-gather with gradient support along specified dim.

        If dim != 0, transposes to gather along dim 0, then transposes back.
        """
        if self.dim != 0:
            tensor = tensor.transpose(0, self.dim)
            result = all_gather_with_gradient(tensor, self.group)
            result = result.transpose(0, self.dim)
            return result
        return all_gather_with_gradient(tensor, self.group)


# ==============================================================================
# ReduceScatterWithGradient
# ==============================================================================


class _ReduceScatterWithGradientFn(Function):
    """Autograd function for reduce-scatter with gradient support.

    Forward: reduce-scatter input tensor.
    Backward: all-gather gradients from all ranks.
    """

    @staticmethod
    def forward(
        ctx, tensor: Tensor, group: Optional[dist.ProcessGroup]
    ) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        ctx.input_shape = tensor.shape

        output = torch.zeros(
            tensor.shape[0] // ctx.world_size,
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        chunks = list(tensor.chunk(ctx.world_size, dim=0))
        chunks = [c.contiguous() for c in chunks]
        dist.reduce_scatter(output, chunks, group=group)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        grad_input_list = [torch.zeros_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(grad_input_list, grad_output, group=ctx.group)
        grad_input = torch.cat(grad_input_list, dim=0).contiguous()
        return grad_input, None


def reduce_scatter_with_gradient(
    tensor: Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """Reduce-scatter with gradient flow.

    In the forward pass, the input tensor is reduce-scattered.
    In the backward pass, gradients are all-gathered.

    Args:
        tensor: Input tensor to reduce-scatter.
        group: Process group for the operation.

    Returns:
        This rank's chunk of the reduced result.
    """
    if not dist.is_initialized():
        return tensor
    return _ReduceScatterWithGradientFn.apply(tensor, group)


class ReduceScatterWithGradient(nn.Module):
    """Module wrapper for reduce-scatter with gradient support.

    Args:
        group: Process group for the operation.
        dim: Dimension along which to reduce-scatter.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        dim: int = 0,
    ):
        super().__init__()
        self.group = group
        self.dim = dim

    def forward(self, tensor: Tensor) -> Tensor:
        """Reduce-scatter with gradient support along specified dim."""
        if self.dim != 0:
            tensor = tensor.transpose(0, self.dim)
            result = reduce_scatter_with_gradient(tensor, self.group)
            result = result.transpose(0, self.dim)
            return result
        return reduce_scatter_with_gradient(tensor, self.group)


# ==============================================================================
# SequenceParallelGather
# ==============================================================================


class _SeqParallelGatherForward(Function):
    """Forward all-gather for sequence parallelism (no gradient)."""

    @staticmethod
    def forward(ctx, tensor: Tensor, group: dist.ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.world_size = dist.get_world_size(group)
        tensor_list = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list, tensor, group=group)
        return torch.cat(tensor_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        chunk_size = grad_output.shape[0] // ctx.world_size
        rank = dist.get_rank(ctx.group)
        grad_input = grad_output[rank * chunk_size:(rank + 1) * chunk_size].contiguous()
        return grad_input, None


class _SeqParallelScatterReduce(Function):
    """Reduce-scatter for sequence parallelism (no gradient in backward)."""

    @staticmethod
    def forward(ctx, tensor: Tensor, group: dist.ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.world_size = dist.get_world_size(group)
        chunks = list(tensor.chunk(ctx.world_size, dim=0))
        chunks = [c.contiguous() for c in chunks]
        output = torch.zeros_like(chunks[0])
        dist.reduce_scatter(output, chunks, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        tensor_list = [torch.zeros_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list, grad_output, group=ctx.group)
        grad_input = torch.cat(tensor_list, dim=0).contiguous()
        return grad_input, None


class SequenceParallelGather(nn.Module):
    """Sequence dimension gathering for sequence parallelism.

    In sequence parallelism, the sequence dimension is partitioned across
    GPUs. This module provides gather (forward) and scatter (backward)
    operations to reconstruct the full sequence when needed.

    Usage:
        gather = SequenceParallelGather(tp_group)
        full_seq = gather(partial_seq)  # Gather across TP group

        scatter = SequenceParallelGather(tp_group, mode="scatter")
        partial_seq = scatter(full_seq)  # Scatter across TP group
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        dim: int = 0,
        mode: str = "gather",
    ):
        """
        Args:
            group: Process group (typically the TP group).
            dim: Dimension along which to gather/scatter.
            mode: 'gather' for forward gather (backward scatter),
                  'scatter' for forward scatter (backward gather).
        """
        super().__init__()
        self.group = group
        self.dim = dim
        self.mode = mode

    def forward(self, tensor: Tensor) -> Tensor:
        """Apply gather or scatter along the configured dimension.

        Args:
            tensor: Input tensor.

        Returns:
            Gathered or scattered tensor.
        """
        if not dist.is_initialized():
            return tensor
        if self.dim != 0:
            tensor = tensor.transpose(0, self.dim)

        if self.mode == "gather":
            result = _SeqParallelGatherForward.apply(tensor, self.group)
        elif self.mode == "scatter":
            result = _SeqParallelScatterReduce.apply(tensor, self.group)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.dim != 0:
            result = result.transpose(0, self.dim)
        return result


# ==============================================================================
# OverlapCommComputation
# ==============================================================================


class OverlapCommComputation:
    """Overlap communication with computation using CUDA streams.

    Splits tensors into chunks and processes communication for one chunk
    while computing on another, effectively hiding communication latency.

    Attributes:
        _group: Process group for communication.
        _num_chunks: Number of chunks for pipelining.
        _streams: CUDA streams for overlap.
        _events: CUDA events for synchronization.
        _device: Device for operations.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        num_chunks: int = 4,
        device: Optional[torch.device] = None,
    ):
        self._group = group
        self._num_chunks = num_chunks
        self._device = device or (
            torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        )
        self._streams: List[torch.cuda.Stream] = []
        self._events: List[List[torch.cuda.Event]] = []

        if self._device.type == "cuda":
            for _ in range(self._num_chunks):
                stream = torch.cuda.Stream(device=self._device)
                self._streams.append(stream)
                chunk_events = []
                for _ in range(2):
                    chunk_events.append(torch.cuda.Event(enable_timing=True))
                self._events.append(chunk_events)

        self._logger = logging.getLogger(f"{__name__}.OverlapCommComputation")

    def overlap_all_reduce(
        self,
        tensor: Tensor,
        compute_fn: Optional[Callable[[Tensor], Tensor]] = None,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> Tensor:
        """All-reduce with computation overlap.

        Splits the tensor into chunks, starts communication for each chunk,
        and optionally applies a compute function while communication
        is in progress.

        Args:
            tensor: Tensor to all-reduce.
            compute_fn: Optional function to apply between chunks.
            op: Reduce operation.

        Returns:
            The all-reduced tensor.
        """
        if not dist.is_initialized() or self._num_chunks <= 1:
            dist.all_reduce(tensor, op=op, group=self._group)
            return tensor

        if self._device.type != "cuda" or not torch.cuda.is_available():
            dist.all_reduce(tensor, op=op, group=self._group)
            return tensor

        chunks = list(tensor.chunk(self._num_chunks))
        output_chunks = [c.clone() for c in chunks]

        default_stream = torch.cuda.current_stream(self._device)

        for i, (chunk, output_chunk) in enumerate(zip(chunks, output_chunks)):
            stream = self._streams[i % len(self._streams)]

            with torch.cuda.stream(stream):
                comm_input = chunk.clone()
                dist.all_reduce(comm_input, op=op, group=self._group)
                output_chunk.copy_(comm_input)

            if compute_fn is not None and i < len(chunks) - 1:
                compute_fn(chunks[i + 1])

        for i in range(len(chunks)):
            self._events[i][0].record(self._streams[i % len(self._streams)])
            self._events[i][0].synchronize()

        result = torch.cat(output_chunks, dim=0)
        return result

    def overlap_all_gather(
        self,
        tensor: Tensor,
        compute_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        """All-gather with computation overlap.

        Args:
            tensor: Input tensor from this rank.
            compute_fn: Optional function to apply during communication.

        Returns:
            Gathered tensor from all ranks.
        """
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size(self._group)

        if self._device.type != "cuda" or self._num_chunks <= 1 or world_size <= 1:
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor, group=self._group)
            return torch.cat(tensor_list, dim=0)

        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

        for i in range(world_size):
            stream = self._streams[i % len(self._streams)]
            with torch.cuda.stream(stream):
                temp_list = [torch.zeros_like(tensor) for _ in range(world_size)]
                dist.all_gather(temp_list, tensor, group=self._group)
                tensor_list[i].copy_(temp_list[i])

            if compute_fn is not None and i < world_size - 1:
                compute_fn(tensor_list[i])

        for stream in self._streams[:world_size]:
            torch.cuda.current_stream(self._device).wait_stream(stream)

        return torch.cat(tensor_list, dim=0)

    def overlap_reduce_scatter(
        self,
        tensor: Tensor,
        compute_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        """Reduce-scatter with computation overlap.

        Args:
            tensor: Input tensor to reduce-scatter.
            compute_fn: Optional function to apply during communication.

        Returns:
            This rank's chunk of the reduced result.
        """
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size(self._group)
        output_size = tensor.shape[0] // world_size

        if self._device.type != "cuda" or self._num_chunks <= 1:
            chunks = list(tensor.chunk(world_size, dim=0))
            chunks = [c.contiguous() for c in chunks]
            output = torch.zeros(output_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
            dist.reduce_scatter(output, chunks, group=self._group)
            return output

        rank = dist.get_rank(self._group)
        chunks = list(tensor.chunk(world_size, dim=0))
        chunks = [c.contiguous() for c in chunks]
        output = torch.zeros(output_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)

        stream = self._streams[0]
        with torch.cuda.stream(stream):
            dist.reduce_scatter(output, chunks, group=self._group)

        if compute_fn is not None:
            compute_fn(tensor)

        torch.cuda.current_stream(self._device).wait_stream(stream)
        return output

    def overlap_send_recv(
        self,
        send_tensor: Optional[Tensor],
        recv_tensor: Optional[Tensor],
        dst: int,
        src: int,
        compute_fn: Optional[Callable] = None,
    ) -> None:
        """Overlapped send and receive with computation.

        Performs send and receive simultaneously while optionally
        running a compute function.

        Args:
            send_tensor: Tensor to send (None to skip send).
            recv_tensor: Tensor to receive into (None to skip recv).
            dst: Destination rank for send.
            src: Source rank for recv.
            compute_fn: Optional compute function to overlap.
        """
        if self._device.type != "cuda":
            if send_tensor is not None:
                dist.send(send_tensor, dst=dst, group=self._group)
            if recv_tensor is not None:
                dist.recv(recv_tensor, src=src, group=self._group)
            return

        stream = self._streams[0]
        send_work = None
        recv_work = None

        with torch.cuda.stream(stream):
            if send_tensor is not None:
                send_work = dist.isend(send_tensor.contiguous(), dst=dst, group=self._group)
            if recv_tensor is not None:
                recv_work = dist.irecv(recv_tensor, src=src, group=self._group)

        if compute_fn is not None:
            compute_fn()

        if send_work is not None:
            send_work.wait()
        if recv_work is not None:
            recv_work.wait()

    def pipeline_all_reduce(
        self,
        tensor: Tensor,
        compute_fn: Callable[[Tensor], Tensor],
        num_stages: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Pipelined all-reduce with interleaved computation.

        Performs multiple stages of compute-communicate pipeline
        for maximum overlap.

        Args:
            tensor: Tensor to all-reduce.
            compute_fn: Function to apply to each chunk.
            num_stages: Number of pipeline stages.

        Returns:
            Tuple of (communication result, compute result).
        """
        stages = num_stages or self._num_chunks
        chunks = list(tensor.chunk(stages))
        comm_results = [c.clone() for c in chunks]
        compute_results = []

        for i, chunk in enumerate(chunks):
            stream_idx = i % len(self._streams)
            stream = self._streams[stream_idx]

            compute_result = compute_fn(chunk)
            compute_results.append(compute_result)

            with torch.cuda.stream(stream):
                dist.all_reduce(comm_results[i], group=self._group)

        for stream in self._streams:
            torch.cuda.current_stream(self._device).wait_stream(stream)

        comm_tensor = torch.cat(comm_results, dim=0)
        compute_tensor = torch.cat(compute_results, dim=0)
        return comm_tensor, compute_tensor


# ==============================================================================
# Communication Optimizer
# ==============================================================================


class CommunicationOptimizer:
    """Optimizes distributed communication through compression and batching.

    Provides tensor compression (FP16, INT8, INT4) to reduce communication
    volume, and batching of small operations into larger ones to amortize
    communication overhead.

    Attributes:
        _compression: Default compression type.
        _bucket_size: Maximum bucket size for batched operations.
        _pending_tensors: Tensors pending to be batched.
        _pending_works: Pending async work handles.
        _group: Process group.
        _world_size: World size.
    """

    def __init__(
        self,
        compression: CommunicationCompressionType = CommunicationCompressionType.NONE,
        bucket_size_mb: int = 25,
        group: Optional[dist.ProcessGroup] = None,
        max_pending: int = 100,
    ):
        self._compression = compression
        self._bucket_size = bucket_size_mb * 1024 * 1024
        self._pending_tensors: List[Tensor] = []
        self._pending_original_shapes: List[Tuple[int, ...]] = []
        self._pending_original_dtypes: List[torch.dtype] = []
        self._pending_works: List[dist.Work] = []
        self._group = group
        self._world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self._max_pending = max_pending
        self._stats = {
            "compress_calls": 0,
            "decompress_calls": 0,
            "bytes_saved": 0,
            "batched_ops": 0,
            "individual_ops": 0,
        }
        self._logger = logging.getLogger(f"{__name__}.CommunicationOptimizer")

    @property
    def compression(self) -> CommunicationCompressionType:
        return self._compression

    @compression.setter
    def compression(self, value: CommunicationCompressionType) -> None:
        self._compression = value

    def compress_tensor(
        self,
        tensor: Tensor,
        compression_type: Optional[CommunicationCompressionType] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compress a tensor for efficient communication.

        Args:
            tensor: Input tensor to compress.
            compression_type: Override compression type for this tensor.

        Returns:
            Tuple of (compressed_tensor, metadata) where metadata contains
            information needed for decompression.

        Raises:
            ValueError: If compression type is not supported.
        """
        ct = compression_type or self._compression
        if ct == CommunicationCompressionType.NONE:
            return tensor.clone(), {"compression": "none"}

        self._stats["compress_calls"] += 1
        original_size = tensor.element_size() * tensor.numel()

        if ct == CommunicationCompressionType.FP16:
            compressed = tensor.to(torch.float16)
            metadata = {
                "compression": "fp16",
                "original_dtype": str(tensor.dtype),
                "original_shape": tuple(tensor.shape),
            }
        elif ct == CommunicationCompressionType.BF16:
            compressed = tensor.to(torch.bfloat16)
            metadata = {
                "compression": "bf16",
                "original_dtype": str(tensor.dtype),
                "original_shape": tuple(tensor.shape),
            }
        elif ct == CommunicationCompressionType.INT8:
            if tensor.is_floating_point():
                abs_max = tensor.abs().max().clamp(min=1e-8)
                scale = abs_max / 127.0
                quantized = (tensor / scale).clamp(-128, 127).to(torch.int8)
                compressed = quantized
                metadata = {
                    "compression": "int8",
                    "original_dtype": str(tensor.dtype),
                    "original_shape": tuple(tensor.shape),
                    "scale": scale.item(),
                }
            else:
                compressed = tensor.to(torch.int8)
                metadata = {
                    "compression": "int8",
                    "original_dtype": str(tensor.dtype),
                    "original_shape": tuple(tensor.shape),
                }
        elif ct == CommunicationCompressionType.UINT8:
            if tensor.is_floating_point():
                tensor_min = tensor.min()
                tensor_max = tensor.max()
                range_val = (tensor_max - tensor_min).clamp(min=1e-8)
                normalized = (tensor - tensor_min) / range_val * 255.0
                quantized = normalized.clamp(0, 255).to(torch.uint8)
                compressed = quantized
                metadata = {
                    "compression": "uint8",
                    "original_dtype": str(tensor.dtype),
                    "original_shape": tuple(tensor.shape),
                    "min_val": tensor_min.item(),
                    "range_val": range_val.item(),
                }
            else:
                compressed = tensor.to(torch.uint8)
                metadata = {
                    "compression": "uint8",
                    "original_dtype": str(tensor.dtype),
                    "original_shape": tuple(tensor.shape),
                }
        elif ct == CommunicationCompressionType.INT4:
            if tensor.is_floating_point():
                abs_max = tensor.abs().max().clamp(min=1e-8)
                scale = abs_max / 7.0
                quantized = (tensor / scale).clamp(-8, 7).to(torch.int8)
                packed = self._pack_int4(quantized)
                compressed = packed
                metadata = {
                    "compression": "int4",
                    "original_dtype": str(tensor.dtype),
                    "original_shape": tuple(tensor.shape),
                    "scale": scale.item(),
                    "original_numel": tensor.numel(),
                }
            else:
                raise ValueError(
                    f"INT4 compression not supported for dtype {tensor.dtype}"
                )
        elif ct == CommunicationCompressionType.POW2:
            if tensor.is_floating_point():
                sign = torch.sign(tensor)
                log2 = torch.log2(tensor.abs().clamp(min=1e-30))
                mantissa_bits = torch.clamp(log2.frac() * 256, 0, 255).to(torch.uint8)
                exponent = log2.floor().to(torch.int8)
                compressed = torch.stack([mantissa_bits, exponent.to(torch.uint8)], dim=-1)
                metadata = {
                    "compression": "pow2",
                    "original_dtype": str(tensor.dtype),
                    "original_shape": tuple(tensor.shape),
                    "sign_meaningful": (sign != 0).any().item(),
                }
            else:
                raise ValueError(
                    f"Power-of-two compression not supported for dtype {tensor.dtype}"
                )
        elif ct == CommunicationCompressionType.TOPK:
            flat = tensor.reshape(-1)
            k = max(1, flat.numel() // 10)
            values, indices = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[indices] = True
            sparse_values = flat[mask].to(torch.float16)
            sparse_indices = indices.to(torch.int32)
            compressed = torch.stack([sparse_values, sparse_indices.float()], dim=-1)
            metadata = {
                "compression": "topk",
                "original_dtype": str(tensor.dtype),
                "original_shape": tuple(tensor.shape),
                "k": k,
                "original_numel": flat.numel(),
            }
        else:
            raise ValueError(f"Unsupported compression type: {ct}")

        compressed_size = compressed.element_size() * compressed.numel()
        self._stats["bytes_saved"] += original_size - compressed_size

        return compressed, metadata

    def decompress_tensor(
        self,
        compressed: Tensor,
        original_shape: Tuple[int, ...],
        original_dtype: torch.dtype,
        metadata: Dict[str, Any],
    ) -> Tensor:
        """Decompress a tensor back to its original form.

        Args:
            compressed: The compressed tensor.
            original_shape: Original tensor shape.
            original_dtype: Original tensor dtype.
            metadata: Metadata from the compression step.

        Returns:
            Decompressed tensor.
        """
        self._stats["decompress_calls"] += 1
        ct_str = metadata.get("compression", "none")

        if ct_str == "none":
            return compressed.reshape(original_shape).to(original_dtype)

        if ct_str == "fp16" or ct_str == "bf16":
            return compressed.reshape(original_shape).to(original_dtype)

        if ct_str == "int8":
            if original_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                scale = metadata.get("scale", 1.0)
                if isinstance(scale, (int, float)):
                    scale = torch.tensor(scale, dtype=original_dtype, device=compressed.device)
                decompressed = compressed.to(original_dtype) * scale
                return decompressed.reshape(original_shape)
            return compressed.reshape(original_shape).to(original_dtype)

        if ct_str == "uint8":
            if original_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                min_val = metadata.get("min_val", 0.0)
                range_val = metadata.get("range_val", 1.0)
                if isinstance(min_val, (int, float)):
                    min_val = torch.tensor(min_val, dtype=original_dtype, device=compressed.device)
                if isinstance(range_val, (int, float)):
                    range_val = torch.tensor(range_val, dtype=original_dtype, device=compressed.device)
                decompressed = compressed.to(original_dtype) / 255.0 * range_val + min_val
                return decompressed.reshape(original_shape)
            return compressed.reshape(original_shape).to(original_dtype)

        if ct_str == "int4":
            unpacked = self._unpack_int4(compressed, metadata.get("original_numel", 0))
            scale = metadata.get("scale", 1.0)
            if isinstance(scale, (int, float)):
                scale = torch.tensor(scale, dtype=original_dtype, device=compressed.device)
            decompressed = unpacked.to(original_dtype) * scale
            return decompressed.reshape(original_shape)

        if ct_str == "pow2":
            parts = compressed.reshape(-1, 2)
            mantissa = parts[:, 0].to(torch.float32) / 256.0
            exponent = parts[:, 1].to(torch.float32) - 128
            decompressed = (2.0 ** (exponent + mantissa))
            return decompressed.reshape(original_shape).to(original_dtype)

        if ct_str == "topk":
            k = metadata.get("k", 1)
            original_numel = metadata.get("original_numel", 1)
            parts = compressed.reshape(-1, 2)
            values = parts[:, 0].to(original_dtype)
            indices = parts[:, 1].to(torch.long)
            result = torch.zeros(original_numel, dtype=original_dtype, device=compressed.device)
            result.scatter_(0, indices, values)
            return result.reshape(original_shape)

        warnings.warn(f"Unknown compression type: {ct_str}, returning as-is")
        return compressed.reshape(original_shape).to(original_dtype)

    @staticmethod
    def _pack_int4(tensor: Tensor) -> Tensor:
        """Pack int8 tensor into int4 (2 values per byte)."""
        flat = tensor.reshape(-1)
        if flat.numel() % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])
        even = flat[0::2] & 0x0F
        odd = (flat[1::2] & 0x0F) << 4
        packed = (even | odd).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_int4(packed: Tensor, original_numel: int) -> Tensor:
        """Unpack int4 (2 values per byte) to int8."""
        flat = packed.reshape(-1).to(torch.int8)
        low = (flat & 0x0F).to(torch.int8)
        high = ((flat >> 4) & 0x0F).to(torch.int8)
        result = torch.zeros(flat.numel() * 2, dtype=torch.int8, device=packed.device)
        result[0::2] = low
        result[1::2] = high
        if original_numel > 0 and original_numel < result.numel():
            result = result[:original_numel]
        return result

    def batched_all_reduce(
        self,
        tensors: List[Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> List[Tensor]:
        """Batch multiple all-reduce operations into fewer larger ones.

        Concatenates tensors that fit within a bucket, performs all-reduce
        on the concatenated tensor, then splits the result back.

        Args:
            tensors: List of tensors to all-reduce.
            op: Reduce operation.

        Returns:
            List of all-reduced tensors.
        """
        if not dist.is_initialized():
            return tensors

        if not tensors:
            return []

        if len(tensors) == 1:
            dist.all_reduce(tensors[0], op=op, group=self._group)
            return tensors

        buckets = self._bucket_tensors(tensors)

        if len(buckets) == 1:
            bucket_tensor, bucket_info = buckets[0]
            dist.all_reduce(bucket_tensor, op=op, group=self._group)
            self._stats["batched_ops"] += 1
            return self._unbucket_tensor(bucket_tensor, bucket_info)
        else:
            self._stats["individual_ops"] += len(tensors)
            for tensor in tensors:
                dist.all_reduce(tensor, op=op, group=self._group)
            return tensors

    def batched_all_gather(
        self,
        tensors: List[Tensor],
    ) -> List[Tensor]:
        """Batch multiple all-gather operations.

        Args:
            tensors: List of tensors to all-gather (one per rank).

        Returns:
            List of all-gathered tensors.
        """
        if not dist.is_initialized() or not tensors:
            return tensors

        if len(tensors) == 1:
            tensor_list = [torch.zeros_like(tensors[0]) for _ in range(self._world_size)]
            dist.all_gather(tensor_list, tensors[0], group=self._group)
            return [torch.cat(tensor_list, dim=0)]

        results = []
        for tensor in tensors:
            tensor_list = [torch.zeros_like(tensor) for _ in range(self._world_size)]
            dist.all_gather(tensor_list, tensor, group=self._group)
            results.append(torch.cat(tensor_list, dim=0))
        return results

    def _bucket_tensors(
        self, tensors: List[Tensor]
    ) -> List[Tuple[Tensor, List[Dict[str, Any]]]]:
        """Bucket tensors for batched communication.

        Groups tensors that fit within the bucket size together.

        Args:
            tensors: List of tensors to bucket.

        Returns:
            List of (bucketed_tensor, bucket_info) tuples.
        """
        buckets = []
        current_bucket_parts = []
        current_size = 0
        current_info = []

        for tensor in tensors:
            tensor_bytes = tensor.element_size() * tensor.numel()

            if current_size + tensor_bytes <= self._bucket_size and current_bucket_parts:
                current_bucket_parts.append(tensor.detach().flatten().contiguous())
                current_info.append({
                    "shape": tuple(tensor.shape),
                    "dtype": tensor.dtype,
                    "numel": tensor.numel(),
                    "offset": current_size,
                    "bytes": tensor_bytes,
                })
                current_size += tensor_bytes
            else:
                if current_bucket_parts:
                    bucket_tensor = torch.cat(current_bucket_parts, dim=0)
                    buckets.append((bucket_tensor, current_info))

                current_bucket_parts = [tensor.detach().flatten().contiguous()]
                current_info = [{
                    "shape": tuple(tensor.shape),
                    "dtype": tensor.dtype,
                    "numel": tensor.numel(),
                    "offset": 0,
                    "bytes": tensor_bytes,
                }]
                current_size = tensor_bytes

        if current_bucket_parts:
            bucket_tensor = torch.cat(current_bucket_parts, dim=0)
            buckets.append((bucket_tensor, current_info))

        return buckets

    def _unbucket_tensor(
        self,
        bucket_tensor: Tensor,
        bucket_info: List[Dict[str, Any]],
    ) -> List[Tensor]:
        """Split a bucketed tensor back into individual tensors.

        Args:
            bucket_tensor: The bucketed tensor.
            bucket_info: Information about each tensor in the bucket.

        Returns:
            List of individual tensors.
        """
        results = []
        for info in bucket_info:
            numel = info["numel"]
            offset = info["offset"] // bucket_tensor.element_size()
            shape = info["shape"]
            dtype = info["dtype"]
            flat_tensor = bucket_tensor[offset:offset + numel]
            results.append(flat_tensor.reshape(shape).to(dtype))
        return results

    def add_to_batch(self, tensor: Tensor) -> bool:
        """Add a tensor to the pending batch.

        Args:
            tensor: Tensor to add.

        Returns:
            True if the batch was flushed, False if the tensor was queued.
        """
        self._pending_tensors.append(tensor.detach())
        self._pending_original_shapes.append(tuple(tensor.shape))
        self._pending_original_dtypes.append(tensor.dtype)

        total_size = sum(t.element_size() * t.numel() for t in self._pending_tensors)
        if total_size >= self._bucket_size or len(self._pending_tensors) >= self._max_pending:
            self.flush_batch()
            return True
        return False

    def flush_batch(self) -> List[Tensor]:
        """Flush all pending tensors as a batched all-reduce.

        Returns:
            List of all-reduced tensors.
        """
        if not self._pending_tensors:
            return []

        tensors = self._pending_tensors
        original_shapes = self._pending_original_shapes
        original_dtypes = self._pending_original_dtypes

        self._pending_tensors = []
        self._pending_original_shapes = []
        self._pending_original_dtypes = []

        flat_parts = [t.flatten().contiguous() for t in tensors]
        bucket = torch.cat(flat_parts, dim=0)

        dist.all_reduce(bucket, op=dist.ReduceOp.SUM, group=self._group)

        offsets = [0]
        for t in flat_parts:
            offsets.append(offsets[-1] + t.numel())

        results = []
        for i, t in enumerate(tensors):
            flat_result = bucket[offsets[i]:offsets[i + 1]]
            results.append(flat_result.reshape(original_shapes[i]).to(original_dtypes[i]))

        self._stats["batched_ops"] += 1
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get compression and batching statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "compress_calls": 0,
            "decompress_calls": 0,
            "bytes_saved": 0,
            "batched_ops": 0,
            "individual_ops": 0,
        }


# ==============================================================================
# Utility Functions
# ==============================================================================


def reduce_scatter_fp32(
    tensor: Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Reduce-scatter in FP32 precision for numerical stability.

    Converts FP16/BF16 tensors to FP32 before reduce-scatter,
    then converts back.

    Args:
        tensor: Input tensor (modified in-place).
        group: Process group.
    """
    if not dist.is_initialized():
        return

    original_dtype = tensor.dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        fp32_tensor = tensor.float()
        chunks = list(fp32_tensor.chunk(dist.get_world_size(group), dim=0))
        chunks = [c.contiguous() for c in chunks]
        output = torch.zeros_like(chunks[0])
        dist.reduce_scatter(output, chunks, group=group)
        tensor.copy_(output.to(original_dtype))
    else:
        chunks = list(tensor.chunk(dist.get_world_size(group), dim=0))
        chunks = [c.contiguous() for c in chunks]
        output = torch.zeros_like(chunks[0])
        dist.reduce_scatter(output, chunks, group=group)
        tensor.copy_(output)


def all_gather_fp32(
    tensor: Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """All-gather in FP32 precision.

    Args:
        tensor: Input tensor.
        group: Process group.

    Returns:
        Gathered tensor in the original dtype.
    """
    if not dist.is_initialized():
        return tensor

    original_dtype = tensor.dtype
    fp32_tensor = tensor.float() if original_dtype in (torch.float16, torch.bfloat16) else tensor

    tensor_list = [torch.zeros_like(fp32_tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(tensor_list, fp32_tensor, group=group)

    result = torch.cat(tensor_list, dim=0)
    if original_dtype in (torch.float16, torch.bfloat16):
        result = result.to(original_dtype)
    return result


def copy_to_device(tensor: Tensor, device: torch.device) -> Tensor:
    """Copy tensor to device with proper gradient tracking."""
    return tensor.to(device)


def split_tensor_along_dim(
    tensor: Tensor,
    dim: int,
    num_chunks: int,
) -> List[Tensor]:
    """Split a tensor along a dimension into equal chunks.

    Args:
        tensor: Input tensor.
        dim: Dimension to split along.
        num_chunks: Number of chunks.

    Returns:
        List of tensors.

    Raises:
        ValueError: If the dimension is not divisible by num_chunks.
    """
    if tensor.shape[dim] % num_chunks != 0:
        raise ValueError(
            f"Dimension {dim} with size {tensor.shape[dim]} is not "
            f"divisible by {num_chunks}"
        )
    return list(tensor.chunk(num_chunks, dim=dim))


def gather_tensor_along_dim(
    tensors: List[Tensor],
    dim: int = 0,
) -> Tensor:
    """Gather tensors along a dimension.

    Args:
        tensors: List of tensors to gather.
        dim: Dimension along which to gather.

    Returns:
        Concatenated tensor.
    """
    return torch.cat(tensors, dim=dim)


def ensure_contiguous(tensor: Tensor) -> Tensor:
    """Ensure a tensor is contiguous."""
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor
