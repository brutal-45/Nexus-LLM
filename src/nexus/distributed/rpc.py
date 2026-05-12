# Copyright (c) 2024 Nexus LLM Contributors
# SPDX-License-Identifier: Apache-2.0
"""RPC framework for distributed training.

Provides a complete RPC framework for remote procedure calls between
distributed training workers. Supports synchronous calls, async calls,
fire-and-forget semantics, remote module execution, parameter server
functionality, profiling, and retry logic with exponential backoff.

Built on top of torch.distributed primitives for maximum portability.
"""

from __future__ import annotations

import abc
import asyncio
import copy
import functools
import hashlib
import io
import logging
import os
import pickle
import queue
import struct
import threading
import time
import traceback
import uuid
import warnings
import weakref
from collections import defaultdict, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
)

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ==============================================================================
# Exceptions
# ==============================================================================


class RPCError(Exception):
    """Base exception for RPC errors."""
    def __init__(self, message: str, src: str = "", dst: str = ""):
        super().__init__(message)
        self.message = message
        self.src = src
        self.dst = dst

    def __str__(self) -> str:
        if self.src and self.dst:
            return f"RPCError [{self.src} -> {self.dst}]: {self.message}"
        return f"RPCError: {self.message}"


class RPCTimeoutError(RPCError):
    """Raised when an RPC call times out."""
    def __init__(self, message: str, timeout: float = 0.0, src: str = "", dst: str = ""):
        super().__init__(message, src, dst)
        self.timeout = timeout


class RPCConnectionError(RPCError):
    """Raised when RPC connection fails."""
    pass


class RPCRemoteError(RPCError):
    """Raised when the remote side of an RPC call raises an exception."""
    def __init__(self, message: str, remote_traceback: str = "", src: str = "", dst: str = ""):
        super().__init__(message, src, dst)
        self.remote_traceback = remote_traceback


class RPCSerializationError(RPCError):
    """Raised when serialization/deserialization of RPC data fails."""
    pass


class RPCNotFoundError(RPCError):
    """Raised when a remote function or module is not found."""
    pass


# ==============================================================================
# Data Structures
# ==============================================================================


class MessageStatus(Enum):
    """Status of an RPC message."""
    PENDING = "pending"
    SENT = "sent"
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Type of RPC message."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"
    BROADCAST = "broadcast"


@dataclass
class RPCMessage:
    """An RPC message sent between workers.

    Attributes:
        id: Unique message identifier.
        msg_type: Type of the message.
        src_name: Name of the sending worker.
        dst_name: Name of the destination worker.
        method_name: Name of the remote method to call.
        args: Positional arguments for the remote method.
        kwargs: Keyword arguments for the remote method.
        status: Current status of the message.
        timestamp: Creation timestamp.
        timeout: Timeout in seconds for this message.
        requires_response: Whether this message requires a response.
        correlation_id: ID linking request to response.
        payload: Raw binary payload (for efficiency).
        metadata: Additional metadata.
        priority: Message priority (higher = more urgent).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    msg_type: MessageType = MessageType.REQUEST
    src_name: str = ""
    dst_name: str = ""
    method_name: str = ""
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: MessageStatus = MessageStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0
    requires_response: bool = True
    correlation_id: str = ""
    payload: bytes = b""
    metadata: Dict[str, str] = field(default_factory=dict)
    priority: int = 0

    def to_bytes(self) -> bytes:
        """Serialize the message to bytes."""
        try:
            data = pickle.dumps({
                "id": self.id,
                "msg_type": self.msg_type.value,
                "src_name": self.src_name,
                "dst_name": self.dst_name,
                "method_name": self.method_name,
                "args": self.args,
                "kwargs": self.kwargs,
                "status": self.status.value,
                "timestamp": self.timestamp,
                "timeout": self.timeout,
                "requires_response": self.requires_response,
                "correlation_id": self.correlation_id,
                "metadata": self.metadata,
                "priority": self.priority,
            }, protocol=pickle.HIGHEST_PROTOCOL)
            header = struct.pack("!QI", len(data), hash(self.id) % (2**32))
            return header + data
        except (pickle.PicklingError, TypeError) as e:
            raise RPCSerializationError(
                f"Failed to serialize RPCMessage: {e}"
            ) from e

    @classmethod
    def from_bytes(cls, data: bytes) -> "RPCMessage":
        """Deserialize a message from bytes."""
        try:
            header_size = struct.calcsize("!QI")
            if len(data) < header_size:
                raise ValueError(f"Data too short: {len(data)} < {header_size}")
            _, _hash = struct.unpack_from("!QI", data)
            msg_data = pickle.loads(data[header_size:])
            msg = cls(
                id=msg_data["id"],
                msg_type=MessageType(msg_data["msg_type"]),
                src_name=msg_data["src_name"],
                dst_name=msg_data["dst_name"],
                method_name=msg_data["method_name"],
                args=msg_data.get("args", ()),
                kwargs=msg_data.get("kwargs", {}),
                status=MessageStatus(msg_data.get("status", "pending")),
                timestamp=msg_data.get("timestamp", time.time()),
                timeout=msg_data.get("timeout", 30.0),
                requires_response=msg_data.get("requires_response", True),
                correlation_id=msg_data.get("correlation_id", ""),
                metadata=msg_data.get("metadata", {}),
                priority=msg_data.get("priority", 0),
            )
            return msg
        except (pickle.UnpicklingError, KeyError, ValueError) as e:
            raise RPCSerializationError(
                f"Failed to deserialize RPCMessage: {e}"
            ) from e

    def copy(self) -> "RPCMessage":
        return copy.deepcopy(self)

    @property
    def size_bytes(self) -> int:
        return len(self.to_bytes())


@dataclass
class RPCResponse:
    """Response to an RPC call.

    Attributes:
        id: Unique response identifier.
        request_id: ID of the original request.
        src_name: Name of the responding worker.
        result: The return value from the remote call.
        error: Error message if the call failed.
        traceback_str: Remote traceback if the call failed.
        status: Status of the response.
        tensors: List of tensors to include in the response.
        timestamp: Creation timestamp.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    src_name: str = ""
    result: Any = None
    error: str = ""
    traceback_str: str = ""
    status: MessageStatus = MessageStatus.COMPLETED
    tensors: List[torch.Tensor] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_message(self) -> RPCMessage:
        """Convert to an RPCMessage for transport."""
        msg = RPCMessage(
            msg_type=MessageType.RESPONSE if not self.error else MessageType.ERROR,
            src_name=self.src_name,
            method_name="",
            args=(self.result,) if not self.error else (),
            kwargs={"error": self.error, "traceback": self.traceback_str},
            status=self.status,
            correlation_id=self.request_id,
        )
        return msg

    @classmethod
    def from_message(cls, msg: RPCMessage) -> "RPCResponse":
        """Create an RPCResponse from an RPCMessage."""
        error = msg.kwargs.get("error", "") if msg.msg_type == MessageType.ERROR else ""
        traceback_str = msg.kwargs.get("traceback", "") if msg.msg_type == MessageType.ERROR else ""
        result = msg.args[0] if msg.args else None
        return cls(
            request_id=msg.correlation_id,
            src_name=msg.src_name,
            result=result,
            error=error,
            traceback_str=traceback_str,
            status=msg.status,
            timestamp=msg.timestamp,
        )


@dataclass
class RemoteCallable:
    """A callable registered for remote execution.

    Attributes:
        func: The function to execute.
        name: Registered name.
        owner_name: Name of the worker that owns this callable.
        is_method: Whether this is a bound method.
        docstring: Documentation string.
    """
    func: Callable
    name: str
    owner_name: str = ""
    is_method: bool = False
    docstring: str = ""

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"RemoteCallable(name={self.name!r}, owner={self.owner_name!r})"


@dataclass
class RPCMetrics:
    """Metrics for RPC operations.

    Attributes:
        total_calls: Total number of RPC calls made.
        successful_calls: Number of successful calls.
        failed_calls: Number of failed calls.
        timed_out_calls: Number of timed out calls.
        total_bytes_sent: Total bytes sent.
        total_bytes_received: Total bytes received.
        total_latency_ms: Total latency in milliseconds.
        avg_latency_ms: Average latency per call.
        min_latency_ms: Minimum latency observed.
        max_latency_ms: Maximum latency observed.
        p50_latency_ms: 50th percentile latency.
        p99_latency_ms: 99th percentile latency.
        calls_by_destination: Call counts per destination.
        calls_by_method: Call counts per method.
        error_counts: Counts of each error type.
    """
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timed_out_calls: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    calls_by_destination: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls_by_method: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _latencies: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_call(
        self,
        dst_name: str,
        method_name: str,
        latency_ms: float,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        success: bool = True,
        error_type: str = "",
        timed_out: bool = False,
    ) -> None:
        """Record metrics for a single RPC call."""
        with self._lock:
            self.total_calls += 1
            if success:
                self.successful_calls += 1
            elif timed_out:
                self.timed_out_calls += 1
            else:
                self.failed_calls += 1
                if error_type:
                    self.error_counts[error_type] += 1

            self.total_bytes_sent += bytes_sent
            self.total_bytes_received += bytes_received
            self.total_latency_ms += latency_ms
            self._latencies.append(latency_ms)
            self.min_latency_ms = min(self.min_latency_ms, latency_ms)
            self.max_latency_ms = max(self.max_latency_ms, latency_ms)
            self.avg_latency_ms = self.total_latency_ms / self.total_calls
            self.calls_by_destination[dst_name] += 1
            self.calls_by_method[method_name] += 1
            if len(self._latencies) > 1:
                sorted_lat = sorted(self._latencies)
                n = len(sorted_lat)
                self.p50_latency_ms = sorted_lat[n // 2]
                self.p99_latency_ms = sorted_lat[int(n * 0.99)]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_calls = 0
            self.successful_calls = 0
            self.failed_calls = 0
            self.timed_out_calls = 0
            self.total_bytes_sent = 0
            self.total_bytes_received = 0
            self.total_latency_ms = 0.0
            self.avg_latency_ms = 0.0
            self.min_latency_ms = float("inf")
            self.max_latency_ms = 0.0
            self.p50_latency_ms = 0.0
            self.p99_latency_ms = 0.0
            self._latencies.clear()
            self.calls_by_destination.clear()
            self.calls_by_method.clear()
            self.error_counts.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "timed_out_calls": self.timed_out_calls,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float("inf") else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "bandwidth_mbps": (
                self.total_bytes_sent / (self.total_latency_ms / 1000.0 / 1e6)
                if self.total_latency_ms > 0 else 0.0
            ),
            "calls_by_destination": dict(self.calls_by_destination),
            "calls_by_method": dict(self.calls_by_method),
            "error_counts": dict(self.error_counts),
        }

    def summary(self) -> str:
        d = self.to_dict()
        lines = [
            f"RPC Metrics Summary:",
            f"  Total Calls: {d['total_calls']}",
            f"  Successful: {d['successful_calls']}",
            f"  Failed: {d['failed_calls']}",
            f"  Timed Out: {d['timed_out_calls']}",
            f"  Avg Latency: {d['avg_latency_ms']:.2f} ms",
            f"  P50 Latency: {d['p50_latency_ms']:.2f} ms",
            f"  P99 Latency: {d['p99_latency_ms']:.2f} ms",
            f"  Min/Max Latency: {d['min_latency_ms']:.2f} / {d['max_latency_ms']:.2f} ms",
            f"  Bytes Sent: {d['total_bytes_sent'] / 1024 / 1024:.2f} MB",
            f"  Bytes Received: {d['total_bytes_received'] / 1024 / 1024:.2f} MB",
            f"  Bandwidth: {d['bandwidth_mbps']:.2f} MB/s",
        ]
        return "\n".join(lines)


# ==============================================================================
# RPC Retry Policy
# ==============================================================================


class RPCRetryPolicy:
    """Handles RPC failures with configurable retry logic.

    Supports exponential backoff with jitter, circuit breaker pattern,
    and per-error-type retry configuration.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff.
        jitter: Whether to add random jitter to delays.
        retryable_errors: Set of error types that are retryable.
        circuit_breaker_threshold: Number of failures before circuit opens.
        circuit_breaker_reset_time: Time in seconds before circuit resets.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[Set[type]] = None,
        circuit_breaker_threshold: int = 10,
        circuit_breaker_reset_time: float = 60.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_errors = retryable_errors or {
            RPCConnectionError,
            RPCTimeoutError,
            RPCError,
        }
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_reset_time = circuit_breaker_reset_time
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._circuit_open: Dict[str, bool] = defaultdict(bool)
        self._circuit_open_time: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for the given attempt number using exponential backoff."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )
        if self.jitter:
            import random
            delay *= random.uniform(0.5, 1.5)
        return max(0.0, delay)

    def should_retry(self, error: Exception, dst_name: str = "") -> bool:
        """Check if the error is retryable."""
        for retryable_type in self.retryable_errors:
            if isinstance(error, retryable_type):
                return True
        if isinstance(error, (ConnectionError, TimeoutError, OSError)):
            return True
        return False

    def is_circuit_open(self, dst_name: str) -> bool:
        """Check if the circuit breaker is open for a destination."""
        with self._lock:
            if not self._circuit_open.get(dst_name, False):
                return False
            open_time = self._circuit_open_time.get(dst_name, 0)
            if time.time() - open_time > self.circuit_breaker_reset_time:
                self._circuit_open[dst_name] = False
                self._failure_counts[dst_name] = 0
                return False
            return True

    def record_failure(self, dst_name: str) -> None:
        """Record a failure for circuit breaker tracking."""
        with self._lock:
            self._failure_counts[dst_name] += 1
            if self._failure_counts[dst_name] >= self.circuit_breaker_threshold:
                self._circuit_open[dst_name] = True
                self._circuit_open_time[dst_name] = time.time()
                logger.warning(
                    f"Circuit breaker OPEN for {dst_name} after "
                    f"{self._failure_counts[dst_name]} failures"
                )

    def record_success(self, dst_name: str) -> None:
        """Record a successful call, resetting failure count."""
        with self._lock:
            self._failure_counts[dst_name] = 0
            if self._circuit_open[dst_name]:
                self._circuit_open[dst_name] = False
                logger.info(f"Circuit breaker CLOSED for {dst_name}")

    def get_retry_context(
        self, error: Exception, dst_name: str
    ) -> Dict[str, Any]:
        """Get context information about the retry situation."""
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "dst_name": dst_name,
            "circuit_open": self.is_circuit_open(dst_name),
            "failure_count": self._failure_counts.get(dst_name, 0),
        }

    def reset(self) -> None:
        """Reset all circuit breaker state."""
        with self._lock:
            self._failure_counts.clear()
            self._circuit_open.clear()
            self._circuit_open_time.clear()


# ==============================================================================
# RPC Backend (Abstract)
# ==============================================================================


class RPCBackend(abc.ABC):
    """Abstract base class for RPC communication backends.

    Defines the interface that all RPC backends must implement for
    sending and receiving messages between distributed workers.
    """

    @abc.abstractmethod
    def initialize(
        self,
        name: str,
        rank: int,
        world_size: int,
        init_method: str = "env://",
        timeout: float = 30.0,
    ) -> None:
        """Initialize the RPC backend.

        Args:
            name: Unique name for this worker.
            rank: Global rank of this worker.
            world_size: Total number of workers.
            init_method: Initialization method URL.
            timeout: Timeout for initialization.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def send_message(self, message: RPCMessage) -> None:
        """Send a message to the destination worker.

        Args:
            message: The message to send.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recv_message(self, timeout: float = 30.0) -> RPCMessage:
        """Receive a message from any worker.

        Args:
            timeout: Maximum time to wait for a message.

        Returns:
            The received message.

        Raises:
            RPCTimeoutError: If no message is received within timeout.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """Send a tensor to a specific rank.

        Args:
            tensor: Tensor to send.
            dst_rank: Destination rank.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recv_tensor(
        self, src_rank: int, dtype: torch.dtype = torch.float32, shape: Tuple[int, ...] = (0,)
    ) -> torch.Tensor:
        """Receive a tensor from a specific rank.

        Args:
            src_rank: Source rank.
            dtype: Expected tensor dtype.
            shape: Expected tensor shape.

        Returns:
            Received tensor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shut down the RPC backend and release resources."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_initialized(self) -> bool:
        """Check if the backend is initialized."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_name(self) -> str:
        """Get the name of this worker."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_rank(self) -> int:
        """Get the rank of this worker."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_world_size(self) -> int:
        """Get the world size."""
        raise NotImplementedError


# ==============================================================================
# Process Group RPC Backend
# ==============================================================================


class ProcessGroupRPC(RPCBackend):
    """PyTorch distributed process group based RPC backend.

    Uses torch.distributed primitives (send/recv, broadcast) for
    reliable message passing between workers. Supports both
    GPU tensors and CPU tensors.

    Attributes:
        _name: Name of this worker.
        _rank: Global rank.
        _world_size: Total number of workers.
        _initialized: Whether the backend is initialized.
        _process_group: The default process group.
        _message_queue: Thread-safe queue for incoming messages.
        _recv_thread: Background thread for receiving messages.
        _running: Whether the receive thread is running.
        _backend: torch.distributed backend name.
        _lock: Thread lock for thread-safe operations.
        _peer_names: Mapping from rank to worker name.
        _name_to_rank: Mapping from worker name to rank.
    """

    def __init__(
        self,
        backend: str = "gloo",
        max_queue_size: int = 10000,
    ):
        self._name = ""
        self._rank = 0
        self._world_size = 1
        self._initialized = False
        self._process_group: Optional[dist.ProcessGroup] = None
        self._message_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False
        self._backend = backend
        self._lock = threading.Lock()
        self._peer_names: Dict[int, str] = {}
        self._name_to_rank: Dict[str, int] = {}
        self._request_handlers: Dict[str, Callable] = {}
        self._name_exchange_done = False
        self._logger = logging.getLogger(f"{__name__}.ProcessGroupRPC[{id(self)}]")

    def initialize(
        self,
        name: str,
        rank: int,
        world_size: int,
        init_method: str = "env://",
        timeout: float = 30.0,
    ) -> None:
        """Initialize the process group RPC backend.

        Sets up the distributed process group, exchanges worker names
        across all ranks, and starts the background receive thread.
        """
        if self._initialized:
            self._logger.warning("ProcessGroupRPC already initialized")
            return

        self._name = name
        self._rank = rank
        self._world_size = world_size

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self._backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout) if timeout else None,
            )

        self._process_group = dist.group.WORLD

        self._exchange_names()
        self._start_recv_thread()

        self._initialized = True
        self._logger.info(
            f"ProcessGroupRPC initialized: name={name}, rank={rank}, "
            f"world_size={world_size}, backend={self._backend}"
        )

    def _exchange_names(self) -> None:
        """Exchange worker names across all ranks."""
        name_bytes = self._name.encode("utf-8")
        name_tensor = torch.zeros(256, dtype=torch.uint8)
        for i, b in enumerate(name_bytes[:256]):
            name_tensor[i] = b

        gathered = [torch.zeros(256, dtype=torch.uint8) for _ in range(self._world_size)]
        dist.all_gather(gathered, name_tensor, group=self._process_group)

        for r in range(self._world_size):
            tensor_bytes = gathered[r]
            name_len = 0
            for i in range(256):
                if tensor_bytes[i] == 0:
                    name_len = i
                    break
            peer_name = tensor_bytes[:name_len].numpy().tobytes().decode("utf-8")
            self._peer_names[r] = peer_name
            self._name_to_rank[peer_name] = r

        self._name_exchange_done = True
        self._logger.debug(f"Name exchange complete: {self._name_to_rank}")

    def _start_recv_thread(self) -> None:
        """Start the background message receiving thread."""
        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name=f"rpc-recv-{self._name}",
            daemon=True,
        )
        self._recv_thread.start()

    def _recv_loop(self) -> None:
        """Background loop for receiving messages from any rank.

        Uses polling to check for incoming messages from all peers.
        Each iteration checks if any rank has data available by
        attempting a non-blocking receive with a small timeout.
        """
        while self._running:
            for src_rank in range(self._world_size):
                if src_rank == self._rank:
                    continue
                try:
                    length_tensor = torch.zeros(1, dtype=torch.long)
                    req = dist.irecv(length_tensor, src=src_rank, group=self._process_group)
                    if req is not None:
                        completed, _ = req.test()
                        if completed:
                            length = int(length_tensor.item())
                            if length > 0:
                                data_tensor = torch.zeros(length, dtype=torch.uint8)
                                data_req = dist.irecv(
                                    data_tensor, src=src_rank, group=self._process_group
                                )
                                if data_req is not None:
                                    data_completed, _ = data_req.test()
                                    if data_completed:
                                        data_bytes = bytes(data_tensor.numpy().tobytes())
                                        message = RPCMessage.from_bytes(data_bytes)
                                        self._message_queue.put(message)
                                else:
                                    data_tensor2 = torch.zeros(length, dtype=torch.uint8)
                                    dist.recv(
                                        data_tensor2, src=src_rank,
                                        group=self._process_group
                                    )
                                    data_bytes = bytes(data_tensor2.numpy().tobytes())
                                    message = RPCMessage.from_bytes(data_bytes)
                                    self._message_queue.put(message)
                except dist.DistBackendError as e:
                    if self._running:
                        self._logger.debug(f"Non-blocking recv from rank {src_rank}: {e}")
                except Exception as e:
                    if self._running:
                        self._logger.warning(f"Error receiving from rank {src_rank}: {e}")

            time.sleep(0.001)

    def send_message(self, message: RPCMessage) -> None:
        """Send a message to the destination worker.

        Serializes the message, sends its length followed by the data.

        Args:
            message: The message to send.

        Raises:
            RPCConnectionError: If the destination is unknown.
            RPCSerializationError: If serialization fails.
        """
        if not self._initialized:
            raise RPCError("Backend not initialized")

        dst_rank = self._name_to_rank.get(message.dst_name)
        if dst_rank is None:
            raise RPCConnectionError(
                f"Unknown destination: {message.dst_name}",
                src=self._name,
                dst=message.dst_name,
            )

        try:
            data = message.to_bytes()
            length_tensor = torch.tensor([len(data)], dtype=torch.long)
            dist.send(length_tensor, dst=dst_rank, group=self._process_group)

            data_tensor = torch.frombuffer(data, dtype=torch.uint8).clone()
            dist.send(data_tensor, dst=dst_rank, group=self._process_group)
        except dist.DistBackendError as e:
            raise RPCConnectionError(
                f"Failed to send message to {message.dst_name}: {e}",
                src=self._name,
                dst=message.dst_name,
            ) from e
        except Exception as e:
            if isinstance(e, RPCError):
                raise
            raise RPCSerializationError(
                f"Send error: {e}", src=self._name, dst=message.dst_name
            ) from e

    def recv_message(self, timeout: float = 30.0) -> RPCMessage:
        """Receive a message from the queue.

        Args:
            timeout: Maximum time to wait.

        Returns:
            The received message.

        Raises:
            RPCTimeoutError: If no message arrives within timeout.
        """
        if not self._initialized:
            raise RPCError("Backend not initialized")
        try:
            message = self._message_queue.get(timeout=timeout)
            return message
        except queue.Empty:
            raise RPCTimeoutError(
                f"No message received within {timeout}s",
                timeout=timeout,
                src=self._name,
            )

    def send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> None:
        """Send a tensor to a specific rank."""
        if not self._initialized:
            raise RPCError("Backend not initialized")
        if tensor.is_cuda and self._backend == "gloo":
            tensor = tensor.cpu()
        dist.send(tensor.contiguous(), dst=dst_rank, group=self._process_group)

    def recv_tensor(
        self,
        src_rank: int,
        dtype: torch.dtype = torch.float32,
        shape: Tuple[int, ...] = (0,),
    ) -> torch.Tensor:
        """Receive a tensor from a specific rank."""
        if not self._initialized:
            raise RPCError("Backend not initialized")
        tensor = torch.zeros(*shape, dtype=dtype)
        dist.recv(tensor, src=src_rank, group=self._process_group)
        return tensor

    def broadcast_message(self, message: RPCMessage, src_rank: int = 0) -> RPCMessage:
        """Broadcast a message from src_rank to all ranks."""
        if not self._initialized:
            raise RPCError("Backend not initialized")
        if self._rank == src_rank:
            data = message.to_bytes()
            length = len(data)
        else:
            length = 0

        length_tensor = torch.tensor([length], dtype=torch.long)
        dist.broadcast(length_tensor, src=src_rank, group=self._process_group)
        length = int(length_tensor.item())

        if length == 0:
            data_tensor = torch.zeros(length, dtype=torch.uint8)
            dist.broadcast(data_tensor, src=src_rank, group=self._process_group)
            return RPCMessage.from_bytes(bytes(data_tensor.numpy().tobytes()))
        else:
            if self._rank == src_rank:
                data = message.to_bytes()
                data_tensor = torch.frombuffer(data, dtype=torch.uint8).clone()
            else:
                data_tensor = torch.zeros(length, dtype=torch.uint8)
            dist.broadcast(data_tensor, src=src_rank, group=self._process_group)
            return RPCMessage.from_bytes(bytes(data_tensor.numpy().tobytes()))

    def shutdown(self) -> None:
        """Shut down the RPC backend."""
        self._running = False
        if self._recv_thread is not None and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=5.0)
        self._initialized = False
        self._logger.info("ProcessGroupRPC shut down")

    def is_initialized(self) -> bool:
        return self._initialized

    def get_name(self) -> str:
        return self._name

    def get_rank(self) -> int:
        return self._rank

    def get_world_size(self) -> int:
        return self._world_size

    def name_to_rank(self, name: str) -> int:
        """Convert a worker name to its rank."""
        rank = self._name_to_rank.get(name)
        if rank is None:
            raise RPCNotFoundError(f"Unknown worker name: {name}")
        return rank

    def rank_to_name(self, rank: int) -> str:
        """Convert a rank to its worker name."""
        name = self._peer_names.get(rank)
        if name is None:
            raise RPCNotFoundError(f"Unknown rank: {rank}")
        return name


def _ensure_timedelta_or_none(timeout: Optional[float]) -> Any:
    """Convert seconds to datetime.timedelta or return None."""
    if timeout is not None:
        return timedelta(seconds=timeout)
    return None


from datetime import timedelta


# ==============================================================================
# RPC Service
# ==============================================================================


class RPCService:
    """RPC server/client implementation for distributed training.

    Provides a high-level API for remote procedure calls between workers.
    Supports synchronous calls, asynchronous calls, fire-and-forget calls,
    remote module execution, and parameter server functionality.

    The RPCService manages registered functions, handles incoming requests,
    and provides retry logic and profiling for all calls.

    Usage:
        # On worker 0:
        service = RPCService()
        service.start_server("worker0", rank=0, world_size=4)

        @service.remote
        def compute(x, y):
            return x + y

        # On worker 1:
        service = RPCService()
        service.start_server("worker1", rank=1, world_size=4)
        result = service.call("worker0", "compute", (10, 20))
        assert result == 30
    """

    _global_instance: Optional["RPCService"] = None
    _global_lock = threading.Lock()

    def __init__(
        self,
        backend: RPCBackend = None,
        timeout: float = 30.0,
        max_workers: int = 16,
        retry_policy: Optional[RPCRetryPolicy] = None,
        profile: bool = False,
    ):
        self._backend: Optional[RPCBackend] = backend
        self._timeout = timeout
        self._initialized = False
        self._name = ""
        self._rank = 0
        self._world_size = 1
        self._registered_funcs: Dict[str, RemoteCallable] = {}
        self._remote_modules: Dict[str, "RemoteModule"] = {}
        self._pending_futures: Dict[str, Future] = {}
        self._pending_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._retry_policy = retry_policy or RPCRetryPolicy()
        self._metrics = RPCMetrics()
        self._profiler: Optional[RPCProfiler] = RPCProfiler() if profile else None
        self._request_handlers: Dict[str, Callable] = {}
        self._shutdown_event = threading.Event()
        self._handler_thread: Optional[threading.Thread] = None
        self._remote_modules_lock = threading.Lock()
        self._registered_funcs_lock = threading.Lock()
        self._logger = logging.getLogger(f"{__name__}.RPCService[{id(self)}]")
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    @classmethod
    def get_instance(cls) -> "RPCService":
        """Get the global RPC service instance."""
        with cls._global_lock:
            if cls._global_instance is None:
                cls._global_instance = cls()
            return cls._global_instance

    @classmethod
    def set_instance(cls, instance: "RPCService") -> None:
        """Set the global RPC service instance."""
        with cls._global_lock:
            cls._global_instance = instance

    def start_server(
        self,
        name: str,
        rank: int,
        world_size: int,
        init_method: str = "env://",
        timeout: float = 30.0,
    ) -> None:
        """Start the RPC server.

        Initializes the backend, registers built-in handlers,
        and starts the request handling thread.

        Args:
            name: Unique name for this worker.
            rank: Global rank.
            world_size: Total number of workers.
            init_method: Initialization method URL.
            timeout: Initialization timeout.
        """
        if self._initialized:
            self._logger.warning(f"RPC server already running as {self._name}")
            return

        if self._backend is None:
            use_nccl = world_size > 1 and torch.cuda.is_available()
            backend_name = "nccl" if use_nccl else "gloo"
            self._backend = ProcessGroupRPC(backend=backend_name)

        self._name = name
        self._rank = rank
        self._world_size = world_size
        self._timeout = timeout

        self._backend.initialize(name, rank, world_size, init_method, timeout)

        self._register_builtin_handlers()

        self._handler_thread = threading.Thread(
            target=self._request_handler_loop,
            name=f"rpc-handler-{name}",
            daemon=True,
        )
        self._handler_thread.start()

        self._initialized = True
        RPCService.set_instance(self)

        self._logger.info(
            f"RPC server started: name={name}, rank={rank}, "
            f"world_size={world_size}"
        )

    def _register_builtin_handlers(self) -> None:
        """Register built-in RPC handlers."""
        self._register_handler("_ping", self._handle_ping)
        self._register_handler("_get_registered_funcs", self._handle_get_registered)
        self._register_handler("_get_name", self._handle_get_name)
        self._register_handler("_get_info", self._handle_get_info)
        self._register_handler("_shutdown", self._handle_shutdown_request)

    def _register_handler(self, name: str, handler: Callable) -> None:
        """Register a request handler."""
        self._request_handlers[name] = handler

    def connect(self, name: str, rank: int) -> None:
        """Connect to a remote RPC peer.

        Validates that the peer exists and is reachable.

        Args:
            name: Name of the remote worker.
            rank: Rank of the remote worker.
        """
        if not self._initialized:
            raise RPCError("RPC service not initialized")

        try:
            response = self.call(name, "_ping", (self._name,), timeout=5.0)
            if response != "pong":
                raise RPCConnectionError(
                    f"Unexpected ping response from {name}: {response}"
                )
            self._logger.info(f"Connected to remote peer: {name} (rank={rank})")
        except RPCTimeoutError:
            self._logger.warning(f"Timeout connecting to {name}, will retry on first call")
        except RPCError as e:
            self._logger.warning(f"Could not verify connection to {name}: {e}")

    def call(
        self,
        remote_name: str,
        method: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Synchronous remote procedure call.

        Sends a request to the remote worker, waits for the response,
        and returns the result. Raises on error or timeout.

        Args:
            remote_name: Name of the remote worker.
            method: Name of the remote method to call.
            args: Positional arguments.
            kwargs: Keyword arguments.
            timeout: Call-specific timeout (overrides default).

        Returns:
            The return value from the remote method.

        Raises:
            RPCTimeoutError: If the call times out.
            RPCRemoteError: If the remote method raises an exception.
            RPCError: For other RPC errors.
        """
        if not self._initialized:
            raise RPCError("RPC service not initialized")

        kwargs = kwargs or {}
        effective_timeout = timeout or self._timeout
        start_time = time.time()
        last_error = None

        for attempt in range(self._retry_policy.max_retries + 1):
            if self._retry_policy.is_circuit_open(remote_name):
                raise RPCError(
                    f"Circuit breaker is open for {remote_name}, "
                    f"too many recent failures"
                )

            try:
                correlation_id = str(uuid.uuid4())
                message = RPCMessage(
                    msg_type=MessageType.REQUEST,
                    src_name=self._name,
                    dst_name=remote_name,
                    method_name=method,
                    args=args,
                    kwargs=kwargs,
                    timeout=effective_timeout,
                    correlation_id=correlation_id,
                )

                bytes_sent = message.size_bytes
                self._backend.send_message(message)

                if self._profiler:
                    self._profiler.record_send(remote_name, method, bytes_sent)

                deadline = time.time() + effective_timeout
                remaining = effective_timeout

                while remaining > 0 and not self._shutdown_event.is_set():
                    try:
                        response_msg = self._backend.recv_message(timeout=min(remaining, 1.0))
                    except RPCTimeoutError:
                        remaining = deadline - time.time()
                        continue

                    if response_msg.correlation_id == correlation_id:
                        if response_msg.msg_type == MessageType.ERROR:
                            error_msg = response_msg.kwargs.get("error", "Unknown error")
                            tb = response_msg.kwargs.get("traceback", "")
                            raise RPCRemoteError(
                                error_msg,
                                remote_traceback=tb,
                                src=remote_name,
                                dst=self._name,
                            )

                        result = response_msg.args[0] if response_msg.args else None
                        bytes_received = response_msg.size_bytes
                        latency_ms = (time.time() - start_time) * 1000

                        self._metrics.record_call(
                            dst_name=remote_name,
                            method_name=method,
                            latency_ms=latency_ms,
                            bytes_sent=bytes_sent,
                            bytes_received=bytes_received,
                            success=True,
                        )

                        if self._profiler:
                            self._profiler.record_recv(
                                remote_name, method, bytes_received, latency_ms
                            )

                        self._retry_policy.record_success(remote_name)
                        return result

                    else:
                        self._pending_futures.setdefault(
                            response_msg.correlation_id, Future()
                        )
                        future = self._pending_futures[response_msg.correlation_id]
                        if response_msg.msg_type == MessageType.ERROR:
                            error_msg = response_msg.kwargs.get("error", "")
                            tb = response_msg.kwargs.get("traceback", "")
                            future.set_exception(
                                RPCRemoteError(
                                    error_msg,
                                    remote_traceback=tb,
                                    src=response_msg.src_name,
                                    dst=self._name,
                                )
                            )
                        else:
                            future.set_result(
                                response_msg.args[0] if response_msg.args else None
                            )

                raise RPCTimeoutError(
                    f"RPC call to {remote_name}.{method} timed out after {effective_timeout}s",
                    timeout=effective_timeout,
                    src=self._name,
                    dst=remote_name,
                )

            except (RPCTimeoutError, RPCConnectionError) as e:
                last_error = e
                if attempt < self._retry_policy.max_retries:
                    delay = self._retry_policy.compute_delay(attempt)
                    self._logger.warning(
                        f"RPC call {remote_name}.{method} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    self._retry_policy.record_failure(remote_name)
                    time.sleep(delay)
                    continue
                break

            except RPCRemoteError:
                self._metrics.record_call(
                    dst_name=remote_name,
                    method_name=method,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error_type="RPCRemoteError",
                )
                raise

            except Exception as e:
                if isinstance(e, RPCError):
                    raise
                self._metrics.record_call(
                    dst_name=remote_name,
                    method_name=method,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error_type=type(e).__name__,
                )
                raise RPCError(
                    f"Unexpected error in RPC call to {remote_name}.{method}: {e}",
                    src=self._name,
                    dst=remote_name,
                ) from e

        latency_ms = (time.time() - start_time) * 1000
        self._metrics.record_call(
            dst_name=remote_name,
            method_name=method,
            latency_ms=latency_ms,
            success=False,
            error_type=type(last_error).__name__ if last_error else "unknown",
            timed_out=isinstance(last_error, RPCTimeoutError),
        )
        self._retry_policy.record_failure(remote_name)
        if last_error:
            raise last_error
        raise RPCError(f"RPC call to {remote_name}.{method} failed after all retries")

    def call_async(
        self,
        remote_name: str,
        method: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Future:
        """Asynchronous remote procedure call.

        Submits the RPC call to a thread pool and returns a Future.
        The caller can await the result or add callbacks.

        Args:
            remote_name: Name of the remote worker.
            method: Name of the remote method.
            args: Positional arguments.
            kwargs: Keyword arguments.
            timeout: Call-specific timeout.

        Returns:
            A Future that will contain the result.
        """
        future: Future = Future()

        def _async_call():
            try:
                result = self.call(remote_name, method, args, kwargs, timeout)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        self._executor.submit(_async_call)
        return future

    def call_remote(
        self,
        remote_name: str,
        method: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Fire-and-forget remote procedure call.

        Submits the call without waiting for a result. Optionally
        registers a callback to be called when the result is ready.

        Args:
            remote_name: Name of the remote worker.
            method: Name of the remote method.
            args: Positional arguments.
            kwargs: Keyword arguments.
            callback: Optional callback(result) to call on completion.
        """
        def _fire_and_forget():
            try:
                result = self.call(remote_name, method, args, kwargs)
                if callback is not None:
                    callback(result)
            except Exception as e:
                self._logger.error(
                    f"Fire-and-forget call to {remote_name}.{method} failed: {e}"
                )

        self._executor.submit(_fire_and_forget)

    def remote(self, func: Callable) -> Callable:
        """Decorator to register a function for remote execution.

        Usage:
            @service.remote
            def my_function(x, y):
                return x + y
        """
        self.register(func)
        return func

    def register(self, func: Callable, name: Optional[str] = None) -> None:
        """Register a function for remote execution.

        Args:
            func: The function to register.
            name: Optional name (defaults to function __name__).
        """
        if not callable(func):
            raise RPCError(f"Cannot register non-callable: {type(func)}")

        func_name = name or func.__name__
        callable_obj = RemoteCallable(
            func=func,
            name=func_name,
            owner_name=self._name,
            is_method=False,
            docstring=func.__doc__ or "",
        )

        with self._registered_funcs_lock:
            self._registered_funcs[func_name] = callable_obj

        self._logger.debug(f"Registered remote function: {func_name}")

    def remote_module(self, module: "torch.nn.Module", name: Optional[str] = None) -> "RemoteModule":
        """Wrap a module for remote execution.

        Args:
            module: The module to wrap.
            name: Optional remote name.

        Returns:
            A RemoteModule wrapper.
        """
        remote = RemoteModule(
            module=module,
            rpc_service=self,
            name=name,
            owner_rank=self._rank,
        )
        with self._remote_modules_lock:
            self._remote_modules[name or module.__class__.__name__] = remote
        return remote

    def _request_handler_loop(self) -> None:
        """Main loop for handling incoming RPC requests.

        Runs in a background thread, continuously receiving messages
        from the backend and dispatching them to registered handlers.
        """
        while not self._shutdown_event.is_set():
            try:
                message = self._backend.recv_message(timeout=1.0)
            except RPCTimeoutError:
                continue
            except Exception as e:
                if not self._shutdown_event.is_set():
                    self._logger.error(f"Error receiving message: {e}")
                continue

            if message.msg_type == MessageType.SHUTDOWN:
                self._shutdown_event.set()
                break

            self._executor.submit(self._handle_request, message)

    def _handle_request(self, message: RPCMessage) -> None:
        """Handle a single incoming RPC request."""
        start_time = time.time()
        method_name = message.method_name
        response_msg: Optional[RPCMessage] = None

        try:
            handler = self._request_handlers.get(method_name)
            if handler is not None:
                result = handler(*message.args, **message.kwargs)
                response_msg = RPCMessage(
                    msg_type=MessageType.RESPONSE,
                    src_name=self._name,
                    dst_name=message.src_name,
                    method_name="",
                    args=(result,) if result is not None else (),
                    status=MessageStatus.COMPLETED,
                    correlation_id=message.correlation_id,
                )
            else:
                remote_callable = self._registered_funcs.get(method_name)
                if remote_callable is None:
                    raise RPCNotFoundError(
                        f"Method '{method_name}' not found on {self._name}",
                        src=message.src_name,
                        dst=self._name,
                    )

                result = remote_callable(*message.args, **message.kwargs)

                if isinstance(result, torch.Tensor):
                    response_msg = RPCMessage(
                        msg_type=MessageType.RESPONSE,
                        src_name=self._name,
                        dst_name=message.src_name,
                        args=(result,),
                        status=MessageStatus.COMPLETED,
                        correlation_id=message.correlation_id,
                    )
                elif isinstance(result, (list, tuple)) and any(
                    isinstance(r, torch.Tensor) for r in result
                ):
                    response_msg = RPCMessage(
                        msg_type=MessageType.RESPONSE,
                        src_name=self._name,
                        dst_name=message.src_name,
                        args=(result,),
                        status=MessageStatus.COMPLETED,
                        correlation_id=message.correlation_id,
                    )
                else:
                    response_msg = RPCMessage(
                        msg_type=MessageType.RESPONSE,
                        src_name=self._name,
                        dst_name=message.src_name,
                        args=(result,),
                        status=MessageStatus.COMPLETED,
                        correlation_id=message.correlation_id,
                    )

        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"Error handling {method_name}: {e}\n{tb}")
            response_msg = RPCMessage(
                msg_type=MessageType.ERROR,
                src_name=self._name,
                dst_name=message.src_name,
                kwargs={"error": str(e), "traceback": tb},
                status=MessageStatus.FAILED,
                correlation_id=message.correlation_id,
            )

        if response_msg is not None and message.requires_response:
            try:
                self._backend.send_message(response_msg)
            except Exception as e:
                self._logger.error(f"Failed to send response: {e}")

        latency_ms = (time.time() - start_time) * 1000
        if self._profiler:
            self._profiler.record_server_handler(
                method_name, latency_ms, response_msg.status if response_msg else MessageStatus.FAILED
            )

    def _handle_ping(self, sender_name: str) -> str:
        """Handle a ping request."""
        return "pong"

    def _handle_get_registered(self) -> List[str]:
        """Handle a request to list registered functions."""
        return list(self._registered_funcs.keys())

    def _handle_get_name(self) -> str:
        """Handle a request to get this worker's name."""
        return self._name

    def _handle_get_info(self) -> Dict[str, Any]:
        """Handle a request for worker info."""
        return {
            "name": self._name,
            "rank": self._rank,
            "world_size": self._world_size,
            "registered_funcs": list(self._registered_funcs.keys()),
            "remote_modules": list(self._remote_modules.keys()),
        }

    def _handle_shutdown_request(self, sender_name: str) -> str:
        """Handle a shutdown request."""
        self._logger.info(f"Shutdown requested by {sender_name}")
        self._shutdown_event.set()
        return "shutting_down"

    def shutdown(self) -> None:
        """Shut down the RPC service."""
        self._shutdown_event.set()

        if self._handler_thread is not None and self._handler_thread.is_alive():
            self._handler_thread.join(timeout=5.0)

        self._executor.shutdown(wait=True, cancel_futures=True)

        if self._backend is not None:
            self._backend.shutdown()

        self._initialized = False
        self._registered_funcs.clear()
        self._remote_modules.clear()
        self._pending_futures.clear()

        with RPCService._global_lock:
            if RPCService._global_instance is self:
                RPCService._global_instance = None

        self._logger.info("RPC service shut down")

    def get_metrics(self) -> RPCMetrics:
        """Get RPC metrics."""
        return self._metrics

    def get_profiler(self) -> Optional["RPCProfiler"]:
        """Get the RPC profiler."""
        return self._profiler

    def get_name(self) -> str:
        return self._name

    def get_rank(self) -> int:
        return self._rank

    def get_world_size(self) -> int:
        return self._world_size

    def is_initialized(self) -> bool:
        return self._initialized


# ==============================================================================
# Remote Module
# ==============================================================================


class RemoteModule:
    """Wraps a nn.Module for remote execution.

    Allows a module to be used on a remote GPU as if it were local.
    Forward passes are sent as RPC calls to the remote worker.

    Attributes:
        module: The wrapped module.
        rpc_service: The RPC service used for communication.
        name: Registered name for this remote module.
        owner_rank: Rank of the worker that owns this module.
        device: Device where the module resides.
        is_local: Whether this module is on the local worker.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        rpc_service: Optional[RPCService] = None,
        name: Optional[str] = None,
        owner_rank: int = 0,
        device: str = "cpu",
    ):
        self._module = module
        self._rpc_service = rpc_service
        self._name = name or module.__class__.__name__
        self._owner_rank = owner_rank
        self._device = device
        self._is_local = True
        self._module.eval()
        self._cache: Dict[str, Any] = {}
        self._parameter_cache: Optional[Dict[str, torch.Tensor]] = None
        self._state_dict_cache: Optional[Dict[str, Any]] = None
        self._logger = logging.getLogger(f"{__name__}.RemoteModule[{self._name}]")

        if rpc_service is not None:
            self._is_local = rpc_service.get_rank() == owner_rank

    @property
    def is_local(self) -> bool:
        return self._is_local

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> str:
        return self._device

    def forward(self, *args, **kwargs) -> Any:
        """Forward input to the remote module.

        If the module is local, executes directly. Otherwise, sends
        an RPC call to the owning worker.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            The output of the forward pass.
        """
        if self._is_local:
            with torch.no_grad():
                return self._module(*args, **kwargs)

        if self._rpc_service is None:
            raise RPCError("No RPC service configured for remote module")

        self._logger.debug(f"Remote forward: {self._name}")
        result = self._rpc_service.call(
            remote_name=self._get_owner_name(),
            method=f"_rm_forward_{self._name}",
            args=args,
            kwargs=kwargs,
        )
        return result

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def _get_owner_name(self) -> str:
        """Get the name of the worker that owns this module."""
        if self._rpc_service is None:
            raise RPCError("No RPC service configured")
        backend = self._rpc_service._backend
        if hasattr(backend, "rank_to_name"):
            return backend.rank_to_name(self._owner_rank)
        return f"rank{self._owner_rank}"

    def parameters(self) -> List[torch.Tensor]:
        """Retrieve parameters from the remote module.

        Returns:
            List of parameter tensors.
        """
        if self._is_local:
            return list(self._module.parameters())

        if self._rpc_service is None:
            raise RPCError("No RPC service configured")

        param_data = self._rpc_service.call(
            remote_name=self._get_owner_name(),
            method=f"_rm_params_{self._name}",
        )

        params = []
        if isinstance(param_data, dict):
            for name, tensor_info in param_data.items():
                if isinstance(tensor_info, torch.Tensor):
                    params.append(tensor_info)
                elif isinstance(tensor_info, dict):
                    shape = tensor_info.get("shape", ())
                    dtype_str = tensor_info.get("dtype", "float32")
                    dtype = getattr(torch, dtype_str, torch.float32)
                    data = tensor_info.get("data", None)
                    if data is not None:
                        t = torch.tensor(data, dtype=dtype).reshape(shape)
                        params.append(t)
        return params

    def state_dict(self) -> Dict[str, Any]:
        """Retrieve the state dict from the remote module.

        Returns:
            The remote module's state dict.
        """
        if self._is_local:
            return self._module.state_dict()

        if self._rpc_service is None:
            raise RPCError("No RPC service configured")

        return self._rpc_service.call(
            remote_name=self._get_owner_name(),
            method=f"_rm_state_dict_{self._name}",
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load a state dict onto the remote module.

        Args:
            state_dict: State dict to load.
        """
        if self._is_local:
            self._module.load_state_dict(state_dict)
            return

        if self._rpc_service is None:
            raise RPCError("No RPC service configured")

        self._rpc_service.call(
            remote_name=self._get_owner_name(),
            method=f"_rm_load_state_dict_{self._name}",
            args=(state_dict,),
        )

    def train(self) -> "RemoteModule":
        """Set the remote module to training mode."""
        if self._is_local:
            self._module.train()
        return self

    def eval(self) -> "RemoteModule":
        """Set the remote module to evaluation mode."""
        if self._is_local:
            self._module.eval()
        return self

    def to(self, device: str) -> "RemoteModule":
        """Move the remote module to a device."""
        self._device = device
        if self._is_local:
            self._module.to(device)
        return self

    def register_handlers(self, service: "RPCService") -> None:
        """Register RPC handlers for this remote module."""
        def _forward_handler(*args, **kwargs):
            with torch.no_grad():
                return self._module(*args, **kwargs)

        def _params_handler():
            params = {}
            for name, param in self._module.named_parameters():
                params[name] = {
                    "shape": tuple(param.shape),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "data": param.detach().cpu().numpy().tolist(),
                }
            return params

        def _state_dict_handler():
            return self._module.state_dict()

        def _load_state_dict_handler(state_dict):
            self._module.load_state_dict(state_dict)
            return "ok"

        def _train_handler():
            self._module.train()
            return "ok"

        def _eval_handler():
            self._module.eval()
            return "ok"

        service._register_handler(f"_rm_forward_{self._name}", _forward_handler)
        service._register_handler(f"_rm_params_{self._name}", _params_handler)
        service._register_handler(f"_rm_state_dict_{self._name}", _state_dict_handler)
        service._register_handler(f"_rm_load_state_dict_{self._name}", _load_state_dict_handler)
        service._register_handler(f"_rm_train_{self._name}", _train_handler)
        service._register_handler(f"_rm_eval_{self._name}", _eval_handler)

        self._logger.debug(f"Registered RPC handlers for remote module: {self._name}")

    def num_parameters(self) -> int:
        """Get the total number of parameters."""
        if self._is_local:
            return sum(p.numel() for p in self._module.parameters())
        return sum(p.numel() for p in self.parameters())


# ==============================================================================
# Parameter Server
# ==============================================================================


class ParameterServer:
    """Centralized parameter server for distributed training.

    Provides a key-value store for parameters and gradients that
    can be accessed by all workers. Supports push/pull/increment
    operations with optional persistence.

    The parameter server runs as a service that can be accessed
    through the RPC framework.

    Attributes:
        _store: The key-value parameter store.
        _locks: Per-key locks for thread safety.
        _version: Version counter for each key.
        _history: History of updates for each key.
        _persistent: Whether to persist to disk.
        _store_path: Path for persistent storage.
        _init_lock: Lock for initialization.
    """

    def __init__(
        self,
        persistent: bool = False,
        store_path: str = "/tmp/nexus_param_server",
        max_history: int = 100,
        sync_interval: float = 0.0,
    ):
        self._store: Dict[str, torch.Tensor] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._version: Dict[str, int] = defaultdict(int)
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._persistent = persistent
        self._store_path = store_path
        self._max_history = max_history
        self._sync_interval = sync_interval
        self._init_lock = threading.Lock()
        self._rpc_service: Optional[RPCService] = None
        self._is_server = False
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = {
            "push_count": 0,
            "pull_count": 0,
            "increment_count": 0,
            "sync_count": 0,
            "total_bytes_stored": 0,
        }
        self._logger = logging.getLogger(f"{__name__}.ParameterServer[{id(self)}]")

    def start(self, rpc_service: RPCService, is_server: bool = False) -> None:
        """Start the parameter server.

        Args:
            rpc_service: The RPC service to use for communication.
            is_server: Whether this instance is the server.
        """
        self._rpc_service = rpc_service
        self._is_server = is_server

        if is_server:
            self._register_handlers(rpc_service)

        if self._persistent and self._sync_interval > 0:
            self._running = True
            self._sync_thread = threading.Thread(
                target=self._sync_loop,
                name="param-server-sync",
                daemon=True,
            )
            self._sync_thread.start()

        self._logger.info(
            f"Parameter server started: is_server={is_server}, "
            f"persistent={self._persistent}"
        )

    def _register_handlers(self, service: RPCService) -> None:
        """Register RPC handlers for parameter server operations."""
        service._register_handler("_ps_push", self._handle_push)
        service._register_handler("_ps_pull", self._handle_pull)
        service._register_handler("_ps_increment", self._handle_increment)
        service._register_handler("_ps_sync", self._handle_sync)
        service._register_handler("_ps_delete", self._handle_delete)
        service._register_handler("_ps_keys", self._handle_keys)
        service._register_handler("_ps_info", self._handle_info)
        service._register_handler("_ps_clear", self._handle_clear)
        service._register_handler("_ps_get_version", self._handle_get_version)
        service._register_handler("_ps_push_many", self._handle_push_many)

    def push(self, key: str, tensor: torch.Tensor) -> int:
        """Push a tensor to the parameter server.

        Args:
            key: The parameter key.
            tensor: The tensor to store.

        Returns:
            The version number after the push.
        """
        if self._is_server:
            return self._local_push(key, tensor)

        if self._rpc_service is None:
            raise RPCError("No RPC service configured for parameter server")

        result = self._rpc_service.call(
            remote_name=self._get_server_name(),
            method="_ps_push",
            args=(key, tensor),
        )
        return result

    def pull(self, key: str) -> torch.Tensor:
        """Pull a tensor from the parameter server.

        Args:
            key: The parameter key.

        Returns:
            The stored tensor.

        Raises:
            RPCNotFoundError: If the key does not exist.
        """
        if self._is_server:
            return self._local_pull(key)

        if self._rpc_service is None:
            raise RPCError("No RPC service configured for parameter server")

        return self._rpc_service.call(
            remote_name=self._get_server_name(),
            method="_ps_pull",
            args=(key,),
        )

    def increment(self, key: str, delta: torch.Tensor) -> int:
        """Incrementally update a parameter.

        Adds delta to the current value stored under key.

        Args:
            key: The parameter key.
            delta: The tensor to add.

        Returns:
            The version number after the increment.
        """
        if self._is_server:
            return self._local_increment(key, delta)

        if self._rpc_service is None:
            raise RPCError("No RPC service configured for parameter server")

        return self._rpc_service.call(
            remote_name=self._get_server_name(),
            method="_ps_increment",
            args=(key, delta),
        )

    def sync(self) -> Dict[str, Any]:
        """Synchronize all parameters.

        Triggers persistence of all parameters to disk.

        Returns:
            Status dictionary with sync information.
        """
        if self._is_server:
            return self._local_sync()

        if self._rpc_service is None:
            raise RPCError("No RPC service configured for parameter server")

        return self._rpc_service.call(
            remote_name=self._get_server_name(),
            method="_ps_sync",
        )

    def _local_push(self, key: str, tensor: torch.Tensor) -> int:
        """Local push operation."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        with self._locks[key]:
            self._store[key] = tensor.detach().clone()
            self._version[key] += 1
            version = self._version[key]

            self._history[key].append({
                "version": version,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "timestamp": time.time(),
                "size_bytes": tensor.element_size() * tensor.numel(),
            })
            if len(self._history[key]) > self._max_history:
                self._history[key] = self._history[key][-self._max_history:]

        self._stats["push_count"] += 1
        self._update_bytes_stored()
        return version

    def _local_pull(self, key: str) -> torch.Tensor:
        """Local pull operation."""
        if key not in self._store:
            raise RPCNotFoundError(f"Key not found: {key}")

        with self._locks[key]:
            tensor = self._store[key].clone()

        self._stats["pull_count"] += 1
        return tensor

    def _local_increment(self, key: str, delta: torch.Tensor) -> int:
        """Local increment operation."""
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(delta)

        with self._locks[key]:
            if key not in self._store:
                self._store[key] = torch.zeros_like(delta)
            self._store[key] = self._store[key] + delta.detach()
            self._version[key] += 1
            version = self._version[key]

        self._stats["increment_count"] += 1
        return version

    def _local_sync(self) -> Dict[str, Any]:
        """Local sync operation — persist to disk if enabled."""
        self._stats["sync_count"] += 1
        result = {
            "keys": list(self._store.keys()),
            "num_keys": len(self._store),
            "total_bytes": self._stats["total_bytes_stored"],
        }

        if self._persistent:
            try:
                self._persist_to_disk()
                result["persisted"] = True
            except Exception as e:
                result["persisted"] = False
                result["error"] = str(e)

        return result

    def _handle_push(self, key: str, tensor: torch.Tensor) -> int:
        return self._local_push(key, tensor)

    def _handle_pull(self, key: str) -> torch.Tensor:
        return self._local_pull(key)

    def _handle_increment(self, key: str, delta: torch.Tensor) -> int:
        return self._local_increment(key, delta)

    def _handle_sync(self) -> Dict[str, Any]:
        return self._local_sync()

    def _handle_delete(self, key: str) -> bool:
        """Delete a key from the parameter server."""
        if key in self._store:
            with self._locks[key]:
                del self._store[key]
                self._history[key].clear()
            return True
        return False

    def _handle_keys(self) -> List[str]:
        """List all keys in the parameter server."""
        return list(self._store.keys())

    def _handle_info(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the parameter server or a specific key."""
        if key is not None:
            if key not in self._store:
                raise RPCNotFoundError(f"Key not found: {key}")
            tensor = self._store[key]
            return {
                "key": key,
                "version": self._version[key],
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "size_bytes": tensor.element_size() * tensor.numel(),
                "history_count": len(self._history[key]),
            }

        total_bytes = sum(
            t.element_size() * t.numel() for t in self._store.values()
        )
        return {
            "num_keys": len(self._store),
            "keys": list(self._store.keys()),
            "total_bytes": total_bytes,
            "stats": dict(self._stats),
        }

    def _handle_clear(self) -> bool:
        """Clear all parameters."""
        self._store.clear()
        self._version.clear()
        self._history.clear()
        return True

    def _handle_get_version(self, key: str) -> int:
        """Get the version of a key."""
        return self._version.get(key, 0)

    def _handle_push_many(self, kv_pairs: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Push multiple key-value pairs."""
        versions = {}
        for key, tensor in kv_pairs.items():
            versions[key] = self._local_push(key, tensor)
        return versions

    def _update_bytes_stored(self) -> None:
        """Update the total bytes stored counter."""
        self._stats["total_bytes_stored"] = sum(
            t.element_size() * t.numel() for t in self._store.values()
        )

    def _get_server_name(self) -> str:
        """Get the name of the parameter server worker."""
        if self._rpc_service is None:
            raise RPCError("No RPC service configured")
        backend = self._rpc_service._backend
        if hasattr(backend, "rank_to_name"):
            return backend.rank_to_name(0)
        return "rank0"

    def _persist_to_disk(self) -> None:
        """Persist all parameters to disk."""
        if not self._persistent:
            return

        os.makedirs(self._store_path, exist_ok=True)
        metadata = {
            "timestamp": time.time(),
            "num_keys": len(self._store),
            "keys": {},
        }

        for key, tensor in self._store.items():
            path = os.path.join(self._store_path, f"{self._safe_key(key)}.pt")
            torch.save(tensor, path)
            metadata["keys"][key] = {
                "path": path,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "version": self._version[key],
            }

        meta_path = os.path.join(self._store_path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_from_disk(self) -> None:
        """Load parameters from disk."""
        meta_path = os.path.join(self._store_path, "metadata.json")
        if not os.path.exists(meta_path):
            return

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        for key, info in metadata.get("keys", {}).items():
            path = info["path"]
            if os.path.exists(path):
                self._store[key] = torch.load(path, map_location="cpu", weights_only=True)
                self._version[key] = info.get("version", 0)

    def _safe_key(self, key: str) -> str:
        """Convert a key to a safe filename."""
        return hashlib.md5(key.encode()).hexdigest()

    def _sync_loop(self) -> None:
        """Background loop for periodic sync."""
        while self._running:
            time.sleep(self._sync_interval)
            try:
                self._local_sync()
            except Exception as e:
                self._logger.error(f"Periodic sync failed: {e}")

    def shutdown(self) -> None:
        """Shut down the parameter server."""
        self._running = False
        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        if self._persistent:
            try:
                self._persist_to_disk()
            except Exception as e:
                self._logger.error(f"Failed to persist on shutdown: {e}")
        self._logger.info("Parameter server shut down")

    def get_stats(self) -> Dict[str, Any]:
        """Get parameter server statistics."""
        self._update_bytes_stored()
        return dict(self._stats)

    def get_version(self, key: str) -> int:
        """Get the version number for a key."""
        if self._is_server:
            return self._version.get(key, 0)
        if self._rpc_service is not None:
            return self._rpc_service.call(
                remote_name=self._get_server_name(),
                method="_ps_get_version",
                args=(key,),
            )
        return 0

    def get_keys(self) -> List[str]:
        """Get all keys in the parameter server."""
        if self._is_server:
            return list(self._store.keys())
        if self._rpc_service is not None:
            return self._rpc_service.call(
                remote_name=self._get_server_name(),
                method="_ps_keys",
            )
        return []


# ==============================================================================
# RPC Profiler
# ==============================================================================


class RPCProfiler:
    """Profiles RPC call latency and bandwidth.

    Tracks send/receive operations, computes statistics, and provides
    methods for analyzing RPC performance.

    Attributes:
        _send_records: Records of send operations.
        _recv_records: Records of receive operations.
        _handler_records: Records of server-side handler execution.
        _lock: Thread lock.
        _enabled: Whether profiling is active.
    """

    def __init__(self):
        self._send_records: List[Dict[str, Any]] = []
        self._recv_records: List[Dict[str, Any]] = []
        self._handler_records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._enabled = True
        self._max_records = 100000
        self._start_time = time.time()

    def record_send(
        self, dst: str, method: str, bytes_sent: int, timestamp: Optional[float] = None
    ) -> None:
        """Record a send operation."""
        if not self._enabled:
            return
        with self._lock:
            self._send_records.append({
                "timestamp": timestamp or time.time(),
                "dst": dst,
                "method": method,
                "bytes_sent": bytes_sent,
            })
            if len(self._send_records) > self._max_records:
                self._send_records = self._send_records[-self._max_records // 2:]

    def record_recv(
        self,
        dst: str,
        method: str,
        bytes_received: int,
        latency_ms: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a receive operation."""
        if not self._enabled:
            return
        with self._lock:
            self._recv_records.append({
                "timestamp": timestamp or time.time(),
                "dst": dst,
                "method": method,
                "bytes_received": bytes_received,
                "latency_ms": latency_ms,
                "bandwidth_mbps": (
                    bytes_received / (latency_ms / 1000.0) / 1e6
                    if latency_ms > 0 else 0
                ),
            })
            if len(self._recv_records) > self._max_records:
                self._recv_records = self._recv_records[-self._max_records // 2:]

    def record_server_handler(
        self,
        method: str,
        latency_ms: float,
        status: MessageStatus = MessageStatus.COMPLETED,
    ) -> None:
        """Record a server-side handler execution."""
        if not self._enabled:
            return
        with self._lock:
            self._handler_records.append({
                "timestamp": time.time(),
                "method": method,
                "latency_ms": latency_ms,
                "status": status.value,
            })
            if len(self._handler_records) > self._max_records:
                self._handler_records = self._handler_records[-self._max_records // 2:]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of profiling data."""
        with self._lock:
            send_summary = self._summarize_sends()
            recv_summary = self._summarize_recvs()
            handler_summary = self._summarize_handlers()

            return {
                "uptime_seconds": time.time() - self._start_time,
                "total_sends": len(self._send_records),
                "total_recvs": len(self._recv_records),
                "total_handler_calls": len(self._handler_records),
                "send_summary": send_summary,
                "recv_summary": recv_summary,
                "handler_summary": handler_summary,
            }

    def _summarize_sends(self) -> Dict[str, Any]:
        """Summarize send records."""
        if not self._send_records:
            return {}

        total_bytes = sum(r["bytes_sent"] for r in self._send_records)
        by_dst: Dict[str, List[float]] = defaultdict(list)
        for r in self._send_records:
            by_dst[r["dst"]].append(r["bytes_sent"])

        return {
            "total_bytes": total_bytes,
            "avg_bytes": total_bytes / len(self._send_records),
            "by_destination": {
                dst: {
                    "count": len(recs),
                    "total_bytes": sum(recs),
                    "avg_bytes": sum(recs) / len(recs),
                }
                for dst, recs in by_dst.items()
            },
        }

    def _summarize_recvs(self) -> Dict[str, Any]:
        """Summarize receive records."""
        if not self._recv_records:
            return {}

        latencies = [r["latency_ms"] for r in self._recv_records]
        total_bytes = sum(r["bytes_received"] for r in self._recv_records)
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return {
            "total_bytes": total_bytes,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": sorted_lat[0],
            "max_latency_ms": sorted_lat[-1],
            "p50_latency_ms": sorted_lat[n // 2],
            "p95_latency_ms": sorted_lat[int(n * 0.95)],
            "p99_latency_ms": sorted_lat[int(n * 0.99)],
            "avg_bandwidth_mbps": (
                total_bytes / (sum(latencies) / 1000.0) / 1e6
                if sum(latencies) > 0 else 0
            ),
        }

    def _summarize_handlers(self) -> Dict[str, Any]:
        """Summarize handler records."""
        if not self._handler_records:
            return {}

        latencies = [r["latency_ms"] for r in self._handler_records]
        by_method: Dict[str, List[float]] = defaultdict(list)
        by_status: Dict[str, int] = defaultdict(int)
        for r in self._handler_records:
            by_method[r["method"]].append(r["latency_ms"])
            by_status[r["status"]] += 1

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": sorted_lat[0],
            "max_latency_ms": sorted_lat[-1],
            "p50_latency_ms": sorted_lat[n // 2],
            "p99_latency_ms": sorted_lat[int(n * 0.99)],
            "by_method": {
                method: {
                    "count": len(recs),
                    "avg_latency_ms": sum(recs) / len(recs),
                }
                for method, recs in by_method.items()
            },
            "by_status": dict(by_status),
        }

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._send_records.clear()
            self._recv_records.clear()
            self._handler_records.clear()
            self._start_time = time.time()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def print_summary(self) -> None:
        """Print a human-readable summary of profiling data."""
        summary = self.get_summary()
        lines = [
            "=" * 60,
            "RPC Profiler Summary",
            "=" * 60,
            f"Uptime: {summary['uptime_seconds']:.1f}s",
            f"Total Sends: {summary['total_sends']}",
            f"Total Receives: {summary['total_recvs']}",
            f"Total Handler Calls: {summary['total_handler_calls']}",
        ]

        recv = summary.get("recv_summary", {})
        if recv:
            lines.extend([
                "",
                "Receive Latency:",
                f"  Avg: {recv['avg_latency_ms']:.2f} ms",
                f"  P50: {recv['p50_latency_ms']:.2f} ms",
                f"  P99: {recv['p99_latency_ms']:.2f} ms",
                f"  Min: {recv['min_latency_ms']:.2f} ms",
                f"  Max: {recv['max_latency_ms']:.2f} ms",
                f"  Avg Bandwidth: {recv['avg_bandwidth_mbps']:.2f} MB/s",
            ])

        handler = summary.get("handler_summary", {})
        if handler and "by_method" in handler:
            lines.extend(["", "Handler Latency by Method:"])
            for method, stats in handler["by_method"].items():
                lines.append(
                    f"  {method}: avg={stats['avg_latency_ms']:.2f}ms, "
                    f"count={stats['count']}"
                )

        print("\n".join(lines))
