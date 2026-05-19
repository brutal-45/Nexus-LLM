"""Batch Processor for Nexus-LLM.

Provides batch request processing capabilities, grouping individual
inference requests into efficient batches for higher throughput.
Supports configurable batch sizes, timeout-based flushing, and
automatic batch assembly from queued requests.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BatchStatus(str, Enum):
    """Status of a batch."""
    ASSEMBLING = "assembling"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some items succeeded, some failed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatchItem:
    """A single item within a batch."""
    item_id: str = ""
    payload: Any = None
    result: Any = None
    success: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.item_id:
            self.item_id = str(uuid.uuid4())[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "success": self.success,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class Batch:
    """A collection of items to be processed together."""
    batch_id: str = ""
    items: List[BatchItem] = field(default_factory=list)
    status: BatchStatus = BatchStatus.ASSEMBLING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.batch_id:
            self.batch_id = f"batch-{str(uuid.uuid4())[:8]}"

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def success_count(self) -> int:
        return sum(1 for item in self.items if item.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for item in self.items if not item.success and item.error is not None)

    @property
    def is_full(self) -> bool:
        return False  # Determined by BatchProcessor config

    @property
    def total_latency_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    def add_item(self, payload: Any, item_id: Optional[str] = None) -> BatchItem:
        """Add an item to the batch.

        Args:
            payload: Item payload.
            item_id: Optional item ID.

        Returns:
            The created BatchItem.
        """
        item = BatchItem(item_id=item_id or str(uuid.uuid4())[:12], payload=payload)
        self.items.append(item)
        return item

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "size": self.size,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "status": self.status.value,
            "created_at": self.created_at,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class BatchConfig:
    """Configuration for the batch processor."""
    max_batch_size: int = 32
    max_wait_time: float = 0.5  # Seconds before flushing a partial batch
    max_item_size: int = 8192   # Max payload size in bytes (approximate)
    auto_flush: bool = True
    concurrency: int = 4        # Max concurrent batches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_batch_size": self.max_batch_size,
            "max_wait_time": self.max_wait_time,
            "max_item_size": self.max_item_size,
            "auto_flush": self.auto_flush,
            "concurrency": self.concurrency,
        }


# ---------------------------------------------------------------------------
# Batch Processor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """Batch request processor for improved inference throughput.

    Collects individual inference requests and groups them into
    batches for efficient processing.  Batches are flushed when
    they reach maximum size or when the wait timeout expires.

    Example::

        processor = BatchProcessor(
            config=BatchConfig(max_batch_size=16, max_wait_time=0.2),
            handler=my_model_batch_infer,
        )
        processor.start()

        # Submit individual items
        item_id = processor.submit({"prompt": "Hello"})
        result = processor.get_result(item_id, timeout=10.0)

        processor.stop()
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        handler: Optional[Callable[[List[Any]], List[Any]]] = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            config: Batch processing configuration.
            handler: Function that processes a list of payloads and returns
                     a list of results (same order as input).
        """
        self._config = config or BatchConfig()
        self._handler = handler

        self._current_batch: Optional[Batch] = None
        self._completed_batches: Dict[str, Batch] = {}
        self._item_to_batch: Dict[str, str] = {}  # item_id -> batch_id
        self._results: Dict[str, Any] = {}         # item_id -> result
        self._pending_futures: Dict[str, threading.Event] = {}

        self._lock = threading.RLock()
        self._batch_condition = threading.Condition(self._lock)

        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._active_batches = 0
        self._total_batches = 0
        self._total_items = 0

    @property
    def is_running(self) -> bool:
        """Whether the processor is active."""
        return self._running

    @property
    def current_batch_size(self) -> int:
        """Number of items in the current assembling batch."""
        with self._lock:
            return self._current_batch.size if self._current_batch else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return
        self._running = True

        if self._config.auto_flush:
            self._flush_thread = threading.Thread(
                target=self._auto_flush_loop,
                daemon=True,
                name="batch-flush",
            )
            self._flush_thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the batch processor.

        Args:
            timeout: Seconds to wait for in-flight batches.
        """
        self._running = False

        # Flush any remaining items
        with self._lock:
            if self._current_batch and self._current_batch.size > 0:
                self._dispatch_batch(self._current_batch)
                self._current_batch = None

        # Wait for active batches
        deadline = time.time() + timeout
        while self._active_batches > 0 and time.time() < deadline:
            time.sleep(0.1)

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)

        # Signal all pending futures
        for event in self._pending_futures.values():
            event.set()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(self, payload: Any, item_id: Optional[str] = None) -> str:
        """Submit a single item for batch processing.

        Args:
            payload: The item payload.
            item_id: Optional custom item ID.

        Returns:
            The item ID for tracking the result.

        Raises:
            RuntimeError: If the processor is not running.
        """
        if not self._running:
            raise RuntimeError("Batch processor is not running")

        iid = item_id or str(uuid.uuid4())[:12]

        with self._lock:
            # Create a new batch if needed
            if self._current_batch is None:
                self._current_batch = Batch()

            # Add item to current batch
            item = self._current_batch.add_item(payload, item_id=iid)
            self._item_to_batch[iid] = self._current_batch.batch_id
            self._total_items += 1

            # Register a future for this item
            self._pending_futures[iid] = threading.Event()

            # Flush if batch is full
            if self._current_batch.size >= self._config.max_batch_size:
                self._dispatch_batch(self._current_batch)
                self._current_batch = None
            else:
                self._batch_condition.notify()

        return iid

    def submit_batch(self, payloads: List[Any]) -> List[str]:
        """Submit multiple items for batch processing.

        Args:
            payloads: List of item payloads.

        Returns:
            List of item IDs.
        """
        return [self.submit(p) for p in payloads]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_result(self, item_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for and retrieve a result for a specific item.

        Args:
            item_id: The item ID returned by submit().
            timeout: Maximum seconds to wait. None waits forever.

        Returns:
            The result for the item.

        Raises:
            TimeoutError: If the result is not available in time.
            KeyError: If the item ID is not found.
        """
        event = self._pending_futures.get(item_id)
        if event is None:
            raise KeyError(f"Item '{item_id}' not found")

        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Result for item '{item_id}' not available within timeout")

        if item_id in self._results:
            return self._results[item_id]

        # Check for error
        batch_id = self._item_to_batch.get(item_id)
        if batch_id:
            batch = self._completed_batches.get(batch_id)
            if batch:
                for item in batch.items:
                    if item.item_id == item_id and item.error:
                        raise RuntimeError(f"Item failed: {item.error}")

        return None

    def is_ready(self, item_id: str) -> bool:
        """Check if a result is ready for an item.

        Args:
            item_id: The item ID.

        Returns:
            True if the result is available.
        """
        return item_id in self._results or self._pending_futures.get(item_id, threading.Event()).is_set()

    # ------------------------------------------------------------------
    # Batch dispatching
    # ------------------------------------------------------------------

    def _dispatch_batch(self, batch: Batch) -> None:
        """Dispatch a batch for processing."""
        batch.status = BatchStatus.PROCESSING
        batch.started_at = time.time()
        self._active_batches += 1
        self._total_batches += 1

        t = threading.Thread(
            target=self._process_batch,
            args=(batch,),
            daemon=True,
            name=f"batch-{batch.batch_id}",
        )
        t.start()

    def _process_batch(self, batch: Batch) -> None:
        """Process a single batch."""
        try:
            if self._handler is not None:
                payloads = [item.payload for item in batch.items]
                results = self._handler(payloads)

                # Assign results to items
                for i, item in enumerate(batch.items):
                    if i < len(results):
                        item.result = results[i]
                        item.success = True
                    else:
                        item.error = "Missing result"
                        item.success = False
            else:
                # No handler registered - mark items as no-op
                for item in batch.items:
                    item.result = None
                    item.success = True

            batch.status = BatchStatus.COMPLETED if batch.failure_count == 0 else BatchStatus.PARTIAL

        except Exception as e:
            batch.status = BatchStatus.FAILED
            for item in batch.items:
                item.error = str(e)
                item.success = False

        finally:
            batch.completed_at = time.time()

            with self._lock:
                self._active_batches -= 1
                self._completed_batches[batch.batch_id] = batch

                # Store results and signal futures
                for item in batch.items:
                    if item.success:
                        self._results[item.item_id] = item.result
                    event = self._pending_futures.pop(item.item_id, None)
                    if event:
                        event.set()

    # ------------------------------------------------------------------
    # Auto-flush
    # ------------------------------------------------------------------

    def _auto_flush_loop(self) -> None:
        """Periodically flush batches that have been waiting too long."""
        while self._running:
            try:
                with self._batch_condition:
                    self._batch_condition.wait(timeout=self._config.max_wait_time)

                    if (self._current_batch and
                        self._current_batch.size > 0 and
                        self._running):
                        age = time.time() - self._current_batch.created_at
                        if age >= self._config.max_wait_time:
                            self._dispatch_batch(self._current_batch)
                            self._current_batch = None

            except Exception:
                pass

    # ------------------------------------------------------------------
    # Manual flush
    # ------------------------------------------------------------------

    def flush(self) -> Optional[str]:
        """Manually flush the current batch.

        Returns:
            The batch ID of the flushed batch, or None if no pending items.
        """
        with self._lock:
            if self._current_batch and self._current_batch.size > 0:
                batch_id = self._current_batch.batch_id
                self._dispatch_batch(self._current_batch)
                self._current_batch = None
                return batch_id
            return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics.

        Returns:
            Dictionary with throughput and batch metrics.
        """
        with self._lock:
            total_items_processed = sum(
                b.size for b in self._completed_batches.values()
                if b.status in (BatchStatus.COMPLETED, BatchStatus.PARTIAL)
            )
            total_latency = sum(
                b.total_latency_ms for b in self._completed_batches.values()
                if b.completed_at is not None
            )
            avg_batch_latency = total_latency / max(self._total_batches, 1)

            return {
                "running": self._running,
                "active_batches": self._active_batches,
                "total_batches": self._total_batches,
                "total_items_submitted": self._total_items,
                "total_items_processed": total_items_processed,
                "pending_items": self.current_batch_size,
                "avg_batch_latency_ms": round(avg_batch_latency, 2),
                "config": self._config.to_dict(),
            }
