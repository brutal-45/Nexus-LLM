"""Worker pool for concurrent task execution."""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from nexus_llm.workers.config import WorkerConfig
from nexus_llm.workers.queue import TaskQueue
from nexus_llm.workers.worker import Worker, WorkerStatus


class WorkerPool:
    """Pool of workers that process tasks from a priority queue.

    Uses :class:`concurrent.futures.ThreadPoolExecutor` under the hood,
    augmented with a priority-based :class:`TaskQueue` and per-worker
    heartbeat monitoring.

    Example::

        pool = WorkerPool(WorkerConfig(num_workers=4))
        future = pool.submit(lambda: expensive_computation())
        result = future.result()

        results = pool.map(str.upper, ["hello", "world"])
    """

    def __init__(self, config: Optional[WorkerConfig] = None) -> None:
        self._config = config or WorkerConfig()
        self._queue = TaskQueue(max_size=self._config.max_queue_size)
        self._workers: List[Worker] = []
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.RLock()
        self._shutdown = False

        self._init_workers()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> WorkerConfig:
        """Return the pool configuration."""
        return self._config

    @property
    def queue(self) -> TaskQueue:
        """Return the task queue."""
        return self._queue

    @property
    def num_workers(self) -> int:
        """Return the number of workers."""
        return len(self._workers)

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit(
        self,
        task: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future:
        """Submit a callable for execution by the worker pool.

        Args:
            task: Callable to execute.
            *args, **kwargs: Arguments forwarded to *task*.

        Returns:
            A :class:`~concurrent.futures.Future` for the result.

        Raises:
            RuntimeError: If the pool has been shut down.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("WorkerPool has been shut down.")

            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._config.num_workers,
                    thread_name_prefix="nxl-worker",
                )

            return self._executor.submit(task, *args, **kwargs)

    def map(
        self,
        func: Callable,
        items: Sequence[Any],
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Apply *func* to every item in *items* concurrently.

        Args:
            func: Callable to apply.
            items: Sequence of inputs.
            timeout: Maximum seconds per item.

        Returns:
            A list of results in the same order as *items*.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("WorkerPool has been shut down.")

            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._config.num_workers,
                    thread_name_prefix="nxl-worker",
                )

        futures = [self.submit(func, item) for item in items]
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            results.append(future.result())

        # Re-order results to match input order
        ordered: List[Any] = [None] * len(items)
        future_to_idx = dict(zip(futures, range(len(items))))
        for future in futures:
            ordered[future_to_idx[future]] = future.result()

        return ordered

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool.

        Args:
            wait: If ``True``, block until all submitted tasks complete.
        """
        with self._lock:
            self._shutdown = True

        for worker in self._workers:
            worker.stop()

        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    # ------------------------------------------------------------------
    # Worker status
    # ------------------------------------------------------------------

    def worker_status(self) -> Dict[str, WorkerStatus]:
        """Return a mapping of worker IDs to their current status."""
        with self._lock:
            return {
                worker.worker_id: worker.status for worker in self._workers
            }

    def active_count(self) -> int:
        """Return the number of workers currently busy."""
        return sum(
            1 for w in self._workers if w.status == WorkerStatus.BUSY
        )

    def idle_count(self) -> int:
        """Return the number of workers currently idle."""
        return sum(
            1 for w in self._workers if w.status == WorkerStatus.IDLE
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_workers(self) -> None:
        """Create worker instances according to configuration."""
        for i in range(self._config.num_workers):
            worker = Worker(
                worker_id=f"worker-{i}",
                heartbeat_interval=self._config.heartbeat_interval,
            )
            self._workers.append(worker)

    def __enter__(self) -> WorkerPool:  # noqa: D105
        return self

    def __exit__(self, *exc: Any) -> None:  # noqa: D105
        self.shutdown(wait=True)
