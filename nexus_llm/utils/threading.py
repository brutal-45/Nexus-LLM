"""Threading utilities: thread pool, background worker, thread-safe queue, async bridge."""

import os
import queue
import threading
import asyncio
import logging
import time
from typing import Optional, Callable, Any, List, Dict, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ThreadPool:
    """A simple thread pool for parallel task execution."""

    def __init__(
        self,
        num_workers: Optional[int] = None,
        max_queue_size: int = 0,
    ):
        """Initialize the thread pool.

        Args:
            num_workers: Number of worker threads. Defaults to CPU count.
            max_queue_size: Maximum size of the task queue (0 for unlimited).
        """
        self.num_workers = num_workers or os.cpu_count() or 4
        self._task_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._results: Dict[int, Any] = {}
        self._errors: Dict[int, Exception] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._shutdown = False
        self._workers: List[threading.Thread] = []

        self._start_workers()

    def _start_workers(self):
        """Start worker threads."""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"ThreadPool-Worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self):
        """Worker thread main loop."""
        while not self._shutdown:
            try:
                task_id, func, args, kwargs = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                result = func(*args, **kwargs)
                with self._lock:
                    self._results[task_id] = result
            except Exception as e:
                with self._lock:
                    self._errors[task_id] = e
            finally:
                self._task_queue.task_done()

    def submit(self, func: Callable, *args, **kwargs) -> int:
        """Submit a task to the thread pool.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Task ID for retrieving results.
        """
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1

        self._task_queue.put((task_id, func, args, kwargs))
        return task_id

    def submit_many(self, tasks: List[tuple]) -> List[int]:
        """Submit multiple tasks.

        Args:
            tasks: List of (func, args, kwargs) tuples.

        Returns:
            List of task IDs.
        """
        task_ids = []
        for task in tasks:
            func = task[0]
            args = task[1] if len(task) > 1 else ()
            kwargs = task[2] if len(task) > 2 else {}
            task_ids.append(self.submit(func, *args, **kwargs))
        return task_ids

    def get_result(self, task_id: int, timeout: Optional[float] = None) -> Any:
        """Get the result of a submitted task.

        Args:
            task_id: Task ID returned by submit().
            timeout: Maximum time to wait for the result.

        Returns:
            Task result.

        Raises:
            Exception: If the task raised an exception.
            TimeoutError: If the result is not available within timeout.
        """
        start = time.time()
        while True:
            with self._lock:
                if task_id in self._results:
                    return self._results.pop(task_id)
                if task_id in self._errors:
                    raise self._errors.pop(task_id)

            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"Result for task {task_id} not available within {timeout}s")
            time.sleep(0.01)

    def get_all_results(self, task_ids: List[int], timeout: Optional[float] = None) -> List[Any]:
        """Get results for multiple tasks.

        Args:
            task_ids: List of task IDs.
            timeout: Maximum time to wait for each result.

        Returns:
            List of results in the same order as task_ids.
        """
        return [self.get_result(tid, timeout) for tid in task_ids]

    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Apply a function to each item in parallel.

        Args:
            func: Function to apply.
            items: List of items.

        Returns:
            List of results.
        """
        task_ids = [self.submit(func, item) for item in items]
        return self.get_all_results(task_ids)

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool.

        Args:
            wait: Whether to wait for pending tasks to complete.
        """
        self._shutdown = True
        if wait:
            self._task_queue.join()
        for worker in self._workers:
            worker.join(timeout=1.0)
        logger.info("Thread pool shut down.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


class BackgroundWorker:
    """A background thread that processes tasks from a queue."""

    def __init__(
        self,
        handler: Callable[[Any], Any],
        name: str = "BackgroundWorker",
        max_queue_size: int = 1000,
    ):
        """Initialize the background worker.

        Args:
            handler: Function to process each task.
            name: Worker thread name.
            max_queue_size: Maximum queue size.
        """
        self.handler = handler
        self.name = name
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._processed = 0
        self._errors = 0

    def start(self):
        """Start the background worker."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=self.name,
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Background worker '{self.name}' started.")

    def _run(self):
        """Main worker loop."""
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.handler(task)
                self._processed += 1
            except Exception as e:
                self._errors += 1
                logger.error(f"Background worker '{self.name}' error: {e}")
            finally:
                self._queue.task_done()

    def submit(self, task: Any):
        """Submit a task for background processing.

        Args:
            task: Task data to process.
        """
        self._queue.put(task)

    def stop(self, wait: bool = True):
        """Stop the background worker.

        Args:
            wait: Whether to wait for pending tasks.
        """
        self._running = False
        if wait and self._thread:
            self._queue.join()
            self._thread.join(timeout=5.0)
        logger.info(
            f"Background worker '{self.name}' stopped. "
            f"Processed: {self._processed}, Errors: {self._errors}"
        )

    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return self._queue.qsize()


class ThreadSafeQueue(Generic[T]):
    """A thread-safe queue with additional features like priority and batching."""

    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._count = 0

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None):
        """Add an item to the queue."""
        self._queue.put(item, block=block, timeout=timeout)
        with self._lock:
            self._count += 1

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        """Get an item from the queue."""
        item = self._queue.get(block=block, timeout=timeout)
        return item

    def task_done(self):
        """Mark a task as done."""
        self._queue.task_done()

    def get_batch(self, batch_size: int, timeout: float = 0.1) -> List[T]:
        """Get a batch of items from the queue.

        Args:
            batch_size: Maximum number of items to get.
            timeout: Timeout for each individual get operation.

        Returns:
            List of items (may be smaller than batch_size).
        """
        batch = []
        for _ in range(batch_size):
            try:
                item = self._queue.get(block=len(batch) == 0, timeout=timeout)
                batch.append(item)
            except queue.Empty:
                break
        return batch

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    def join(self):
        """Block until all items have been processed."""
        self._queue.join()


class AsyncBridge:
    """Bridge between synchronous and asynchronous code."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the async event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )
        self._thread.start()
        logger.info("AsyncBridge started.")

    def _run_loop(self):
        """Run the event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def call_async(self, coro) -> asyncio.Future:
        """Submit a coroutine to the event loop from a sync context.

        Args:
            coro: Coroutine to execute.

        Returns:
            Future that can be used to get the result.
        """
        if self._loop is None:
            raise RuntimeError("AsyncBridge not started. Call start() first.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def call_sync(self, coro, timeout: Optional[float] = None) -> Any:
        """Call an async function synchronously.

        Args:
            coro: Coroutine to execute.
            timeout: Timeout in seconds.

        Returns:
            Coroutine result.
        """
        future = self.call_async(coro)
        return future.result(timeout=timeout)

    def stop(self):
        """Stop the async event loop."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("AsyncBridge stopped.")
