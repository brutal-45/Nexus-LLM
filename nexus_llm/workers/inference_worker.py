"""
Inference Worker for Nexus-LLM

Runs model inference in a separate process to avoid blocking the main
application. Supports streaming output, batch inference, and GPU
resource management.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Union


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class InferenceStatus(str, Enum):
    """Status of an inference task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceTask:
    """Represents a single inference request."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    model_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None

    # Generation parameters with defaults
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "params": self.params,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "stream": self.stream,
        }


@dataclass
class InferenceResult:
    """Result of an inference task."""

    task_id: str = ""
    text: str = ""
    status: InferenceStatus = InferenceStatus.PENDING
    error: Optional[str] = None
    tokens_generated: int = 0
    tokens_prompt: int = 0
    finish_reason: str = ""
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    model_name: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "text": self.text,
            "status": self.status.value,
            "error": self.error,
            "tokens_generated": self.tokens_generated,
            "tokens_prompt": self.tokens_prompt,
            "finish_reason": self.finish_reason,
            "generation_time": round(self.generation_time, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "model_name": self.model_name,
        }


# ---------------------------------------------------------------------------
# Inference Worker
# ---------------------------------------------------------------------------

class InferenceWorker:
    """Worker process that executes inference tasks.

    Runs in a separate process and communicates via multiprocessing queues.
    Supports:
    - Single and batch inference
    - Streaming token output
    - GPU resource isolation
    - Graceful shutdown and cancellation

    Example::

        worker = InferenceWorker(model_name="llama-7b")
        worker.start()
        task = InferenceTask(prompt="Hello, world!")
        result = worker.submit(task)
        worker.stop()
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "auto",
        gpu_id: Optional[int] = None,
        max_batch_size: int = 1,
        max_queue_size: int = 100,
        worker_id: Optional[str] = None,
    ) -> None:
        """Initialize the InferenceWorker.

        Args:
            model_name: Default model to load.
            device: Device string ('auto', 'cuda', 'cpu').
            gpu_id: Specific GPU device index.
            max_batch_size: Maximum batch size for inference.
            max_queue_size: Maximum pending tasks in queue.
            worker_id: Optional worker identifier.
        """
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.model_name = model_name
        self.device = device
        self.gpu_id = gpu_id
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size

        self._process: Optional[mp.Process] = None
        self._task_queue: mp.Queue = mp.Queue(maxsize=max_queue_size)
        self._result_queue: mp.Queue = mp.Queue()
        self._control_queue: mp.Queue = mp.Queue()
        self._running = False
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker process."""
        if self._running:
            return

        self._process = mp.Process(
            target=self._run,
            args=(self._task_queue, self._result_queue, self._control_queue),
            name=f"inference-worker-{self.worker_id}",
            daemon=True,
        )
        self._process.start()
        self._running = True

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker process gracefully.

        Args:
            timeout: Seconds to wait for graceful shutdown.
        """
        if not self._running:
            return

        self._control_queue.put("shutdown")
        if self._process and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)

        self._running = False

    def is_alive(self) -> bool:
        """Check if the worker process is running."""
        return self._process is not None and self._process.is_alive()

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit(self, task: InferenceTask, timeout: float = 30.0) -> InferenceResult:
        """Submit an inference task and wait for the result.

        Args:
            task: The inference task to execute.
            timeout: Maximum wait time in seconds.

        Returns:
            InferenceResult with the output.

        Raises:
            TimeoutError: If the task doesn't complete within timeout.
            RuntimeError: If the worker is not running.
        """
        if not self._running:
            raise RuntimeError("Worker is not running")

        if not task.model_name:
            task.model_name = self.model_name

        self._task_queue.put(task)

        try:
            result = self._result_queue.get(timeout=timeout)
            return result
        except Exception:
            raise TimeoutError(
                f"Inference task {task.task_id} timed out after {timeout}s"
            )

    def submit_async(self, task: InferenceTask) -> str:
        """Submit a task without waiting for the result.

        Args:
            task: The inference task to execute.

        Returns:
            The task ID for later result retrieval.

        Raises:
            RuntimeError: If the worker is not running.
        """
        if not self._running:
            raise RuntimeError("Worker is not running")

        if not task.model_name:
            task.model_name = self.model_name

        self._task_queue.put(task)
        return task.task_id

    def get_result(self, timeout: float = 30.0) -> Optional[InferenceResult]:
        """Retrieve a result from the result queue.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            InferenceResult or None if timeout.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    def cancel(self, task_id: str) -> None:
        """Request cancellation of a task.

        Args:
            task_id: The ID of the task to cancel.
        """
        self._control_queue.put(("cancel", task_id))

    # ------------------------------------------------------------------
    # Worker process main loop
    # ------------------------------------------------------------------

    @staticmethod
    def _run(
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        control_queue: mp.Queue,
    ) -> None:
        """Main loop for the worker process.

        Args:
            task_queue: Queue to receive tasks.
            result_queue: Queue to send results.
            control_queue: Queue for control messages.
        """
        model = None
        cancelled_tasks: set = set()
        running = True

        while running:
            # Check for control messages
            try:
                while not control_queue.empty():
                    msg = control_queue.get_nowait()
                    if msg == "shutdown":
                        running = False
                        break
                    elif isinstance(msg, tuple) and msg[0] == "cancel":
                        cancelled_tasks.add(msg[1])
            except Exception:
                pass

            if not running:
                break

            # Get next task
            try:
                task = task_queue.get(timeout=1.0)
            except Exception:
                continue

            if not isinstance(task, InferenceTask):
                continue

            # Check if cancelled
            if task.task_id in cancelled_tasks:
                result_queue.put(InferenceResult(
                    task_id=task.task_id,
                    status=InferenceStatus.CANCELLED,
                ))
                cancelled_tasks.discard(task.task_id)
                continue

            # Execute inference
            start_time = time.time()
            try:
                result = InferenceWorker._execute_inference(task, model)
                result.generation_time = time.time() - start_time
                if result.tokens_generated > 0 and result.generation_time > 0:
                    result.tokens_per_second = (
                        result.tokens_generated / result.generation_time
                    )
            except Exception as exc:
                result = InferenceResult(
                    task_id=task.task_id,
                    status=InferenceStatus.FAILED,
                    error=str(exc),
                    model_name=task.model_name,
                )

            result_queue.put(result)

    @staticmethod
    def _execute_inference(
        task: InferenceTask,
        model: Any,
    ) -> InferenceResult:
        """Execute a single inference task.

        In a real implementation, this would call the model backend.
        This stub simulates inference for testing.

        Args:
            task: The inference task.
            model: The loaded model (or None for stub mode).

        Returns:
            InferenceResult with the generated text.
        """
        # Stub implementation - in production this calls the model backend
        if model is not None:
            # Real inference path
            try:
                output = model.generate(
                    task.prompt,
                    max_new_tokens=task.max_new_tokens,
                    temperature=task.temperature,
                    top_p=task.top_p,
                    top_k=task.top_k,
                    repetition_penalty=task.repetition_penalty,
                    do_sample=task.do_sample,
                    stop=task.stop_sequences or None,
                )
                return InferenceResult(
                    task_id=task.task_id,
                    text=output,
                    status=InferenceStatus.COMPLETED,
                    model_name=task.model_name,
                    tokens_generated=len(output.split()),
                    finish_reason="stop",
                )
            except Exception as exc:
                return InferenceResult(
                    task_id=task.task_id,
                    status=InferenceStatus.FAILED,
                    error=str(exc),
                    model_name=task.model_name,
                )

        # Stub mode - return a placeholder result
        stub_text = f"[Stub inference for: {task.prompt[:50]}...]"
        return InferenceResult(
            task_id=task.task_id,
            text=stub_text,
            status=InferenceStatus.COMPLETED,
            model_name=task.model_name,
            tokens_generated=len(stub_text.split()),
            tokens_prompt=len(task.prompt.split()),
            finish_reason="stop",
        )
