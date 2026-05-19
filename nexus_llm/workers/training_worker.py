"""
Training Worker for Nexus-LLM

Runs model training in a separate process, enabling non-blocking
fine-tuning with progress tracking, checkpointing, and early stopping.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TrainingStatus(str, Enum):
    """Status of a training task."""
    PENDING = "pending"
    LOADING_DATA = "loading_data"
    INITIALIZING = "initializing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TrainingMetrics:
    """Metrics snapshot during training."""

    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    epoch_progress: float = 0.0  # 0.0 to 1.0
    overall_progress: float = 0.0  # 0.0 to 1.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    samples_per_second: float = 0.0
    eta_seconds: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "total_steps": self.total_steps,
            "train_loss": round(self.train_loss, 6),
            "eval_loss": round(self.eval_loss, 6) if self.eval_loss is not None else None,
            "learning_rate": f"{self.learning_rate:.2e}",
            "epoch_progress": round(self.epoch_progress, 4),
            "overall_progress": round(self.overall_progress, 4),
            "gpu_memory_used_gb": round(self.gpu_memory_used_gb, 2),
            "gpu_memory_total_gb": round(self.gpu_memory_total_gb, 2),
            "samples_per_second": round(self.samples_per_second, 2),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds else None,
        }


@dataclass
class TrainingTask:
    """Represents a training job configuration."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    dataset_path: str = ""
    output_dir: str = "./output"
    preset: str = "standard"

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False

    # Data parameters
    max_seq_length: int = 2048
    train_split: str = "train"
    eval_split: str = "validation"
    eval_steps: int = 500

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    save_strategy: str = "steps"

    # LoRA parameters (optional)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Resume
    resume_from_checkpoint: Optional[str] = None

    # Callbacks
    progress_callback: Optional[Callable[[TrainingMetrics], None]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "preset": self.preset,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r if self.use_lora else None,
        }


@dataclass
class TrainingResult:
    """Result of a completed training job."""

    task_id: str = ""
    status: TrainingStatus = TrainingStatus.PENDING
    error: Optional[str] = None
    output_dir: str = ""
    best_model_path: Optional[str] = None
    final_train_loss: Optional[float] = None
    final_eval_loss: Optional[float] = None
    total_epochs: int = 0
    total_steps: int = 0
    total_training_time: float = 0.0
    best_eval_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    checkpoints: List[str] = field(default_factory=list)
    model_name: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "error": self.error,
            "output_dir": self.output_dir,
            "best_model_path": self.best_model_path,
            "final_train_loss": round(self.final_train_loss, 6) if self.final_train_loss else None,
            "final_eval_loss": round(self.final_eval_loss, 6) if self.final_eval_loss else None,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "total_training_time": round(self.total_training_time, 2),
            "best_eval_loss": round(self.best_eval_loss, 6) if self.best_eval_loss else None,
            "best_epoch": self.best_epoch,
            "checkpoints": self.checkpoints,
            "model_name": self.model_name,
        }


# ---------------------------------------------------------------------------
# Training Worker
# ---------------------------------------------------------------------------

class TrainingWorker:
    """Worker process that executes training tasks.

    Runs training in a separate process to keep the main application
    responsive. Supports progress tracking via callbacks, checkpointing,
    early stopping, and graceful cancellation.

    Example::

        worker = TrainingWorker()
        worker.start()
        task = TrainingTask(
            model_name="llama-7b",
            dataset_path="./data/train.jsonl",
            num_train_epochs=3,
        )
        result = worker.submit(task)
        worker.stop()
    """

    def __init__(
        self,
        gpu_id: Optional[int] = None,
        max_queue_size: int = 10,
        worker_id: Optional[str] = None,
    ) -> None:
        """Initialize the TrainingWorker.

        Args:
            gpu_id: Specific GPU device to use.
            max_queue_size: Maximum pending tasks.
            worker_id: Optional worker identifier.
        """
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.gpu_id = gpu_id

        self._process: Optional[mp.Process] = None
        self._task_queue: mp.Queue = mp.Queue(maxsize=max_queue_size)
        self._result_queue: mp.Queue = mp.Queue()
        self._metrics_queue: mp.Queue = mp.Queue()
        self._control_queue: mp.Queue = mp.Queue()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the training worker process."""
        if self._running:
            return

        self._process = mp.Process(
            target=self._run,
            args=(
                self._task_queue,
                self._result_queue,
                self._metrics_queue,
                self._control_queue,
            ),
            name=f"training-worker-{self.worker_id}",
            daemon=True,
        )
        self._process.start()
        self._running = True

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker process.

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
                self._process.join(timeout=10.0)

        self._running = False

    def is_alive(self) -> bool:
        """Check if the worker is running."""
        return self._process is not None and self._process.is_alive()

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def submit(
        self,
        task: TrainingTask,
        timeout: Optional[float] = None,
    ) -> TrainingResult:
        """Submit a training task and wait for completion.

        Training tasks are long-running; use timeout carefully.
        For long training, prefer submit_async() + poll_results().

        Args:
            task: The training task configuration.
            timeout: Maximum wait time. None means wait indefinitely.

        Returns:
            TrainingResult with final training output.

        Raises:
            RuntimeError: If the worker is not running.
            TimeoutError: If the timeout is exceeded.
        """
        if not self._running:
            raise RuntimeError("Worker is not running")

        self._task_queue.put(task)

        try:
            result = self._result_queue.get(timeout=timeout)
            return result
        except Exception:
            raise TimeoutError("Training task timed out")

    def submit_async(self, task: TrainingTask) -> str:
        """Submit a training task without waiting.

        Args:
            task: The training task configuration.

        Returns:
            The task ID for later result retrieval.
        """
        if not self._running:
            raise RuntimeError("Worker is not running")

        self._task_queue.put(task)
        return task.task_id

    def get_metrics(self) -> Optional[TrainingMetrics]:
        """Retrieve the latest training metrics.

        Returns:
            TrainingMetrics if available, None otherwise.
        """
        try:
            return self._metrics_queue.get_nowait()
        except Exception:
            return None

    def get_result(self, timeout: float = 5.0) -> Optional[TrainingResult]:
        """Retrieve a training result if available.

        Args:
            timeout: Wait time in seconds.

        Returns:
            TrainingResult or None.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    def cancel(self, task_id: str) -> None:
        """Request cancellation of a training task.

        Args:
            task_id: The ID of the training task.
        """
        self._control_queue.put(("cancel", task_id))

    def pause(self, task_id: str) -> None:
        """Pause a running training task.

        Args:
            task_id: The ID of the training task.
        """
        self._control_queue.put(("pause", task_id))

    def resume(self, task_id: str) -> None:
        """Resume a paused training task.

        Args:
            task_id: The ID of the training task.
        """
        self._control_queue.put(("resume", task_id))

    # ------------------------------------------------------------------
    # Worker process main loop
    # ------------------------------------------------------------------

    @staticmethod
    def _run(
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        metrics_queue: mp.Queue,
        control_queue: mp.Queue,
    ) -> None:
        """Main loop for the training worker process."""
        cancelled_tasks: set = set()
        paused_tasks: set = set()
        running = True

        while running:
            # Process control messages
            try:
                while not control_queue.empty():
                    msg = control_queue.get_nowait()
                    if msg == "shutdown":
                        running = False
                        break
                    elif isinstance(msg, tuple):
                        action, task_id = msg
                        if action == "cancel":
                            cancelled_tasks.add(task_id)
                        elif action == "pause":
                            paused_tasks.add(task_id)
                        elif action == "resume":
                            paused_tasks.discard(task_id)
            except Exception:
                pass

            if not running:
                break

            # Get next task
            try:
                task = task_queue.get(timeout=1.0)
            except Exception:
                continue

            if not isinstance(task, TrainingTask):
                continue

            # Check cancellation
            if task.task_id in cancelled_tasks:
                result_queue.put(TrainingResult(
                    task_id=task.task_id,
                    status=TrainingStatus.CANCELLED,
                    model_name=task.model_name,
                ))
                cancelled_tasks.discard(task.task_id)
                continue

            # Execute training
            start_time = time.time()
            try:
                result = TrainingWorker._execute_training(
                    task, metrics_queue, cancelled_tasks, paused_tasks
                )
                result.total_training_time = time.time() - start_time
            except Exception as exc:
                result = TrainingResult(
                    task_id=task.task_id,
                    status=TrainingStatus.FAILED,
                    error=str(exc),
                    model_name=task.model_name,
                )

            result_queue.put(result)

    @staticmethod
    def _execute_training(
        task: TrainingTask,
        metrics_queue: mp.Queue,
        cancelled_tasks: set,
        paused_tasks: set,
    ) -> TrainingResult:
        """Execute a training task.

        In production, this would use the HuggingFace Trainer or a custom
        training loop. This stub simulates training progress.

        Args:
            task: Training configuration.
            metrics_queue: Queue for sending progress metrics.
            cancelled_tasks: Set of cancelled task IDs.
            paused_tasks: Set of paused task IDs.

        Returns:
            TrainingResult with final output.
        """
        # Stub: simulate training epochs
        total_steps = 1000  # Simulated
        best_eval_loss = float("inf")
        best_epoch = 0
        checkpoints: List[str] = []

        for epoch in range(1, task.num_train_epochs + 1):
            # Check cancellation
            if task.task_id in cancelled_tasks:
                return TrainingResult(
                    task_id=task.task_id,
                    status=TrainingStatus.CANCELLED,
                    model_name=task.model_name,
                    output_dir=task.output_dir,
                    total_epochs=epoch - 1,
                    checkpoints=checkpoints,
                )

            # Simulate epoch steps
            for step in range(1, total_steps + 1):
                # Check pause
                while task.task_id in paused_tasks:
                    time.sleep(1.0)
                    if task.task_id in cancelled_tasks:
                        return TrainingResult(
                            task_id=task.task_id,
                            status=TrainingStatus.CANCELLED,
                            model_name=task.model_name,
                        )

                # Send metrics (throttled)
                if step % 100 == 0:
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        step=step,
                        total_steps=total_steps * task.num_train_epochs,
                        train_loss=2.5 / (epoch + step * 0.001),
                        eval_loss=2.6 / (epoch + step * 0.001) if step % 500 == 0 else None,
                        learning_rate=task.learning_rate * (1 - step / total_steps),
                        epoch_progress=step / total_steps,
                        overall_progress=(epoch - 1 + step / total_steps) / task.num_train_epochs,
                        samples_per_second=45.0,
                    )
                    try:
                        metrics_queue.put_nowait(metrics)
                    except Exception:
                        pass

            # Simulate checkpoint save
            ckpt_path = f"{task.output_dir}/checkpoint-epoch-{epoch}"
            checkpoints.append(ckpt_path)

            # Simulate eval
            eval_loss = 2.0 / (epoch + 1)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch

        return TrainingResult(
            task_id=task.task_id,
            status=TrainingStatus.COMPLETED,
            model_name=task.model_name,
            output_dir=task.output_dir,
            best_model_path=f"{task.output_dir}/best_model",
            final_train_loss=0.5 / task.num_train_epochs,
            final_eval_loss=best_eval_loss,
            total_epochs=task.num_train_epochs,
            total_steps=total_steps * task.num_train_epochs,
            best_eval_loss=best_eval_loss,
            best_epoch=best_epoch,
            checkpoints=checkpoints,
        )
