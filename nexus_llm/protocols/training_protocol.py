"""Nexus-LLM Training Protocol.

Defines the request/response structures and protocol handler for
model training and fine-tuning operations.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TrainingStatus(Enum):
    """Status of a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingRequest:
    """Request to start a training job.

    Attributes:
        model: Base model name or path.
        dataset: Path to training dataset.
        output_dir: Directory for checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        lora_rank: LoRA rank (0 disables LoRA).
        lora_alpha: LoRA alpha parameter.
        validation_split: Fraction of data for validation.
        max_seq_length: Maximum sequence length.
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        job_id: Unique job identifier.
        metadata: Additional request metadata.
    """

    model: str = ""
    dataset: str = ""
    output_dir: str = "./output"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    lora_rank: int = 8
    lora_alpha: int = 16
    validation_split: float = 0.1
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = False
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "validation_split": self.validation_split,
            "max_seq_length": self.max_seq_length,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "job_id": self.job_id,
        }


@dataclass
class TrainingMetrics:
    """Metrics from a training run.

    Attributes:
        step: Current training step.
        epoch: Current epoch.
        train_loss: Training loss.
        val_loss: Validation loss.
        learning_rate: Current learning rate.
        tokens_seen: Total tokens processed.
    """

    step: int = 0
    epoch: float = 0.0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    tokens_seen: int = 0


@dataclass
class TrainingResponse:
    """Response from a training job.

    Attributes:
        job_id: Job identifier.
        status: Current job status.
        metrics: Latest training metrics.
        output_model_path: Path to the trained model (if completed).
        error: Error message (if failed).
        created: Job creation timestamp.
        updated: Last update timestamp.
    """

    job_id: str = ""
    status: TrainingStatus = TrainingStatus.PENDING
    metrics: Optional[TrainingMetrics] = None
    output_model_path: Optional[str] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "metrics": self.metrics.__dict__ if self.metrics else None,
            "output_model_path": self.output_model_path,
            "error": self.error,
            "created": self.created,
            "updated": self.updated,
        }


class TrainingProtocol(ABC):
    """Abstract protocol handler for training operations.

    Concrete implementations must provide methods for starting,
    monitoring, and controlling training jobs.
    """

    @abstractmethod
    def start_training(self, request: TrainingRequest) -> TrainingResponse:
        """Start a training job.

        Args:
            request: The training request.

        Returns:
            A TrainingResponse with job details.
        """
        ...

    @abstractmethod
    def get_status(self, job_id: str) -> TrainingResponse:
        """Get the status of a training job.

        Args:
            job_id: The job identifier.

        Returns:
            A TrainingResponse with current status.
        """
        ...

    @abstractmethod
    def cancel_training(self, job_id: str) -> TrainingResponse:
        """Cancel a running training job.

        Args:
            job_id: The job identifier.

        Returns:
            A TrainingResponse confirming cancellation.
        """
        ...

    def validate_request(self, request: TrainingRequest) -> List[str]:
        """Validate a training request.

        Args:
            request: The request to validate.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        if not request.model:
            errors.append("Model not specified")
        if not request.dataset:
            errors.append("Dataset not specified")
        if request.epochs <= 0:
            errors.append("Epochs must be positive")
        if request.batch_size <= 0:
            errors.append("Batch size must be positive")
        if request.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        return errors
