"""Nexus-LLM Enums Module.

Provides enumeration types used throughout the Nexus-LLM framework
for type-safe categorization of model types, devices, precision,
tasks, chat roles, and training stages.
"""

from enum import Enum


class ModelType(str, Enum):
    """Types of language models supported by Nexus-LLM."""

    CAUSAL_LM = "causal_lm"
    SEQ2SEQ_LM = "seq2seq_lm"
    MASKED_LM = "masked_lm"
    INSTRUCTION = "instruction"
    CHAT = "chat"
    CODE = "code"
    EMBEDDING = "embedding"
    RLHF = "rlhf"
    DPO = "dpo"
    MULTIMODAL = "multimodal"

    def __str__(self) -> str:
        return self.value

    @property
    def description(self) -> str:
        """Get a human-readable description of the model type."""
        descriptions = {
            ModelType.CAUSAL_LM: "Causal Language Model (autoregressive)",
            ModelType.SEQ2SEQ_LM: "Sequence-to-Sequence Language Model",
            ModelType.MASKED_LM: "Masked Language Model (bidirectional)",
            ModelType.INSTRUCTION: "Instruction-tuned Model",
            ModelType.CHAT: "Chat-tuned Model",
            ModelType.CODE: "Code Generation Model",
            ModelType.EMBEDDING: "Embedding Model",
            ModelType.RLHF: "RLHF-trained Model",
            ModelType.DPO: "DPO-trained Model",
            ModelType.MULTIMODAL: "Multimodal Model",
        }
        return descriptions.get(self, self.value)


class DeviceType(str, Enum):
    """Device types for model inference and training."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    TPU = "tpu"
    XPU = "xpu"

    def __str__(self) -> str:
        return self.value

    @property
    def description(self) -> str:
        """Get a human-readable description of the device type."""
        descriptions = {
            DeviceType.AUTO: "Automatic device selection",
            DeviceType.CPU: "CPU (Central Processing Unit)",
            DeviceType.CUDA: "NVIDIA GPU (CUDA)",
            DeviceType.MPS: "Apple Metal Performance Shaders",
            DeviceType.TPU: "Google TPU",
            DeviceType.XPU: "Intel XPU",
        }
        return descriptions.get(self, self.value)

    @classmethod
    def detect(cls) -> "DeviceType":
        """Detect the best available device.

        Returns:
            The detected DeviceType.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return cls.CUDA
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return cls.MPS
        except ImportError:
            pass
        return cls.CPU


class PrecisionType(str, Enum):
    """Precision types for model weights and computation."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ_INT4 = "gptq_int4"
    GPTQ_INT8 = "gptq_int8"
    AWQ_INT4 = "awq_int4"
    MIXED = "mixed"

    def __str__(self) -> str:
        return self.value

    @property
    def bits(self) -> int:
        """Get the number of bits per parameter."""
        bits_map = {
            PrecisionType.FP32: 32,
            PrecisionType.FP16: 16,
            PrecisionType.BF16: 16,
            PrecisionType.INT8: 8,
            PrecisionType.INT4: 4,
            PrecisionType.GPTQ_INT4: 4,
            PrecisionType.GPTQ_INT8: 8,
            PrecisionType.AWQ_INT4: 4,
            PrecisionType.MIXED: 16,
        }
        return bits_map.get(self, 32)

    @property
    def is_quantized(self) -> bool:
        """Check if this precision type is quantized."""
        return self in (
            PrecisionType.INT8,
            PrecisionType.INT4,
            PrecisionType.GPTQ_INT4,
            PrecisionType.GPTQ_INT8,
            PrecisionType.AWQ_INT4,
        )


class TaskType(str, Enum):
    """Types of tasks that can be performed with LLMs."""

    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    CHAT = "chat"
    INSTRUCTION_FOLLOWING = "instruction_following"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    BENCHMARKING = "benchmarking"

    def __str__(self) -> str:
        return self.value


class ChatRole(str, Enum):
    """Roles in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

    def __str__(self) -> str:
        return self.value


class MessageType(str, Enum):
    """Types of message content."""

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    FILE = "file"
    ERROR = "error"
    SYSTEM = "system"

    def __str__(self) -> str:
        return self.value


class TrainingStage(str, Enum):
    """Stages of the training process."""

    INITIALIZING = "initializing"
    PREPARING_DATA = "preparing_data"
    LOADING_MODEL = "loading_model"
    CONFIGURING = "configuring"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SAVING_CHECKPOINT = "saving_checkpoint"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    def __str__(self) -> str:
        return self.value

    @property
    def is_active(self) -> bool:
        """Check if this stage represents an active training state."""
        return self in (
            TrainingStage.INITIALIZING,
            TrainingStage.PREPARING_DATA,
            TrainingStage.LOADING_MODEL,
            TrainingStage.CONFIGURING,
            TrainingStage.TRAINING,
            TrainingStage.EVALUATING,
            TrainingStage.SAVING_CHECKPOINT,
        )

    @property
    def is_terminal(self) -> bool:
        """Check if this stage represents a terminal state."""
        return self in (
            TrainingStage.COMPLETED,
            TrainingStage.FAILED,
            TrainingStage.INTERRUPTED,
        )
