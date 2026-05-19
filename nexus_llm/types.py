"""Nexus-LLM Type Definitions Module.

Provides type definitions using dataclasses and TypedDict for structured
data throughout the Nexus-LLM framework. All core data structures used
for messages, conversations, configurations, and model information are
defined here.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from nexus_llm.enums import ChatRole, DeviceType, MessageType, ModelType, PrecisionType, TaskType, TrainingStage


@dataclass
class Message:
    """Represents a single chat message.

    Attributes:
        role: The role of the message sender (system/user/assistant).
        content: The text content of the message.
        name: Optional name of the sender.
        message_type: Type of message (text/code/image).
        timestamp: When the message was created.
        metadata: Additional message metadata.
    """

    role: ChatRole
    content: str
    name: Optional[str] = None
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary.

        Returns:
            Dictionary representation of the message.
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "name": self.name,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary.

        Args:
            data: Dictionary with message data.

        Returns:
            A new Message instance.
        """
        return cls(
            role=ChatRole(data["role"]),
            content=data["content"],
            name=data.get("name"),
            message_type=MessageType(data.get("message_type", "text")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    """Represents a conversation with message history.

    Attributes:
        id: Unique conversation identifier.
        messages: List of messages in the conversation.
        model: Model name used for this conversation.
        system_prompt: The system prompt for the conversation.
        created_at: When the conversation was created.
        updated_at: When the conversation was last updated.
        metadata: Additional conversation metadata.
    """

    id: str = ""
    messages: List[Message] = field(default_factory=list)
    model: str = ""
    system_prompt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate a unique ID if none is provided."""
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())

    def add_message(self, role: ChatRole, content: str, **kwargs: Any) -> Message:
        """Add a message to the conversation.

        Args:
            role: The role of the message sender.
            content: The message content.
            **kwargs: Additional message attributes.

        Returns:
            The newly created Message.
        """
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_history(self, limit: Optional[int] = None) -> List[Message]:
        """Get conversation history.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            List of messages, optionally limited.
        """
        if limit:
            return self.messages[-limit:]
        return self.messages

    def clear_history(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the conversation to a dictionary.

        Returns:
            Dictionary representation of the conversation.
        """
        return {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "model": self.model,
            "system_prompt": self.system_prompt,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create a Conversation from a dictionary.

        Args:
            data: Dictionary with conversation data.

        Returns:
            A new Conversation instance.
        """
        return cls(
            id=data.get("id", ""),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling parameter.
        top_k: Top-k sampling parameter.
        repetition_penalty: Penalty for repeating tokens.
        length_penalty: Penalty based on output length.
        no_repeat_ngram_size: Size of n-grams that cannot be repeated.
        num_beams: Number of beams for beam search.
        do_sample: Whether to use sampling.
        seed: Random seed for reproducibility.
        stop_sequences: Sequences that stop generation.
    """

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    do_sample: bool = True
    seed: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "seed": self.seed,
            "stop_sequences": self.stop_sequences,
        }


@dataclass
class ModelInfo:
    """Metadata about a language model.

    Attributes:
        name: Model name or identifier.
        model_type: Type of the model.
        full_name: Full model name (e.g., HuggingFace repo ID).
        size: Model size description (e.g., '7B').
        parameter_count: Number of parameters.
        context_length: Maximum context length.
        device: Device the model is loaded on.
        precision: Precision of the model weights.
        description: Human-readable description.
        license: Model license.
        local_path: Local filesystem path if downloaded.
        is_loaded: Whether the model is currently loaded in memory.
        tags: Tags for categorizing the model.
    """

    name: str = ""
    model_type: ModelType = ModelType.CAUSAL_LM
    full_name: str = ""
    size: str = ""
    parameter_count: Optional[int] = None
    context_length: int = 2048
    device: DeviceType = DeviceType.AUTO
    precision: PrecisionType = PrecisionType.FP16
    description: str = ""
    license: str = ""
    local_path: Optional[str] = None
    is_loaded: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "full_name": self.full_name,
            "size": self.size,
            "parameter_count": self.parameter_count,
            "context_length": self.context_length,
            "device": self.device.value,
            "precision": self.precision.value,
            "description": self.description,
            "license": self.license,
            "local_path": self.local_path,
            "is_loaded": self.is_loaded,
            "tags": self.tags,
        }


@dataclass
class TrainingConfig:
    """Configuration for model training/fine-tuning.

    Attributes:
        model: Base model name or path.
        dataset: Path to training dataset.
        output_dir: Directory for saving checkpoints.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        lora_rank: LoRA rank (0 disables LoRA).
        lora_alpha: LoRA alpha parameter.
        lora_dropout: LoRA dropout rate.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_seq_length: Maximum sequence length.
        fp16: Whether to use FP16 mixed precision.
        bf16: Whether to use BF16 mixed precision.
        device: Device to train on.
        warmup_steps: Number of warmup steps.
        weight_decay: Weight decay coefficient.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        logging_steps: Log every N steps.
        validation_split: Fraction of data for validation.
        seed: Random seed.
        resume_from: Path to checkpoint to resume from.
    """

    model: str = ""
    dataset: str = ""
    output_dir: str = "./output"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = False
    device: str = "auto"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    validation_split: float = 0.1
    seed: int = 42
    resume_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model": self.model,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "device": self.device,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "validation_split": self.validation_split,
            "seed": self.seed,
            "resume_from": self.resume_from,
        }


@dataclass
class EvalConfig:
    """Configuration for model evaluation.

    Attributes:
        model: Model name or path to evaluate.
        benchmark: Benchmark dataset name.
        tasks: List of evaluation tasks.
        output_dir: Directory for saving results.
        device: Device to evaluate on.
        batch_size: Evaluation batch size.
        num_fewshot: Number of few-shot examples.
        limit: Limit number of evaluation examples.
        fp16: Use FP16 for evaluation.
        bf16: Use BF16 for evaluation.
        save_predictions: Whether to save model predictions.
    """

    model: str = ""
    benchmark: Optional[str] = None
    tasks: Optional[List[str]] = None
    output_dir: str = "./eval_results"
    device: str = "auto"
    batch_size: int = 8
    num_fewshot: int = 0
    limit: Optional[int] = None
    fp16: bool = False
    bf16: bool = False
    save_predictions: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "tasks": self.tasks,
            "output_dir": self.output_dir,
            "device": self.device,
            "batch_size": self.batch_size,
            "num_fewshot": self.num_fewshot,
            "limit": self.limit,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "save_predictions": self.save_predictions,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for inference benchmarking.

    Attributes:
        model: Model name or path to benchmark.
        device: Device to benchmark on.
        batch_sizes: List of batch sizes to test.
        seq_lengths: List of sequence lengths to test.
        warmup: Number of warmup iterations.
        iterations: Number of benchmark iterations.
        output_file: Path to save results JSON.
    """

    model: str = ""
    device: str = "auto"
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    warmup: int = 3
    iterations: int = 10
    output_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model": self.model,
            "device": self.device,
            "batch_sizes": self.batch_sizes,
            "seq_lengths": self.seq_lengths,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "output_file": self.output_file,
        }


@dataclass
class ServerConfig:
    """Configuration for the inference server.

    Attributes:
        host: Server host address.
        port: Server port number.
        model: Model name or path to serve.
        workers: Number of worker processes.
        device: Device for inference.
        api_key: API key for authentication.
        cors: Whether to enable CORS.
        reload: Whether to enable auto-reload.
        ssl_certfile: Path to SSL certificate file.
        ssl_keyfile: Path to SSL key file.
        log_level: Logging level.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    model: str = ""
    workers: int = 1
    device: str = "auto"
    api_key: Optional[str] = None
    cors: bool = False
    reload: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    log_level: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "host": self.host,
            "port": self.port,
            "model": self.model,
            "workers": self.workers,
            "device": self.device,
            "api_key": "***" if self.api_key else None,
            "cors": self.cors,
            "reload": self.reload,
            "ssl_certfile": self.ssl_certfile,
            "ssl_keyfile": self.ssl_keyfile,
            "log_level": self.log_level,
        }


@dataclass
class DownloadConfig:
    """Configuration for model downloading.

    Attributes:
        model_name: Name of the model to download.
        source: Download source (huggingface/local).
        output_dir: Directory to save the downloaded model.
        revision: Model revision/branch.
        quantize: Quantization format.
        token: API token for authentication.
        verify: Whether to verify file hashes.
    """

    model_name: str = ""
    source: str = "huggingface"
    output_dir: Optional[str] = None
    revision: Optional[str] = None
    quantize: Optional[str] = None
    token: Optional[str] = None
    verify: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model_name": self.model_name,
            "source": self.source,
            "output_dir": self.output_dir,
            "revision": self.revision,
            "quantize": self.quantize,
            "token": "***" if self.token else None,
            "verify": self.verify,
        }


@dataclass
class ChatConfig:
    """Configuration for chat sessions.

    Attributes:
        model: Model name or path to use.
        system_prompt: System prompt for the conversation.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        max_tokens: Maximum tokens to generate.
        device: Device for inference.
        use_history: Whether to maintain conversation history.
        single_prompt: If set, send this prompt and exit.
    """

    model: str = ""
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    device: Optional[str] = None
    use_history: bool = True
    single_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "use_history": self.use_history,
            "single_prompt": self.single_prompt,
        }


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        model: Model name.
        device: Device used.
        batch_size: Batch size tested.
        seq_length: Sequence length tested.
        avg_latency_ms: Average latency in milliseconds.
        min_latency_ms: Minimum latency in milliseconds.
        max_latency_ms: Maximum latency in milliseconds.
        throughput_samples_per_sec: Throughput in samples per second.
        throughput_tokens_per_sec: Throughput in tokens per second.
        memory_used_mb: GPU/CPU memory used in MB.
    """

    model: str = ""
    device: str = ""
    batch_size: int = 1
    seq_length: int = 128
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_used_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model": self.model,
            "device": self.device,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_used_mb": self.memory_used_mb,
        }


@dataclass
class EvalResult:
    """Result from an evaluation run.

    Attributes:
        model: Model evaluated.
        task: Evaluation task name.
        metric_name: Name of the metric.
        metric_value: Value of the metric.
        num_examples: Number of examples evaluated.
        config: Evaluation configuration used.
    """

    model: str = ""
    task: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    num_examples: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "model": self.model,
            "task": self.task,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "num_examples": self.num_examples,
            "config": self.config,
        }
