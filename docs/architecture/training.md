# Training Architecture

Deep dive into the training pipeline architecture: training loop, data processing pipeline, optimization strategies, and distributed training coordination.

---

## Overview

The Nexus-LLM training system is built on top of HuggingFace Transformers, PEFT, and TRL, providing a unified interface for full fine-tuning, LoRA, and QLoRA. The architecture separates the training loop, data pipeline, and optimization into distinct, composable components.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Training Architecture                         │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Training Orchestrator                     │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │  │
│  │  │  Config   │ │  Job     │ │ Callback │ │  Checkpoint   │ │  │
│  │  │  Resolver │ │  Manager │ │ Manager  │ │  Manager      │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                               │                                    │
│  ┌────────────────────────────┼────────────────────────────────┐ │
│  │                     Training Loop                            │ │
│  │  ┌──────────┐ ┌──────────┐ │ ┌──────────┐ ┌──────────────┐ │ │
│  │  │  Forward │ │  Loss    │ │ │ Backward │ │  Optimizer   │ │ │
│  │  │  Pass    │ │  Compute │ │ │ Pass     │ │  Step        │ │ │
│  │  └──────────┘ └──────────┘ │ └──────────┘ └──────────────┘ │ │
│  │                             │                                │ │
│  │  ┌──────────┐ ┌──────────┐ │ ┌──────────────────────────────┐│ │
│  │  │  LR      │ │ Gradient │ │ │  Gradient Accumulation      ││ │
│  │  │ Scheduler│ │ Clipping │ │ │  Buffer                     ││ │
│  │  └──────────┘ └──────────┘ │ └──────────────────────────────┘│ │
│  └────────────────────────────┼────────────────────────────────┘ │
│                               │                                    │
│  ┌────────────────────────────┼────────────────────────────────┐ │
│  │                     Data Pipeline                            │ │
│  │  ┌──────────┐ ┌──────────┐ │ ┌──────────┐ ┌──────────────┐ │ │
│  │  │  Loader  │ │ Processor│ │ │ Tokenizer│ │  Collator    │ │ │
│  │  └──────────┘ └──────────┘ │ └──────────┘ └──────────────┘ │ │
│  └────────────────────────────┼────────────────────────────────┘ │
│                               │                                    │
│  ┌────────────────────────────┼────────────────────────────────┐ │
│  │                   Optimization Layer                         │ │
│  │  ┌──────────┐ ┌──────────┐ │ ┌──────────┐ ┌──────────────┐ │ │
│  │  │  Mixed   │ │  DeepSpeed│ │ │  FSDP    │ │  Gradient    │ │ │
│  │  │  Precision│ │  ZeRO    │ │ │  Sharding│ │  Checkpoint  │ │ │
│  │  └──────────┘ └──────────┘ │ └──────────┘ └──────────────┘ │ │
│  └────────────────────────────┼────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Training Orchestrator

The orchestrator is the top-level component that coordinates all aspects of a training run.

### Job Lifecycle

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│ Queued  │───▶│Preparing│───▶│ Running │───▶│Completing│───▶│Completed │
└─────────┘    └─────────┘    └────┬────┘    └──────────┘    └──────────┘
                                   │
                                   ├── Error ──▶ Failed
                                   └── Cancel ─▶ Cancelled
```

```python
class TrainingOrchestrator:
    """Coordinates the entire training process."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.job_manager = JobManager()
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        self.callback_manager = CallbackManager()

    async def submit_job(self, request: TrainingRequest) -> TrainingJob:
        """Submit a new training job."""
        # Validate request
        self._validate_request(request)

        # Resolve configuration (merge defaults, user config, and request)
        resolved_config = self._resolve_config(request)

        # Create job
        job = TrainingJob(
            id=f"train_{uuid4().hex[:8]}",
            config=resolved_config,
            status="queued",
            created_at=datetime.utcnow(),
        )

        # Enqueue
        await self.job_manager.enqueue(job)
        return job

    async def run_job(self, job: TrainingJob) -> TrainingResult:
        """Execute a training job."""
        job.status = "preparing"
        self._notify_status(job)

        try:
            # Prepare data
            train_dataset, eval_dataset = await self._prepare_data(job.config)

            # Prepare model
            model, tokenizer = await self._prepare_model(job.config)

            # Prepare trainer
            trainer = self._create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=job.config,
            )

            # Run training
            job.status = "running"
            self._notify_status(job)

            result = trainer.train()

            # Finalize
            job.status = "completed"
            job.result = result
            self._notify_status(job)

            return result

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self._notify_status(job)
            raise
```

### Config Resolution

Configuration is resolved by merging multiple sources:

```python
class ConfigResolver:
    """Merges training configuration from multiple sources."""

    def resolve(self, request: TrainingRequest) -> TrainingConfig:
        """
        Priority (highest to lowest):
        1. Request parameters (from API/CLI)
        2. User config file
        3. Default config file
        4. Built-in defaults
        """
        config = self.built_in_defaults.copy()

        # Layer: default config file
        if self.default_config:
            config = self._deep_merge(config, self.default_config)

        # Layer: user config file
        if self.user_config:
            config = self._deep_merge(config, self.user_config)

        # Layer: request parameters
        if request.hyperparameters:
            config = self._deep_merge(config, request.hyperparameters)

        # Validate
        self._validate(config)
        return TrainingConfig(**config)
```

---

## Training Loop

### Core Loop

```python
class TrainingLoop:
    """The core training loop with gradient accumulation and mixed precision."""

    def __init__(self, model, optimizer, scheduler, config: TrainingConfig):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.grad_buffer = GradBuffer(model)

    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            # Forward pass
            with self._autocast():
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward pass
            self._backward(loss)

            # Accumulate gradients
            self.grad_buffer.accumulate()

            # Optimizer step (every N steps)
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Callbacks
                self._on_step_end(step, loss.item() * self.config.gradient_accumulation_steps)

            total_loss += loss.item()

        return total_loss / len(dataloader)
```

### Mixed Precision

```python
class MixedPrecisionManager:
    """Manages FP16, BF16, and FP32 training."""

    def __init__(self, config: PrecisionConfig):
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        elif self.bf16:
            self.scaler = None  # BF16 doesn't need a scaler

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                yield
        elif self.bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                yield
        else:
            yield

    def backward(self, loss, optimizer):
        """Perform backward pass with appropriate precision handling."""
        if self.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
```

---

## Data Pipeline

### Architecture

```
Raw Data Source
      │
      ▼
┌───────────┐
│  Loader    │  Load from JSONL, CSV, HuggingFace, etc.
└─────┬─────┘
      │
      ▼
┌───────────┐
│  Validator │  Check schema, filter bad examples
└─────┬─────┘
      │
      ▼
┌───────────┐
│  Splitter  │  Train/eval split
└─────┬─────┘
      │
      ▼
┌───────────┐
│ Preprocessor│  Clean text, normalize, deduplicate
└─────┬─────┘
      │
      ▼
┌───────────┐
│  Tokenizer │  Convert text → token IDs
└─────┬─────┘
      │
      ▼
┌───────────┐
│  Collator  │  Batch and pad sequences
└─────┬─────┘
      │
      ▼
┌───────────┐
│  DataLoader│  Feed batches to training loop
└───────────┘
```

### Data Loader

```python
class TrainingDataLoader:
    """Loads and prepares training data from multiple sources."""

    def load(self, source: DataSource) -> Dataset:
        """Load data from the specified source."""
        if source.type == "jsonl":
            return self._load_jsonl(source.path)
        elif source.type == "csv":
            return self._load_csv(source.path)
        elif source.type == "huggingface":
            return self._load_huggingface(source.dataset_name, source.split)
        elif source.type == "parquet":
            return self._load_parquet(source.path)
        else:
            raise ValueError(f"Unsupported data source type: {source.type}")

    def _load_jsonl(self, path: str) -> Dataset:
        """Load from a JSONL file."""
        records = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    records.append(self._normalize_record(record))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")

        return Dataset.from_list(records)
```

### Data Collator

```python
class DataCollator:
    """Prepares batches for training with padding and masking."""

    def __init__(self, tokenizer, max_length: int = 2048, padding: str = "longest"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __call__(self, features: list[dict]) -> dict:
        """Collate a list of examples into a batch."""
        # Extract sequences
        input_ids = [f["input_ids"][:self.max_length] for f in features]

        # Pad to uniform length
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (same as input_ids, with padding tokens set to -100)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask out the prompt portion for instruction-tuned models
        if "prompt_lengths" in features[0]:
            for i, prompt_len in enumerate(f["prompt_lengths"] for f in features):
                labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch
```

---

## Optimization Layer

### Gradient Checkpointing

Reduces memory by recomputing activations during the backward pass:

```python
class GradientCheckpointing:
    """Enables gradient checkpointing to trade compute for memory."""

    @staticmethod
    def enable(model, config: CheckpointConfig):
        """
        Memory savings:
        - 7B model: ~28GB → ~16GB (43% reduction)
        - 13B model: ~52GB → ~30GB (42% reduction)

        Speed cost: ~25% slower due to recomputation
        """
        if config.enabled:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                }
            )
```

### DeepSpeed Integration

```python
class DeepSpeedManager:
    """Manages DeepSpeed ZeRO optimization stages."""

    STAGE_CONFIGS = {
        1: {  # Shard optimizer states across GPUs
            "zero_optimization": {"stage": 1},
        },
        2: {  # Shard optimizer + gradient states
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"},
            },
        },
        3: {  # Shard optimizer + gradient + parameters
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
        },
    }

    def get_config(self, stage: int, custom: dict = None) -> dict:
        """Generate DeepSpeed configuration for a given stage."""
        config = self.STAGE_CONFIGS[stage].copy()
        if custom:
            config = self._deep_merge(config, custom)
        return config
```

### FSDP (Fully Sharded Data Parallel)

```python
class FSDPConfig:
    """Configuration for PyTorch FSDP training."""

    @staticmethod
    def get_wrapping_policy(model_name: str):
        """Get the appropriate FSDP wrapping policy for a model."""
        from torch.distributed.fsdp import (
            MixedPrecision,
            ShardingStrategy,
        )

        policies = {
            "llama": LlamaFSDPPolicy,
            "mistral": MistralFSDPPolicy,
            "phi": PhiFSDPPolicy,
            "default": DefaultFSDPPolicy,
        }

        return policies.get(model_name, policies["default"])

    @staticmethod
    def get_mixed_precision(dtype: str = "bf16"):
        """Configure mixed precision for FSDP."""
        if dtype == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
```

---

## Callback System

Callbacks hook into the training loop for logging, checkpointing, and early stopping.

### Built-in Callbacks

```python
class LoggingCallback(TrainerCallback):
    """Logs training metrics to console, file, and external services."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
        }

        # Console
        logger.info(
            f"Step {state.global_step} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"LR: {metrics['learning_rate']:.2e}"
        )

        # File
        self.metrics_logger.log(metrics)

        # External (TensorBoard, W&B, MLflow)
        if "tensorboard" in args.report_to:
            self.tb_writer.add_scalar("train/loss", metrics["loss"], state.global_step)


class EarlyStoppingCallback(TrainerCallback):
    """Stops training when a metric stops improving."""

    def __init__(self, patience: int = 3, threshold: float = 0.01):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = float("inf")
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_metric = metrics.get("eval_loss", float("inf"))

        if current_metric < self.best_metric - self.threshold:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                control.should_training_stop = True
                logger.info(f"Early stopping triggered after {self.patience} evaluations without improvement")


class WebSocketProgressCallback(TrainerCallback):
    """Streams training progress via WebSocket for real-time monitoring."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.ws_manager.broadcast({
            "type": "training.metric",
            "data": {
                "job_id": self.job_id,
                "step": state.global_step,
                "epoch": state.epoch,
                "metrics": logs,
            }
        })
```

### Custom Callbacks

```python
# Users can define custom callbacks in Python
class CustomCallback(TrainerCallback):
    """Example custom callback."""

    def on_step_end(self, args, state, control, **kwargs):
        # Do something after each step
        if state.global_step % 100 == 0:
            # Run custom evaluation
            pass

    def on_save(self, args, state, control, **kwargs):
        # Do something when a checkpoint is saved
        pass
```

---

## Checkpoint Management

### Checkpoint Strategy

```
checkpoints/
└── my_training/
    ├── checkpoint-500/
    │   ├── adapter_config.json
    │   ├── adapter_model.bin
    │   ├── trainer_state.json
    │   ├── training_args.bin
    │   ├── optimizer.pt
    │   ├── rng_state.pth
    │   └── scheduler.pt
    ├── checkpoint-1000/
    │   └── ...
    ├── checkpoint-1500/
    │   └── ...
    └── best_checkpoint/
        └── ...  → symlink to best performing checkpoint
```

### Checkpoint Manager

```python
class CheckpointManager:
    """Manages training checkpoints with rotation and best-model tracking."""

    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.checkpoints: list[Path] = []

    def save(self, trainer, state, metrics: dict = None):
        """Save a checkpoint."""
        checkpoint_dir = Path(self.output_dir) / f"checkpoint-{state.global_step}"
        trainer.save_model(str(checkpoint_dir))
        trainer.save_state()

        self.checkpoints.append(checkpoint_dir)

        # Track best model
        if metrics and self._is_best(metrics):
            best_dir = Path(self.output_dir) / "best_checkpoint"
            if best_dir.is_symlink():
                best_dir.unlink()
            best_dir.symlink_to(checkpoint_dir)

        # Rotate old checkpoints
        self._rotate()

    def _rotate(self):
        """Remove old checkpoints to stay within save_total_limit."""
        while len(self.checkpoints) > self.save_total_limit:
            oldest = self.checkpoints.pop(0)
            if oldest.is_symlink():
                continue  # Don't delete best model symlink
            shutil.rmtree(oldest)
```

---

## Memory Estimation

### Memory Budget Calculator

```python
class MemoryEstimator:
    """Estimates GPU memory requirements for training configurations."""

    # Approximate memory per parameter (bytes)
    PARAM_MEMORY = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    def estimate(self, config: TrainingConfig) -> MemoryEstimate:
        """Estimate total GPU memory needed."""
        model_params = self._count_parameters(config.model)
        param_bytes = self.PARAM_MEMORY[config.dtype]

        # Model weights
        model_memory = model_params * param_bytes

        # Optimizer states (AdamW = 2x model size for FP32 moments)
        if config.method == "lora":
            trainable_params = self._count_lora_parameters(config)
            optimizer_memory = trainable_params * 8  # FP32 moments
        else:
            optimizer_memory = model_memory * 2

        # Gradients
        if config.method == "lora":
            gradient_memory = trainable_params * param_bytes
        else:
            gradient_memory = model_memory

        # Activations (depends on batch size and seq length)
        activation_memory = self._estimate_activations(
            config.model, config.batch_size, config.max_seq_length
        )

        total = model_memory + optimizer_memory + gradient_memory + activation_memory

        return MemoryEstimate(
            model_gb=model_memory / 1e9,
            optimizer_gb=optimizer_memory / 1e9,
            gradients_gb=gradient_memory / 1e9,
            activations_gb=activation_memory / 1e9,
            total_gb=total / 1e9,
            recommended_gpu_gb=total / 1e9 * 1.2,  # 20% overhead
        )
```

### Quick Reference

| Config | 7B Model | 8B Model | 13B Model | 70B Model |
|--------|----------|----------|-----------|-----------|
| Full FP16, BS=4 | ~56 GB | ~64 GB | ~104 GB | ~560 GB |
| LoRA FP16, BS=4 | ~18 GB | ~20 GB | ~34 GB | ~180 GB |
| QLoRA 4-bit, BS=4 | ~8 GB | ~9 GB | ~14 GB | ~52 GB |
| QLoRA 4-bit, BS=1 | ~5 GB | ~6 GB | ~9 GB | ~40 GB |
