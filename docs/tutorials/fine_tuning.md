# Fine-Tuning Tutorial

This tutorial walks you through fine-tuning a Nexus-LLM model using LoRA (Low-Rank Adaptation).

## When to Fine-Tune?

Fine-tuning is useful when you need the model to:
- Follow a specific output format consistently
- Learn domain-specific knowledge and terminology
- Adopt a particular tone or style
- Improve on a narrow task (e.g., medical Q&A, legal summarization)

If your use case can be solved with prompting or RAG, start there first — fine-tuning is more expensive and requires curated data.

## Prerequisites

- A GPU with at least 16GB VRAM (24GB+ recommended)
- A training dataset in JSONL format
- Basic familiarity with Python and the command line

## Step 1: Prepare Your Dataset

Nexus-LLM expects training data in JSONL format with instruction-output pairs:

```jsonl
{"instruction": "Explain our return policy", "output": "Our return policy allows..."}
{"instruction": "How do I reset my password?", "output": "To reset your password..."}
{"instruction": "What are the shipping options?", "output": "We offer three shipping options..."}
```

Load and validate your dataset:

```python
from nexus_llm.training import DatasetLoader, DataPreprocessor

loader = DatasetLoader()
dataset = loader.load_jsonl(
    path="data/train.jsonl",
    validation_split=0.1,
    seed=42,
)

print(f"Training: {len(dataset.train)} samples")
print(f"Validation: {len(dataset.val)} samples")
```

Preprocess for the training format:

```python
preprocessor = DataPreprocessor(
    format="chatml",
    max_length=2048,
    truncate=True,
)

train_data = preprocessor.process(dataset.train)
val_data = preprocessor.process(dataset.val)
```

## Step 2: Configure LoRA

LoRA dramatically reduces the number of trainable parameters by injecting low-rank matrices:

```python
from nexus_llm.training import LoRAConfig

lora_config = LoRAConfig(
    r=16,                    # LoRA rank — start with 16
    lora_alpha=32,           # Scaling factor — typically 2x rank
    target_modules=[
        "q_proj",            # Apply to attention projections
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,       # Regularization
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Choosing LoRA Rank

| Rank | Trainable Params | Best For |
|------|-----------------|----------|
| 8 | ~4M | Simple tasks, small datasets (<500 samples) |
| 16 | ~8M | General purpose (recommended starting point) |
| 32 | ~16M | Complex tasks, larger datasets (>5K samples) |
| 64 | ~32M | Very complex tasks, lots of data (>10K samples) |

## Step 3: Train

```python
from nexus_llm.training import Trainer

trainer = Trainer(
    base_model="nexus-7b-chat",
    lora_config=lora_config,
    output_dir="./checkpoints/my-finetune",
    training_args={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "fp16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 3,
    },
)

trainer.train(
    train_dataset=train_data,
    eval_dataset=val_data,
)
```

## Step 4: Monitor Training

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir ./checkpoints/my-finetune/runs
```

Key metrics to watch:
- **Training loss**: Should decrease steadily
- **Validation loss**: Should decrease; if it increases, you may be overfitting
- **Learning rate**: Should follow the cosine schedule

## Step 5: Save and Test

```python
# Save the LoRA adapter
trainer.save_adapter("./my_adapter")

# Test the fine-tuned model
from nexus_llm import InferenceEngine

engine = InferenceEngine(
    model_name="nexus-7b-chat",
    adapter_path="./my_adapter",
)

response = engine.chat("Explain our return policy")
print(response.text)
```

## Step 6: (Optional) Merge Adapter

For deployment without the adapter overhead:

```python
trainer.merge_and_save("./models/my-merged-model")
```

The merged model can be loaded directly without specifying an adapter path.

## Tips for Better Results

1. **Data quality > quantity**: 500 clean examples beat 5,000 noisy ones
2. **Balance your dataset**: Avoid over-representing one type of instruction
3. **Use a low learning rate**: 2e-4 is a safe starting point
4. **Watch for overfitting**: Stop when validation loss stops improving
5. **Test thoroughly**: Fine-tuned models can lose general capabilities
6. **Apply safety alignment**: After domain fine-tuning, consider DPO alignment

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM during training | Reduce batch size, enable gradient checkpointing, use `dtype="int8"` for the base model |
| Training loss doesn't decrease | Check data format, increase learning rate, verify preprocessor output |
| Overfitting | Add dropout, reduce LoRA rank, get more diverse data |
| Model forgets general knowledge | Use a lower learning rate, fewer epochs, or apply DPO after fine-tuning |
