# Training Guide

Learn how to train and fine-tune large language models with Nexus-LLM.

## Overview

Nexus-LLM supports multiple training methods:

| Method | Description | GPU Memory | Best For |
|---|---|---|---|
| **LoRA** | Low-Rank Adaptation | ~8GB for 7B | Quick fine-tuning, limited resources |
| **QLoRA** | Quantized LoRA (4-bit) | ~6GB for 7B | Very limited GPU memory |
| **Full** | Full parameter fine-tuning | ~60GB for 7B | Maximum quality |
| **DPO** | Direct Preference Optimization | ~16GB for 7B | Preference alignment |
| **ORPO** | Odds Ratio Preference Optimization | ~16GB for 7B | Alignment without reference model |
| **RLHF** | Reinforcement Learning from HF | ~80GB for 7B | Advanced alignment |

---

## Quick Start: LoRA Fine-Tuning

### 1. Prepare Your Data

Create a JSONL file with training examples in chat format:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write a Python function to reverse a string."}, {"role": "assistant", "content": "def reverse_string(s: str) -> str:\n    return s[::-1]"}]}
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Explain list comprehension in Python."}, {"role": "assistant", "content": "List comprehension is a concise way to create lists in Python..."}]}
```

### 2. Run Training

```bash
nexus-llm train \
  --model meta-llama/Llama-3.1-8B \
  --data ./data/training_data.jsonl \
  --method lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --output ./checkpoints/my-lora
```

### 3. Use the Fine-Tuned Model

```bash
# Serve with LoRA adapter
nexus-llm serve \
  --model meta-llama/Llama-3.1-8B \
  --lora ./checkpoints/my-lora

# Or merge LoRA weights
nexus-llm merge-lora \
  --base-model meta-llama/Llama-3.1-8B \
  --lora ./checkpoints/my-lora \
  --output ./models/my-merged-model
```

---

## Training Data Formats

### Chat Format (Recommended)

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Alpaca Format

```jsonl
{"instruction": "Translate to French", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"}
```

### Instruction Format

```jsonl
{"instruction": "Write a poem about the moon.", "output": "Silver light upon the sea..."}
```

### DPO Format (Preference Data)

```jsonl
{"prompt": "Explain machine learning.", "chosen": "Machine learning is a branch of AI that...", "rejected": "ML is when computers do stuff."}
```

---

## Training Parameters

### LoRA Configuration

| Parameter | Default | Range | Description |
|---|---|---|---|
| `--lora-rank` | 16 | 1-256 | Rank of the LoRA update matrices |
| `--lora-alpha` | 32 | 1-512 | Scaling factor (typically 2x rank) |
| `--lora-dropout` | 0.05 | 0.0-0.5 | Dropout probability |
| `--lora-target-modules` | all linear | — | Which modules to apply LoRA to |

**Recommended LoRA ranks by model size:**

| Model Size | LoRA Rank | LoRA Alpha | Trainable Params |
|---|---|---|---|
| 1-3B | 8-16 | 16-32 | ~10-40M |
| 7-13B | 16-32 | 32-64 | ~20-80M |
| 30-70B | 32-64 | 64-128 | ~40-160M |

### Training Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--learning-rate` | 2e-4 | Learning rate (use 1e-4 for full fine-tuning) |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 4 | Batch size per GPU |
| `--gradient-accumulation` | 8 | Steps to accumulate gradients |
| `--warmup-ratio` | 0.1 | Fraction of steps for LR warmup |
| `--weight-decay` | 0.01 | L2 regularization |
| `--max-seq-length` | 2048 | Maximum sequence length |
| `--lr-scheduler` | cosine | Learning rate scheduler |

### Effective Batch Size

```
effective_batch_size = batch_size × gradient_accumulation × num_gpus
```

For example: `4 × 8 × 2 = 64` samples per optimization step.

---

## Distributed Training

### DeepSpeed ZeRO

```bash
# ZeRO Stage 2 (recommended for most cases)
nexus-llm train \
  --model meta-llama/Llama-3.1-70B \
  --data ./data/train.jsonl \
  --method lora \
  --deepspeed zero2 \
  --gpus 4

# ZeRO Stage 3 (for very large models)
nexus-llm train \
  --model meta-llama/Llama-3.1-70B \
  --data ./data/train.jsonl \
  --method lora \
  --deepspeed zero3 \
  --gpus 8
```

**DeepSpeed Stage Comparison:**

| Stage | Sharded | Memory Savings | Communication | Best For |
|---|---|---|---|---|
| ZeRO-1 | Optimizer states | ~4x | Low | Quick start |
| ZeRO-2 | Optimizer + Gradients | ~8x | Medium | Most cases |
| ZeRO-3 | All states + Parameters | ~N_gpu | High | Very large models |

### FSDP (Fully Sharded Data Parallel)

```bash
nexus-llm train \
  --model meta-llama/Llama-3.1-70B \
  --data ./data/train.jsonl \
  --method lora \
  --fsdp full_shard \
  --gpus 8
```

---

## Preference Alignment

### DPO Training

```bash
nexus-llm train \
  --model meta-llama/Llama-3.1-8B \
  --data ./data/preference_data.jsonl \
  --method dpo \
  --learning-rate 5e-5 \
  --dpo-beta 0.1 \
  --epochs 1 \
  --output ./checkpoints/dpo-model
```

### ORPO Training

```bash
nexus-llm train \
  --model meta-llama/Llama-3.1-8B \
  --data ./data/preference_data.jsonl \
  --method orpo \
  --orpo-beta 0.1 \
  --epochs 1 \
  --output ./checkpoints/orpo-model
```

### RLHF with PPO

```bash
nexus-llm train \
  --model meta-llama/Llama-3.1-8B \
  --data ./data/prompts.jsonl \
  --method rlhf \
  --reward-model ./checkpoints/reward-model \
  --ppo-epochs 4 \
  --kl-coeff 0.1 \
  --output ./checkpoints/rlhf-model
```

---

## Monitoring Training

### TensorBoard

```bash
# Training with TensorBoard logging
nexus-llm train \
  --model meta-llama/Llama-3.1-8B \
  --data ./data/train.jsonl \
  --method lora \
  --logging-dir ./runs \
  --report-to tensorboard

# View TensorBoard
tensorboard --logdir ./runs
```

### Weights & Biases

```bash
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=nexus-llm

nexus-llm train \
  --model meta-llama/Llama-3.1-8B \
  --data ./data/train.jsonl \
  --method lora \
  --report-to wandb
```

---

## Evaluation

### Built-in Benchmarks

```bash
# Evaluate on MMLU
nexus-llm evaluate \
  --model ./checkpoints/my-lora \
  --benchmark mmlu

# Evaluate on GSM8K
nexus-llm evaluate \
  --model ./checkpoints/my-lora \
  --benchmark gsm8k

# Evaluate on HumanEval (code)
nexus-llm evaluate \
  --model ./checkpoints/my-lora \
  --benchmark humaneval

# Custom evaluation data
nexus-llm evaluate \
  --model ./checkpoints/my-lora \
  --data ./data/eval_data.jsonl \
  --metrics accuracy,bleu,rouge
```

---

## Best Practices

1. **Start with LoRA** — It's faster and uses less memory. Move to full fine-tuning only if needed.
2. **Use a good learning rate** — 2e-4 for LoRA, 1e-4 for QLoRA, 5e-6 for full fine-tuning.
3. **Warm up properly** — Use a warmup ratio of 0.05-0.1.
4. **Use cosine scheduling** — Cosine annealing with warm restarts often works best.
5. **Monitor for overfitting** — Watch training and validation loss curves.
6. **Save checkpoints frequently** — Use `--save-steps 500` to avoid losing progress.
7. **Use a validation set** — Always hold out 10% of data for validation.
8. **Start with 3 epochs** — More epochs often lead to overfitting with LoRA.
