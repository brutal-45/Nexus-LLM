# Nexus-LLM Training Documentation

This guide covers everything you need to know about fine-tuning language models with Nexus-LLM. Whether you're adapting a model to your domain, teaching it a new task, or improving its performance on specific outputs, this document will walk you through the process from start to finish.

---

## 1. Quick Start Fine-Tuning

Fine-tuning a model with Nexus-LLM is designed to be as simple as possible. With LoRA (Low-Rank Adaptation), you can fine-tune a 7B parameter model on a single consumer GPU with as little as 8GB of VRAM. Here's the fastest path from zero to a fine-tuned model.

### The Three-Command Fine-Tune

```bash
# Step 1: Prepare your dataset (convert to Nexus-LLM format)
nexus-llm dataset prepare --input ./my_data.jsonl --format alpaca --output ./data/processed

# Step 2: Start fine-tuning
nexus-llm train --model mistral-7b --data ./data/processed --epochs 3 --output ./models/my-finetune

# Step 3: Use your fine-tuned model
nexus-llm chat --model ./models/my-finetune
```

That's it! Behind the scenes, Nexus-LLM:
1. Downloads the base model if not cached
2. Applies LoRA adapters to the attention layers
3. Trains only the adapter weights (saving memory and time)
4. Saves the adapter separately from the base model
5. Merges the adapter on load for seamless inference

### Quick-Tune with a Single Command

For even faster iteration, use the `quick-tune` command with sensible defaults:

```bash
nexus-llm quick-tune \
  --model phi-2 \
  --data ./training_data.jsonl \
  --task instruction \
  --output ./models/my-model
```

This automatically:
- Detects your data format
- Selects optimal hyperparameters for the task
- Uses 4-bit quantization if GPU memory is limited
- Saves both the adapter and a merged model

### What You'll Need

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for 7B models)
- **Disk Space**: 2-5x the model size for checkpoints and intermediate files
- **Data**: At least 100 examples for basic fine-tuning (1,000+ recommended for good results)
- **Time**: Varies from minutes (small model, small dataset) to hours (large model, large dataset)

---

## 2. Preparing Your Dataset

The quality of your fine-tuned model is directly determined by the quality of your training data. This section covers best practices for dataset preparation, from data collection to cleaning and formatting.

### Data Collection Guidelines

**Quality over quantity**: 500 high-quality, carefully curated examples will produce better results than 50,000 low-quality ones. Focus on examples that are representative of the tasks you want the model to perform.

**Diversity**: Ensure your dataset covers the full range of inputs and outputs you expect the model to handle. If you're fine-tuning for customer support, include different types of queries, different tones, and edge cases.

**Consistency**: Maintain a consistent format, style, and level of detail across all examples. Inconsistent training data leads to unpredictable model behavior.

**Size recommendations by task**:

| Task Type | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| Style adaptation | 50 | 200-500 | 1,000+ |
| Task-specific (single) | 100 | 500-1,000 | 5,000+ |
| Task-specific (multi) | 200 | 1,000-5,000 | 10,000+ |
| Domain adaptation | 500 | 2,000-10,000 | 50,000+ |
| Chat/conversation | 200 | 1,000-5,000 | 20,000+ |

### Data Cleaning

Before training, clean your data thoroughly:

```bash
# Use the built-in data cleaner
nexus-llm dataset clean --input ./raw_data.jsonl --output ./clean_data.jsonl

# With specific cleaning options
nexus-llm dataset clean \
  --input ./raw_data.jsonl \
  --output ./clean_data.jsonl \
  --remove-duplicates \
  --min-length 10 \
  --max-length 2048 \
  --filter-language en \
  --remove-urls \
  --normalize-whitespace
```

The cleaning process:
1. **Removes duplicates** — Exact and near-duplicate entries that could bias training
2. **Filters by length** — Removes extremely short or long entries
3. **Normalizes text** — Fixes whitespace, unicode issues, and encoding problems
4. **Validates format** — Ensures all entries conform to the expected schema
5. **Removes PII** — Optionally strips email addresses, phone numbers, and other sensitive data

### Data Validation

Always validate your dataset before training:

```bash
# Validate dataset format and statistics
nexus-llm dataset validate --input ./clean_data.jsonl --format alpaca

# Output includes:
# - Total examples: 1,247
# - Average input length: 42 tokens
# - Average output length: 186 tokens
# - Max sequence length: 892 tokens
# - Format errors: 0
# - Duplicate entries: 3 (removed)
```

### Train/Test Split

Nexus-LLM can automatically split your data:

```bash
nexus-llm dataset split \
  --input ./clean_data.jsonl \
  --train-ratio 0.9 \
  --val-ratio 0.1 \
  --seed 42 \
  --output ./data/split
```

This creates `train.jsonl` and `val.jsonl` in the output directory. The validation set is used to monitor overfitting during training.

---

## 3. Data Formats

Nexus-LLM supports three primary data formats for fine-tuning. Each format is optimized for different types of tasks and model architectures.

### Alpaca Format

The Alpaca format is the most widely supported and versatile. It supports an optional `input` field for providing context alongside the instruction.

```json
[
  {
    "instruction": "Classify the following email as spam or not spam.",
    "input": "Subject: You've won $1,000,000! Click here to claim your prize now!",
    "output": "Spam"
  },
  {
    "instruction": "Write a professional email declining a meeting invitation.",
    "input": "",
    "output": "Dear [Name],\n\nThank you for the invitation to the meeting on [date]. Unfortunately, I have a prior commitment and will not be able to attend. I would appreciate it if you could share the meeting notes or any key decisions made.\n\nBest regards,\n[Your Name]"
  }
]
```

**Best for**: Instruction-following tasks, classification, generation tasks with optional context.

**Format fields**:
- `instruction` (required): The task description
- `input` (optional): Additional context or input data
- `output` (required): The expected model response

### Chat Format

The Chat format supports multi-turn conversations with role-based messages. This is the native format for chat-tuned models.

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful cooking assistant."
      },
      {
        "role": "user",
        "content": "How do I make a basic vinaigrette?"
      },
      {
        "role": "assistant",
        "content": "A basic vinaigrette is simple! The classic ratio is 3 parts oil to 1 part acid (vinegar or lemon juice). Here's how:\n\n1. Combine 3 tbsp olive oil and 1 tbsp vinegar\n2. Add a pinch of salt and pepper\n3. Whisk until emulsified\n4. Optional: add mustard, herbs, or garlic for flavor"
      },
      {
        "role": "user",
        "content": "Can I use balsamic vinegar?"
      },
      {
        "role": "assistant",
        "content": "Absolutely! Balsamic vinegar makes a delicious vinaigrette with a slightly sweeter, more complex flavor. It pairs especially well with salads containing fruit, nuts, or cheese. You might want to use a 2:1 oil-to-vinegar ratio with balsamic since it's less acidic than white vinegar."
      }
    ]
  }
]
```

**Best for**: Multi-turn dialogue, conversation modeling, customer support bots.

**Format fields**:
- `messages` (required): Array of message objects
- Each message has `role` (system/user/assistant) and `content`

### Instruction Format

The Instruction format is the simplest format, designed for straightforward prompt-response pairs without additional context.

```json
[
  {
    "prompt": "Explain the concept of recursion in programming.",
    "response": "Recursion is a programming technique where a function calls itself to solve a problem. A recursive function typically has two parts: a base case (the simplest possible solution that stops the recursion) and a recursive case (where the function calls itself with a smaller or simpler input).\n\nFor example, calculating factorial:\n- Base case: 0! = 1\n- Recursive case: n! = n × (n-1)!\n\nRecursion is elegant for problems with self-similar structure, like tree traversal, but can cause stack overflow if the base case is never reached."
  },
  {
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris."
  }
]
```

**Best for**: Simple Q&A, knowledge injection, style transfer, single-turn tasks.

**Format fields**:
- `prompt` (required): The input text/question
- `response` (required): The expected output

### Converting Between Formats

Nexus-LLM can convert between data formats:

```bash
# Convert Alpaca to Chat format
nexus-llm dataset convert \
  --input ./data/alpaca_data.json \
  --from alpaca \
  --to chat \
  --output ./data/chat_data.json

# Convert CSV to Alpaca format
nexus-llm dataset convert \
  --input ./data/spreadsheet.csv \
  --from csv \
  --to alpaca \
  --instruction-column "question" \
  --output-column "answer" \
  --output ./data/alpaca_data.json
```

### Custom Data Loaders

For complex or proprietary data formats, you can write a custom data loader:

```python
from nexus_llm.training.dataset import BaseDataset

class MyCustomDataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        # Load your custom format
        with open(data_path) as f:
            self.data = [self._parse_line(line) for line in f]

    def _parse_line(self, line):
        # Convert your format to the standard format
        parts = line.strip().split("|||")
        return {
            "instruction": parts[0],
            "input": parts[1] if len(parts) > 2 else "",
            "output": parts[-1]
        }
```

---

## 4. LoRA Configuration

LoRA (Low-Rank Adaptation) is the core technique that makes fine-tuning large models practical. Instead of updating all model parameters, LoRA adds small trainable rank-decomposition matrices to specific layers, dramatically reducing memory usage and training time while maintaining quality.

### How LoRA Works

When fine-tuning a model with LoRA:
1. The original model weights are **frozen** (not updated)
2. Small **adapter matrices** (A and B) are added to target layers
3. Only these adapter matrices are trained
4. The adapters are typically 0.1-1% the size of the original model
5. At inference, adapters can be merged with the base model for zero overhead

### LoRA Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lora_r` | 8 | 1-256 | Rank of the LoRA decomposition. Higher values = more capacity but more parameters. Start with 8-16 for most tasks. |
| `lora_alpha` | 16 | 1-512 | Scaling factor for LoRA updates. Typically set to 2× `lora_r`. Controls the magnitude of the adaptation. |
| `lora_dropout` | 0.05 | 0-0.5 | Dropout probability for LoRA layers. Helps prevent overfitting. Use 0.05-0.1 for small datasets. |
| `lora_target_modules` | auto | - | Which layers to apply LoRA to. "auto" targets all linear layers in attention and MLP blocks. |

### Choosing LoRA Rank

The rank (`lora_r`) is the most important LoRA parameter:

| Task Complexity | Recommended Rank | Adapter Size (7B model) |
|----------------|-----------------|------------------------|
| Style/tone adaptation | 4-8 | ~4-8 MB |
| Single-task instruction | 8-16 | ~8-16 MB |
| Multi-task instruction | 16-32 | ~16-32 MB |
| Domain adaptation | 32-64 | ~32-64 MB |
| Complex reasoning | 64-128 | ~64-128 MB |

### Setting LoRA Configuration

```bash
# Via CLI flags
nexus-llm train \
  --model mistral-7b \
  --data ./data/processed \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --output ./models/my-model

# Via configuration file
nexus-llm config --set training.lora_r 16
nexus-llm config --set training.lora_alpha 32
nexus-llm config --set training.lora_dropout 0.05
```

### Advanced LoRA Settings

For fine-grained control over which modules receive LoRA adapters:

```yaml
training:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  lora_bias: none           # Whether to train bias terms: none, all, or lora_only
  lora_modules_to_save:     # Non-LoRA modules that should be trained
    - embed_tokens
    - lm_head
```

### QLoRA (Quantized LoRA)

QLoRA combines 4-bit quantization with LoRA for maximum memory efficiency:

```bash
# Fine-tune a 7B model with only ~6GB VRAM
nexus-llm train \
  --model mistral-7b \
  --data ./data/processed \
  --quantization 4bit \
  --lora-r 16 \
  --output ./models/my-model
```

QLoRA quantization options:
- `4bit`: NF4 quantization (recommended) — best quality-to-memory ratio
- `8bit`: LLM.int8() quantization — slightly more memory but sometimes more stable

---

## 5. Training Parameters

Understanding training parameters is crucial for achieving good results. This section covers each parameter in detail and provides guidance on tuning them for your specific use case.

### Core Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-4 | How fast the model adapts. Too high causes instability; too low makes training slow. LoRA typically uses higher rates than full fine-tuning. |
| `num_epochs` | 3 | Number of passes through the entire dataset. More epochs risk overfitting; monitor validation loss. |
| `batch_size` | 4 | Examples processed per gradient update. Larger batches provide more stable gradients but use more memory. |
| `gradient_accumulation_steps` | 4 | Accumulate gradients over multiple steps before updating. Effective batch size = batch_size × gradient_accumulation_steps. |
| `warmup_steps` | 100 | Number of steps to linearly increase learning rate from 0. Prevents instability at the start of training. |
| `max_seq_length` | 512 | Maximum sequence length for training. Longer sequences use more memory. Set based on your data's length distribution. |
| `save_steps` | 500 | Save a checkpoint every N steps. Useful for resuming and selecting the best checkpoint. |
| `eval_steps` | 500 | Evaluate on validation data every N steps. Essential for monitoring overfitting. |

### Learning Rate Guide

| Scenario | Recommended LR | Notes |
|----------|---------------|-------|
| LoRA fine-tune (small dataset) | 1e-4 to 3e-4 | Standard range for LoRA |
| LoRA fine-tune (large dataset) | 5e-5 to 1e-4 | Lower rate for more data |
| QLoRA (4-bit) | 1e-4 to 2e-4 | Slightly lower for stability |
| Full fine-tune | 1e-5 to 5e-5 | Much lower for full weights |
| Continual pretraining | 5e-6 to 2e-5 | Very low for large-scale adaptation |

### Batch Size and Memory

Effective batch size matters more than per-GPU batch size. Use gradient accumulation to simulate larger batches:

| GPU VRAM | Model Size | Batch Size | Grad Accum | Effective Batch |
|----------|-----------|------------|------------|----------------|
| 8 GB | 1-3B | 4 | 4 | 16 |
| 8 GB | 7B (4-bit) | 2 | 8 | 16 |
| 16 GB | 7B | 4 | 4 | 16 |
| 24 GB | 7B | 8 | 2 | 16 |
| 24 GB | 13B | 4 | 4 | 16 |
| 80 GB | 7B | 16 | 1 | 16 |

### Setting Parameters

```bash
# Via CLI
nexus-llm train \
  --model mistral-7b \
  --data ./data/processed \
  --learning-rate 2e-4 \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 100 \
  --max-seq-length 512 \
  --save-steps 500 \
  --eval-steps 500 \
  --output ./models/my-model

# Via config file (config/default_config.yaml)
nexus-llm config --set training.learning_rate 0.0002
nexus-llm config --set training.num_epochs 3
nexus-llm config --set training.batch_size 4
```

### Advanced Parameters

```yaml
training:
  # Optimizer
  optimizer: adamw_torch       # adamw_torch, adamw_8bit, adafactor
  weight_decay: 0.01           # L2 regularization to prevent overfitting
  max_grad_norm: 1.0           # Gradient clipping for stability

  # Learning rate schedule
  lr_scheduler_type: cosine    # cosine, linear, constant, constant_with_warmup

  # Precision
  fp16: false                  # Use FP16 mixed precision
  bf16: true                   # Use BF16 mixed precision (Ampere+ GPUs)

  # Checkpointing
  save_total_limit: 3          # Keep only the N best checkpoints
  load_best_model_at_end: true # Load the best checkpoint (by eval loss)
  metric_for_best_model: eval_loss

  # Reproducibility
  seed: 42
  dataloader_num_workers: 4
  dataloader_pin_memory: true

  # Logging
  logging_steps: 10
  report_to: tensorboard       # tensorboard, wandb, none
  run_name: "nexus-lora-finetune"
```

---

## 6. Monitoring Training

Monitoring your training run is essential for catching problems early and ensuring your model converges properly. Nexus-LLM provides multiple ways to track training progress.

### Terminal Output

During training, Nexus-LLM displays a live progress dashboard:

```
╭─────────────────────────────────────────────────────────╮
│  Nexus-LLM Training                           Run #3   │
├─────────────────────────────────────────────────────────┤
│  Model: mistral-7b-instruct  |  LoRA r=16, α=32        │
│  Dataset: 2,847 train / 316 val examples                │
├─────────────────────────────────────────────────────────┤
│  Epoch 2/3  │  Step 1,423/4,270  │  ████████░░  33%    │
│                                                         │
│  Train Loss: 0.8234  │  Val Loss: 0.9127  │  ↓ Good    │
│  Learning Rate: 1.2e-4  │  Grad Norm: 0.43             │
│  Speed: 3.2 samples/s  │  ETA: 28 min                  │
│  GPU: 12.4/16.0 GB (77%)  │  Temp: 71°C               │
╰─────────────────────────────────────────────────────────╯
```

### TensorBoard Integration

For detailed training visualization:

```bash
# Enable TensorBoard logging
nexus-llm train \
  --model mistral-7b \
  --data ./data/processed \
  --report-to tensorboard \
  --output ./models/my-model

# Start TensorBoard
tensorboard --logdir ./models/my-model/runs
```

TensorBoard shows:
- Training and validation loss curves
- Learning rate schedule
- Gradient norms
- GPU memory usage over time
- Token-level metrics

### Weights & Biases (W&B)

For team-based experiment tracking:

```bash
# Login to W&B
wandb login

# Train with W&B logging
nexus-llm train \
  --model mistral-7b \
  --data ./data/processed \
  --report-to wandb \
  --run-name "my-experiment-v3" \
  --output ./models/my-model
```

### Key Metrics to Watch

| Metric | Healthy Range | Warning Signs |
|--------|--------------|---------------|
| Training loss | Steadily decreasing | Stuck or increasing |
| Validation loss | Decreasing, then plateauing | Increasing (overfitting) |
| Train/val gap | Small (< 0.5) | Large gap (> 1.0) = overfitting |
| Gradient norm | 0.1-10 | Very high (> 100) = instability |
| GPU utilization | > 80% | < 50% = bottleneck |
| Learning rate | Per schedule | Spikes or drops |

### Detecting and Fixing Problems

**Overfitting** (train loss ↓, val loss ↑):
- Reduce `num_epochs`
- Increase `lora_dropout` (0.1-0.2)
- Decrease `lora_r`
- Add more training data
- Use data augmentation

**Underfitting** (both losses remain high):
- Increase `lora_r`
- Increase `learning_rate`
- Train for more epochs
- Check data quality

**Training instability** (loss spikes, NaN values):
- Reduce `learning_rate` by 10x
- Enable `bf16: true` (more stable than fp16)
- Reduce `batch_size` and increase `gradient_accumulation_steps`
- Check for bad data (empty strings, very long sequences)

---

## 7. Using Fine-Tuned Models

After training completes, you need to load and use your fine-tuned model. Nexus-LLM makes this seamless whether you're using the terminal, CLI, or API.

### Understanding the Output Structure

After training, your output directory contains:

```
./models/my-finetune/
├── adapter_config.json       # LoRA adapter configuration
├── adapter_model.safetensors # LoRA adapter weights
├── tokenizer.json            # Tokenizer files
├── tokenizer_config.json
├── special_tokens_map.json
├── checkpoint-500/           # Intermediate checkpoints
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── ...
├── checkpoint-1000/
│   └── ...
├── runs/                     # TensorBoard logs
│   └── ...
└── training_args.bin         # Training configuration
```

### Loading the Fine-Tuned Model

```bash
# Load with adapter (recommended — uses less disk space)
nexus-llm chat --model ./models/my-finetune

# The base model is loaded first, then the adapter is applied automatically
# This requires the base model to be available (cached or local)
```

### Merging the Adapter

For production deployment or to eliminate the adapter loading overhead, merge the adapter with the base model:

```bash
# Merge adapter into the base model
nexus-llm model merge \
  --base-model mistral-7b-instruct \
  --adapter ./models/my-finetune \
  --output ./models/my-finetune-merged

# Use the merged model (no base model dependency)
nexus-llm chat --model ./models/my-finetune-merged
```

### Using Multiple Adapters

Nexus-LLM supports loading different adapters for different tasks:

```bash
# Load the base model with a specific adapter
nexus-llm chat \
  --model mistral-7b-instruct \
  --adapter ./models/code-adapter \
  --adapter ./models/style-adapter

# Switch adapters in the chat session
nexus-llm> /adapter ./models/qa-adapter
Switched to adapter: qa-adapter
```

### Serving Fine-Tuned Models

Deploy your fine-tuned model as an API:

```bash
# Serve the fine-tuned model
nexus-llm serve --model ./models/my-finetune-merged --port 8000

# Or serve with the adapter (auto-loads base model)
nexus-llm serve --model mistral-7b-instruct --adapter ./models/my-finetune --port 8000
```

### Sharing Models

Share your fine-tuned model with others:

```bash
# Push to HuggingFace Hub
nexus-llm model push \
  --model ./models/my-finetune \
  --repo-id username/my-nexus-model \
  --private

# Create a model card
nexus-llm model card \
  --model ./models/my-finetune \
  --output ./models/my-finetune/README.md \
  --base-model mistral-7b-instruct \
  --description "Fine-tuned for customer support Q&A"
```

### Evaluating Fine-Tuned Models

Always evaluate your model before deploying:

```bash
# Run automated evaluation
nexus-llm evaluate \
  --model ./models/my-finetune \
  --benchmark instruction-following \
  --output ./eval_results

# Compare with base model
nexus-llm evaluate \
  --model ./models/my-finetune \
  --baseline mistral-7b-instruct \
  --test-data ./data/test.jsonl \
  --metrics accuracy,fluency,relevance
```

### Best Practices for Production

1. **Always merge adapters** before production deployment for faster loading
2. **Quantize the merged model** (4-bit or 8-bit) for reduced memory footprint
3. **A/B test** your fine-tuned model against the base model with real users
4. **Monitor outputs** in production — set up logging for generated responses
5. **Version your adapters** — keep track of which adapter version is deployed
6. **Regular retraining** — schedule periodic fine-tuning with fresh data to prevent drift
