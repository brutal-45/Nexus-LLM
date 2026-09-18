# Fine-Tuning Guide

Learn how to fine-tune language models using LoRA and QLoRA with Nexus-LLM. These parameter-efficient methods let you adapt large models with limited GPU memory and training data.

---

## Why Parameter-Efficient Fine-Tuning?

Full fine-tuning updates all model parameters, which requires:
- 40GB+ VRAM for a 7B model
- Thousands of training examples
- Hours of training time

Parameter-efficient methods like LoRA train only a small number of additional parameters while keeping the base model frozen. This means:
- **8GB VRAM** is enough for QLoRA on a 7B model
- **100–500 examples** can produce good results
- **Training completes in minutes** instead of hours

---

## LoRA (Low-Rank Adaptation)

### How LoRA Works

LoRA adds low-rank decomposition matrices (called adapters) alongside the original weight matrices. During training, only these small adapter matrices are updated while the base model remains frozen.

```
Original:  y = Wx        (W is frozen, e.g., 4096 × 4096)
LoRA:      y = Wx + BAx  (B is 4096 × r, A is r × 4096, r << 4096)
```

With `r = 8`, the number of trainable parameters drops from ~16M per layer to ~65K — a 245x reduction.

### LoRA Configuration

#### Key Parameters

| Parameter | Default | Description | Recommendations |
|-----------|---------|-------------|-----------------|
| `r` (rank) | 8 | Rank of the low-rank matrices | 4–8 for simple tasks, 16–64 for complex tasks |
| `lora_alpha` | 16 | Scaling factor for LoRA updates | Typically 2× the rank |
| `lora_dropout` | 0.05 | Dropout probability on LoRA layers | 0.05–0.1 for regularization |
| `target_modules` | `["q_proj", "v_proj"]` | Which layers to apply LoRA to | See below |

#### Target Modules

Different model architectures have different layer names. Here are recommended target modules:

| Model Family | Target Modules | Description |
|--------------|---------------|-------------|
| LLaMA / Mistral | `q_proj, v_proj, k_proj, o_proj` | Attention projections |
| LLaMA (full) | `q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj` | Attention + MLP |
| GPT-2 | `c_attn, c_proj` | Attention layers |
| Phi | `q_proj, v_proj, k_proj, dense` | Attention projections |
| Qwen | `c_attn, c_proj` | Attention layers |

**Rule of thumb:** Targeting only `q_proj` and `v_proj` works well for most tasks. Adding `k_proj` and `o_proj` improves quality at the cost of more parameters. Adding MLP layers (`gate_proj`, `up_proj`, `down_proj`) gives the best results but uses the most memory.

### Running LoRA Fine-Tuning

```bash
# Basic LoRA fine-tuning
./scripts/train.sh \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset ./data/training_data.jsonl \
  --method lora \
  --rank 8 \
  --alpha 16 \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-5

# Advanced: target more modules for better quality
./scripts/train.sh \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset ./data/training_data.jsonl \
  --method lora \
  --rank 16 \
  --alpha 32 \
  --target-modules "q_proj,v_proj,k_proj,o_proj" \
  --epochs 5 \
  --batch-size 2 \
  --grad-accum 8 \
  --lr 1e-4

# Quick experiment with low rank
./scripts/train.sh \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset ./data/training_data.jsonl \
  --method lora \
  --rank 4 \
  --alpha 8 \
  --epochs 1 \
  --batch-size 8
```

### LoRA Configuration File

```yaml
# config/training/lora.yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"

method: "lora"

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  bias: "none"
  task_type: "CAUSAL_LM"
  modules_to_save:
    - embed_tokens
    - lm_head

hyperparameters:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  fp16: true
  max_grad_norm: 1.0

data:
  max_seq_length: 2048

output:
  dir: "./checkpoints/lora_experiment"
```

---

## QLoRA (Quantized LoRA)

### How QLoRA Works

QLoRA combines LoRA with 4-bit quantization of the base model. The frozen base model weights are stored in 4-bit NormalFloat format, dramatically reducing memory usage while LoRA adapters (in BF16) capture the task-specific adaptations.

```
Base model:  4-bit quantized  →  ~4GB for 7B model (vs ~14GB in FP16)
LoRA adapters: BF16           →  ~30MB for rank-8
Total:                        →  ~4GB + 30MB ≈ 4GB
```

### QLoRA Configuration

```bash
# QLoRA fine-tuning on a single GPU with 8GB VRAM
./scripts/train.sh \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset ./data/training_data.jsonl \
  --method qlora \
  --rank 8 \
  --alpha 16 \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-5
```

### QLoRA Configuration File

```yaml
# config/training/qlora.yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"

method: "qlora"

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  bias: "none"
  task_type: "CAUSAL_LM"

hyperparameters:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  fp16: false
  bf16: true
  gradient_checkpointing: true
  max_grad_norm: 0.3

data:
  max_seq_length: 2048

output:
  dir: "./checkpoints/qlora_experiment"
```

### Memory Comparison

| Method | 7B Model | 8B Model | 13B Model | 70B Model |
|--------|----------|----------|-----------|-----------|
| Full (FP16) | ~28 GB | ~32 GB | ~52 GB | ~280 GB |
| LoRA (FP16) | ~16 GB | ~18 GB | ~30 GB | ~160 GB |
| QLoRA (4-bit) | ~6 GB | ~7 GB | ~10 GB | ~48 GB |

---

## Merging Adapters

After LoRA/QLoRA training, you have a base model and an adapter. For deployment, you can either:

1. **Merge them** into a single model (simpler deployment)
2. **Keep them separate** and load at runtime (more flexible)

### Merge into Base Model

```bash
# Merge adapter weights into the base model
nexus merge-adapter \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter ./checkpoints/lora_experiment \
  --output ./models/my_finetuned_model
```

### Merge Multiple Adapters

You can combine multiple LoRA adapters trained for different tasks:

```bash
nexus merge-adapters \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapters ./checkpoints/math_adapter,./checkpoints/code_adapter \
  --weights 0.6,0.4 \
  --output ./models/combined_model
```

### Load Adapter at Runtime

```bash
# Use the adapter without merging
./scripts/run.sh \
  --mode chat \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter ./checkpoints/lora_experiment
```

### Export Merged Model

```bash
# Export to different formats
nexus export --model ./models/my_finetuned_model --format gguf --quantization q4_k_m --output ./models/my_model.gguf
nexus export --model ./models/my_finetuned_model --format safetensors --output ./models/my_model_safe
```

---

## Advanced Techniques

### Multi-Rank LoRA

Use different ranks for different modules:

```yaml
lora:
  rank_pattern:
    q_proj: 16
    v_proj: 16
    k_proj: 8
    o_proj: 8
    gate_proj: 4
  alpha_pattern:
    q_proj: 32
    v_proj: 32
    k_proj: 16
    o_proj: 16
    gate_proj: 8
```

### LoRA+ (Improved Initialization)

```yaml
lora:
  r: 8
  lora_alpha: 16
  use_lora_plus: true
  lora_plus_scale: 0.1  # Learning rate ratio (B/A)
```

### RSLoRA (Rank-Stabilized LoRA)

```yaml
lora:
  r: 64
  lora_alpha: 16
  use_rslora: true  # Automatically scales alpha by sqrt(r)
```

### DoRA (Weight-Decomposed LoRA)

```yaml
lora:
  r: 8
  lora_alpha: 16
  use_dora: true  # Decompose weights into magnitude and direction
```

---

## Choosing Hyperparameters

### Rank Selection Guide

| Task Complexity | Recommended Rank | LoRA Alpha | Target Modules |
|----------------|-----------------|------------|----------------|
| Simple style transfer | 4 | 8 | q_proj, v_proj |
| Single-domain QA | 8 | 16 | q_proj, v_proj |
| Multi-domain instruction | 16 | 32 | q_proj, v_proj, k_proj, o_proj |
| Complex reasoning | 32 | 64 | All attention + MLP |
| Code generation | 16–32 | 32–64 | q_proj, v_proj, k_proj, o_proj |

### Learning Rate Guide

| Method | Recommended LR | Scheduler |
|--------|---------------|-----------|
| LoRA (rank 8) | 2e-5 to 1e-4 | Cosine |
| LoRA (rank 16+) | 1e-4 to 3e-4 | Cosine |
| QLoRA | 1e-4 to 2e-4 | Cosine with 0.1 warmup |
| Full fine-tuning | 5e-6 to 2e-5 | Linear with 0.05 warmup |

---

## Checklist Before Training

- [ ] Dataset is in JSONL format with consistent schema
- [ ] Dataset has 100+ examples (500+ recommended)
- [ ] Training and evaluation data are split (90/10 or 80/20)
- [ ] No duplicate or near-duplicate examples
- [ ] GPU has enough VRAM for the chosen method
- [ ] Output directory has enough disk space
- [ ] Base model is downloaded and accessible
- [ ] HuggingFace token is set for gated models
