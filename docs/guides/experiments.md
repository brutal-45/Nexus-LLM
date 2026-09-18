# Experiment Tracking Guide

This guide covers experiment tracking, hyperparameter management, and ablation studies in Nexus-LLM.

## Overview

Nexus-LLM includes a built-in experiment tracking system that helps you organize, compare, and reproduce model training and evaluation experiments. It integrates with popular tracking backends like MLflow and Weights & Biases.

```
Experiment → Runs → Metrics/Artifacts
    ↓
Hyperparameters (tracked automatically)
    ↓
Comparisons & Ablations
```

---

## Quick Start

### Creating an Experiment

```bash
# Create a new experiment
nexus-llm experiment create \
  --name "llama3-lora-finetune" \
  --description "LoRA fine-tuning experiments on Llama 3.1 8B"

# List all experiments
nexus-llm experiment list

# Show experiment details
nexus-llm experiment show llama3-lora-finetune
```

### Running an Experiment

```bash
# Start a training run with automatic tracking
nexus-llm train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/training_data.jsonl \
  --method lora \
  --experiment llama3-lora-finetune \
  --run-name "lora-r16-alpha32"

# All hyperparameters and metrics are tracked automatically
```

### Using the Python SDK

```python
from nexus_llm.experiments import Experiment, Run

# Create or load an experiment
exp = Experiment.create(
    name="llama3-lora-finetune",
    description="LoRA fine-tuning experiments on Llama 3.1 8B"
)

# Start a run
run = exp.start_run(name="lora-r16-alpha32", tags=["lora", "baseline"])

# Log hyperparameters
run.log_params({
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
})

# Log metrics during training
for epoch in range(3):
    train_loss = train_one_epoch(model, dataloader)
    eval_loss = evaluate(model, eval_dataloader)

    run.log_metrics({
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "epoch": epoch + 1,
    })

# Log artifacts
run.log_artifact("checkpoints/best_model", type="model")
run.log_artifact("eval_report.json", type="report")

# Finish the run
run.finish(status="completed")
```

---

## Experiments

### Experiment Structure

An experiment groups related runs that share the same goal:

```
experiment/
├── name: "llama3-lora-finetune"
├── description: "LoRA fine-tuning experiments"
├── created_at: "2024-01-15T10:00:00Z"
├── tags: ["lora", "llama3"]
└── runs/
    ├── run-001: "lora-r16-alpha32"
    ├── run-002: "lora-r32-alpha64"
    ├── run-003: "lora-r16-alpha16"
    └── run-004: "full-finetune-baseline"
```

### Managing Experiments

```bash
# List experiments
nexus-llm experiment list

# Output:
# ID   Name                      Runs  Best Eval Loss  Created
# ─────────────────────────────────────────────────────────────
# 1    llama3-lora-finetune        4       0.342       2024-01-15
# 2    qwen-qlora-chat             2       0.521       2024-01-18
# 3    embedding-comparison        6       N/A         2024-01-20

# Show experiment details with all runs
nexus-llm experiment show llama3-lora-finetune

# Delete an experiment (and all its runs)
nexus-llm experiment delete llama3-lora-finetune --confirm

# Archive an experiment (keeps data, hides from listings)
nexus-llm experiment archive llama3-lora-finetune

# Restore an archived experiment
nexus-llm experiment restore llama3-lora-finetune
```

### Experiment Comparison

Compare runs across experiments:

```bash
# Compare all runs in an experiment
nexus-llm experiment compare llama3-lora-finetune

# Compare specific runs
nexus-llm experiment compare \
  --runs run-001,run-002,run-003 \
  --metrics eval_loss,train_loss

# Output:
# ┌──────────┬───────────┬──────────┬──────────┬───────────┐
# │ Run      │ LoRA r    │ LR       │ Eval Loss│ Train Loss│
# ├──────────┼───────────┼──────────┼──────────┼───────────┤
# │ run-001  │ 16        │ 2e-4     │ 0.342    │ 0.218     │
# │ run-002  │ 32        │ 2e-4     │ 0.338    │ 0.195     │
# │ run-003  │ 16        │ 1e-4     │ 0.351    │ 0.245     │
# │ run-004  │ full      │ 2e-5     │ 0.325    │ 0.180     │
# └──────────┴───────────┴──────────┴──────────┴───────────┘
```

### Experiment API

```bash
# Create experiment via API
curl -X POST http://localhost:8000/v1/experiments \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama3-lora-finetune",
    "description": "LoRA fine-tuning experiments on Llama 3.1 8B",
    "tags": ["lora", "llama3"]
  }'

# List experiments
curl http://localhost:8000/v1/experiments \
  -H "Authorization: Bearer nxs_sk_abc123"

# Get experiment details
curl http://localhost:8000/v1/experiments/llama3-lora-finetune \
  -H "Authorization: Bearer nxs_sk_abc123"
```

---

## Runs

### Run Lifecycle

```
created → running → completed
                 → failed
                 → stopped (manually)
```

### Creating Runs

```python
from nexus_llm.experiments import Experiment

exp = Experiment.load("llama3-lora-finetune")

# Start a run with automatic tracking
run = exp.start_run(
    name="lora-r16-alpha32",
    tags=["lora", "baseline"],
    # Automatically capture environment info
    capture_env=True,
    # Automatically capture git info
    capture_git=True,
)

# The run automatically captures:
# - Python version and packages
# - GPU model and CUDA version
# - Git commit, branch, and diff
# - System info (CPU, RAM, disk)
```

### Logging Metrics

```python
# Log a single metric
run.log_metric("train_loss", 0.523, step=100)

# Log multiple metrics at once
run.log_metrics({
    "train_loss": 0.523,
    "train_accuracy": 0.891,
    "learning_rate": 1.8e-4,
}, step=100)

# Log metrics with custom x-axis
run.log_metric("eval_loss", 0.342, step=500, x_axis="tokens")

# Log time-series data
import time
start = time.time()
# ... training step ...
run.log_metric("step_duration_ms", (time.time() - start) * 1000)
```

### Logging Parameters

```python
# Log hyperparameters
run.log_params({
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "method": "lora",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "lr_scheduler": "cosine",
    "seed": 42,
    "fp16": True,
    "dataset": "training_data.jsonl",
})

# Log a single parameter
run.log_param("optimizer", "adamw_torch")
```

### Logging Artifacts

```python
# Log a model checkpoint
run.log_artifact(
    path="checkpoints/best_model",
    type="model",
    description="Best model checkpoint (lowest eval loss)"
)

# Log a file
run.log_artifact(
    path="eval_report.json",
    type="report"
)

# Log a plot or image
run.log_artifact(
    path="training_curve.png",
    type="plot"
)

# Log custom metadata with artifact
run.log_artifact(
    path="config.yaml",
    type="config",
    metadata={"source": "auto-generated", "version": 2}
)
```

### Run Tags

Tags help organize and filter runs:

```python
# Add tags to a run
run.add_tags(["lora", "baseline", "production-candidate"])

# Remove a tag
run.remove_tag("baseline")

# Filter runs by tag
runs = exp.list_runs(tags=["lora"])
```

---

## Hyperparameters

### Automatic Hyperparameter Tracking

When using Nexus-LLM's training CLI, all hyperparameters are automatically logged:

```bash
nexus-llm train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/training_data.jsonl \
  --method lora \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --num-epochs 3 \
  --experiment llama3-lora-finetune \
  --run-name "lora-r16-alpha32"
```

All specified parameters plus defaults are captured in the run.

### Hyperparameter Search

Run a grid search or random search over hyperparameters:

```bash
# Grid search
nexus-llm experiment search \
  --experiment llama3-lora-finetune \
  --method grid \
  --params '{
    "lora_r": [8, 16, 32],
    "lora_alpha": [16, 32, 64],
    "learning_rate": [1e-4, 2e-4, 5e-4]
  }'

# Random search (sample 10 combinations)
nexus-llm experiment search \
  --experiment llama3-lora-finetune \
  --method random \
  --num-runs 10 \
  --params '{
    "lora_r": {"type": "int", "low": 4, "high": 64},
    "lora_alpha": {"type": "int", "low": 8, "high": 128},
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": true}
  }'
```

### Hyperparameter Search via Python

```python
from nexus_llm.experiments import Experiment, HyperparamSearch

exp = Experiment.load("llama3-lora-finetune")

search = HyperparamSearch(
    experiment=exp,
    method="bayesian",  # grid, random, bayesian
    objective="eval_loss",
    direction="minimize",
    num_trials=20,
    params={
        "lora_r": {"type": "int", "low": 4, "high": 64},
        "lora_alpha": {"type": "int", "low": 8, "high": 128},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "lora_dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.2},
    },
)

# Run the search
best_run = search.run()

print(f"Best run: {best_run.name}")
print(f"Best eval loss: {best_run.get_metric('eval_loss')}")
print(f"Best params: {best_run.params}")
```

---

## Ablation Studies

Ablation studies help you understand the contribution of individual components by systematically removing or changing them.

### Creating an Ablation Study

```python
from nexus_llm.experiments import AblationStudy

study = AblationStudy(
    name="lora-ablation",
    experiment="llama3-lora-finetune",
    description="Ablation study on LoRA hyperparameters",
    baseline_params={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "num_epochs": 3,
    },
)

# Define ablation groups
study.add_group(
    name="lora_rank",
    description="Effect of LoRA rank",
    variable="lora_r",
    values=[4, 8, 16, 32, 64],
)

study.add_group(
    name="lora_alpha",
    description="Effect of LoRA alpha (scaling factor)",
    variable="lora_alpha",
    values=[8, 16, 32, 64, 128],
)

study.add_group(
    name="dropout",
    description="Effect of LoRA dropout",
    variable="lora_dropout",
    values=[0.0, 0.05, 0.1, 0.2, 0.3],
)

study.add_group(
    name="target_modules",
    description="Effect of target module selection",
    variable="target_modules",
    values=[
        ["q_proj", "v_proj"],
        ["q_proj", "v_proj", "k_proj", "o_proj"],
        ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ],
)

# Run the ablation study
results = study.run()

# View results
print(results.summary())
```

### Ablation Results

```bash
# View ablation results
nexus-llm experiment ablation show lora-ablation

# Output:
# Ablation Study: lora-ablation
# Baseline: lora_r=16, lora_alpha=32, lora_dropout=0.05
#
# Group: lora_rank (variable: lora_r)
# ┌───────┬───────────┬────────────┬───────────┐
# │ r     │ Eval Loss │ Train Loss │ Δ Loss    │
# ├───────┼───────────┼────────────┼───────────┤
# │ 4     │ 0.421     │ 0.312      │ +0.079    │
# │ 8     │ 0.378     │ 0.268      │ +0.036    │
# │ 16 ★  │ 0.342     │ 0.218      │ baseline  │
# │ 32    │ 0.338     │ 0.195      │ -0.004    │
# │ 64    │ 0.341     │ 0.188      │ -0.001    │
# └───────┴───────────┴────────────┴───────────┘
# Conclusion: r=32 provides marginal improvement over r=16.
#             r=64 shows no improvement over r=32.
```

### CLI Ablation

```bash
# Create ablation study
nexus-llm experiment ablation create \
  --name lora-ablation \
  --experiment llama3-lora-finetune \
  --baseline-params '{"lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4}'

# Add ablation groups
nexus-llm experiment ablation add-group \
  --study lora-ablation \
  --name lora_rank \
  --variable lora_r \
  --values 4,8,16,32,64

# Run the ablation
nexus-llm experiment ablation run lora-ablation

# View results
nexus-llm experiment ablation results lora-ablation

# Generate a report
nexus-llm experiment ablation report lora-ablation \
  --output ablation_report.pdf
```

---

## Integration with External Trackers

### MLflow Integration

```yaml
# In training_config.yaml or via CLI
tracking:
  backend: "mlflow"
  mlflow:
    tracking_uri: "http://localhost:5000"
    experiment_name: "nexus-llm-experiments"
    artifact_location: "s3://mlflow/artifacts"
```

```bash
# Start training with MLflow tracking
nexus-llm train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/training_data.jsonl \
  --tracking mlflow \
  --mlflow-uri http://localhost:5000
```

### Weights & Biases Integration

```yaml
tracking:
  backend: "wandb"
  wandb:
    project: "nexus-llm"
    entity: "my-team"
    tags: ["lora", "llama3"]
```

```bash
# Start training with W&B tracking
nexus-llm train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/training_data.jsonl \
  --tracking wandb \
  --wandb-project nexus-llm \
  --wandb-entity my-team
```

### TensorBoard Integration

```yaml
tracking:
  backend: "tensorboard"
  tensorboard:
    log_dir: "runs/"
    flush_secs: 30
```

```bash
# Start training with TensorBoard logging
nexus-llm train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/training_data.jsonl \
  --tracking tensorboard

# Launch TensorBoard
tensorboard --logdir runs/
```

### Multi-Backend Tracking

Track to multiple backends simultaneously:

```bash
nexus-llm train \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/training_data.jsonl \
  --tracking tensorboard,wandb \
  --wandb-project nexus-llm
```

---

## Experiment Reports

### Generating Reports

```bash
# Generate a report for an experiment
nexus-llm experiment report llama3-lora-finetune \
  --format html \
  --output report.html

# Generate a comparison report across runs
nexus-llm experiment report llama3-lora-finetune \
  --format pdf \
  --runs run-001,run-002,run-004 \
  --metrics eval_loss,train_loss \
  --output comparison.pdf

# Generate markdown report
nexus-llm experiment report llama3-lora-finetune \
  --format markdown \
  --output report.md
```

### Report Contents

A typical report includes:
- Experiment metadata (name, description, tags)
- Run comparison table
- Training curves (loss over epochs)
- Hyperparameter importance analysis
- Best run selection with rationale
- Resource usage summary (GPU hours, memory)
- Artifacts and their locations

---

## Best Practices

1. **Name experiments descriptively** — Use a consistent naming convention like `{model}-{method}-{task}`
2. **Tag runs consistently** — Use tags for easy filtering (e.g., `baseline`, `production`, `experimental`)
3. **Log all hyperparameters** — Even defaults; this ensures reproducibility
4. **Use seeds and log them** — Set and record random seeds for reproducibility
5. **Log artifacts at key points** — Best checkpoint, final checkpoint, eval reports
6. **Compare against a baseline** — Always include a baseline run in your experiment
7. **Run ablations systematically** — Change one variable at a time
8. **Version your datasets** — Log dataset version/hash in run parameters
9. **Track compute costs** — Log GPU hours and memory usage for cost analysis
10. **Clean up old experiments** — Archive completed experiments regularly

---

## Related Documentation

- [Training Guide](training.md) — Fine-tuning and training workflows
- [Fine-Tuning Guide](fine_tuning.md) — LoRA, QLoRA, and full fine-tuning
- [Evaluation Guide](../api/rest.md) — Model evaluation endpoints
- [Configuration Guide](configuration.md) — Configuration file reference
