# Nexus-LLM Model Documentation

This document covers everything you need to know about using models with Nexus-LLM — from choosing the right model to downloading, switching, and optimizing them for your use case.

---

## 1. Available Models

Nexus-LLM supports 40 pre-configured models across multiple architectures and size ranges. Each model entry includes the model identifier, parameter count, VRAM requirements, and a brief description.

### GPT-2 Family (OpenAI)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `gpt2` | 124M | ~1 GB | The original GPT-2 base model. Fast and lightweight, suitable for simple text generation and testing. Limited in quality for complex tasks. |
| `gpt2-medium` | 355M | ~2 GB | Medium-sized GPT-2 variant. Good balance of speed and quality for general-purpose text generation. The default model for Nexus-LLM. |
| `gpt2-large` | 774M | ~3 GB | Large GPT-2 variant. Produces more coherent long-form text than smaller variants. |
| `gpt2-xl` | 1.5B | ~6 GB | The largest GPT-2 variant. Best quality in the GPT-2 family but requires more memory. Good for creative writing tasks. |

### DialoGPT Family (Microsoft)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `dialogpt-small` | 117M | ~1 GB | Conversational model trained on Reddit. Lightweight option for chat applications. |
| `dialogpt-medium` | 345M | ~2 GB | Medium conversational model. Better dialogue quality than the small variant. |
| `dialogpt-large` | 762M | ~3 GB | Large conversational model. Most natural dialogue flow in the DialoGPT family. |

### Phi Family (Microsoft)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `phi-1` | 1.3B | ~4 GB | Compact model trained for code generation. Excels at Python programming tasks. |
| `phi-1.5` | 1.3B | ~4 GB | General-purpose variant of Phi. Strong reasoning capabilities for its size. |
| `phi-2` | 2.7B | ~6 GB | Advanced small model with impressive reasoning and code capabilities. Punches well above its weight class. |

### TinyLlama Family

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `tinyllama` | 1.1B | ~2 GB | Compact LLaMA-based model. Great for resource-constrained environments while maintaining decent quality. |
| `tinyllama-chat` | 1.1B | ~2 GB | Instruction-tuned TinyLlama variant. Better at following directions and conversational tasks. |

### Gemma Family (Google)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `gemma-2b` | 2B | ~4 GB | Google's lightweight open model. Strong general performance for its parameter count. |
| `gemma-2b-it` | 2B | ~4 GB | Instruction-tuned Gemma 2B. Optimized for following user instructions and multi-turn conversation. |
| `gemma-7b` | 7B | ~14 GB | Full-sized Gemma model. Competitive with larger models on many benchmarks. |
| `gemma-7b-it` | 7B | ~14 GB | Instruction-tuned Gemma 7B. Excellent for complex instruction following. |

### LLaMA Family (Meta)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `llama-7b` | 7B | ~14 GB | Meta's base LLaMA model. Strong foundation model for general NLP tasks. |
| `llama-7b-chat` | 7B | ~14 GB | Chat-tuned LLaMA 7B. Optimized for dialogue and instruction following. |
| `llama-13b` | 13B | ~26 GB | Larger LLaMA variant. Improved quality for complex reasoning tasks. |
| `llama-13b-chat` | 13B | ~26 GB | Chat-tuned LLaMA 13B. Best dialogue quality in the open LLaMA family. |

### Mistral Family (Mistral AI)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `mistral-7b` | 7B | ~14 GB | Mistral's base model with sliding window attention. Excellent efficiency and quality. |
| `mistral-7b-instruct` | 7B | ~14 GB | Instruction-tuned Mistral 7B. One of the best 7B-class models for instruction following. |
| `mistral-7b-chat` | 7B | ~14 GB | Chat-optimized Mistral variant. Strong conversational abilities with low latency. |

### Qwen Family (Alibaba)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `qwen-1.8b` | 1.8B | ~4 GB | Compact multilingual model with strong Chinese and English capabilities. |
| `qwen-1.8b-chat` | 1.8B | ~4 GB | Chat-tuned Qwen 1.8B. Good for bilingual conversation. |
| `qwen-7b` | 7B | ~14 GB | Full Qwen model with robust multilingual support. |
| `qwen-7b-chat` | 7B | ~14 GB | Chat-tuned Qwen 7B. Excellent for multilingual dialogue and instruction following. |

### FLAN-T5 Family (Google)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `flan-t5-small` | 80M | ~0.5 GB | Ultra-lightweight instruction-tuned model. Best for very constrained environments. |
| `flan-t5-base` | 250M | ~1 GB | Base FLAN-T5 model. Good for simple instruction following and question answering. |
| `flan-t5-large` | 780M | ~3 GB | Large FLAN-T5 variant. Significantly better reasoning than base. |
| `flan-t5-xl` | 3B | ~7 GB | XL FLAN-T5. Strong zero-shot performance on diverse NLP tasks. |
| `flan-t5-xxl` | 11B | ~22 GB | The largest FLAN-T5. Best-in-class for instruction following among T5 models. |

### StableLM Family (Stability AI)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `stablelm-2-zephyr` | 1.6B | ~3 GB | Compact Zephyr-tuned model. Good conversational quality in a small package. |
| `stablelm-2-1.6b` | 1.6B | ~3 GB | Base StableLM 2 model. Versatile for its size. |
| `stablelm-2-12b` | 12B | ~24 GB | Large StableLM variant. Competitive with other 12B-class models. |
| `stablelm-2-12b-chat` | 12B | ~24 GB | Chat-tuned StableLM 12B. Strong dialogue capabilities. |

### Pythia Family (EleutherAI)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `pythia-70m` | 70M | ~0.5 GB | Smallest Pythia model. Useful for research and very fast prototyping. |
| `pythia-160m` | 160M | ~1 GB | Small Pythia variant. Good for testing pipelines. |
| `pythia-410m` | 410M | ~2 GB | Medium Pythia model. Decent quality for lightweight applications. |
| `pythia-1b` | 1B | ~3 GB | 1B Pythia variant. Good balance for research and light production use. |
| `pythia-1.4b` | 1.4B | ~4 GB | Larger Pythia model. Improved coherence over smaller variants. |
| `pythia-2.8b` | 2.8B | ~6 GB | The largest standard Pythia. Best quality in the family. |

### OPT Family (Meta)

| Model | Parameters | VRAM | Description |
|-------|-----------|------|-------------|
| `opt-125m` | 125M | ~1 GB | Smallest OPT model. Fast but limited quality. Good for testing. |
| `opt-350m` | 350M | ~2 GB | Small OPT variant. Better quality than 125M while remaining lightweight. |
| `opt-1.3b` | 1.3B | ~4 GB | Medium OPT model. Reasonable quality for general text generation. |
| `opt-2.7b` | 2.7B | ~6 GB | Large OPT model. Good for tasks requiring more coherence and reasoning. |
| `opt-6.7b` | 6.7B | ~14 GB | Very large OPT. Competitive quality but requires significant resources. |

---

## 2. How to Choose a Model

Choosing the right model depends on three main factors: **your hardware**, **your task**, and **your quality requirements**. Here's a systematic approach to making the right choice.

### Step 1: Check Your Hardware

The most important constraint is your GPU's VRAM (or system RAM if using CPU mode):

| Available VRAM | Recommended Models |
|----------------|-------------------|
| < 2 GB | `gpt2`, `flan-t5-small`, `pythia-70m`, `opt-125m` |
| 2-4 GB | `gpt2-medium`, `dialogpt-medium`, `tinyllama`, `phi-1.5` |
| 4-8 GB | `phi-2`, `gemma-2b-it`, `qwen-1.8b-chat`, `flan-t5-xl` |
| 8-16 GB | `mistral-7b-instruct`, `gemma-7b-it`, `llama-7b-chat` |
| 16-32 GB | `llama-13b-chat`, `stablelm-2-12b-chat`, `flan-t5-xxl` |
| 32+ GB | Any model; consider running multiple instances |

### Step 2: Match Your Task

Different models excel at different tasks:
- **Conversation/Chat**: Look for models with `-chat` or `-instruct` suffixes
- **Code Generation**: `phi-1`, `phi-2`, `mistral-7b-instruct`
- **Creative Writing**: `gpt2-xl`, `mistral-7b`, `llama-13b`
- **Question Answering**: `flan-t5-xl`, `gemma-7b-it`, `mistral-7b-instruct`
- **Multilingual**: `qwen-7b-chat`, `gemma-7b-it`
- **Summarization**: `flan-t5-large`, `mistral-7b-instruct`

### Step 3: Consider the Trade-offs

- **Smaller models** are faster and use less memory but produce lower-quality output
- **Larger models** generate better text but require more resources and are slower
- **Instruction-tuned models** (`-it`, `-instruct`, `-chat`) follow directions better but may be less creative
- **Base models** are more flexible for fine-tuning but require better prompting

---

## 3. Model Categories Explained

Nexus-LLM organizes models into categories based on their training methodology and intended use:

### Base Models

Base models are trained on large text corpora using next-token prediction. They generate text continuations and are the most flexible for fine-tuning, but require careful prompting for specific tasks. Examples: `gpt2`, `mistral-7b`, `llama-7b`, `pythia-2.8b`.

**Best for**: Fine-tuning, research, creative text generation, when you need maximum flexibility.

### Instruction-Tuned Models

These models undergo additional training on instruction-following datasets (like FLAN, Alpaca, or Open-Orca). They learn to understand and execute user instructions rather than just continuing text. Identified by `-it` or `-instruct` suffixes. Examples: `gemma-2b-it`, `mistral-7b-instruct`, `flan-t5-base`.

**Best for**: Question answering, task execution, summarization, analysis, when you want the model to follow directions precisely.

### Chat-Tuned Models

Chat models are specifically trained for multi-turn dialogue with instruction and alignment training. They understand conversational context, maintain persona consistency, and know when to ask clarifying questions. Identified by `-chat` suffixes. Examples: `dialogpt-medium`, `llama-7b-chat`, `qwen-7b-chat`, `tinyllama-chat`.

**Best for**: Interactive chatbots, virtual assistants, customer service bots, any application requiring natural conversation flow.

### Code Models

Code models are specialized for programming tasks. They're trained on code repositories and can generate, explain, debug, and refactor code. Examples: `phi-1`, `phi-2`.

**Best for**: Code generation, code explanation, debugging assistance, documentation generation.

### Multilingual Models

Models trained with significant multilingual data that perform well across multiple languages. Examples: `qwen-7b-chat` (Chinese, English), `gemma-7b-it` (multilingual).

**Best for**: Applications serving users in multiple languages, translation assistance, cross-lingual tasks.

---

## 4. Downloading Models

Nexus-LLM automatically downloads models from HuggingFace on first use, but you can also pre-download models for offline use or faster startup.

### Automatic Download (Recommended)

Simply use a model and it downloads automatically:

```bash
# This will download the model on first run
nexus-llm chat --model phi-2
```

Models are cached in `~/.cache/huggingface/hub/` by default. You can change the cache directory:

```bash
nexus-llm config --set model.cache_dir /path/to/cache
```

### Pre-Download Models

Download a model without running it:

```bash
# Download a specific model
nexus-llm model download phi-2

# Download with specific quantization
nexus-llm model download mistral-7b-instruct --quantization 4bit
```

### Download All Popular Models

```bash
# Download the recommended set for your hardware
nexus-llm model download --recommended

# Download all small models (< 2B parameters)
nexus-llm model download --category small
```

### Using HuggingFace CLI

For more control over downloads:

```bash
# Install the HuggingFace Hub CLI
pip install huggingface_hub

# Download a model
huggingface-cli download microsoft/phi-2

# Download to a specific directory
huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2
```

### Using Local Models

You can use models stored on your local filesystem:

```bash
# Point to a local model directory
nexus-llm chat --model ./models/my-fine-tuned-model

# Or set it in config
nexus-llm config --set model.name ./models/my-fine-tuned-model
```

The local directory must contain at minimum: `config.json`, `tokenizer.json` (or `tokenizer_config.json`), and model weight files (`pytorch_model.bin`, `model.safetensors`, or sharded variants).

### Managing Disk Space

Model files can be large. Here are some tips:

```bash
# List all cached models and their sizes
nexus-llm model list --cached

# Remove a cached model
nexus-llm model remove gpt2-medium

# Show total cache size
du -sh ~/.cache/huggingface/hub/
```

---

## 5. Switching Models

Nexus-LLM makes it easy to switch between models on the fly, whether you're using the terminal, CLI, or API server.

### Switching in Terminal Chat

While in an interactive chat session, use the `/model` command:

```
nexus-llm> /model phi-2
Loading phi-2...
Model loaded: phi-2 (2.7B parameters, 4.2 GB VRAM)

nexus-llm> /model
Current model: phi-2

nexus-llm> /model mistral-7b-instruct
Unloading phi-2...
Loading mistral-7b-instruct...
Model loaded: mistral-7b-instruct (7.2B parameters, 13.8 GB VRAM)
```

### Switching via CLI

Specify the model with the `--model` flag on any command:

```bash
# Chat with a specific model
nexus-llm chat --model gemma-2b-it

# Generate text with a specific model
nexus-llm generate --model phi-2 --prompt "Write a haiku about programming"

# Start the server with a specific model
nexus-llm serve --model mistral-7b-instruct
```

### Switching via Configuration

Set the default model in your configuration file:

```bash
# Set the default model
nexus-llm config --set model.name phi-2

# This model will be used when no --model flag is specified
nexus-llm chat
```

Or edit `config/default_config.yaml` directly:

```yaml
model:
  name: phi-2
  # ... other settings
```

### Switching via API

When running the API server, you can switch models through the API:

```bash
# Load a new model
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "phi-2"}'

# List loaded models
curl http://localhost:8000/v1/models

# Unload a model
curl -X POST http://localhost:8000/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2-medium"}'
```

### Hot-Swapping Models

The server supports loading multiple models simultaneously (memory permitting). You can specify which model to use per request:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2",
    "prompt": "Explain recursion",
    "max_length": 200
  }'
```

---

## 6. Recommended Models by Use Case

Here are our top recommendations for common use cases, considering both quality and resource requirements.

### General Chat and Q&A

| Use Case | Best Model | Runner-Up | Budget Option |
|----------|-----------|-----------|---------------|
| General chat | `mistral-7b-instruct` | `gemma-7b-it` | `tinyllama-chat` |
| Factual Q&A | `gemma-7b-it` | `mistral-7b-instruct` | `flan-t5-large` |
| Instruction following | `mistral-7b-instruct` | `llama-7b-chat` | `gemma-2b-it` |

### Code and Development

| Use Case | Best Model | Runner-Up | Budget Option |
|----------|-----------|-----------|---------------|
| Code generation | `phi-2` | `mistral-7b-instruct` | `phi-1.5` |
| Code explanation | `phi-2` | `qwen-7b-chat` | `phi-1` |
| Debugging help | `mistral-7b-instruct` | `phi-2` | `gemma-2b-it` |

### Creative and Content

| Use Case | Best Model | Runner-Up | Budget Option |
|----------|-----------|-----------|---------------|
| Creative writing | `llama-13b-chat` | `mistral-7b` | `gpt2-xl` |
| Brainstorming | `mistral-7b-instruct` | `gemma-7b-it` | `tinyllama-chat` |
| Storytelling | `llama-13b` | `mistral-7b` | `gpt2-large` |

### Analysis and Research

| Use Case | Best Model | Runner-Up | Budget Option |
|----------|-----------|-----------|---------------|
| Summarization | `mistral-7b-instruct` | `flan-t5-xl` | `flan-t5-large` |
| Analysis | `gemma-7b-it` | `mistral-7b-instruct` | `phi-2` |
| Data interpretation | `flan-t5-xl` | `mistral-7b-instruct` | `flan-t5-large` |

### Multilingual

| Use Case | Best Model | Runner-Up | Budget Option |
|----------|-----------|-----------|---------------|
| Chinese/English | `qwen-7b-chat` | `qwen-1.8b-chat` | `qwen-1.8b` |
| European languages | `gemma-7b-it` | `mistral-7b-instruct` | `gemma-2b-it` |
| Translation | `qwen-7b-chat` | `gemma-7b-it` | `flan-t5-large` |

### Resource-Constrained Environments

| Environment | Best Model | VRAM Needed |
|-------------|-----------|-------------|
| Raspberry Pi / Edge | `pythia-70m` | ~0.5 GB |
| Old laptop (CPU only) | `gpt2` | ~1 GB RAM |
| 4 GB GPU | `phi-2` | ~4 GB |
| 8 GB GPU | `mistral-7b-instruct` (4-bit) | ~4 GB |
| 8 GB GPU (fp16) | `gemma-2b-it` | ~4 GB |

### Production Deployment

For production API servers where reliability and consistency matter:

1. **`mistral-7b-instruct`** — Best overall quality-to-cost ratio. Handles diverse tasks well.
2. **`gemma-7b-it`** — Excellent instruction following with Google's safety training.
3. **`phi-2`** — Best for code-heavy workloads with limited resources.
4. **`flan-t5-xl`** — Best for structured output tasks (classification, extraction, QA).

When deploying in production, always use quantization (4-bit or 8-bit) to reduce memory footprint and increase throughput with minimal quality loss.
