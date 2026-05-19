# Advanced Tutorial

Master advanced Nexus-LLM features: distributed training across multiple GPUs, model quantization for efficient deployment, and building custom agents with tool use.

---

## Prerequisites

- Completed [Beginner](./beginner.md) and [Intermediate](./intermediate.md) tutorials
- Familiarity with fine-tuning and RAG concepts
- Multi-GPU setup (for distributed training sections)
- Comfortable with Python and command line

**Time to complete:** ~90 minutes

---

## Part 1: Distributed Training

Train models across multiple GPUs or multiple machines for faster training and larger models.

### Multi-GPU Training (Single Machine)

Use `accelerate` for seamless multi-GPU training:

#### Configure Accelerate

```bash
accelerate config
```

Answer the prompts:

```
- How many different machines will you use? 1
- How many processes on each machine? 2
- What GPU IDs? 0,1
- Do you wish to use FP16 or BF16? bf16
```

This creates `~/.cache/huggingface/accelerate/default_config.yaml`.

#### Run Distributed Training

```bash
accelerate launch main.py \
  --mode train \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --dataset data/large_dataset.jsonl \
  --method lora \
  --rank 16 \
  --alpha 32 \
  --target-modules "q_proj,v_proj,k_proj,o_proj" \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --bf16 \
  --output-dir checkpoints/70b_lora
```

#### Manual Accelerate Config

Create `config/accelerate/multi_gpu.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_type: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Run with:

```bash
accelerate launch --config_file config/accelerate/multi_gpu.yaml main.py --mode train ...
```

### Multi-Node Training (Multiple Machines)

For very large models, distribute across multiple machines:

#### On the Master Node

```bash
accelerate launch \
  --config_file config/accelerate/multi_node.yaml \
  --main_process_ip 192.168.1.100 \
  --main_process_port 29500 \
  --num_machines 2 \
  --machine_rank 0 \
  main.py --mode train \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --dataset data/large_dataset.jsonl \
    --method lora \
    --epochs 3 \
    --output-dir checkpoints/70b_distributed
```

#### On the Worker Node

```bash
accelerate launch \
  --config_file config/accelerate/multi_node.yaml \
  --main_process_ip 192.168.1.100 \
  --main_process_port 29500 \
  --num_machines 2 \
  --machine_rank 1 \
  main.py --mode train \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --dataset data/large_dataset.jsonl \
    --method lora \
    --epochs 3 \
    --output-dir checkpoints/70b_distributed
```

### DeepSpeed Integration

For even larger models, use DeepSpeed ZeRO optimization:

```json
// config/deepspeed/zero3.json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

Run with DeepSpeed:

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/deepspeed/zero3.json \
  main.py --mode train \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --dataset data/large_dataset.jsonl \
    --method lora \
    --output-dir checkpoints/70b_deepspeed
```

---

## Part 2: Model Quantization

Quantization reduces model size and memory usage with minimal quality loss.

### Overview of Quantization Methods

| Method | Bits | Size Reduction | Quality Loss | Speed | Use Case |
|--------|------|---------------|-------------|-------|----------|
| FP16 | 16 | 1x (baseline) | None | Fast | Training, best quality |
| BF16 | 16 | 1x | None | Fast | Training (Ampere+ GPUs) |
| 8-bit (LLM.int8()) | 8 | ~2x | Minimal | Moderate | Memory-constrained inference |
| 4-bit (NF4/QLoRA) | 4 | ~4x | Low | Slower | QLoRA training, efficient inference |
| GPTQ | 4 | ~4x | Low | Fast | GPU inference (post-training) |
| AWQ | 4 | ~4x | Low | Fast | GPU inference (post-training) |
| GGUF (llama.cpp) | 2–8 | 2–8x | Varies | Moderate | CPU/edge inference |

### GPTQ Quantization

Quantize a model after training for efficient GPU inference:

```bash
nexus quantize \
  --model ./models/my_finetuned_model \
  --method gptq \
  --bits 4 \
  --group-size 128 \
  --desc-act true \
  --damp-percent 0.1 \
  --calibration-dataset "wikitext2" \
  --calibration-samples 128 \
  --output ./models/my_model_gptq
```

Use the quantized model:

```bash
./scripts/run.sh --mode chat --model ./models/my_model_gptq
```

### AWQ Quantization

Alternative 4-bit quantization with activation-aware weighting:

```bash
nexus quantize \
  --model ./models/my_finetuned_model \
  --method awq \
  --bits 4 \
  --group-size 128 \
  --zero-point true \
  --version gemma \
  --calibration-dataset "wikitext2" \
  --calibration-samples 128 \
  --output ./models/my_model_awq
```

### GGUF Export (for llama.cpp)

Export to GGUF format for CPU, Metal, or Vulkan inference:

```bash
nexus export \
  --model ./models/my_finetuned_model \
  --format gguf \
  --quantization q4_k_m \
  --output ./models/my_model.gguf
```

Available GGUF quantizations:

| Quantization | Description | Size (7B) | Quality |
|-------------|-------------|-----------|---------|
| `q8_0` | 8-bit | ~7.7 GB | Excellent |
| `q5_k_m` | 5-bit K-quants medium | ~5.1 GB | Very Good |
| `q4_k_m` | 4-bit K-quants medium | ~4.4 GB | Good |
| `q3_k_m` | 3-bit K-quants medium | ~3.5 GB | Acceptable |
| `q2_k` | 2-bit K-quants | ~2.9 GB | Degraded |

Then run with llama.cpp:

```bash
./llama-cli -m ./models/my_model.gguf -p "Hello, how are you?" -n 256
```

### On-the-Fly Quantization

Load models in 4-bit or 8-bit without pre-quantizing:

```yaml
# config/user.yaml
model:
  quantization: "4bit"
  quantization_config:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
```

---

## Part 3: Custom Agents with Tools

Build sophisticated agents that can use tools, plan multi-step workflows, and maintain state.

### Building a Research Agent

Create an agent that can search the web, read pages, and synthesize findings:

```python
# agents/research_agent.py
from nexus_llm.agents import Agent, tool
from typing import Optional

class ResearchAgent(Agent):
    """An agent that performs deep research on a topic."""

    name = "researcher"
    description = "Deep research agent with web search and analysis"

    model = "meta-llama/Llama-3.1-8B-Instruct"

    system_prompt = """You are a thorough research analyst. When given a research question:

1. **Plan**: Break the question into sub-questions
2. **Search**: Use web_search to find relevant information
3. **Read**: Use web_reader to extract details from promising URLs
4. **Analyze**: Synthesize findings across sources
5. **Report**: Write a comprehensive answer with citations

Always cite your sources. If sources conflict, note the disagreement.
Be honest about what you don't know."""

    max_iterations = 10

    @tool(description="Search the web for information")
    def web_search(self, query: str, num_results: int = 5) -> str:
        """Search the web and return top results."""
        import httpx

        response = httpx.post(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.config.get("search_api_key", "")},
            params={"q": query, "count": num_results}
        )

        results = response.json().get("web", {}).get("results", [])
        formatted = []
        for r in results:
            formatted.append(f"- [{r['title']}]({r['url']})\n  {r.get('description', '')}")

        return "\n\n".join(formatted) if formatted else "No results found."

    @tool(description="Read and extract text content from a URL")
    def web_reader(self, url: str, max_length: int = 5000) -> str:
        """Read a web page and extract its text content."""
        import httpx
        from bs4 import BeautifulSoup

        try:
            response = httpx.get(url, timeout=15, follow_redirects=True)
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, navigation
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            return text[:max_length]
        except Exception as e:
            return f"Error reading {url}: {e}"

    @tool(description="Save a research note for later reference")
    def save_note(self, key: str, content: str) -> str:
        """Save a note to the agent's working memory."""
        self.memory.working[key] = content
        return f"Note saved under key: {key}"

    @tool(description="Retrieve a previously saved note")
    def get_note(self, key: str) -> str:
        """Retrieve a saved note from working memory."""
        return self.memory.working.get(key, f"No note found for key: {key}")
```

### Building a Code Agent with Sandbox

```python
# agents/code_agent.py
import subprocess
import tempfile
import os
from nexus_llm.agents import Agent, tool

class CodeAgent(Agent):
    """An agent that can write and execute code safely."""

    name = "coder"
    description = "Code writing and execution agent"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    system_prompt = """You are a coding assistant. You can write and execute Python code.

Follow this workflow:
1. Understand the problem
2. Plan your approach
3. Write the code
4. Execute and test it
5. Debug if needed
6. Explain the solution

Always explain what you're doing before writing code. When code fails,
analyze the error message and fix it."""

    max_iterations = 15

    @tool(description="Execute Python code and return the output")
    def execute_python(self, code: str, timeout: int = 30) -> str:
        """Execute Python code in a temporary sandbox."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            output += f"Return code: {result.returncode}"

            return output
        except subprocess.TimeoutExpired:
            return f"Error: Execution timed out after {timeout} seconds"
        finally:
            os.unlink(temp_path)

    @tool(description="Install a Python package")
    def install_package(self, package: str) -> str:
        """Install a Python package using pip."""
        result = subprocess.run(
            ["pip", "install", package],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return f"Successfully installed {package}"
        return f"Error installing {package}:\n{result.stderr}"
```

### Registering Custom Agents

Add your agent to the configuration:

```yaml
# config/default.yaml
agents:
  researcher:
    module: "agents.research_agent"
    class: "ResearchAgent"
    config:
      search_api_key: "${BRAVE_SEARCH_API_KEY}"
    enabled: true

  coder:
    module: "agents.code_agent"
    class: "CodeAgent"
    config: {}
    enabled: true
```

Use the agent:

```bash
./scripts/run.sh --mode chat --agent researcher
```

---

## Part 4: Production Optimization

### Speculative Decoding

Use a smaller draft model to speed up inference:

```yaml
inference:
  speculative_decoding:
    enabled: true
    draft_model: "microsoft/Phi-3-mini-4k-instruct"
    max_draft_tokens: 5
    acceptance_threshold: 0.9
```

### Continuous Batching

Serve multiple concurrent requests efficiently:

```yaml
server:
  continuous_batching:
    enabled: true
    max_batch_size: 32
    max_waiting_tokens: 20
    scheduling_policy: "fcfs"    # fcfs, priority, shortest_first
```

### KV Cache Optimization

```yaml
inference:
  kv_cache:
    type: "paged"               # paged, unified
    max_cache_length: 2048
    block_size: 16
    gpu_memory_utilization: 0.9
```

### Model Sharding

For models too large for a single GPU:

```yaml
model:
  device_map: "auto"
  max_memory:
    0: "24GiB"
    1: "24GiB"
    2: "24GiB"
    3: "24GiB"
```

---

## Part 5: Monitoring and Observability

### Prometheus + Grafana Stack

```yaml
# docker-compose.monitoring.yml
version: "3.8"
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  grafana-data:
```

### Key Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|----------------|-------------|
| `nexus_inference_latency_p99` | > 5s | Tail latency too high |
| `nexus_gpu_memory_used_percent` | > 90% | GPU memory approaching limit |
| `nexus_request_error_rate` | > 1% | Error rate elevated |
| `nexus_queue_depth` | > 50 | Too many queued requests |
| `nexus_tokens_per_second` | < 100 | Throughput degraded |

---

## What's Next?

You've mastered advanced Nexus-LLM features! You now know how to:

- ✅ Train models across multiple GPUs and machines
- ✅ Quantize models for efficient deployment
- ✅ Build custom agents with tools
- ✅ Optimize for production workloads
- ✅ Set up monitoring and observability

Explore more:

- **[Architecture Documentation](../architecture/overview.md)** — Understand the system internals
- **[Plugin Guide](../guides/plugins.md)** — Extend Nexus-LLM with plugins
- **[Deployment Guide](../guides/deployment.md)** — Production deployment strategies
