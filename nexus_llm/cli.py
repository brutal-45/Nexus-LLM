"""Nexus-LLM CLI Module.

Full command-line interface using Click with all commands and options.
Provides the main CLI group and all subcommands for chat, serve, train,
evaluation, model management, and configuration.
"""

import sys
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from nexus_llm.__version__ import __version__, get_version_info
from nexus_llm.constants import (
    APP_NAME,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name=APP_NAME)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output.")
@click.option("--config", "-c", default=None, type=click.Path(), help="Path to configuration file.")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False), help="Set logging level.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config: Optional[str], log_level: Optional[str]) -> None:
    """Nexus-LLM: A powerful LLM framework for training, serving, and chatting.

    Use one of the available commands to get started. For detailed help
    on each command, use: nexus-llm COMMAND --help
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config"] = config
    ctx.obj["log_level"] = log_level

    if verbose and quiet:
        console.print("[bold red]Error:[/bold red] Cannot use both --verbose and --quiet.")
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model name or path to use.")
@click.option("--system", "-s", default=None, help="System prompt for the conversation.")
@click.option("--temperature", "-t", default=TEMPERATURE, type=float, help="Sampling temperature.")
@click.option("--top-p", default=TOP_P, type=float, help="Top-p (nucleus) sampling parameter.")
@click.option("--top-k", default=TOP_K, type=int, help="Top-k sampling parameter.")
@click.option("--max-tokens", default=MAX_TOKENS, type=int, help="Maximum tokens to generate.")
@click.option("--device", default=None, help="Device to use (auto/cpu/cuda/mps).")
@click.option("--no-history", is_flag=True, help="Disable conversation history.")
@click.option("--prompt", "-p", default=None, help="Non-interactive: send a single prompt and exit.")
@click.pass_context
def chat(
    ctx: click.Context,
    model: str,
    system: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    device: Optional[str],
    no_history: bool,
    prompt: Optional[str],
) -> None:
    """Start an interactive chat session with an LLM.

    Engage in a multi-turn conversation with a language model. Use --system
    to set the system prompt and --no-history to start fresh each turn.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config: Dict[str, Any] = {
        "model": model,
        "system_prompt": system,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "device": device,
        "use_history": not no_history,
        "single_prompt": prompt,
    }
    app.run_chat(config)


@cli.command()
@click.option("--host", "-h", default=DEFAULT_HOST, help="Server host address.")
@click.option("--port", "-p", default=DEFAULT_PORT, type=int, help="Server port number.")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model name or path to serve.")
@click.option("--workers", "-w", default=1, type=int, help="Number of worker processes.")
@click.option("--device", default=None, help="Device to use (auto/cpu/cuda/mps).")
@click.option("--api-key", default=None, help="API key for authentication.")
@click.option("--cors", is_flag=True, help="Enable CORS headers.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
@click.option("--ssl-certfile", default=None, type=click.Path(), help="SSL certificate file.")
@click.option("--ssl-keyfile", default=None, type=click.Path(), help="SSL key file.")
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    model: str,
    workers: int,
    device: Optional[str],
    api_key: Optional[str],
    cors: bool,
    reload: bool,
    ssl_certfile: Optional[str],
    ssl_keyfile: Optional[str],
) -> None:
    """Start the LLM inference server.

    Launches a FastAPI-based server that provides REST API and WebSocket
    endpoints for model inference.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config: Dict[str, Any] = {
        "host": host,
        "port": port,
        "model": model,
        "workers": workers,
        "device": device,
        "api_key": api_key,
        "cors": cors,
        "reload": reload,
        "ssl_certfile": ssl_certfile,
        "ssl_keyfile": ssl_keyfile,
    }
    app.run_serve(config)


@cli.command()
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Base model name or path.")
@click.option("--dataset", "-d", required=True, help="Path to training dataset.")
@click.option("--output", "-o", default="./output", help="Output directory for checkpoints.")
@click.option("--epochs", "-e", default=3, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=8, type=int, help="Training batch size.")
@click.option("--learning-rate", "-lr", default=2e-5, type=float, help="Learning rate.")
@click.option("--lora-rank", default=8, type=int, help="LoRA rank (0 to disable).")
@click.option("--lora-alpha", default=16, type=int, help="LoRA alpha parameter.")
@click.option("--lora-dropout", default=0.05, type=float, help="LoRA dropout rate.")
@click.option("--gradient-accumulation", default=1, type=int, help="Gradient accumulation steps.")
@click.option("--max-seq-length", default=2048, type=int, help="Maximum sequence length.")
@click.option("--fp16", is_flag=True, help="Use FP16 mixed precision training.")
@click.option("--bf16", is_flag=True, help="Use BF16 mixed precision training.")
@click.option("--device", default=None, help="Device to use.")
@click.option("--warmup-steps", default=100, type=int, help="Number of warmup steps.")
@click.option("--weight-decay", default=0.01, type=float, help="Weight decay.")
@click.option("--save-steps", default=500, type=int, help="Save checkpoint every N steps.")
@click.option("--eval-steps", default=500, type=int, help="Evaluate every N steps.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def train(ctx: click.Context, **kwargs: Any) -> None:
    """Fine-tune an LLM on a dataset.

    Train or fine-tune a language model using LoRA/QLoRA for parameter-efficient
    fine-tuning, or full fine-tuning with --lora-rank 0.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    app.run_train(kwargs)


@cli.command("train-data")
@click.option("--input", "-i", required=True, help="Input data file or directory.")
@click.option("--output", "-o", required=True, help="Output directory for processed data.")
@click.option("--format", "-f", default="jsonl", type=click.Choice(["jsonl", "json", "parquet", "csv"]), help="Output format.")
@click.option("--split-ratio", default="0.9,0.05,0.05", help="Train/val/test split ratio (comma-separated).")
@click.option("--tokenizer", default=None, help="Tokenizer to use for token counting.")
@click.option("--max-length", default=2048, type=int, help="Maximum sequence length.")
@click.option("--shuffle", is_flag=True, help="Shuffle data before splitting.")
@click.option("--seed", default=42, type=int, help="Random seed for shuffling.")
@click.pass_context
def train_data(
    ctx: click.Context,
    input: str,
    output: str,
    format: str,
    split_ratio: str,
    tokenizer: Optional[str],
    max_length: int,
    shuffle: bool,
    seed: int,
) -> None:
    """Prepare and process training data.

    Converts raw data into the format required for training, with options
    for splitting, shuffling, and token counting.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    ratios = [float(r.strip()) for r in split_ratio.split(",")]
    config: Dict[str, Any] = {
        "input_path": input,
        "output_path": output,
        "format": format,
        "split_ratio": ratios,
        "tokenizer_name": tokenizer,
        "max_length": max_length,
        "shuffle": shuffle,
        "seed": seed,
    }
    app.run_train_data(config)


@cli.command()
@click.option("--source", "-s", default="huggingface", type=click.Choice(["huggingface", "local"]), help="Model source.")
@click.option("--list-available", is_flag=True, help="List available models.")
@click.option("--filter", default=None, help="Filter models by name pattern.")
@click.option("--details", is_flag=True, help="Show detailed model information.")
@click.pass_context
def models(ctx: click.Context, source: str, list_available: bool, filter: Optional[str], details: bool) -> None:
    """List and manage available models.

    Display information about available models, their sizes, and sources.
    Use --list-available to see all supported models.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config: Dict[str, Any] = {
        "source": source,
        "list_available": list_available,
        "filter": filter,
        "details": details,
    }
    app.run_models(config)


@cli.command()
@click.argument("model_name")
@click.option("--source", "-s", default="huggingface", type=click.Choice(["huggingface", "local"]), help="Model source.")
@click.option("--output", "-o", default=None, help="Output directory for downloaded model.")
@click.option("--revision", default=None, help="Model revision/branch to download.")
@click.option("--quantize", default=None, type=click.Choice(["gptq", "awq", "bitsandbytes", "none"]), help="Quantization format.")
@click.option("--token", default=None, help="Hugging Face API token.")
@click.option("--no-verify", is_flag=True, help="Skip hash verification after download.")
@click.pass_context
def download(
    ctx: click.Context,
    model_name: str,
    source: str,
    output: Optional[str],
    revision: Optional[str],
    quantize: Optional[str],
    token: Optional[str],
    no_verify: bool,
) -> None:
    """Download a model from a remote source.

    Downloads a model from Hugging Face Hub or other sources. Supports
    quantization and hash verification.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config: Dict[str, Any] = {
        "model_name": model_name,
        "source": source,
        "output_dir": output,
        "revision": revision,
        "quantize": quantize,
        "token": token,
        "verify": not no_verify,
    }
    app.run_download(config)


@cli.command("eval")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model to evaluate.")
@click.option("--benchmark", "-b", default=None, help="Benchmark dataset to use.")
@click.option("--tasks", "-t", default=None, help="Comma-separated evaluation tasks.")
@click.option("--output", "-o", default="./eval_results", help="Output directory for results.")
@click.option("--device", default=None, help="Device to use.")
@click.option("--batch-size", default=8, type=int, help="Evaluation batch size.")
@click.option("--num-fewshot", default=0, type=int, help="Number of few-shot examples.")
@click.option("--limit", default=None, type=int, help="Limit number of examples.")
@click.option("--fp16", is_flag=True, help="Use FP16 for evaluation.")
@click.option("--bf16", is_flag=True, help="Use BF16 for evaluation.")
@click.option("--save-predictions", is_flag=True, help="Save model predictions.")
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    model: str,
    benchmark: Optional[str],
    tasks: Optional[str],
    output: str,
    device: Optional[str],
    batch_size: int,
    num_fewshot: int,
    limit: Optional[int],
    fp16: bool,
    bf16: bool,
    save_predictions: bool,
) -> None:
    """Evaluate an LLM on benchmarks.

    Run evaluation on standard benchmarks like MMLU, HellaSwag, ARC,
    TruthfulQA, and more.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    task_list = tasks.split(",") if tasks else None
    config: Dict[str, Any] = {
        "model": model,
        "benchmark": benchmark,
        "tasks": task_list,
        "output_dir": output,
        "device": device,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "fp16": fp16,
        "bf16": bf16,
        "save_predictions": save_predictions,
    }
    app.run_eval(config)


@cli.command()
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model to benchmark.")
@click.option("--device", default=None, help="Device to use.")
@click.option("--batch-sizes", default="1,2,4,8", help="Comma-separated batch sizes.")
@click.option("--seq-lengths", default="128,256,512,1024", help="Comma-separated sequence lengths.")
@click.option("--warmup", default=3, type=int, help="Number of warmup iterations.")
@click.option("--iterations", default=10, type=int, help="Number of benchmark iterations.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file for results (JSON).")
@click.pass_context
def benchmark(
    ctx: click.Context,
    model: str,
    device: Optional[str],
    batch_sizes: str,
    seq_lengths: str,
    warmup: int,
    iterations: int,
    output: Optional[str],
) -> None:
    """Benchmark LLM inference performance.

    Measure inference latency and throughput across different batch sizes
    and sequence lengths.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config: Dict[str, Any] = {
        "model": model,
        "device": device,
        "batch_sizes": [int(b.strip()) for b in batch_sizes.split(",")],
        "seq_lengths": [int(s.strip()) for s in seq_lengths.split(",")],
        "warmup": warmup,
        "iterations": iterations,
        "output_file": output,
    }
    app.run_benchmark(config)


@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--list", "list_all", is_flag=True, help="List all configuration values.")
@click.option("--unset", is_flag=True, help="Unset a configuration key.")
@click.option("--reset", is_flag=True, help="Reset configuration to defaults.")
@click.option("--show-source", is_flag=True, help="Show where each config value comes from.")
@click.pass_context
def config(
    ctx: click.Context,
    key: Optional[str],
    value: Optional[str],
    list_all: bool,
    unset: bool,
    reset: bool,
    show_source: bool,
) -> None:
    """View or modify configuration settings.

    Get, set, or list configuration values. Configuration is loaded from
    files, environment variables, and defaults.
    """
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config_data: Dict[str, Any] = {
        "key": key,
        "value": value,
        "list_all": list_all,
        "unset": unset,
        "reset": reset,
        "show_source": show_source,
    }
    app.run_config(config_data)


@cli.command("info")
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display information about the Nexus-LLM installation.

    Shows version, Python info, installed dependencies, and available hardware.
    """
    version_info = get_version_info()

    panel_content = Text()
    panel_content.append(f"Nexus-LLM v{__version__}\n", style="bold cyan")
    panel_content.append(f"Author: {version_info['author']}\n")
    panel_content.append(f"License: {version_info['license']}\n")
    panel_content.append(f"Release: {version_info['release_status']}\n")
    panel_content.append(f"Python: {sys.version}\n")

    console.print(Panel(panel_content, title="Nexus-LLM Info", border_style="cyan"))

    # Check dependencies
    table = Table(title="Dependency Status")
    table.add_column("Package", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Status", style="yellow")

    dependencies = [
        "torch", "transformers", "accelerate", "peft", "datasets",
        "tokenizers", "fastapi", "uvicorn", "click", "rich",
        "pyyaml", "numpy", "pandas",
    ]

    for dep in dependencies:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "installed")
            table.add_row(dep, ver, "[green]✓[/green]")
        except ImportError:
            table.add_row(dep, "-", "[red]✗[/red]")

    console.print(table)

    # Check hardware
    try:
        import torch
        if torch.cuda.is_available():
            console.print(f"\n[bold green]CUDA available:[/bold green] {torch.cuda.get_device_name(0)}")
            console.print(f"[dim]CUDA version: {torch.version.cuda}[/dim]")
            console.print(f"[dim]GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB[/dim]")
        else:
            console.print("\n[yellow]CUDA not available - using CPU[/yellow]")
    except ImportError:
        pass
