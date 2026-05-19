#!/usr/bin/env python3
"""Nexus-LLM CLI Entry Point.

Main command-line interface for the Nexus-LLM framework providing
chat, serve, train, evaluation, and model management capabilities.
"""

import sys
import click
from rich.console import Console

from nexus_llm import __version__
from nexus_llm.constants import (
    APP_NAME,
    DEFAULT_MODEL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
)
from nexus_llm.cli import cli as nexus_cli

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name=APP_NAME)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--config", "-c", default=None, help="Path to configuration file.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: str | None) -> None:
    """Nexus-LLM: A powerful LLM framework for training, serving, and chatting."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config


@cli.command()
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model name or path to use.")
@click.option("--system", "-s", default=None, help="System prompt for the conversation.")
@click.option("--temperature", "-t", default=TEMPERATURE, type=float, help="Sampling temperature.")
@click.option("--top-p", default=TOP_P, type=float, help="Top-p (nucleus) sampling.")
@click.option("--top-k", default=TOP_K, type=int, help="Top-k sampling.")
@click.option("--max-tokens", default=MAX_TOKENS, type=int, help="Maximum tokens to generate.")
@click.option("--device", default=None, help="Device to use (auto/cpu/cuda/mps).")
@click.option("--no-history", is_flag=True, help="Disable conversation history.")
@click.pass_context
def chat(
    ctx: click.Context,
    model: str,
    system: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    device: str | None,
    no_history: bool,
) -> None:
    """Start an interactive chat session with an LLM."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config = {
        "model": model,
        "system_prompt": system,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "device": device,
        "use_history": not no_history,
    }
    app.run_chat(config)


@cli.command()
@click.option("--host", "-h", default=DEFAULT_HOST, help="Server host.")
@click.option("--port", "-p", default=DEFAULT_PORT, type=int, help="Server port.")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model name or path to use.")
@click.option("--workers", "-w", default=1, type=int, help="Number of worker processes.")
@click.option("--device", default=None, help="Device to use (auto/cpu/cuda/mps).")
@click.option("--api-key", default=None, help="API key for authentication.")
@click.option("--cors", is_flag=True, help="Enable CORS.")
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    model: str,
    workers: int,
    device: str | None,
    api_key: str | None,
    cors: bool,
) -> None:
    """Start the LLM inference server."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config = {
        "host": host,
        "port": port,
        "model": model,
        "workers": workers,
        "device": device,
        "api_key": api_key,
        "cors": cors,
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
@click.option("--lora-alpha", default=16, type=int, help="LoRA alpha.")
@click.option("--gradient-accumulation", default=1, type=int, help="Gradient accumulation steps.")
@click.option("--max-seq-length", default=2048, type=int, help="Maximum sequence length.")
@click.option("--fp16", is_flag=True, help="Use FP16 mixed precision.")
@click.option("--bf16", is_flag=True, help="Use BF16 mixed precision.")
@click.option("--device", default=None, help="Device to use.")
@click.pass_context
def train(ctx: click.Context, **kwargs) -> None:
    """Fine-tune an LLM on a dataset."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    app.run_train(kwargs)


@cli.command("train-data")
@click.option("--input", "-i", required=True, help="Input data file or directory.")
@click.option("--output", "-o", required=True, help="Output directory for processed data.")
@click.option("--format", "-f", default="jsonl", help="Output format (jsonl/json/parquet).")
@click.option("--split-ratio", default="0.9,0.05,0.05", help="Train/val/test split ratio.")
@click.option("--tokenizer", default=None, help="Tokenizer to use for token counting.")
@click.option("--max-length", default=2048, type=int, help="Maximum sequence length.")
@click.pass_context
def train_data(
    ctx: click.Context,
    input: str,
    output: str,
    format: str,
    split_ratio: str,
    tokenizer: str | None,
    max_length: int,
) -> None:
    """Prepare and process training data."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    ratios = [float(r.strip()) for r in split_ratio.split(",")]
    config = {
        "input_path": input,
        "output_path": output,
        "format": format,
        "split_ratio": ratios,
        "tokenizer_name": tokenizer,
        "max_length": max_length,
    }
    app.run_train_data(config)


@cli.command()
@click.option("--source", "-s", default="huggingface", help="Model source (huggingface/local).")
@click.option("--list-available", is_flag=True, help="List available models.")
@click.option("--filter", default=None, help="Filter models by name pattern.")
@click.pass_context
def models(ctx: click.Context, source: str, list_available: bool, filter: str | None) -> None:
    """List and manage available models."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config = {
        "source": source,
        "list_available": list_available,
        "filter": filter,
    }
    app.run_models(config)


@cli.command()
@click.argument("model_name")
@click.option("--source", "-s", default="huggingface", help="Model source.")
@click.option("--output", "-o", default=None, help="Output directory for downloaded model.")
@click.option("--revision", default=None, help="Model revision/branch to download.")
@click.option("--quantize", default=None, help="Quantization format (gptq/awq/bitsandbytes).")
@click.pass_context
def download(
    ctx: click.Context,
    model_name: str,
    source: str,
    output: str | None,
    revision: str | None,
    quantize: str | None,
) -> None:
    """Download a model from a remote source."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config = {
        "model_name": model_name,
        "source": source,
        "output_dir": output,
        "revision": revision,
        "quantize": quantize,
    }
    app.run_download(config)


@cli.command("eval")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model to evaluate.")
@click.option("--benchmark", "-b", default=None, help="Benchmark dataset to use.")
@click.option("--tasks", "-t", default=None, help="Comma-separated evaluation tasks.")
@click.option("--output", "-o", default="./eval_results", help="Output directory.")
@click.option("--device", default=None, help="Device to use.")
@click.option("--batch-size", default=8, type=int, help="Evaluation batch size.")
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    model: str,
    benchmark: str | None,
    tasks: str | None,
    output: str,
    device: str | None,
    batch_size: int,
) -> None:
    """Evaluate an LLM on benchmarks."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    task_list = tasks.split(",") if tasks else None
    config = {
        "model": model,
        "benchmark": benchmark,
        "tasks": task_list,
        "output_dir": output,
        "device": device,
        "batch_size": batch_size,
    }
    app.run_eval(config)


@cli.command()
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model to benchmark.")
@click.option("--device", default=None, help="Device to use.")
@click.option("--batch-sizes", default="1,2,4,8", help="Comma-separated batch sizes.")
@click.option("--seq-lengths", default="128,256,512,1024", help="Comma-separated sequence lengths.")
@click.option("--warmup", default=3, type=int, help="Number of warmup iterations.")
@click.option("--iterations", default=10, type=int, help="Number of benchmark iterations.")
@click.option("--output", "-o", default=None, help="Output file for results.")
@click.pass_context
def benchmark(
    ctx: click.Context,
    model: str,
    device: str | None,
    batch_sizes: str,
    seq_lengths: str,
    warmup: int,
    iterations: int,
    output: str | None,
) -> None:
    """Benchmark LLM inference performance."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config = {
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
@click.pass_context
def config(
    ctx: click.Context,
    key: str | None,
    value: str | None,
    list_all: bool,
    unset: bool,
) -> None:
    """View or modify configuration settings."""
    from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=ctx.obj.get("config"))
    config_data = {
        "key": key,
        "value": value,
        "list_all": list_all,
        "unset": unset,
    }
    app.run_config(config_data)


def main() -> None:
    """Main entry point for Nexus-LLM CLI."""
    try:
        cli(standalone_mode=True)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
