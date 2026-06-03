"""Click-based CLI entry point for Nexus-LLM.

Provides the ``nexus-llm`` command with subcommands for chat, serving,
training, model management, configuration, and system info.

Run as:
    python -m nexus_llm          (via __main__.py)
    nexus-llm                    (after pip install)
"""

import os
import platform
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from nexus_llm import __version__
from nexus_llm.core.config import Settings, get_settings, DEFAULT_CONFIG_PATH
from nexus_llm.core.exceptions import NexusLLMError
from nexus_llm.core.model_catalog import (
    MODEL_CATALOG,
    get_model_info,
    list_models,
    list_categories,
    get_recommended_models,
)
from nexus_llm.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Rich console (shared across commands)
# ---------------------------------------------------------------------------

console = Console()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Display the Nexus-LLM startup banner using Rich."""
    banner_text = Text()
    banner_text.append("Nexus-LLM", style="bold cyan")
    banner_text.append(f"  v{__version__}\n", style="dim")
    banner_text.append("Terminal-based LLM chat with local inference backend", style="italic dim")

    panel = Panel(
        banner_text,
        box=box.HEAVY,
        border_style="cyan",
        expand=False,
        padding=(1, 4),
    )
    console.print(panel)
    console.print()


# ---------------------------------------------------------------------------
# Error handling decorator
# ---------------------------------------------------------------------------

def _handle_errors(func):
    """Decorator that wraps a Click command with graceful error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NexusLLMError as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
            sys.exit(1)
        except ImportError as exc:
            missing = str(exc).replace("No module named '", "").replace("'", "")
            console.print(f"\n[bold red]Missing dependency:[/bold red] {missing}")
            console.print(f"  Run: [cyan]pip install {missing}[/cyan]")
            console.print(f"  Or:  [cyan]pip install -e .[/cyan]\n")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n\n[bold]Interrupted.[/bold] Goodbye from Nexus-LLM!\n")
            sys.exit(0)
        except Exception as exc:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {exc}\n")
            raise
    # Preserve Click metadata so --help etc. still work
    wrapper.__doc__ = func.__doc__
    wrapper = click.pass_context(wrapper) if hasattr(func, '__click_params__') else wrapper
    import functools
    return functools.wraps(func)(wrapper)


# ===================================================================
# Root group — ``nexus-llm`` (defaults to chat)
# ===================================================================

@click.group(invoke_without_command=True)
@click.option(
    "--version", "-V",
    is_flag=True,
    help="Show the Nexus-LLM version and exit.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with full tracebacks.",
)
@click.option(
    "--config", "-c",
    "config_path",
    envvar="NEXUS_CONFIG",
    default=None,
    type=click.Path(exists=False),
    help="Path to a YAML configuration file.",
)
@click.pass_context
def cli(ctx, version, debug, config_path):
    """Nexus-LLM — Terminal-based LLM chat with local inference.

    Run without a subcommand to start an interactive chat session.

    \b
    Quick start:
        nexus-llm                  # open chat (default)
        nexus-llm chat             # same as above
        nexus-llm serve            # start API server
        nexus-llm models           # list available models
        nexus-llm info             # show system info
    """
    # --version
    if version:
        console.print(f"Nexus-LLM v{__version__}")
        return

    # Ensure a context object exists for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["config_path"] = config_path

    # Initialise settings
    try:
        settings = get_settings(config_path=config_path)
    except Exception:
        settings = Settings()
    if debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    ctx.obj["settings"] = settings

    # Setup logging
    setup_logger(
        name="nexus_llm",
        level=settings.log_level,
        log_file=settings.log_file if not debug else None,
    )

    # No subcommand → default to chat
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


# ===================================================================
# chat — Interactive chat
# ===================================================================

@cli.command()
@click.option(
    "--model", "-m",
    default=None,
    help="Model ID to load on startup (e.g. gpt2-medium, phi-2).",
)
@click.option(
    "--device", "-d",
    default=None,
    type=click.Choice(["auto", "cpu", "cuda", "mps"], case_sensitive=False),
    help="Compute device for inference.",
)
@click.option(
    "--precision", "-p",
    default=None,
    type=click.Choice(["fp32", "fp16", "bf16", "8bit", "4bit"], case_sensitive=False),
    help="Model precision / quantisation mode.",
)
@click.option(
    "--theme",
    default=None,
    type=click.Choice(["dark", "light", "monokai", "solarized"]),
    help="Terminal colour theme.",
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable streaming output (single-shot generation).",
)
@click.option(
    "--system-prompt",
    default=None,
    help="Set a custom system prompt for the session.",
)
@click.pass_context
@_handle_errors  # type: ignore[arg-type]
def chat(ctx, model, device, precision, theme, no_stream, system_prompt):
    """Start an interactive chat session with a local LLM.

    \b
    Examples:
        nexus-llm chat
        nexus-llm chat --model phi-2 --device cuda
        nexus-llm chat --precision fp16 --no-stream
    """
    _print_banner()

    settings: Settings = ctx.obj["settings"]

    # Apply CLI overrides to settings
    if model is not None:
        settings.model.name = model
    if device is not None:
        settings.model.device = device
    if precision is not None:
        settings.model.precision = precision
    if theme is not None:
        settings.terminal.theme = theme
    if no_stream:
        settings.terminal.streaming = False

    # Launch the terminal chat
    from nexus_llm.terminal.chat import TerminalChat

    chat_session = TerminalChat(settings=settings)

    # Inject system prompt if provided
    if system_prompt:
        chat_session.ctx["system_prompt"] = system_prompt

    chat_session.run()


# ===================================================================
# serve — API server
# ===================================================================

@cli.command()
@click.option(
    "--host", "-h",
    default=None,
    help="Bind address (default: 127.0.0.1).",
)
@click.option(
    "--port", "-p",
    default=None,
    type=int,
    help="Bind port (default: 8000).",
)
@click.option(
    "--workers", "-w",
    default=None,
    type=int,
    help="Number of uvicorn workers (default: 1).",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Model ID to pre-load on server startup.",
)
@click.option(
    "--device", "-d",
    default=None,
    type=click.Choice(["auto", "cpu", "cuda", "mps"], case_sensitive=False),
    help="Compute device for the pre-loaded model.",
)
@click.option(
    "--precision",
    default=None,
    type=click.Choice(["fp32", "fp16", "bf16", "8bit", "4bit"], case_sensitive=False),
    help="Precision mode for the pre-loaded model.",
)
@click.option(
    "--reload",
    "uvicorn_reload",
    is_flag=True,
    help="Enable uvicorn auto-reload (development mode).",
)
@click.pass_context
@_handle_errors  # type: ignore[arg-type]
def serve(ctx, host, port, workers, model, device, precision, uvicorn_reload):
    """Start the Nexus-LLM API server.

    Launches a uvicorn server backed by FastAPI with full REST and
    WebSocket endpoints for model management and text generation.

    \b
    Examples:
        nexus-llm serve
        nexus-llm serve --host 0.0.0.0 --port 8080
        nexus-llm serve --model gpt2-medium --device cuda
    """
    _print_banner()

    settings: Settings = ctx.obj["settings"]

    # Apply CLI overrides
    if host is not None:
        settings.server.host = host
    if port is not None:
        settings.server.port = port
    if workers is not None:
        settings.server.workers = workers

    from nexus_llm.backend.server import LLMServer

    server = LLMServer(
        host=settings.server.host,
        port=settings.server.port,
        cors_origins=settings.server.cors_origins,
    )

    # Pre-load model if requested
    if model is not None:
        console.print(f"[cyan]Pre-loading model:[/cyan] {model}")
        server.engine.load_model(
            model_id=model,
            device=device or settings.model.device,
            precision=precision or settings.model.precision,
            cache_dir=settings.model.cache_dir,
        )
        console.print(f"[green]Model '{model}' loaded successfully.[/green]\n")

    console.print(
        f"[bold green]Starting Nexus-LLM server[/bold green] "
        f"on http://{settings.server.host}:{settings.server.port}"
    )

    uvicorn_kwargs = {}
    if uvicorn_reload:
        uvicorn_kwargs["reload"] = True
    if workers and workers > 1:
        uvicorn_kwargs["workers"] = workers

    server.run(**uvicorn_kwargs)


# ===================================================================
# train — Fine-tuning
# ===================================================================

@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    help="Model ID from the catalog to fine-tune.",
)
@click.option(
    "--data", "-d",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data (JSONL, JSON, CSV, or HuggingFace dataset ID).",
)
@click.option(
    "--data-format",
    default="auto",
    type=click.Choice(["auto", "alpaca", "chat", "instruction"]),
    help="Format of the training data.",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(),
    help="Directory to save checkpoints and the final model.",
)
@click.option(
    "--epochs", "-e",
    "num_epochs",
    default=None,
    type=int,
    help="Number of training epochs (default: 3).",
)
@click.option(
    "--batch-size", "-b",
    default=None,
    type=int,
    help="Per-device batch size (default: 4).",
)
@click.option(
    "--learning-rate", "-l",
    "lr",
    default=None,
    type=float,
    help="Learning rate (default: 2e-4).",
)
@click.option(
    "--no-lora",
    is_flag=True,
    help="Disable LoRA — train the full model instead.",
)
@click.option(
    "--lora-r",
    default=None,
    type=int,
    help="LoRA rank (default: 8).",
)
@click.option(
    "--resume-from",
    default=None,
    type=click.Path(exists=True),
    help="Path to a checkpoint to resume training from.",
)
@click.pass_context
@_handle_errors  # type: ignore[arg-type]
def train(ctx, model, data_path, data_format, output_dir, num_epochs,
          batch_size, lr, no_lora, lora_r, resume_from):
    """Fine-tune a model with LoRA or full training.

    Wraps HuggingFace Trainer with Nexus-LLM model management and
    LoRA configuration for efficient fine-tuning.

    \b
    Examples:
        nexus-llm train -m gpt2-medium -d data/train.jsonl
        nexus-llm train -m phi-2 -d ./data --epochs 5 --lora-r 16
    """
    _print_banner()

    settings: Settings = ctx.obj["settings"]

    # Apply overrides to training settings
    if output_dir is not None:
        settings.training.output_dir = output_dir
    if num_epochs is not None:
        settings.training.num_epochs = num_epochs
    if batch_size is not None:
        settings.training.batch_size = batch_size
    if lr is not None:
        settings.training.learning_rate = lr
    if lora_r is not None:
        settings.training.lora_r = lora_r

    # Load model + tokenizer
    console.print(f"[cyan]Loading model:[/cyan] {model}")
    from nexus_llm.backend.inference import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(
        model_id=model,
        device=settings.model.device,
        precision=settings.model.precision,
        cache_dir=settings.model.cache_dir,
    )
    console.print(f"[green]Model '{model}' loaded.[/green]\n")

    # Initialise trainer
    from nexus_llm.training.trainer import NexusTrainer

    trainer = NexusTrainer(settings=settings)
    tokenizer = engine.tokenizer_manager.tokenizer

    console.print("[cyan]Starting training...[/cyan]")
    console.print(f"  Data:       {data_path}")
    console.print(f"  Format:     {data_format}")
    console.print(f"  Output:     {settings.training.output_dir}")
    console.print(f"  Epochs:     {settings.training.num_epochs}")
    console.print(f"  Batch size: {settings.training.batch_size}")
    console.print(f"  LoRA:       {'disabled' if no_lora else f'r={settings.training.lora_r}'}")
    console.print()

    result = trainer.train(
        model=engine.model_manager.model,
        tokenizer=tokenizer,
        data_path=data_path,
        data_format=data_format,
        apply_lora=not no_lora,
        lora_r=lora_r,
        resume_from_checkpoint=resume_from,
    )

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"  Final loss:  {result.get('training_loss', 'N/A')}")
    console.print(f"  Steps:       {result.get('global_step', 'N/A')}")
    console.print(f"  Output:      {result.get('output_dir', 'N/A')}")


# ===================================================================
# models — List available models
# ===================================================================

@cli.command(name="models")
@click.option(
    "--category", "-c",
    default=None,
    help="Filter models by category (e.g. gpt2, phi, llama, qwen).",
)
@click.option(
    "--recommended",
    is_flag=True,
    help="Show only recommended models.",
)
@click.option(
    "--json", "as_json",
    is_flag=True,
    help="Output as JSON (useful for scripting).",
)
@click.pass_context
@_handle_errors  # type: ignore[arg-type]
def models(ctx, category, recommended, as_json):
    """List available models in the catalog.

    Shows model ID, name, size, and minimum RAM for each model.
    Use --category to filter by family or --recommended for top picks.

    \b
    Examples:
        nexus-llm models
        nexus-llm models --recommended
        nexus-llm models --category phi
    """
    if recommended:
        model_list = get_recommended_models()
    else:
        model_list = list_models(category=category)

    if not model_list:
        console.print("[yellow]No models found matching the given filters.[/yellow]")
        return

    # JSON output
    if as_json:
        import json as _json
        data = []
        for m in model_list:
            data.append({
                "id": m.id,
                "name": m.name,
                "hf_id": m.hf_id,
                "category": m.category,
                "size": m.size,
                "params": m.params,
                "model_type": m.model_type,
                "recommended": m.recommended,
                "min_ram_gb": m.min_ram_gb,
                "description": m.description,
            })
        console.print_json(_json.dumps(data, indent=2))
        return

    # Rich table
    table = Table(
        title="Available Models" if not recommended else "Recommended Models",
        box=box.ROUNDED,
        show_lines=False,
        title_style="bold cyan",
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Category", style="dim")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Type", style="dim")
    table.add_column("RAM", style="yellow", justify="right")
    table.add_column("★", justify="center")

    for m in model_list:
        star = "[bold yellow]★[/bold yellow]" if m.recommended else ""
        table.add_row(
            m.id,
            m.name,
            m.category,
            m.size,
            m.model_type,
            f"{m.min_ram_gb} GB",
            star,
        )

    console.print(table)
    console.print(
        f"\n[dim]{len(model_list)} model(s) shown. "
        f"Categories: {', '.join(list_categories())}[/dim]"
    )


# ===================================================================
# download — Download a model
# ===================================================================

@cli.command()
@click.argument("model_id")
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(),
    help="Directory to cache the downloaded model.",
)
@click.option(
    "--precision",
    default="fp32",
    type=click.Choice(["fp32", "fp16", "bf16", "8bit", "4bit"], case_sensitive=False),
    help="Precision mode for the downloaded model.",
)
@click.pass_context
@_handle_errors  # type: ignore[arg-type]
def download(ctx, model_id, cache_dir, precision):
    """Download a model from HuggingFace Hub.

    Downloads the model weights and tokenizer so they are available
    for offline use. Use ``nexus-llm models`` to see available IDs.

    \b
    Examples:
        nexus-llm download gpt2-medium
        nexus-llm download phi-2 --cache-dir ./my-models
    """
    settings: Settings = ctx.obj["settings"]

    # Validate model ID
    info = get_model_info(model_id)

    console.print(f"[cyan]Downloading model:[/cyan] {info.name} ({info.hf_id})")
    console.print(f"  Parameters:  {info.params}")
    console.print(f"  Category:    {info.category}")
    console.print(f"  Precision:   {precision}")
    if cache_dir:
        console.print(f"  Cache dir:   {cache_dir}")
    console.print()

    from nexus_llm.backend.model_manager import ModelManager

    manager = ModelManager()
    with console.status("[bold green]Downloading...[/bold green]", spinner="dots"):
        manager.load(
            model_id=model_id,
            device="cpu",
            precision=precision,
            cache_dir=cache_dir or settings.model.cache_dir,
        )

    console.print(f"\n[bold green]Model '{model_id}' downloaded successfully![/bold green]")


# ===================================================================
# config — Show / edit configuration
# ===================================================================

@cli.command()
@click.option(
    "--show",
    "action_show",
    is_flag=True,
    help="Display the current configuration.",
)
@click.option(
    "--path",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to the configuration file to show or edit.",
)
@click.option(
    "--set", "set_values",
    multiple=True,
    nargs=2,
    help="Set a config value: --set model.device cuda",
)
@click.option(
    "--reset",
    is_flag=True,
    help="Reset configuration to defaults.",
)
@click.pass_context
@_handle_errors  # type: ignore[arg-type]
def config(ctx, action_show, config_path, set_values, reset):
    """Show or edit the Nexus-LLM configuration.

    Without options, displays the current configuration. Use --set
    to change individual values or --reset to restore defaults.

    \b
    Examples:
        nexus-llm config
        nexus-llm config --set model.device cuda
        nexus-llm config --set model.name phi-2 --set server.port 8080
        nexus-llm config --reset
    """
    settings: Settings = ctx.obj["settings"]
    target_path = config_path or str(DEFAULT_CONFIG_PATH)

    # Determine the effective action (default: show)
    if reset:
        # Reset to defaults and save
        from nexus_llm.core.config import reset_settings
        reset_settings()
        fresh = Settings()
        fresh.to_yaml(target_path)
        console.print(f"[green]Configuration reset to defaults.[/green]")
        console.print(f"  Saved to: {target_path}")
        return

    if set_values:
        # Apply key=value overrides
        for key, value in set_values:
            _apply_config_override(settings, key, value)
        settings.to_yaml(target_path)
        console.print(f"[green]Configuration updated.[/green]")
        console.print(f"  Saved to: {target_path}")
        for key, value in set_values:
            console.print(f"  {key} = {value}")
        return

    # Default action: show current config
    _show_config(settings, target_path)


def _apply_config_override(settings: Settings, key: str, value: str) -> None:
    """Apply a dotted-key config override (e.g. 'model.device' → 'cuda')."""
    import dataclasses

    parts = key.split(".")
    if len(parts) != 2:
        raise click.BadParameter(
            f"Config key must be in 'section.key' format, got: {key}"
        )

    section_name, attr = parts
    section_map = {
        "model": settings.model,
        "server": settings.server,
        "terminal": settings.terminal,
        "training": settings.training,
    }

    section = section_map.get(section_name)
    if section is None:
        raise click.BadParameter(
            f"Unknown section '{section_name}'. "
            f"Valid sections: {', '.join(section_map)}"
        )

    if not hasattr(section, attr):
        raise click.BadParameter(
            f"Unknown key '{attr}' in section '{section_name}'."
        )

    # Coerce the value to the correct type
    current = getattr(section, attr)
    if isinstance(current, bool):
        setattr(section, attr, value.lower() in ("true", "1", "yes"))
    elif isinstance(current, int):
        setattr(section, attr, int(value))
    elif isinstance(current, float):
        setattr(section, attr, float(value))
    else:
        setattr(section, attr, value)


def _show_config(settings: Settings, config_path: str) -> None:
    """Display the current configuration in a Rich table."""
    import dataclasses

    console.print(
        Panel(
            f"[bold]Configuration[/bold]\n[dim]{config_path}[/dim]",
            box=box.ROUNDED,
            border_style="cyan",
            expand=False,
        )
    )

    for section_name in ("model", "server", "terminal", "training"):
        section = getattr(settings, section_name)
        if not dataclasses.is_dataclass(section):
            continue

        table = Table(
            title=section_name.capitalize(),
            box=box.SIMPLE,
            show_header=True,
            title_style="bold",
        )
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Type", style="dim")

        for field in dataclasses.fields(section):
            val = getattr(section, field.name)
            # Show a clean type name (e.g. "str", "int", "Optional[str]")
            raw_type = field.type
            if isinstance(raw_type, str):
                type_str = raw_type
            elif hasattr(raw_type, "__name__"):
                type_str = raw_type.__name__
            else:
                type_str = str(raw_type).replace("typing.", "")
            table.add_row(field.name, str(val), type_str)

        console.print(table)
        console.print()

    # Top-level settings
    table = Table(title="General", box=box.SIMPLE, show_header=True, title_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("debug", str(settings.debug))
    table.add_row("log_level", settings.log_level)
    table.add_row("log_file", settings.log_file)
    console.print(table)


# ===================================================================
# info — System information
# ===================================================================

@cli.command()
@click.pass_context
@_handle_errors  # type: ignore[arg-type]
def info(ctx):
    """Show system information and hardware capabilities.

    Displays Python version, OS, CPU, RAM, GPU (if available),
    and installed library versions relevant to Nexus-LLM.

    \b
    Example:
        nexus-llm info
    """
    _print_banner()

    settings: Settings = ctx.obj["settings"]

    # --- System table ---
    sys_table = Table(
        title="System Information",
        box=box.ROUNDED,
        show_header=False,
        title_style="bold cyan",
    )
    sys_table.add_column("Key", style="bold white", width=20)
    sys_table.add_column("Value", style="green")

    sys_table.add_row("Nexus-LLM", f"v{__version__}")
    sys_table.add_row("Python", f"{platform.python_version()} ({platform.python_implementation()})")
    sys_table.add_row("OS", f"{platform.system()} {platform.release()}")
    sys_table.add_row("Architecture", platform.machine())
    sys_table.add_row("Platform", platform.platform())

    console.print(sys_table)
    console.print()

    # --- Hardware table ---
    hw_table = Table(
        title="Hardware",
        box=box.ROUNDED,
        show_header=False,
        title_style="bold cyan",
    )
    hw_table.add_column("Key", style="bold white", width=20)
    hw_table.add_column("Value", style="green")

    # CPU
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        hw_table.add_row("CPU Cores", str(cpu_count))
    except Exception:
        hw_table.add_row("CPU Cores", "unknown")

    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        hw_table.add_row("RAM Total", f"{ram.total / (1024**3):.1f} GB")
        hw_table.add_row("RAM Available", f"{ram.available / (1024**3):.1f} GB")
        hw_table.add_row("RAM Used", f"{ram.percent}%")
    except ImportError:
        hw_table.add_row("RAM", "[dim]psutil not installed[/dim]")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                hw_table.add_row(f"GPU {i}", f"{name} ({mem:.1f} GB)")
            hw_table.add_row("CUDA", f"v{torch.version.cuda}")
        else:
            hw_table.add_row("GPU", "No CUDA GPU detected")
    except ImportError:
        hw_table.add_row("GPU", "[dim]PyTorch not installed[/dim]")

    # MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw_table.add_row("MPS", "Available (Apple Silicon)")
    except Exception:
        pass

    console.print(hw_table)
    console.print()

    # --- Libraries table ---
    lib_table = Table(
        title="Installed Libraries",
        box=box.ROUNDED,
        show_header=False,
        title_style="bold cyan",
    )
    lib_table.add_column("Package", style="bold white", width=20)
    lib_table.add_column("Version", style="green")

    for pkg in ("torch", "transformers", "accelerate", "bitsandbytes",
                "peft", "datasets", "click", "rich", "fastapi", "uvicorn",
                "prompt_toolkit", "yaml"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "installed")
            lib_table.add_row(pkg, ver)
        except ImportError:
            lib_table.add_row(pkg, "[dim]not installed[/dim]")

    console.print(lib_table)

    # --- Config path ---
    console.print()
    console.print(f"[dim]Config file: {DEFAULT_CONFIG_PATH}[/dim]")
    console.print(f"[dim]Model cache: {settings.model.cache_dir}[/dim]")


# ===================================================================
# Entry point — for ``from nexus_llm.cli import cli; cli()``
# ===================================================================

def main():
    """Run the Nexus-LLM CLI."""
    cli()


if __name__ == "__main__":
    main()
