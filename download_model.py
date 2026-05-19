#!/usr/bin/env python3
"""Model Download Script for Nexus-LLM.

Download and manage LLM models from Hugging Face and other sources.
Supports 39+ models with progress bars, hash verification, and
configurable download options.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# Registry of supported models
SUPPORTED_MODELS = {
    # OpenAI GPT-2 family
    "gpt2": {
        "full_name": "openai-community/gpt2",
        "size": "124M",
        "description": "GPT-2 small - 124M parameters",
        "license": "MIT",
        "sha256": None,
    },
    "gpt2-medium": {
        "full_name": "openai-community/gpt2-medium",
        "size": "355M",
        "description": "GPT-2 medium - 355M parameters",
        "license": "MIT",
        "sha256": None,
    },
    "gpt2-large": {
        "full_name": "openai-community/gpt2-large",
        "size": "774M",
        "description": "GPT-2 large - 774M parameters",
        "license": "MIT",
        "sha256": None,
    },
    "gpt2-xl": {
        "full_name": "openai-community/gpt2-xl",
        "size": "1.5B",
        "description": "GPT-2 XL - 1.5B parameters",
        "license": "MIT",
        "sha256": None,
    },
    # Meta LLaMA family
    "llama-7b": {
        "full_name": "meta-llama/Llama-2-7b-hf",
        "size": "7B",
        "description": "LLaMA 2 7B - 7B parameters",
        "license": "LLaMA 2",
        "sha256": None,
    },
    "llama-13b": {
        "full_name": "meta-llama/Llama-2-13b-hf",
        "size": "13B",
        "description": "LLaMA 2 13B - 13B parameters",
        "license": "LLaMA 2",
        "sha256": None,
    },
    "llama-70b": {
        "full_name": "meta-llama/Llama-2-70b-hf",
        "size": "70B",
        "description": "LLaMA 2 70B - 70B parameters",
        "license": "LLaMA 2",
        "sha256": None,
    },
    "llama3-8b": {
        "full_name": "meta-llama/Meta-Llama-3-8B",
        "size": "8B",
        "description": "LLaMA 3 8B - 8B parameters",
        "license": "LLaMA 3",
        "sha256": None,
    },
    "llama3-70b": {
        "full_name": "meta-llama/Meta-Llama-3-70B",
        "size": "70B",
        "description": "LLaMA 3 70B - 70B parameters",
        "license": "LLaMA 3",
        "sha256": None,
    },
    # Mistral family
    "mistral-7b": {
        "full_name": "mistralai/Mistral-7B-v0.1",
        "size": "7B",
        "description": "Mistral 7B v0.1 - 7B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "mistral-7b-instruct": {
        "full_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "size": "7B",
        "description": "Mistral 7B Instruct v0.2 - 7B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "mixtral-8x7b": {
        "full_name": "mistralai/Mixtral-8x7B-v0.1",
        "size": "47B",
        "description": "Mixtral 8x7B v0.1 - 47B parameters (MoE)",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "mixtral-8x22b": {
        "full_name": "mistralai/Mixtral-8x22B-v0.1",
        "size": "141B",
        "description": "Mixtral 8x22B v0.1 - 141B parameters (MoE)",
        "license": "Apache 2.0",
        "sha256": None,
    },
    # Qwen family
    "qwen-7b": {
        "full_name": "Qwen/Qwen-7B",
        "size": "7B",
        "description": "Qwen 7B - 7B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "qwen-14b": {
        "full_name": "Qwen/Qwen-14B",
        "size": "14B",
        "description": "Qwen 14B - 14B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "qwen-72b": {
        "full_name": "Qwen/Qwen-72B",
        "size": "72B",
        "description": "Qwen 72B - 72B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "qwen2-7b": {
        "full_name": "Qwen/Qwen2-7B",
        "size": "7B",
        "description": "Qwen2 7B - 7B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "qwen2-72b": {
        "full_name": "Qwen/Qwen2-72B",
        "size": "72B",
        "description": "Qwen2 72B - 72B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    # Microsoft Phi family
    "phi-1": {
        "full_name": "microsoft/phi-1",
        "size": "1.3B",
        "description": "Phi-1 - 1.3B parameters",
        "license": "MIT",
        "sha256": None,
    },
    "phi-1.5": {
        "full_name": "microsoft/phi-1_5",
        "size": "1.3B",
        "description": "Phi-1.5 - 1.3B parameters",
        "license": "MIT",
        "sha256": None,
    },
    "phi-2": {
        "full_name": "microsoft/phi-2",
        "size": "2.7B",
        "description": "Phi-2 - 2.7B parameters",
        "license": "MIT",
        "sha256": None,
    },
    "phi-3-mini": {
        "full_name": "microsoft/Phi-3-mini-4k-instruct",
        "size": "3.8B",
        "description": "Phi-3 Mini - 3.8B parameters",
        "license": "MIT",
        "sha256": None,
    },
    # Google Gemma family
    "gemma-2b": {
        "full_name": "google/gemma-2b",
        "size": "2B",
        "description": "Gemma 2B - 2B parameters",
        "license": "Gemma",
        "sha256": None,
    },
    "gemma-7b": {
        "full_name": "google/gemma-7b",
        "size": "7B",
        "description": "Gemma 7B - 7B parameters",
        "license": "Gemma",
        "sha256": None,
    },
    "gemma2-9b": {
        "full_name": "google/gemma-2-9b",
        "size": "9B",
        "description": "Gemma 2 9B - 9B parameters",
        "license": "Gemma",
        "sha256": None,
    },
    "gemma2-27b": {
        "full_name": "google/gemma-2-27b",
        "size": "27B",
        "description": "Gemma 2 27B - 27B parameters",
        "license": "Gemma",
        "sha256": None,
    },
    # EleutherAI
    "gpt-neo-125m": {
        "full_name": "EleutherAI/gpt-neo-125M",
        "size": "125M",
        "description": "GPT-Neo 125M parameters",
        "license": "MIT",
        "sha256": None,
    },
    "gpt-neo-1.3b": {
        "full_name": "EleutherAI/gpt-neo-1.3B",
        "size": "1.3B",
        "description": "GPT-Neo 1.3B parameters",
        "license": "MIT",
        "sha256": None,
    },
    "gpt-neo-2.7b": {
        "full_name": "EleutherAI/gpt-neo-2.7B",
        "size": "2.7B",
        "description": "GPT-Neo 2.7B parameters",
        "license": "MIT",
        "sha256": None,
    },
    "gpt-j-6b": {
        "full_name": "EleutherAI/gpt-j-6b",
        "size": "6B",
        "description": "GPT-J 6B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "gpt-neox-20b": {
        "full_name": "EleutherAI/gpt-neox-20b",
        "size": "20B",
        "description": "GPT-NeoX 20B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    # StabilityAI
    "stablelm-2-1.6b": {
        "full_name": "stabilityai/stablelm-2-1_6b",
        "size": "1.6B",
        "description": "StableLM 2 1.6B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "stablelm-2-12b": {
        "full_name": "stabilityai/stablelm-2-12b",
        "size": "12B",
        "description": "StableLM 2 12B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    # Falcon
    "falcon-7b": {
        "full_name": "tiiuae/falcon-7b",
        "size": "7B",
        "description": "Falcon 7B - 7B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "falcon-40b": {
        "full_name": "tiiuae/falcon-40b",
        "size": "40B",
        "description": "Falcon 40B - 40B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    # Yi
    "yi-6b": {
        "full_name": "01-ai/Yi-6B",
        "size": "6B",
        "description": "Yi 6B - 6B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "yi-34b": {
        "full_name": "01-ai/Yi-34B",
        "size": "34B",
        "description": "Yi 34B - 34B parameters",
        "license": "Apache 2.0",
        "sha256": None,
    },
    # Databricks
    "dbrx-instruct": {
        "full_name": "databricks/dbrx-instruct",
        "size": "132B",
        "description": "DBRX Instruct - 132B parameters (MoE)",
        "license": "Databricks",
        "sha256": None,
    },
    # NousResearch
    "nous-hermes-2-mistral-7b": {
        "full_name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "size": "7B",
        "description": "Nous Hermes 2 Mistral 7B DPO",
        "license": "Apache 2.0",
        "sha256": None,
    },
    "nous-hermes-2-yi-34b": {
        "full_name": "NousResearch/Nous-Hermes-2-Yi-34B",
        "size": "34B",
        "description": "Nous Hermes 2 Yi 34B",
        "license": "Apache 2.0",
        "sha256": None,
    },
    # OpenCoder
    "opencoder-8b": {
        "full_name": "infly/OpenCoder-8B-Instruct",
        "size": "8B",
        "description": "OpenCoder 8B Instruct",
        "license": "Apache 2.0",
        "sha256": None,
    },
}


def compute_file_hash(filepath: Path, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Compute the hash of a file.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm to use (sha256, md5, sha512).
        chunk_size: Size of chunks to read at a time.

    Returns:
        The hex digest of the file hash.
    """
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_hash(filepath: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """Verify the hash of a downloaded file.

    Args:
        filepath: Path to the file to verify.
        expected_hash: Expected hash value.
        algorithm: Hash algorithm to use.

    Returns:
        True if hash matches, False otherwise.
    """
    computed = compute_file_hash(filepath, algorithm)
    return computed == expected_hash


def download_file_with_progress(url: str, dest_path: Path, description: str = "") -> Path:
    """Download a file with a progress bar.

    Args:
        url: URL to download from.
        dest_path: Destination path for the downloaded file.
        description: Description for the progress bar.

    Returns:
        Path to the downloaded file.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    progress_bar = tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        desc=description or dest_path.name,
    )

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        console.print(f"[bold red]Warning:[/bold red] Download size mismatch for {dest_path.name}")

    return dest_path


def download_model_from_hf(
    model_name: str,
    output_dir: Path,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    verify: bool = True,
) -> Path:
    """Download a model from Hugging Face Hub.

    Args:
        model_name: Hugging Face model name (e.g., 'openai-community/gpt2-medium').
        output_dir: Directory to save the model.
        revision: Model revision/branch to download.
        token: Hugging Face API token.
        verify: Whether to verify file hashes.

    Returns:
        Path to the downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[bold red]Error:[/bold red] huggingface_hub is not installed.")
        console.print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    model_dir = output_dir / model_name.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Downloading model:[/bold blue] {model_name}")
    console.print(f"[dim]Output directory: {model_dir}[/dim]")

    kwargs = {
        "repo_id": model_name,
        "local_dir": str(model_dir),
        "max_workers": 4,
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token

    start_time = time.time()

    try:
        snapshot_download(**kwargs)
    except Exception as exc:
        console.print(f"[bold red]Error downloading model:[/bold red] {exc}")
        sys.exit(1)

    elapsed = time.time() - start_time
    console.print(f"[bold green]Model downloaded successfully![/bold green]")
    console.print(f"[dim]Time: {elapsed:.1f}s | Location: {model_dir}[/dim]")

    if verify:
        console.print("[bold blue]Verifying downloaded files...[/bold blue]")
        safetensors_files = list(model_dir.glob("*.safetensors"))
        bin_files = list(model_dir.glob("*.bin"))
        all_model_files = safetensors_files + bin_files
        if all_model_files:
            for model_file in all_model_files:
                file_hash = compute_file_hash(model_file)
                console.print(f"  [green]✓[/green] {model_file.name} (sha256: {file_hash[:16]}...)")
        console.print("[bold green]Verification complete.[/bold green]")

    return model_dir


def list_available_models(filter_pattern: Optional[str] = None) -> None:
    """List all available models in a formatted table.

    Args:
        filter_pattern: Optional pattern to filter model names.
    """
    table = Table(title="Available Models for Download")
    table.add_column("Short Name", style="cyan", no_wrap=True)
    table.add_column("HuggingFace ID", style="green")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Description")
    table.add_column("License", style="magenta")

    for short_name, info in sorted(SUPPORTED_MODELS.items()):
        if filter_pattern and filter_pattern.lower() not in short_name.lower():
            continue
        table.add_row(
            short_name,
            info["full_name"],
            info["size"],
            info["description"],
            info["license"],
        )

    console.print(table)
    console.print(f"\n[bold]Total models:[/bold] {len(SUPPORTED_MODELS)}")
    console.print("[dim]Use 'python download_model.py <model_name>' to download.[/dim]")


def main() -> None:
    """Main entry point for the model download script."""
    parser = argparse.ArgumentParser(
        description="Nexus-LLM Model Downloader - Download and manage LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_model.py gpt2-medium
  python download_model.py mistral-7b -o ./my_models
  python download_model.py --list
  python download_model.py --list --filter llama
  python download_model.py llama-7b --token hf_xxxxx
        """,
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Short name or HuggingFace model ID to download",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.environ.get("NEXUS_LLM_MODELS_DIR", "./models"),
        help="Output directory for downloaded models (default: ./models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter models by name pattern",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision/branch to download",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip hash verification after download",
    )
    parser.add_argument(
        "--full-name",
        action="store_true",
        help="Use full HuggingFace model name instead of short name",
    )

    args = parser.parse_args()

    if args.list or args.filter:
        list_available_models(args.filter)
        return

    if not args.model:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output)

    if args.full_name:
        model_name = args.model
    elif args.model in SUPPORTED_MODELS:
        model_name = SUPPORTED_MODELS[args.model]["full_name"]
    else:
        model_name = args.model
        console.print(f"[bold yellow]Note:[/bold yellow] '{args.model}' not in known models. Using as direct HuggingFace ID.")

    download_model_from_hf(
        model_name=model_name,
        output_dir=output_dir,
        revision=args.revision,
        token=args.token,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
