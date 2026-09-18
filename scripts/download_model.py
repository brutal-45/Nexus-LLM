#!/usr/bin/env python3
"""Standalone model download script for Nexus-LLM.

Downloads a model from HuggingFace Hub with progress tracking.
Supports all models in the Nexus-LLM catalog.

Usage:
    python scripts/download_model.py                    # download default (gpt2-medium)
    python scripts/download_model.py phi-2              # download a specific model
    python scripts/download_model.py --list             # list available models
    python scripts/download_model.py --list --category phi  # list by category
    python scripts/download_model.py tinyllama --cache-dir ./my-models

Requirements:
    pip install huggingface_hub transformers torch tqdm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model catalog — mirrors nexus_llm.core.model_catalog
# ---------------------------------------------------------------------------
# This is duplicated here so the script can run standalone without the
# full nexus_llm package installed. Keep in sync with model_catalog.py.
# ---------------------------------------------------------------------------

MODELS: dict[str, dict[str, str]] = {
    # GPT-2 Family
    "gpt2":              {"hf_id": "openai-community/gpt2",           "name": "GPT-2 Small",       "category": "gpt2",     "size": "124M"},
    "gpt2-medium":       {"hf_id": "openai-community/gpt2-medium",    "name": "GPT-2 Medium",      "category": "gpt2",     "size": "355M"},
    "gpt2-large":        {"hf_id": "openai-community/gpt2-large",     "name": "GPT-2 Large",       "category": "gpt2",     "size": "774M"},
    "gpt2-xl":           {"hf_id": "openai-community/gpt2-xl",        "name": "GPT-2 XL",          "category": "gpt2",     "size": "1.5B"},
    # DialoGPT Family
    "dialogpt-small":    {"hf_id": "microsoft/DialoGPT-small",        "name": "DialoGPT Small",    "category": "dialogpt", "size": "117M"},
    "dialogpt-medium":   {"hf_id": "microsoft/DialoGPT-medium",       "name": "DialoGPT Medium",   "category": "dialogpt", "size": "345M"},
    "dialogpt-large":    {"hf_id": "microsoft/DialoGPT-large",        "name": "DialoGPT Large",    "category": "dialogpt", "size": "762M"},
    # Phi Family
    "phi-1":             {"hf_id": "microsoft/phi-1",                  "name": "Phi-1",             "category": "phi",      "size": "1.3B"},
    "phi-1.5":           {"hf_id": "microsoft/phi-1_5",               "name": "Phi-1.5",           "category": "phi",      "size": "1.3B"},
    "phi-2":             {"hf_id": "microsoft/phi-2",                  "name": "Phi-2",             "category": "phi",      "size": "2.7B"},
    # Pythia Family
    "pythia-70m":        {"hf_id": "EleutherAI/pythia-70m",           "name": "Pythia 70M",        "category": "pythia",   "size": "70M"},
    "pythia-160m":       {"hf_id": "EleutherAI/pythia-160m",          "name": "Pythia 160M",       "category": "pythia",   "size": "160M"},
    "pythia-410m":       {"hf_id": "EleutherAI/pythia-410m",          "name": "Pythia 410M",       "category": "pythia",   "size": "410M"},
    "pythia-1b":         {"hf_id": "EleutherAI/pythia-1b",            "name": "Pythia 1B",         "category": "pythia",   "size": "1B"},
    "pythia-1.4b":       {"hf_id": "EleutherAI/pythia-1.4b",          "name": "Pythia 1.4B",       "category": "pythia",   "size": "1.4B"},
    "pythia-2.8b":       {"hf_id": "EleutherAI/pythia-2.8b",          "name": "Pythia 2.8B",       "category": "pythia",   "size": "2.8B"},
    # OPT Family
    "opt-125m":          {"hf_id": "facebook/opt-125m",               "name": "OPT 125M",          "category": "opt",      "size": "125M"},
    "opt-350m":          {"hf_id": "facebook/opt-350m",               "name": "OPT 350M",          "category": "opt",      "size": "350M"},
    "opt-1.3b":          {"hf_id": "facebook/opt-1.3b",               "name": "OPT 1.3B",          "category": "opt",      "size": "1.3B"},
    "opt-2.7b":          {"hf_id": "facebook/opt-2.7b",               "name": "OPT 2.7B",          "category": "opt",      "size": "2.7B"},
    # TinyLlama
    "tinyllama":         {"hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama 1.1B", "category": "llama",    "size": "1.1B"},
    # Qwen
    "qwen2.5-0.5b":      {"hf_id": "Qwen/Qwen2.5-0.5B",             "name": "Qwen2.5 0.5B",     "category": "qwen",     "size": "0.5B"},
    "qwen2.5-1.5b":      {"hf_id": "Qwen/Qwen2.5-1.5B",             "name": "Qwen2.5 1.5B",     "category": "qwen",     "size": "1.5B"},
    "qwen2.5-3b":        {"hf_id": "Qwen/Qwen2.5-3B",               "name": "Qwen2.5 3B",       "category": "qwen",     "size": "3B"},
    # SmolLM
    "smollm-135m":       {"hf_id": "HuggingFaceTB/SmolLM-135M",      "name": "SmolLM 135M",      "category": "smollm",   "size": "135M"},
    "smollm-360m":       {"hf_id": "HuggingFaceTB/SmolLM-360M",      "name": "SmolLM 360M",      "category": "smollm",   "size": "360M"},
    "smollm-1.7b":       {"hf_id": "HuggingFaceTB/SmolLM-1.7B",      "name": "SmolLM 1.7B",      "category": "smollm",   "size": "1.7B"},
    # Gemma
    "gemma-2b":          {"hf_id": "google/gemma-2b",                 "name": "Gemma 2B",          "category": "gemma",    "size": "2B"},
    "gemma-2b-it":       {"hf_id": "google/gemma-2b-it",              "name": "Gemma 2B IT",       "category": "gemma",    "size": "2B"},
    # Mamba
    "mamba-130m":        {"hf_id": "state-spaces/mamba-130m",         "name": "Mamba 130M",        "category": "mamba",    "size": "130M"},
    "mamba-370m":        {"hf_id": "state-spaces/mamba-370m",         "name": "Mamba 370M",        "category": "mamba",    "size": "370M"},
    "mamba-790m":        {"hf_id": "state-spaces/mamba-790m",         "name": "Mamba 790M",        "category": "mamba",    "size": "790M"},
    # StableLM
    "stablelm-2-1.6b":   {"hf_id": "stabilityai/stablelm-2-1_6b",    "name": "StableLM 2 1.6B",   "category": "stablelm", "size": "1.6B"},
    "stablelm-2-zephyr": {"hf_id": "stabilityai/stablelm-2-zephyr-1_6b", "name": "StableLM 2 Zephyr", "category": "stablelm", "size": "1.6B"},
    # BLOOM
    "bloom-560m":        {"hf_id": "bigscience/bloom-560m",           "name": "BLOOM 560M",        "category": "bloom",    "size": "560M"},
    "bloom-1b1":         {"hf_id": "bigscience/bloom-1b1",            "name": "BLOOM 1.1B",        "category": "bloom",    "size": "1.1B"},
    "bloom-1b7":         {"hf_id": "bigscience/bloom-1b7",            "name": "BLOOM 1.7B",        "category": "bloom",    "size": "1.7B"},
    # FLAN-T5
    "flan-t5-small":     {"hf_id": "google/flan-t5-small",            "name": "FLAN-T5 Small",     "category": "flan-t5",  "size": "80M"},
    "flan-t5-base":      {"hf_id": "google/flan-t5-base",             "name": "FLAN-T5 Base",      "category": "flan-t5",  "size": "250M"},
    "flan-t5-large":     {"hf_id": "google/flan-t5-large",            "name": "FLAN-T5 Large",     "category": "flan-t5",  "size": "780M"},
}

DEFAULT_MODEL = "gpt2-medium"

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def _info(msg: str) -> None:
    print(f"{CYAN}[INFO]{NC}  {msg}")


def _success(msg: str) -> None:
    print(f"{GREEN}[OK]{NC}    {msg}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC}  {msg}")


def _error(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}")


# ---------------------------------------------------------------------------
# List models
# ---------------------------------------------------------------------------
def list_models(category: Optional[str] = None) -> None:
    """Print the catalog of available models."""
    filtered = MODELS
    if category:
        filtered = {k: v for k, v in MODELS.items() if v["category"] == category}
        if not filtered:
            _error(f"No models found in category '{category}'.")
            categories = sorted({m["category"] for m in MODELS.values()})
            print(f"  {DIM}Available categories: {', '.join(categories)}{NC}")
            return

    print(f"\n{BOLD}Available Models{NC}  ({len(filtered)} total)\n")
    print(f"  {'ID':<22} {'Name':<25} {'Category':<12} {'Size':<8}")
    print(f"  {'─' * 22} {'─' * 25} {'─' * 12} {'─' * 8}")

    for mid, info in sorted(filtered.items(), key=lambda x: x[1]["category"]):
        default_marker = " *" if mid == DEFAULT_MODEL else ""
        print(f"  {CYAN}{mid:<22}{NC} {info['name']:<25} {DIM}{info['category']:<12}{NC} {info['size']:<8}{default_marker}")

    print(f"\n  {DIM}* = default model{NC}")
    print(f"  {DIM}Usage: python scripts/download_model.py <model_id>{NC}\n")


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------
def _make_progress_callback(repo_id: str):
    """Create a progress callback for huggingface_hub downloads."""
    try:
        from tqdm import tqdm
    except ImportError:
        # If tqdm is not available, return a no-op
        return None

    pbar = tqdm(
        desc=f"  Downloading",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour="cyan",
    )

    def callback(progress):
        if hasattr(progress, "completed") and hasattr(progress, "total") and progress.total:
            pbar.total = progress.total
            pbar.update(progress.completed - pbar.n)
        elif hasattr(progress, "n") and hasattr(progress, "total"):
            if progress.total:
                pbar.total = progress.total
            pbar.update(progress.n - pbar.n)

    return pbar, callback


# ---------------------------------------------------------------------------
# Download model
# ---------------------------------------------------------------------------
def download_model(model_id: str, cache_dir: Optional[str] = None) -> None:
    """Download a model from HuggingFace Hub."""
    if model_id not in MODELS:
        _error(f"Unknown model: '{model_id}'")
        available = ", ".join(sorted(MODELS.keys()))
        print(f"  {DIM}Available models: {available}{NC}")
        print(f"  {DIM}Use --list to see the full catalog.{NC}")
        sys.exit(1)

    info = MODELS[model_id]
    hf_id = info["hf_id"]

    print(f"\n{BOLD}{CYAN}Downloading Model{NC}\n")
    print(f"  ID:         {model_id}")
    print(f"  Name:       {info['name']}")
    print(f"  HuggingFace: {hf_id}")
    print(f"  Size:       {info['size']}")
    print(f"  Category:   {info['category']}")
    if cache_dir:
        print(f"  Cache dir:  {cache_dir}")
    print()

    # Try using huggingface_hub first (lighter dependency)
    try:
        from huggingface_hub import snapshot_download

        _info("Using huggingface_hub to download...")

        kwargs = {"repo_id": hf_id}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        result = snapshot_download(**kwargs)
        _success(f"Model '{model_id}' downloaded successfully!")
        print(f"  {DIM}Cached at: {result}{NC}")

    except ImportError:
        # Fall back to transformers
        _warn("huggingface_hub not found. Falling back to transformers...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _info("Downloading model and tokenizer...")
            start = time.time()

            model_kwargs = {}
            if cache_dir:
                model_kwargs["cache_dir"] = cache_dir

            _info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(hf_id, **model_kwargs)
            _success("Tokenizer downloaded.")

            _info("Downloading model weights...")
            model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
            _success("Model weights downloaded.")

            elapsed = time.time() - start
            _success(f"Model '{model_id}' downloaded in {elapsed:.1f}s!")
            if cache_dir:
                print(f"  {DIM}Cached at: {cache_dir}{NC}")

        except ImportError:
            _error("Neither huggingface_hub nor transformers is installed.")
            print(f"  {DIM}Install dependencies with:{NC}")
            print(f"  {CYAN}  pip install huggingface_hub{NC}")
            print(f"  {CYAN}  pip install transformers torch{NC}")
            sys.exit(1)

    except Exception as exc:
        _error(f"Download failed: {exc}")
        print(f"  {DIM}Check your internet connection and try again.{NC}")
        print(f"  {DIM}You may need to authenticate for gated models:{NC}")
        print(f"  {CYAN}  huggingface-cli login{NC}")
        sys.exit(1)

    print(f"\n  {DIM}You can now use this model with:{NC}")
    print(f"  {CYAN}  nexus-llm chat --model {model_id}{NC}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nexus-LLM Model Downloader — download models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_model.py                    # default model\n"
            "  python scripts/download_model.py phi-2              # specific model\n"
            "  python scripts/download_model.py --list             # list all models\n"
            "  python scripts/download_model.py --list --category phi\n"
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Model ID to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit.",
    )
    parser.add_argument(
        "--category", "-c",
        default=None,
        help="Filter model list by category (use with --list).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded models.",
    )

    args = parser.parse_args()

    if args.list:
        list_models(category=args.category)
        return

    download_model(model_id=args.model, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
