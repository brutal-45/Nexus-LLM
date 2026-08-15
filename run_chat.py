#!/usr/bin/env python3
"""Nexus-LLM Chat Runner Script.

Start an interactive chat session with an LLM model.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Main entry point for the chat runner."""
    parser = argparse.ArgumentParser(
        description="Nexus-LLM Chat - Start an interactive chat session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_chat.py
  python run_chat.py --model gpt2-medium
  python run_chat.py --model mistral-7b --system "You are a helpful assistant."
  python run_chat.py --temperature 0.9 --top-p 0.95
        """,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NEXUS_LLM_DEFAULT_MODEL", "gpt2-medium"),
        help="Model name or path to use (default: gpt2-medium)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt for the conversation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("NEXUS_LLM_TEMPERATURE", "0.7")),
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.environ.get("NEXUS_LLM_TOP_P", "0.9")),
        help="Top-p (nucleus) sampling (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("NEXUS_LLM_TOP_K", "50")),
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("NEXUS_LLM_MAX_TOKENS", "2048")),
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("NEXUS_LLM_DEVICE", "auto"),
        help="Device to use: auto/cpu/cuda/mps (default: auto)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable conversation history",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Non-interactive: send a single prompt and exit",
    )

    args = parser.parse_args()

    try:
        from nexus_llm.app import NexusLLMApp
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=args.config)

    config = {
        "model": args.model,
        "system_prompt": args.system,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "device": args.device,
        "use_history": not args.no_history,
        "single_prompt": args.prompt,
    }

    app.run_chat(config)


if __name__ == "__main__":
    main()
