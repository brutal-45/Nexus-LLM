#!/usr/bin/env python3
"""Nexus-LLM Server Runner Script.

Start the FastAPI-based inference server with configurable options.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Main entry point for the server runner."""
    parser = argparse.ArgumentParser(
        description="Nexus-LLM Server - Start the LLM inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py
  python run_server.py --model gpt2-medium --port 8080
  python run_server.py --model mistral-7b --workers 4 --cors
        """,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("NEXUS_LLM_HOST", "0.0.0.0"),
        help="Server host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NEXUS_LLM_PORT", "8000")),
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NEXUS_LLM_DEFAULT_MODEL", "gpt2-medium"),
        help="Model name or path to serve (default: gpt2-medium)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("NEXUS_LLM_WORKERS", "1")),
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("NEXUS_LLM_DEVICE", "auto"),
        help="Device to use: auto/cpu/cuda/mps (default: auto)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NEXUS_LLM_API_KEY", ""),
        help="API key for authentication",
    )
    parser.add_argument(
        "--cors",
        action="store_true",
        default=os.environ.get("NEXUS_LLM_CORS_ENABLED", "false").lower() == "true",
        help="Enable CORS",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=os.environ.get("NEXUS_LLM_SSL_CERTFILE"),
        help="SSL certificate file path",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=os.environ.get("NEXUS_LLM_SSL_KEYFILE"),
        help="SSL key file path",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("NEXUS_LLM_LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    try:
        from nexus_llm.app import NexusLLMApp
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=args.config)

    config = {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "workers": args.workers,
        "device": args.device,
        "api_key": args.api_key,
        "cors": args.cors,
        "ssl_certfile": args.ssl_certfile,
        "ssl_keyfile": args.ssl_keyfile,
        "log_level": args.log_level,
        "reload": args.reload,
    }

    app.run_serve(config)


if __name__ == "__main__":
    main()
