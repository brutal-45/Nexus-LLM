"""Entry point for running Nexus-LLM as a module.

Usage:
    python -m nexus_llm [command] [options]
"""

import sys


def main() -> None:
    """Main entry point for python -m nexus_llm."""
    try:
        from nexus_llm.cli import cli
        cli(standalone_mode=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except ImportError as exc:
        print(f"Error importing Nexus-LLM: {exc}", file=sys.stderr)
        print("Make sure all dependencies are installed:", file=sys.stderr)
        print("  pip install -e .", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
