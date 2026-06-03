"""Entry point for running: python -m nexus_llm"""

import sys

def main():
    try:
        from nexus_llm.cli import cli
        cli()
    except ImportError as e:
        missing = str(e).replace("No module named '", "").replace("'", "")
        print(f"\n  ❌ Missing dependency: {missing}")
        print(f"  Run: pip install {missing}")
        print(f"  Or:  pip install -e .\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  👋 Goodbye from Nexus-LLM!\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
