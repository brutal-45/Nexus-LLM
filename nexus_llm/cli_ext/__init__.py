"""CLI extension framework for Nexus-LLM.

Provides a plugin architecture for extending the CLI with custom
commands, subcommands, hooks, and structured output formatting.
"""

from nexus_llm.cli_ext.extension import CLIExtension
from nexus_llm.cli_ext.plugin import CLIPlugin
from nexus_llm.cli_ext.registry import CommandRegistry
from nexus_llm.cli_ext.output import OutputFormat

__all__ = [
    "CLIExtension",
    "CLIPlugin",
    "CommandRegistry",
    "OutputFormat",
]
