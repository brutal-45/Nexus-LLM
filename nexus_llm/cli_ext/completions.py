"""
Shell Completions for Nexus-LLM CLI

Generates completion scripts for Bash, Zsh, and Fish shells.
Supports command, option, and argument completion with
dynamic model and preset name resolution.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Completion data
# ---------------------------------------------------------------------------

# All CLI commands and their descriptions
COMMANDS: Dict[str, str] = {
    "chat": "Start an interactive chat session",
    "serve": "Start the API server",
    "train": "Fine-tune or train a model",
    "eval": "Evaluate a model",
    "download": "Download a model from HuggingFace Hub",
    "list": "List available models, presets, or plugins",
    "config": "View or modify configuration",
    "plugin": "Manage plugins",
    "rag": "RAG pipeline operations",
    "benchmark": "Run performance benchmarks",
    "version": "Show version information",
    "init": "Initialize a new Nexus-LLM project",
    "export": "Export a model or configuration",
    "quantize": "Quantize a model",
}

# Global options
GLOBAL_OPTIONS: List[Dict[str, str]] = [
    {"flags": "--config -c", "desc": "Path to configuration file"},
    {"flags": "--verbose -v", "desc": "Enable verbose output"},
    {"flags": "--quiet -q", "desc": "Suppress non-error output"},
    {"flags": "--help -h", "desc": "Show help message"},
    {"flags": "--version", "desc": "Show version"},
    {"flags": "--log-level", "desc": "Set log level (debug|info|warning|error)"},
    {"flags": "--no-color", "desc": "Disable colored output"},
]

# Command-specific options
COMMAND_OPTIONS: Dict[str, List[Dict[str, str]]] = {
    "chat": [
        {"flags": "--model -m", "desc": "Model name or path"},
        {"flags": "--preset -p", "desc": "Chat preset (creative|precise|balanced|coding|writing|research)"},
        {"flags": "--system -s", "desc": "System prompt"},
        {"flags": "--temperature -t", "desc": "Sampling temperature"},
        {"flags": "--max-tokens", "desc": "Maximum tokens to generate"},
        {"flags": "--context", "desc": "Context window size"},
        {"flags": "--history", "desc": "Path to chat history file"},
        {"flags": "--no-stream", "desc": "Disable streaming output"},
    ],
    "serve": [
        {"flags": "--host", "desc": "Server host address"},
        {"flags": "--port", "desc": "Server port"},
        {"flags": "--workers -w", "desc": "Number of worker processes"},
        {"flags": "--model -m", "desc": "Model name or path"},
        {"flags": "--preset -p", "desc": "Server preset (development|production|high_performance)"},
        {"flags": "--reload", "desc": "Enable auto-reload for development"},
        {"flags": "--api-key", "desc": "API key for authentication"},
        {"flags": "--cors", "desc": "Allowed CORS origins"},
    ],
    "train": [
        {"flags": "--model -m", "desc": "Base model name or path"},
        {"flags": "--dataset -d", "desc": "Training dataset path"},
        {"flags": "--preset -p", "desc": "Training preset (quick|standard|thorough|lora|qlora)"},
        {"flags": "--output -o", "desc": "Output directory"},
        {"flags": "--epochs -e", "desc": "Number of training epochs"},
        {"flags": "--batch-size -b", "desc": "Training batch size"},
        {"flags": "--learning-rate -l", "desc": "Learning rate"},
        {"flags": "--resume", "desc": "Resume from checkpoint"},
        {"flags": "--gpus -g", "desc": "GPU devices to use"},
    ],
    "eval": [
        {"flags": "--model -m", "desc": "Model name or path"},
        {"flags": "--benchmark -b", "desc": "Benchmark to run"},
        {"flags": "--output -o", "desc": "Output file for results"},
        {"flags": "--tasks", "desc": "Specific evaluation tasks"},
    ],
    "download": [
        {"flags": "--model -m", "desc": "HuggingFace model ID"},
        {"flags": "--output -o", "desc": "Download directory"},
        {"flags": "--quantization -q", "desc": "Quantization format"},
        {"flags": "--revision", "desc": "Model revision/branch"},
    ],
    "list": [
        {"flags": "--type -t", "desc": "List type (models|presets|plugins)"},
        {"flags": "--format -f", "desc": "Output format (table|json|yaml)"},
    ],
    "config": [
        {"flags": "--get", "desc": "Get a configuration value"},
        {"flags": "--set", "desc": "Set a configuration value"},
        {"flags": "--unset", "desc": "Remove a configuration value"},
        {"flags": "--show", "desc": "Show full configuration"},
    ],
    "plugin": [
        {"flags": "--install", "desc": "Install a plugin"},
        {"flags": "--uninstall", "desc": "Uninstall a plugin"},
        {"flags": "--enable", "desc": "Enable a plugin"},
        {"flags": "--disable", "desc": "Disable a plugin"},
        {"flags": "--list -l", "desc": "List installed plugins"},
    ],
    "rag": [
        {"flags": "--index", "desc": "Index documents"},
        {"flags": "--query -q", "desc": "Search query"},
        {"flags": "--collection -c", "desc": "Collection name"},
        {"flags": "--embed-model", "desc": "Embedding model"},
    ],
    "benchmark": [
        {"flags": "--model -m", "desc": "Model name or path"},
        {"flags": "--preset", "desc": "Benchmark preset"},
        {"flags": "--output -o", "desc": "Output file"},
        {"flags": "--iterations -n", "desc": "Number of iterations"},
    ],
    "quantize": [
        {"flags": "--model -m", "desc": "Model name or path"},
        {"flags": "--bits -b", "desc": "Quantization bits (4|8)"},
        {"flags": "--output -o", "desc": "Output path"},
        {"flags": "--format -f", "desc": "Quantization format (gguf|gptq|awq)"},
    ],
    "export": [
        {"flags": "--model -m", "desc": "Model name or path"},
        {"flags": "--format -f", "desc": "Export format (onnx|safetensors|gguf)"},
        {"flags": "--output -o", "desc": "Output path"},
    ],
}

# Static completion candidates
STATIC_CHOICES: Dict[str, List[str]] = {
    "chat_preset": ["creative", "precise", "balanced", "coding", "writing", "research"],
    "server_preset": ["development", "production", "high_performance"],
    "training_preset": ["quick", "standard", "thorough", "lora", "qlora"],
    "log_level": ["debug", "info", "warning", "error", "critical"],
    "list_type": ["models", "presets", "plugins"],
    "output_format": ["table", "json", "yaml"],
    "quant_bits": ["4", "8"],
    "quant_format": ["gguf", "gptq", "awq"],
    "export_format": ["onnx", "safetensors", "gguf"],
    "benchmark_preset": ["default", "quick", "thorough"],
}


# ---------------------------------------------------------------------------
# Completion Generator
# ---------------------------------------------------------------------------

class CompletionGenerator:
    """Generates shell completion scripts for Nexus-LLM.

    Produces completion scripts for Bash, Zsh, and Fish shells
    that support command, option, and argument completion.
    """

    PROGRAM_NAME = "nexus"
    PROGRAM_DESC = "Nexus-LLM — Unified LLM Framework"

    def __init__(
        self,
        program_name: str = "nexus",
        commands: Optional[Dict[str, str]] = None,
        global_options: Optional[List[Dict[str, str]]] = None,
        command_options: Optional[Dict[str, List[Dict[str, str]]]] = None,
        static_choices: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.program_name = program_name
        self.commands = commands or COMMANDS
        self.global_options = global_options or GLOBAL_OPTIONS
        self.command_options = command_options or COMMAND_OPTIONS
        self.static_choices = static_choices or STATIC_CHOICES

    # ------------------------------------------------------------------
    # Bash
    # ------------------------------------------------------------------

    def generate_bash(self) -> str:
        """Generate a Bash completion script."""
        lines = [
            f"#!/usr/bin/env bash",
            f"# Bash completion for {self.program_name}",
            f"# Generated by {self.program_name} completions",
            f"",
            f"_{self.program_name}_completions()",
            f"{{",
            f"  local cur prev words cword",
            f"  _init_completion || return",
            f"",
            f"  # Global options",
            f"  local global_opts='{self._bash_global_opts()}'",
            f"",
            f"  # Commands",
            f"  local commands='{self._bash_commands()}'",
            f"",
        ]

        # Add per-command option blocks
        for cmd, opts in self.command_options.items():
            opt_str = " ".join(
                flag for opt in opts for flag in opt.get("flags", "").split() if flag.startswith("-")
            )
            lines.append(f'  local {cmd}_opts="{opt_str}"')
        lines.append("")

        # Completion logic
        lines.extend([
            f"  # Complete commands",
            f"  if [[ $cword -eq 1 ]]; then",
            f"    COMPREPLY=($(compgen -W \"${{commands}}\" -- \"$cur\"))",
            f"    return",
            f"  fi",
            f"",
            f"  # Complete global options",
            f"  if [[ $cur == -* ]]; then",
            f"    local cmd=\"${{words[1]}}\"",
            f"    local cmd_opts_var=\"${{cmd}}_opts\"",
            f"    local cmd_opts=\"${{!cmd_opts_var}}\"",
            f"    COMPREPLY=($(compgen -W \"${{global_opts}} ${{cmd_opts}}\" -- \"$cur\"))",
            f"    return",
            f"  fi",
            f"",
            f"  # Option argument completion",
            f"  case \"$prev\" in",
        ])

        # Option-specific completions
        choices_map = {
            "--preset -p": "chat_preset",
            "--preset": "chat_preset",
            "--type -t": "list_type",
            "--type": "list_type",
            "--format -f": "output_format",
            "--format": "output_format",
            "--bits -b": "quant_bits",
            "--bits": "quant_bits",
            "--log-level": "log_level",
        }
        for opt_key, choice_key in choices_map.items():
            choices = self.static_choices.get(choice_key, [])
            if choices:
                for opt in opt_key.split():
                    lines.append(f'    {opt})')
                    lines.append(f'      COMPREPLY=($(compgen -W "{" ".join(choices)}" -- "$cur"))')
                    lines.append(f'      return')
                    lines.append(f'      ;;')

        lines.extend([
            f"  esac",
            f"}}",
            f"",
            f"complete -F _{self.program_name}_completions {self.program_name}",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Zsh
    # ------------------------------------------------------------------

    def generate_zsh(self) -> str:
        """Generate a Zsh completion script."""
        lines = [
            f"#compdef {self.program_name}",
            f"# Zsh completion for {self.program_name}",
            f"# Generated by {self.program_name} completions",
            f"",
            f"_{self.program_name}() {{",
            f"  local -a commands",
            f"  local -a global_opts",
            f"",
        ]

        # Commands
        lines.append("  commands=(")
        for cmd, desc in self.commands.items():
            lines.append(f'    "{cmd}:{desc}"')
        lines.append("  )")
        lines.append("")

        # Global options
        lines.append("  global_opts=(")
        for opt in self.global_options:
            flags = opt.get("flags", "")
            desc = opt.get("desc", "")
            parts = flags.split()
            if len(parts) == 2:
                lines.append('    "' + parts[0] + '[' + desc + ']" "' + parts[1] + '[' + desc + ']"')
            elif len(parts) == 1:
                lines.append('    "' + parts[0] + '[' + desc + ']"')
        lines.append("  )")
        lines.append("")

        # Main dispatch
        lines.extend([
            f"  _arguments -C \\",
            f"    \"${{global_opts[@]}}\" \\",
            f"    \"1: :->command\" \\",
            f"    \"*::arg:->args\"",
            f"",
            f"  case $state in",
            f"    command)",
            f"      _describe 'command' commands",
            f"      ;;",
            f"    args)",
            f"      local cmd=\"${{words[1]}}\"",
            f"      case $cmd in",
        ])

        # Per-command subcompletion
        for cmd, opts in self.command_options.items():
            lines.append(f"        {cmd})")
            lines.append(f"          local -a {cmd}_opts")
            lines.append(f"          {cmd}_opts=(")
            for opt in opts:
                flags = opt.get("flags", "")
                desc = opt.get("desc", "")
                parts = flags.split()
                if len(parts) == 2:
                    lines.append('    "' + parts[0] + '[' + desc + ']" "' + parts[1] + '[' + desc + ']"')
                elif len(parts) == 1:
                    lines.append('            "' + parts[0] + '[' + desc + ']"')
            lines.append(f"          )")
            lines.append(f"          _arguments \"${{{cmd}_opts[@]}}\"")
            lines.append(f"          ;;")

        lines.extend([
            f"      esac",
            f"      ;;",
            f"  esac",
            f"}}",
            f"",
            f"_{self.program_name}",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fish
    # ------------------------------------------------------------------

    def generate_fish(self) -> str:
        """Generate a Fish completion script."""
        lines = [
            f"# Fish completion for {self.program_name}",
            f"# Generated by {self.program_name} completions",
            f"",
        ]

        # Global options
        for opt in self.global_options:
            flags = opt.get("flags", "")
            desc = opt.get("desc", "")
            for flag in flags.split():
                lines.append(
                    f'complete -c {self.program_name} -n "__fish_use_subcommand" '
                    f'-l {flag.lstrip("-")} '
                    f'-d "{desc}"'
                )
        lines.append("")

        # Commands
        for cmd, desc in self.commands.items():
            lines.append(
                f'complete -c {self.program_name} -n "__fish_use_subcommand" '
                f'-a "{cmd}" -d "{desc}"'
            )
        lines.append("")

        # Per-command options
        for cmd, opts in self.command_options.items():
            for opt in opts:
                flags = opt.get("flags", "")
                desc = opt.get("desc", "")
                for flag in flags.split():
                    if flag.startswith("--"):
                        lines.append(
                            f'complete -c {self.program_name} -n "__fish_seen_subcommand_from {cmd}" '
                            f'-l {flag.lstrip("--")} '
                            f'-d "{desc}"'
                        )
                    elif flag.startswith("-"):
                        lines.append(
                            f'complete -c {self.program_name} -n "__fish_seen_subcommand_from {cmd}" '
                            f'-s {flag.lstrip("-")} '
                            f'-d "{desc}"'
                        )
            lines.append("")

        # Static choice completions
        choice_option_map = {
            "chat_preset": [("chat", "--preset"), ("chat", "-p")],
            "server_preset": [("serve", "--preset"), ("serve", "-p")],
            "training_preset": [("train", "--preset"), ("train", "-p")],
            "log_level": [("", "--log-level")],
            "list_type": [("list", "--type"), ("list", "-t")],
            "output_format": [("list", "--format"), ("list", "-f")],
            "quant_bits": [("quantize", "--bits"), ("quantize", "-b")],
            "quant_format": [("quantize", "--format"), ("quantize", "-f")],
            "export_format": [("export", "--format"), ("export", "-f")],
        }

        for choice_key, mappings in choice_option_map.items():
            choices = self.static_choices.get(choice_key, [])
            if not choices:
                continue
            for cmd, opt in mappings:
                cmd_cond = (
                    f'-n "__fish_seen_subcommand_from {cmd}"'
                    if cmd
                    else ""
                )
                opt_flag = (
                    f'-l {opt.lstrip("--")}'
                    if opt.startswith("--")
                    else f'-s {opt.lstrip("-")}'
                )
                choices_str = " ".join(choices)
                lines.append(
                    f'complete -c {self.program_name} {cmd_cond} '
                    f'{opt_flag} -a "{choices_str}"'
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bash_commands(self) -> str:
        return " ".join(self.commands.keys())

    def _bash_global_opts(self) -> str:
        opts = []
        for opt in self.global_options:
            flags = opt.get("flags", "")
            opts.extend(flags.split())
        return " ".join(opts)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_generator = CompletionGenerator()


def generate_bash_completions() -> str:
    """Generate Bash completion script using default configuration."""
    return _generator.generate_bash()


def generate_zsh_completions() -> str:
    """Generate Zsh completion script using default configuration."""
    return _generator.generate_zsh()


def generate_fish_completions() -> str:
    """Generate Fish completion script using default configuration."""
    return _generator.generate_fish()
