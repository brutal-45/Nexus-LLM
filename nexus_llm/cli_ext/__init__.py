"""
Nexus-LLM CLI Extensions Module

Provides shell completions, output formatters, and argument validators
that extend the base CLI functionality.
"""

from nexus_llm.cli_ext.completions import (
    CompletionGenerator,
    generate_bash_completions,
    generate_zsh_completions,
    generate_fish_completions,
)
from nexus_llm.cli_ext.formatters import (
    OutputFormatter,
    TableFormatter,
    JsonFormatter,
    YamlFormatter,
    format_output,
)
from nexus_llm.cli_ext.validators import (
    validate_model_name,
    validate_path,
    validate_url,
    validate_port,
    validate_positive_int,
    validate_float_range,
    validate_choice,
    validate_file_extension,
)

__all__ = [
    "CompletionGenerator",
    "generate_bash_completions",
    "generate_zsh_completions",
    "generate_fish_completions",
    "OutputFormatter",
    "TableFormatter",
    "JsonFormatter",
    "YamlFormatter",
    "format_output",
    "validate_model_name",
    "validate_path",
    "validate_url",
    "validate_port",
    "validate_positive_int",
    "validate_float_range",
    "validate_choice",
    "validate_file_extension",
]
