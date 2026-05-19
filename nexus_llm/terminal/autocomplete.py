"""
Nexus-LLM Autocomplete Module

Provides tab completion for commands, model names, file paths,
and custom completers for the interactive chat terminal.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

from nexus_llm.terminal.commands import CommandRegistry


@dataclass
class CompletionItem:
    """A single completion suggestion."""
    text: str
    display: str = ""
    description: str = ""
    category: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.display:
            self.display = self.text


class AutoCompleter:
    """Meta-completer that aggregates multiple completion sources.

    Combines command, model, file path, and custom completers
    into a unified completion system. Dispatches to the appropriate
    sub-completer based on the current input context.
    """

    def __init__(self) -> None:
        self._command_completer: CommandCompleter | None = None
        self._model_completer: ModelCompleter | None = None
        self._path_completer: PathCompleter | None = None
        self._custom_completers: list[Any] = []

    def set_command_completer(self, completer: CommandCompleter) -> None:
        """Set the command completer.

        Args:
            completer: The CommandCompleter instance.
        """
        self._command_completer = completer

    def set_model_completer(self, completer: ModelCompleter) -> None:
        """Set the model name completer.

        Args:
            completer: The ModelCompleter instance.
        """
        self._model_completer = completer

    def set_path_completer(self, completer: PathCompleter) -> None:
        """Set the file path completer.

        Args:
            completer: The PathCompleter instance.
        """
        self._path_completer = completer

    def add_custom_completer(self, completer: Any) -> None:
        """Add a custom completer.

        Args:
            completer: A completer with a get_completions() method.
        """
        self._custom_completers.append(completer)

    def get_completions(self, text: str, cursor_pos: int) -> list[CompletionItem]:
        """Get completion suggestions for the current input.

        Dispatches to the appropriate sub-completer based on context:
        - Input starting with '/' → command completer
        - Input after '/model' → model completer
        - Input resembling a file path → path completer
        - Otherwise → custom completers

        Args:
            text: Current input text.
            cursor_pos: Cursor position in the text.

        Returns:
            List of CompletionItem suggestions.
        """
        items: list[CompletionItem] = []

        if text.startswith("/"):
            # Command completion
            if self._command_completer:
                items.extend(self._command_completer.get_completions(text, cursor_pos))

            # Check if this is a /model command - add model completions
            parts = text.split(maxsplit=1)
            if len(parts) > 1 and parts[0] in ("/model", "/switch"):
                if self._model_completer:
                    items.extend(self._model_completer.get_completions(parts[1], len(parts[1])))

            # Check for path-accepting commands
            if len(parts) > 1 and parts[0] in ("/load", "/save", "/export", "/import"):
                if self._path_completer:
                    items.extend(self._path_completer.get_completions(parts[1], len(parts[1])))
        else:
            # File path completion
            if self._looks_like_path(text) and self._path_completer:
                items.extend(self._path_completer.get_completions(text, cursor_pos))

            # Custom completers
            for completer in self._custom_completers:
                if hasattr(completer, "get_completions"):
                    items.extend(completer.get_completions(text, cursor_pos))

        # Sort by priority, then alphabetically
        items.sort(key=lambda i: (-i.priority, i.text))
        return items

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        """Check if text looks like a file path.

        Args:
            text: Input text to check.

        Returns:
            True if the text appears to be a file path.
        """
        return text.startswith(("./", "../", "/", "~")) or (
            len(text) > 1 and text[1] == ":" and text[0].isalpha()
        )

    def to_prompt_toolkit_completer(self) -> Any:
        """Convert to a prompt_toolkit Completer for integration.

        Returns:
            A prompt_toolkit Completer, or None if prompt_toolkit is unavailable.
        """
        if not HAS_PROMPT_TOOLKIT:
            return None

        meta_completer = self

        class _PtkCompleter(Completer):
            def get_completions(self, document: Document, complete_event: Any) -> Any:
                text = document.text_before_cursor
                cursor_pos = len(text)
                items = meta_completer.get_completions(text, cursor_pos)
                for item in items:
                    # Calculate the offset from cursor to the completion start
                    word_before_cursor = document.get_word_before_cursor(WORD=True)
                    yield Completion(
                        item.text,
                        start_position=-len(word_before_cursor) if word_before_cursor else 0,
                        display=item.display,
                        display_meta=item.description,
                    )

        return _PtkCompleter()


class CommandCompleter:
    """Completer for slash commands.

    Provides completion suggestions for command names and their arguments,
    based on the registered commands in a CommandRegistry.
    """

    def __init__(self, registry: CommandRegistry | None = None) -> None:
        self._registry = registry

    def set_registry(self, registry: CommandRegistry) -> None:
        """Set the command registry.

        Args:
            registry: The CommandRegistry to use.
        """
        self._registry = registry

    def get_completions(self, text: str, cursor_pos: int) -> list[CompletionItem]:
        """Get command completion suggestions.

        Args:
            text: Input text (starting with /).
            cursor_pos: Cursor position.

        Returns:
            List of CompletionItem suggestions.
        """
        if not self._registry or not text.startswith("/"):
            return []

        items: list[CompletionItem] = []
        partial = text[1:]  # Remove the leading /

        # If there's no space, complete command names
        if " " not in partial:
            for name, info in self._registry.get_commands().items():
                if name.startswith(partial.lower()):
                    items.append(CompletionItem(
                        text="/" + name,
                        display="/" + name,
                        description=info.get("description", ""),
                        category="command",
                        priority=10,
                    ))
        else:
            # Complete command arguments
            cmd_name = partial.split()[0]
            cmd_info = self._registry.get_commands().get(cmd_name)
            if cmd_info and cmd_info.get("usage"):
                items.append(CompletionItem(
                    text=cmd_info["usage"],
                    display=cmd_info["usage"],
                    description=cmd_info.get("description", ""),
                    category="usage",
                    priority=5,
                ))

        return items


class ModelCompleter:
    """Completer for model names.

    Provides completion suggestions for known model names and paths.
    """

    # Common model names for completion
    KNOWN_MODELS: list[str] = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "distilgpt2",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-1.3b",
        "EleutherAI/gpt-neo-2.7b",
        "EleutherAI/gpt-j-6b",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "mistralai/Mistral-7B-v0.1",
        "microsoft/phi-1",
        "microsoft/phi-1_5",
        "microsoft/phi-2",
        "Qwen/Qwen1.5-0.5B",
        "Qwen/Qwen1.5-1.8B",
        "Qwen/Qwen1.5-4B",
        "Qwen/Qwen1.5-7B",
    ]

    def __init__(self, additional_models: list[str] | None = None) -> None:
        self._models = list(self.KNOWN_MODELS)
        if additional_models:
            self._models.extend(additional_models)

    def add_model(self, name: str) -> None:
        """Add a model name to the completion list.

        Args:
            name: Model name or path.
        """
        if name not in self._models:
            self._models.append(name)

    def remove_model(self, name: str) -> None:
        """Remove a model name from the completion list.

        Args:
            name: Model name to remove.
        """
        self._models = [m for m in self._models if m != name]

    def get_completions(self, text: str, cursor_pos: int) -> list[CompletionItem]:
        """Get model name completion suggestions.

        Args:
            text: Partial model name.
            cursor_pos: Cursor position.

        Returns:
            List of CompletionItem suggestions.
        """
        items: list[CompletionItem] = []
        text_lower = text.lower()

        for model in self._models:
            if model.lower().startswith(text_lower):
                items.append(CompletionItem(
                    text=model,
                    display=model,
                    description="model",
                    category="model",
                    priority=5,
                ))

        # Also check for local model paths
        if "/" in text or text.startswith("."):
            parent = os.path.dirname(text)
            base = os.path.basename(text)
            try:
                if os.path.isdir(parent):
                    for entry in os.listdir(parent):
                        if entry.startswith(base):
                            full_path = os.path.join(parent, entry)
                            if os.path.isdir(full_path):
                                items.append(CompletionItem(
                                    text=full_path,
                                    display=full_path,
                                    description="local path",
                                    category="path",
                                    priority=3,
                                ))
            except OSError:
                pass

        return items


class PathCompleter:
    """Completer for file system paths.

    Provides tab completion for file and directory paths,
    supporting relative and absolute paths.
    """

    def __init__(
        self,
        only_directories: bool = False,
        only_files: bool = False,
        file_extensions: list[str] | None = None,
    ) -> None:
        self._only_directories = only_directories
        self._only_files = only_files
        self._file_extensions = file_extensions  # e.g., [".json", ".yaml", ".txt"]

    def get_completions(self, text: str, cursor_pos: int) -> list[CompletionItem]:
        """Get file path completion suggestions.

        Args:
            text: Partial path text.
            cursor_pos: Cursor position.

        Returns:
            List of CompletionItem suggestions.
        """
        items: list[CompletionItem] = []

        # Expand tilde
        expanded = os.path.expanduser(text)

        # Determine the directory and prefix to search
        if os.path.isdir(expanded):
            search_dir = expanded
            prefix = ""
        else:
            search_dir = os.path.dirname(expanded)
            prefix = os.path.basename(expanded)

        if not search_dir:
            search_dir = "."

        try:
            entries = os.listdir(search_dir)
        except OSError:
            return items

        for entry in sorted(entries):
            # Filter by prefix
            if not entry.startswith(prefix):
                continue

            full_path = os.path.join(search_dir, entry)
            is_dir = os.path.isdir(full_path)

            # Apply filters
            if self._only_directories and not is_dir:
                continue
            if self._only_files and is_dir:
                continue
            if self._file_extensions and not is_dir:
                ext = os.path.splitext(entry)[1]
                if ext not in self._file_extensions:
                    continue

            # Build display path relative to original input
            if text.startswith("~"):
                display_path = os.path.join(os.path.dirname(text), entry)
                display_path = display_path.replace(os.path.expanduser("~"), "~", 1)
            else:
                display_path = os.path.join(os.path.dirname(text), entry) if os.path.dirname(text) else entry

            # Clean up path
            display_path = os.path.normpath(display_path)

            suffix = "/" if is_dir else ""
            items.append(CompletionItem(
                text=display_path + suffix,
                display=entry + suffix,
                description="directory" if is_dir else "file",
                category="path",
                priority=5 if is_dir else 3,
            ))

        return items
