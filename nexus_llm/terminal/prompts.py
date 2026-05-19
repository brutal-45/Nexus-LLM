"""
Nexus-LLM Prompt Input Module

Provides advanced prompt input using prompt_toolkit with auto-suggestion,
multiline editing, history browsing, and syntax-aware completion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from prompt_toolkit import PromptSession as PtkSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, WordCompleter
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.styles import Style as PtkStyle

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


DEFAULT_HISTORY_FILE = os.path.expanduser("~/.nexus_llm/prompt_history")


class PromptSession:
    """Advanced prompt input session with history, completion, and styling.

    Provides an interactive input prompt with auto-suggestions from history,
    word completion, and configurable key bindings. Falls back to basic
    input() when prompt_toolkit is unavailable.
    """

    def __init__(
        self,
        history_file: str | None = None,
        enable_auto_suggest: bool = True,
        enable_history_search: bool = True,
        style: dict[str, str] | None = None,
    ) -> None:
        self._history_file = history_file or DEFAULT_HISTORY_FILE
        self._enable_auto_suggest = enable_auto_suggest
        self._enable_history_search = enable_history_search
        self._last_input = ""
        self._input_history: list[str] = []
        self._style = style or {
            "": "#e0e0e0",
            "prompt": "#4488ff bold",
            "completion": "#888888",
            "suggestion": "#555555",
        }

        if HAS_PROMPT_TOOLKIT:
            self._init_ptk_session()

    def _init_ptk_session(self) -> None:
        """Initialize the prompt_toolkit session with features."""
        history_dir = Path(self._history_file).parent
        history_dir.mkdir(parents=True, exist_ok=True)

        file_history = FileHistory(self._history_file)

        auto_suggest = AutoSuggestFromHistory() if self._enable_auto_suggest else None

        style = PtkStyle.from_dict(self._style)

        bindings = KeyBindings()

        @bindings.add("c-l")
        def _clear_screen(event: Any) -> None:
            """Clear the screen on Ctrl+L."""
            event.app.renderer.clear()

        self._ptk_session = PtkSession(
            history=file_history,
            auto_suggest=auto_suggest,
            style=style,
            key_bindings=bindings,
            enable_history_search=self._enable_history_search,
        )

    def get_input(
        self,
        prompt: str = "> ",
        multiline: bool = False,
        completer: Any | None = None,
        default: str = "",
        is_password: bool = False,
    ) -> str:
        """Get user input with all configured features.

        Args:
            prompt: The prompt string to display.
            multiline: Whether to accept multiline input.
            completer: Optional completer for tab completion.
            default: Default value if user presses Enter.
            is_password: Whether to hide input (for passwords).

        Returns:
            The user's input string.
        """
        if HAS_PROMPT_TOOLKIT:
            return self._ptk_input(
                prompt=prompt,
                multiline=multiline,
                completer=completer,
                default=default,
                is_password=is_password,
            )
        return self._basic_input(prompt, default, is_password)

    def _ptk_input(
        self,
        prompt: str,
        multiline: bool,
        completer: Any | None,
        default: str,
        is_password: bool,
    ) -> str:
        """Get input using prompt_toolkit."""
        ptk_completer = completer
        if completer and isinstance(completer, WordCompleter):
            ptk_completer = completer
        elif completer and hasattr(completer, "get_completions"):
            ptk_completer = completer

        try:
            result = self._ptk_session.prompt(
                prompt,
                completer=ptk_completer,
                default=default,
                is_password=is_password,
                multiline=multiline,
            )
            self._last_input = result
            self._input_history.append(result)
            return result
        except KeyboardInterrupt:
            raise
        except EOFError:
            raise

    def _basic_input(self, prompt: str, default: str, is_password: bool) -> str:
        """Fallback input using Python's built-in input()."""
        try:
            if is_password:
                import getpass
                result = getpass.getpass(prompt)
            else:
                display_prompt = prompt
                if default:
                    display_prompt = f"{prompt}[{default}] "
                result = input(display_prompt)
                if not result and default:
                    result = default
            self._last_input = result
            self._input_history.append(result)
            return result
        except EOFError:
            raise

    def get_confirmation(self, prompt: str, default: bool = False) -> bool:
        """Get a yes/no confirmation from the user.

        Args:
            prompt: The question to ask.
            default: Default value if user presses Enter.

        Returns:
            True for yes, False for no.
        """
        suffix = " [Y/n] " if default else " [y/N] "
        try:
            response = self.get_input(prompt + suffix)
            if not response.strip():
                return default
            return response.strip().lower() in ("y", "yes", "1", "true")
        except (KeyboardInterrupt, EOFError):
            return default

    def get_choice(
        self,
        prompt: str,
        choices: list[str],
        default: str | None = None,
    ) -> str:
        """Get a choice from a list of options.

        Args:
            prompt: The question to ask.
            choices: List of valid choices.
            default: Default choice.

        Returns:
            The selected choice string.
        """
        choice_str = "/".join(choices)
        suffix = f" [{choice_str}]"
        if default:
            suffix = f" [{default}] "
        try:
            response = self.get_input(prompt + suffix)
            if not response.strip() and default:
                return default
            if response.strip() in choices:
                return response.strip()
            return choices[0] if choices else ""
        except (KeyboardInterrupt, EOFError):
            return default or (choices[0] if choices else "")

    @property
    def last_input(self) -> str:
        """Get the last entered input."""
        return self._last_input

    @property
    def history(self) -> list[str]:
        """Get the in-memory input history."""
        return list(self._input_history)

    def clear_history(self) -> None:
        """Clear the in-memory input history."""
        self._input_history.clear()
