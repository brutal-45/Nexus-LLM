"""
Nexus-LLM Key Bindings Module

Provides key binding management with Emacs/Vi modes, custom bindings,
and macro support for the terminal interface.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.key_binding.bindings import emacs, vi

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


DEFAULT_KEYBINDS_FILE = os.path.expanduser("~/.nexus_llm/keybinds.json")


class BindingMode(str, Enum):
    """Input mode for key bindings."""
    EMACS = "emacs"
    VI = "vi"


@dataclass
class KeyAction:
    """A single key binding action definition."""
    key: str
    action: str
    description: str = ""
    mode: BindingMode = BindingMode.EMACS
    context: str = "global"
    handler_name: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize to a dictionary."""
        return {
            "key": self.key,
            "action": self.action,
            "description": self.description,
            "mode": self.mode.value,
            "context": self.context,
            "handler_name": self.handler_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyAction:
        """Deserialize from a dictionary."""
        return cls(
            key=data.get("key", ""),
            action=data.get("action", ""),
            description=data.get("description", ""),
            mode=BindingMode(data.get("mode", "emacs")),
            context=data.get("context", "global"),
            handler_name=data.get("handler_name", ""),
        )


@dataclass
class Macro:
    """A recorded sequence of key actions that can be replayed."""
    name: str
    actions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "name": self.name,
            "actions": self.actions,
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Macro:
        """Deserialize from a dictionary."""
        return cls(
            name=data.get("name", ""),
            actions=data.get("actions", []),
            created_at=data.get("created_at", time.time()),
            description=data.get("description", ""),
        )


# Default key bindings
DEFAULT_BINDINGS: list[KeyAction] = [
    # Navigation
    KeyAction("up", "history_prev", "Previous history entry", BindingMode.EMACS, "input"),
    KeyAction("down", "history_next", "Next history entry", BindingMode.EMACS, "input"),
    KeyAction("left", "cursor_left", "Move cursor left", BindingMode.EMACS, "input"),
    KeyAction("right", "cursor_right", "Move cursor right", BindingMode.EMACS, "input"),
    KeyAction("home", "cursor_home", "Move cursor to start", BindingMode.EMACS, "input"),
    KeyAction("end", "cursor_end", "Move cursor to end", BindingMode.EMACS, "input"),

    # Editing
    KeyAction("backspace", "delete_char", "Delete character before cursor", BindingMode.EMACS, "input"),
    KeyAction("delete", "delete_char_forward", "Delete character at cursor", BindingMode.EMACS, "input"),
    KeyAction("c-a", "cursor_home", "Move to start of line", BindingMode.EMACS, "input"),
    KeyAction("c-e", "cursor_end", "Move to end of line", BindingMode.EMACS, "input"),
    KeyAction("c-k", "kill_line", "Kill line from cursor", BindingMode.EMACS, "input"),
    KeyAction("c-u", "kill_line_before", "Kill line before cursor", BindingMode.EMACS, "input"),
    KeyAction("c-w", "kill_word_back", "Kill word before cursor", BindingMode.EMACS, "input"),
    KeyAction("c-y", "yank", "Yank killed text", BindingMode.EMACS, "input"),

    # Multi-line
    KeyAction("enter", "submit_or_newline", "Submit or insert newline", BindingMode.EMACS, "input"),
    KeyAction("c-j", "newline", "Insert newline", BindingMode.EMACS, "input"),
    KeyAction("c-o", "newline", "Insert newline", BindingMode.EMACS, "input"),

    # Completion
    KeyAction("tab", "complete", "Tab completion", BindingMode.EMACS, "input"),
    KeyAction("s-tab", "complete_back", "Backward completion", BindingMode.EMACS, "input"),

    # Session control
    KeyAction("c-c", "interrupt", "Interrupt current operation", BindingMode.EMACS, "global"),
    KeyAction("c-d", "exit", "Exit session", BindingMode.EMACS, "global"),
    KeyAction("c-l", "clear_screen", "Clear screen", BindingMode.EMACS, "global"),
    KeyAction("c-r", "search_history", "Search history", BindingMode.EMACS, "input"),

    # Pane/layout
    KeyAction("c-x o", "focus_next_pane", "Focus next pane", BindingMode.EMACS, "global"),
    KeyAction("c-x 1", "layout_single", "Single pane layout", BindingMode.EMACS, "global"),
    KeyAction("c-x 2", "layout_split_h", "Split horizontally", BindingMode.EMACS, "global"),
    KeyAction("c-x 3", "layout_split_v", "Split vertically", BindingMode.EMACS, "global"),

    # Macro
    KeyAction("c-x (", "macro_start", "Start macro recording", BindingMode.EMACS, "global"),
    KeyAction("c-x )", "macro_stop", "Stop macro recording", BindingMode.EMACS, "global"),
    KeyAction("c-x e", "macro_play", "Play last macro", BindingMode.EMACS, "global"),
]


class KeyBindingManager:
    """Manages key bindings with Emacs/Vi modes and macro support.

    Provides registration, lookup, and persistence of key bindings,
    as well as macro recording and playback. Integrates with
    prompt_toolkit when available.
    """

    def __init__(
        self,
        mode: BindingMode = BindingMode.EMACS,
        config_file: str | None = None,
    ) -> None:
        self._mode = mode
        self._config_file = config_file or DEFAULT_KEYBINDS_FILE
        self._bindings: list[KeyAction] = list(DEFAULT_BINDINGS)
        self._handlers: dict[str, Callable[[], None]] = {}
        self._macros: dict[str, Macro] = {}
        self._recording_macro: Macro | None = None
        self._last_macro: Macro | None = None
        self._ptk_bindings: Any = None

        # Register default handlers
        self._register_default_handlers()

        # Load custom bindings from file
        self._load_bindings()

    @property
    def mode(self) -> BindingMode:
        """Get the current binding mode."""
        return self._mode

    @mode.setter
    def mode(self, value: BindingMode) -> None:
        """Set the binding mode and rebuild bindings."""
        self._mode = value
        self._build_ptk_bindings()

    def register_handler(self, action: str, handler: Callable[[], None]) -> None:
        """Register a handler function for a named action.

        Args:
            action: Action name (e.g., 'clear_screen').
            handler: Callable to invoke when the action is triggered.
        """
        self._handlers[action] = handler

    def add_binding(
        self,
        key: str,
        action: str,
        description: str = "",
        context: str = "global",
        mode: BindingMode | None = None,
    ) -> None:
        """Add a custom key binding.

        Args:
            key: Key sequence (e.g., 'c-x s', 'f5').
            action: Action name to trigger.
            description: Human-readable description.
            context: Context where the binding is active.
            mode: Binding mode (defaults to current mode).
        """
        binding = KeyAction(
            key=key,
            action=action,
            description=description,
            mode=mode or self._mode,
            context=context,
        )
        self._bindings.append(binding)
        self._build_ptk_bindings()

    def remove_binding(self, key: str, context: str = "global") -> bool:
        """Remove a key binding.

        Args:
            key: Key sequence to remove.
            context: Context of the binding.

        Returns:
            True if a binding was found and removed.
        """
        for i, binding in enumerate(self._bindings):
            if binding.key == key and binding.context == context:
                self._bindings.pop(i)
                self._build_ptk_bindings()
                return True
        return False

    def get_binding(self, key: str, context: str = "global") -> KeyAction | None:
        """Look up a key binding.

        Args:
            key: Key sequence.
            context: Binding context.

        Returns:
            The KeyAction, or None if not found.
        """
        for binding in reversed(self._bindings):
            if binding.key == key and binding.context == context and binding.mode == self._mode:
                return binding
        return None

    def get_all_bindings(self, context: str | None = None) -> list[KeyAction]:
        """Get all bindings, optionally filtered by context.

        Args:
            context: Optional context filter.

        Returns:
            List of KeyAction objects.
        """
        if context:
            return [b for b in self._bindings if b.context == context and b.mode == self._mode]
        return [b for b in self._bindings if b.mode == self._mode]

    def get_help_text(self) -> str:
        """Generate help text showing all active bindings.

        Returns:
            Formatted string of key bindings.
        """
        lines = [f"Key Bindings ({self._mode.value} mode)", ""]

        by_context: dict[str, list[KeyAction]] = {}
        for binding in self._bindings:
            if binding.mode != self._mode:
                continue
            ctx = binding.context
            if ctx not in by_context:
                by_context[ctx] = []
            by_context[ctx].append(binding)

        for ctx in sorted(by_context):
            lines.append(f"  [{ctx}]")
            max_key_len = max(len(b.key) for b in by_context[ctx]) if by_context[ctx] else 0
            for binding in sorted(by_context[ctx], key=lambda b: b.key):
                padded_key = binding.key.ljust(max_key_len)
                lines.append(f"    {padded_key}  ─  {binding.description or binding.action}")
            lines.append("")

        return "\n".join(lines)

    # Macro support

    def start_macro(self, name: str = "") -> None:
        """Start recording a macro.

        Args:
            name: Optional macro name.
        """
        macro_name = name or f"macro_{len(self._macros) + 1}"
        self._recording_macro = Macro(name=macro_name)
        self._recording_macro.actions = []

    def stop_macro(self) -> Macro | None:
        """Stop recording the current macro.

        Returns:
            The recorded Macro, or None if not recording.
        """
        if self._recording_macro is None:
            return None
        macro = self._recording_macro
        self._macros[macro.name] = macro
        self._last_macro = macro
        self._recording_macro = None
        return macro

    def record_key(self, key: str) -> None:
        """Record a key press in the current macro.

        Args:
            key: The key that was pressed.
        """
        if self._recording_macro is not None:
            self._recording_macro.actions.append(key)

    def play_macro(self, name: str | None = None) -> list[str] | None:
        """Play back a recorded macro.

        Args:
            name: Macro name (defaults to last recorded macro).

        Returns:
            List of key actions, or None if macro not found.
        """
        if name:
            macro = self._macros.get(name)
        else:
            macro = self._last_macro

        if macro is None:
            return None

        # Execute each action in the macro
        for action_key in macro.actions:
            binding = self.get_binding(action_key)
            if binding and binding.action in self._handlers:
                self._handlers[binding.action]()

        return macro.actions

    def list_macros(self) -> list[str]:
        """List all recorded macro names.

        Returns:
            List of macro name strings.
        """
        return list(self._macros.keys())

    def delete_macro(self, name: str) -> bool:
        """Delete a recorded macro.

        Args:
            name: Macro name.

        Returns:
            True if the macro was found and deleted.
        """
        if name in self._macros:
            del self._macros[name]
            if self._last_macro and self._last_macro.name == name:
                self._last_macro = None
            return True
        return False

    # Persistence

    def save_bindings(self, path: str | None = None) -> str:
        """Save custom bindings to a JSON file.

        Args:
            path: File path (defaults to config file).

        Returns:
            The path the bindings were saved to.
        """
        save_path = path or self._config_file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = {
            "mode": self._mode.value,
            "bindings": [b.to_dict() for b in self._bindings],
            "macros": {name: m.to_dict() for name, m in self._macros.items()},
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return save_path

    def _load_bindings(self) -> None:
        """Load custom bindings from the config file."""
        if not os.path.exists(self._config_file):
            return
        try:
            with open(self._config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "mode" in data:
                self._mode = BindingMode(data["mode"])

            # Merge custom bindings (don't replace defaults)
            custom = data.get("bindings", [])
            existing_keys = {(b.key, b.context, b.mode) for b in self._bindings}
            for binding_data in custom:
                action = KeyAction.from_dict(binding_data)
                key_tuple = (action.key, action.context, action.mode)
                if key_tuple not in existing_keys:
                    self._bindings.append(action)

            # Load macros
            for name, macro_data in data.get("macros", {}).items():
                self._macros[name] = Macro.from_dict(macro_data)

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # prompt_toolkit integration

    def get_ptk_bindings(self) -> Any:
        """Get prompt_toolkit KeyBindings for integration.

        Returns:
            A prompt_toolkit KeyBindings object, or None if unavailable.
        """
        if not HAS_PROMPT_TOOLKIT:
            return None
        if self._ptk_bindings is None:
            self._build_ptk_bindings()
        return self._ptk_bindings

    def _build_ptk_bindings(self) -> None:
        """Build prompt_toolkit KeyBindings from registered bindings."""
        if not HAS_PROMPT_TOOLKIT:
            return

        kb = KeyBindings()

        for binding in self._bindings:
            if binding.mode != self._mode:
                continue
            handler = self._handlers.get(binding.action)
            if handler:
                try:
                    # Convert our key notation to prompt_toolkit format
                    ptk_key = self._convert_key(binding.key)
                    if ptk_key:
                        kb.add(ptk_key)(lambda event, h=handler: h())
                except Exception:
                    pass

        self._ptk_bindings = kb

    @staticmethod
    def _convert_key(key: str) -> str | None:
        """Convert our key notation to prompt_toolkit format.

        Args:
            key: Key string like 'c-x', 'f5', 'up', 'enter'.

        Returns:
            Prompt toolkit key string, or None if not convertible.
        """
        # Direct mappings
        direct_map = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "home": "home",
            "end": "end",
            "enter": "enter",
            "tab": "tab",
            "backspace": "backspace",
            "delete": "delete",
            "escape": "escape",
            "space": "space",
        }

        if key in direct_map:
            return direct_map[key]

        # Convert Ctrl combinations
        if key.startswith("c-") and len(key) == 3:
            return f"c-{key[2]}"

        # Convert Alt combinations
        if key.startswith("a-") and len(key) == 3:
            return f"escape,{key[2]}"

        # Convert Shift+Tab
        if key == "s-tab":
            return "s-tab"

        # Function keys
        if key.startswith("f") and key[1:].isdigit():
            return key

        return None

    def _register_default_handlers(self) -> None:
        """Register default no-op handlers for all standard actions."""
        default_actions = [
            "history_prev", "history_next", "cursor_left", "cursor_right",
            "cursor_home", "cursor_end", "delete_char", "delete_char_forward",
            "kill_line", "kill_line_before", "kill_word_back", "yank",
            "submit_or_newline", "newline", "complete", "complete_back",
            "interrupt", "exit", "clear_screen", "search_history",
            "focus_next_pane", "layout_single", "layout_split_h",
            "layout_split_v", "macro_start", "macro_stop", "macro_play",
        ]
        for action in default_actions:
            self._handlers[action] = lambda: None
