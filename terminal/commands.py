"""Command Handler - Processes terminal commands and special inputs."""

import logging
from typing import Dict, Any, Callable, Optional, List

from terminal.formatter import OutputFormatter
from terminal.history import ChatHistory

logger = logging.getLogger(__name__)


class CommandHandler:
    """
    Handles all slash commands in the terminal chat interface.
    Provides a command registry with validation and help generation.
    """

    def __init__(
        self,
        formatter: OutputFormatter,
        history: ChatHistory,
        inference_engine=None,
        model_manager=None,
        settings=None,
    ):
        self.formatter = formatter
        self.history = history
        self.inference_engine = inference_engine
        self.model_manager = model_manager
        self.settings = settings

        # Command registry
        self._commands: Dict[str, Dict[str, Any]] = {
            "/help": {
                "handler": self._cmd_help,
                "description": "Show available commands",
                "usage": "/help",
            },
            "/quit": {
                "handler": self._cmd_quit,
                "description": "Exit the application",
                "usage": "/quit",
            },
            "/exit": {
                "handler": self._cmd_quit,
                "description": "Exit the application (alias for /quit)",
                "usage": "/exit",
            },
            "/clear": {
                "handler": self._cmd_clear,
                "description": "Clear the current conversation",
                "usage": "/clear",
            },
            "/save": {
                "handler": self._cmd_save,
                "description": "Save the current conversation",
                "usage": "/save",
            },
            "/history": {
                "handler": self._cmd_history,
                "description": "Show saved conversation sessions",
                "usage": "/history",
            },
            "/load": {
                "handler": self._cmd_load,
                "description": "Load a previous session",
                "usage": "/load <session_id>",
            },
            "/search": {
                "handler": self._cmd_search,
                "description": "Search through conversation history",
                "usage": "/search <query>",
            },
            "/export": {
                "handler": self._cmd_export,
                "description": "Export current session",
                "usage": "/export [json|md]",
            },
            "/model": {
                "handler": self._cmd_model,
                "description": "Show current model information",
                "usage": "/model",
            },
            "/switch": {
                "handler": self._cmd_switch,
                "description": "Switch to a different model",
                "usage": "/switch <model_name>",
            },
            "/config": {
                "handler": self._cmd_config,
                "description": "Show current configuration",
                "usage": "/config",
            },
            "/set": {
                "handler": self._cmd_set,
                "description": "Update a configuration setting",
                "usage": "/set <key> <value>",
            },
            "/stats": {
                "handler": self._cmd_stats,
                "description": "Show inference statistics",
                "usage": "/stats",
            },
            "/stop": {
                "handler": self._cmd_stop,
                "description": "Stop current generation",
                "usage": "/stop",
            },
            "/system": {
                "handler": self._cmd_system,
                "description": "Set the system prompt",
                "usage": "/system <prompt>",
            },
            "/tokenize": {
                "handler": self._cmd_tokenize,
                "description": "Count tokens in text",
                "usage": "/tokenize <text>",
            },
            "/reset": {
                "handler": self._cmd_reset,
                "description": "Reset everything to defaults",
                "usage": "/reset",
            },
        }

        self._should_exit = False

    @property
    def should_exit(self) -> bool:
        """Check if the application should exit."""
        return self._should_exit

    def is_command(self, text: str) -> bool:
        """Check if the input text is a command."""
        return text.strip().startswith("/")

    def execute(self, text: str) -> Optional[str]:
        """
        Execute a command and return any output.
        Returns None for commands that handle their own output.
        """
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd not in self._commands:
            self.formatter.print_error(f"Unknown command: {cmd}")
            self.formatter.print_info("Type /help to see available commands.")
            return None

        handler = self._commands[cmd]["handler"]
        try:
            return handler(args)
        except Exception as e:
            self.formatter.print_error(f"Command failed: {e}")
            logger.error(f"Command '{cmd}' failed: {e}")
            return None

    # ---- Command Implementations ----

    def _cmd_help(self, args: str) -> None:
        """Show help information."""
        self.formatter.print_help()

    def _cmd_quit(self, args: str) -> None:
        """Exit the application."""
        self._should_exit = True
        self.formatter.print_info("Saving session and exiting... Goodbye!")
        self.history.save_session()

    def _cmd_clear(self, args: str) -> None:
        """Clear the current conversation."""
        self.history.clear()
        self.formatter.print_success("Conversation cleared. Starting fresh!")

    def _cmd_save(self, args: str) -> None:
        """Save the current conversation."""
        self.history.save_session()
        self.formatter.print_success(
            f"Session saved: {self.history.session_id}"
        )

    def _cmd_history(self, args: str) -> None:
        """Show saved conversation sessions."""
        sessions = self.history.list_sessions()
        self.formatter.print_sessions_list(sessions)

    def _cmd_load(self, args: str) -> None:
        """Load a previous session."""
        session_id = args.strip()
        if not session_id:
            self.formatter.print_error("Please provide a session ID. Use /history to list sessions.")
            return

        if self.history.load_session(session_id):
            self.formatter.print_success(
                f"Session loaded: {session_id} ({self.history.message_count} messages)"
            )
        else:
            self.formatter.print_error(f"Failed to load session: {session_id}")

    def _cmd_search(self, args: str) -> None:
        """Search through conversation history."""
        query = args.strip()
        if not query:
            self.formatter.print_error("Please provide a search query.")
            return

        results = self.history.search_history(query)
        if results:
            self.formatter.print_info(f"Found {len(results)} results:")
            for result in results[:10]:
                self.formatter.console.print(
                    f"  [{result['session_id']}] {result['role']}: {result['content'][:100]}..."
                )
        else:
            self.formatter.print_info("No results found.")

    def _cmd_export(self, args: str) -> None:
        """Export current session."""
        fmt = args.strip().lower() or "json"
        content = self.history.export_session(format=fmt)
        if content:
            self.formatter.console.print(content)
        else:
            self.formatter.print_error("No session to export.")

    def _cmd_model(self, args: str) -> None:
        """Show current model information."""
        if self.model_manager:
            info = self.model_manager.model_info
            self.formatter.print_model_info(info)
        else:
            self.formatter.print_error("Model not loaded yet.")

    def _cmd_switch(self, args: str) -> None:
        """Switch to a different model."""
        model_name = args.strip()
        if not model_name:
            self.formatter.print_error("Please provide a model name.")
            self.formatter.print_info("Examples: gpt2, gpt2-medium, gpt2-large, microsoft/DialoGPT-medium")
            return

        if self.model_manager:
            self.formatter.print_info(f"Switching to model: {model_name}")
            try:
                self.model_manager.unload_model()
                self.model_manager.model_name = model_name
                self.model_manager.load_model()
                self.formatter.print_success(f"Model switched to: {model_name}")
            except Exception as e:
                self.formatter.print_error(f"Failed to switch model: {e}")

    def _cmd_config(self, args: str) -> None:
        """Show current configuration."""
        if self.settings:
            config_dict = self.settings.to_dict()
            self.formatter.console.print(config_dict)
        else:
            self.formatter.print_error("Configuration not available.")

    def _cmd_set(self, args: str) -> None:
        """Update a configuration setting."""
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            self.formatter.print_error("Usage: /set <key> <value>")
            self.formatter.print_info("Example: /set temperature 0.8")
            return

        key, value = parts
        key_map = {
            "temperature": ("inference", "temperature", float),
            "top_p": ("inference", "top_p", float),
            "top_k": ("inference", "top_k", int),
            "max_new_tokens": ("model", "max_new_tokens", int),
            "system_prompt": ("inference", "system_prompt", str),
            "repetition_penalty": ("inference", "repetition_penalty", float),
        }

        if key not in key_map:
            self.formatter.print_error(f"Unknown setting: {key}")
            self.formatter.print_info(f"Available settings: {', '.join(key_map.keys())}")
            return

        section, attr, type_func = key_map[key]
        try:
            typed_value = type_func(value)
            if self.settings:
                obj = getattr(self.settings, section)
                setattr(obj, attr, typed_value)
                self.formatter.print_success(f"Set {key} = {typed_value}")
        except (ValueError, TypeError) as e:
            self.formatter.print_error(f"Invalid value for {key}: {e}")

    def _cmd_stats(self, args: str) -> None:
        """Show inference statistics."""
        if self.inference_engine:
            stats = self.inference_engine.stats
            self.formatter.print_model_info(stats)
        else:
            self.formatter.print_error("No inference statistics available yet.")

    def _cmd_stop(self, args: str) -> None:
        """Stop current generation."""
        if self.inference_engine:
            self.inference_engine.stop_generation()
            self.formatter.print_info("Generation stopped.")
        else:
            self.formatter.print_error("No generation in progress.")

    def _cmd_system(self, args: str) -> None:
        """Set the system prompt."""
        prompt = args.strip()
        if not prompt:
            # Show current system prompt
            if self.settings:
                self.formatter.print_info(f"Current system prompt: {self.settings.inference.system_prompt}")
            else:
                self.formatter.print_error("No system prompt set.")
            return

        if self.settings:
            self.settings.inference.system_prompt = prompt
            self.formatter.print_success("System prompt updated.")

    def _cmd_tokenize(self, args: str) -> None:
        """Count tokens in text."""
        text = args.strip()
        if not text:
            self.formatter.print_error("Please provide text to tokenize.")
            return

        if self.model_manager:
            count = self.model_manager.count_tokens(text)
            self.formatter.print_info(f"Token count: {count}")
        else:
            self.formatter.print_error("Model not loaded yet.")

    def _cmd_reset(self, args: str) -> None:
        """Reset everything to defaults."""
        self.history.clear()
        self.formatter.print_success("Reset complete. Starting fresh with default settings.")
