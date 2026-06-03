"""Command handler for Nexus-LLM terminal chat.

Implements 25+ slash commands that control model management, configuration,
chat history, theming, generation parameters, and server lifecycle.
"""

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from nexus_llm.terminal.formatter import OutputFormatter
from nexus_llm.terminal.history import ChatHistory
from nexus_llm.terminal.themes import Theme, get_theme, list_themes


class CommandHandler:
    """Dispatches slash-commands and delegates to the appropriate handler.

    Every handler receives (args: str, ctx: dict) and returns a string or
    None.  The *ctx* dictionary is populated by the TerminalChat instance
    and contains the live engine, settings, and other shared objects.
    """

    def __init__(self, formatter: OutputFormatter, history: ChatHistory) -> None:
        self.formatter = formatter
        self.history = history
        self._commands: Dict[str, Dict[str, Any]] = {}
        self._register_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, raw_line: str, ctx: Dict[str, Any]) -> Optional[str]:
        """Parse and execute a slash command.

        Args:
            raw_line: The full input line starting with ``/``.
            ctx: Shared context dict from TerminalChat.

        Returns:
            A result string, or None.
        """
        parts = raw_line.strip().split(None, 1)
        if not parts:
            return None

        cmd_name = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        entry = self._commands.get(cmd_name)
        if entry is None:
            self.formatter.print_error(f"Unknown command: {cmd_name}. Type /help for a list.")
            return None

        handler = entry["handler"]
        try:
            return handler(args, ctx)
        except Exception as exc:
            self.formatter.print_error(f"Command failed: {exc}")
            return None

    @property
    def command_names(self) -> List[str]:
        """Sorted list of registered command names."""
        return sorted(self._commands.keys())

    def get_completions(self, partial: str) -> List[str]:
        """Return command names that start with *partial* (for auto-suggest)."""
        return [c for c in self.command_names if c.startswith(partial)]

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _register(self, name: str, description: str, handler: Callable) -> None:
        self._commands[name] = {"description": description, "handler": handler}

    def _register_all(self) -> None:
        """Register all built-in commands."""
        self._register("/help", "Show available commands", self._cmd_help)
        self._register("/model", "Switch to a different model", self._cmd_model)
        self._register("/models", "List available models", self._cmd_models)
        self._register("/load", "Load a model", self._cmd_load)
        self._register("/unload", "Unload current model", self._cmd_unload)
        self._register("/info", "Show current model info", self._cmd_info)
        self._register("/config", "View or set configuration", self._cmd_config)
        self._register("/system", "Set system prompt", self._cmd_system)
        self._register("/clear", "Clear conversation", self._cmd_clear)
        self._register("/history", "Show chat history", self._cmd_history)
        self._register("/save", "Save conversation", self._cmd_save)
        self._register("/load-chat", "Load a saved conversation", self._cmd_load_chat)
        self._register("/export", "Export conversation as Markdown", self._cmd_export)
        self._register("/theme", "Change terminal theme", self._cmd_theme)
        self._register("/themes", "List available themes", self._cmd_themes)
        self._register("/stream", "Toggle streaming (on/off)", self._cmd_stream)
        self._register("/temp", "Set temperature", self._cmd_temp)
        self._register("/topp", "Set top_p", self._cmd_topp)
        self._register("/topk", "Set top_k", self._cmd_topk)
        self._register("/maxlen", "Set max generation length", self._cmd_maxlen)
        self._register("/seed", "Set random seed", self._cmd_seed)
        self._register("/reset", "Reset all settings to defaults", self._cmd_reset)
        self._register("/stats", "Show generation statistics", self._cmd_stats)
        self._register("/train", "Start training on a dataset", self._cmd_train)
        self._register("/server", "Manage server (start/stop)", self._cmd_server)
        self._register("/download", "Download a model", self._cmd_download)
        self._register("/quit", "Exit Nexus-LLM", self._cmd_quit)

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_help(self, args: str, ctx: Dict[str, Any]) -> None:
        """Display all available commands."""
        from rich.table import Table

        table = Table(title="Commands", show_header=True, border_style=self.formatter.theme.border_color)
        table.add_column("Command", style=self.formatter.theme.highlight, min_width=14)
        table.add_column("Description", style=self.formatter.theme.assistant_text)

        for name in self.command_names:
            entry = self._commands[name]
            table.add_row(name, entry["description"])

        self.formatter.console.print(table)

    # -- Model management -----------------------------------------------

    def _cmd_model(self, args: str, ctx: Dict[str, Any]) -> None:
        """Switch to a different model (loads it)."""
        if not args:
            self.formatter.print_error("Usage: /model <model_id>")
            return
        engine = ctx.get("engine")
        if engine is None:
            self.formatter.print_error("Engine not available.")
            return

        model_id = args.strip()
        self.formatter.print_info(f"Loading model: {model_id} ...")
        try:
            settings = ctx.get("settings")
            device = settings.model.device if settings else "auto"
            precision = settings.model.precision if settings else "fp32"
            cache_dir = settings.model.cache_dir if settings else None
            engine.load_model(model_id, device=device, precision=precision, cache_dir=cache_dir)
            self.formatter.print_success(f"Model '{model_id}' loaded successfully.")
            self.history.new_conversation(model=model_id)
        except Exception as exc:
            self.formatter.print_error(str(exc))

    def _cmd_models(self, args: str, ctx: Dict[str, Any]) -> None:
        """List available models."""
        from nexus_llm.core.model_catalog import MODEL_CATALOG
        models = [
            {
                "id": info.id,
                "name": info.name,
                "size": info.size,
                "category": info.category,
                "recommended": info.recommended,
            }
            for info in MODEL_CATALOG.values()
        ]
        models.sort(key=lambda m: m["id"])
        self.formatter.print_models_table(models)

    def _cmd_load(self, args: str, ctx: Dict[str, Any]) -> None:
        """Load a model by ID."""
        if not args:
            self.formatter.print_error("Usage: /load <model_id>")
            return
        self._cmd_model(args, ctx)

    def _cmd_unload(self, args: str, ctx: Dict[str, Any]) -> None:
        """Unload the current model."""
        engine = ctx.get("engine")
        if engine is None:
            self.formatter.print_error("Engine not available.")
            return
        if not engine.model_manager.is_loaded:
            self.formatter.print_info("No model is currently loaded.")
            return
        model_id = engine.model_manager.model_id
        engine.unload_model()
        self.formatter.print_success(f"Model '{model_id}' unloaded.")

    def _cmd_info(self, args: str, ctx: Dict[str, Any]) -> None:
        """Show current model information."""
        engine = ctx.get("engine")
        if engine is None or not engine.model_manager.is_loaded:
            self.formatter.print_info("No model is currently loaded.")
            return
        info = engine.model_manager.get_info()
        self.formatter.print_model_info_table(info)

    # -- Configuration --------------------------------------------------

    def _cmd_config(self, args: str, ctx: Dict[str, Any]) -> None:
        """View or set a configuration value."""
        settings = ctx.get("settings")
        if settings is None:
            self.formatter.print_error("Settings not available.")
            return

        if not args:
            # Show full config
            config_dict = dataclasses.asdict(settings)
            self.formatter.print_config_table(config_dict)
            return

        parts = args.split(None, 1)
        key = parts[0]
        value = parts[1] if len(parts) > 1 else None

        if value is None:
            # Show a single value
            current = self._get_nested_attr(settings, key)
            if current is not None:
                self.formatter.print_info(f"{key} = {current}")
            else:
                self.formatter.print_error(f"Unknown config key: {key}")
            return

        # Set a value
        try:
            self._set_nested_attr(settings, key, value)
            self.formatter.print_success(f"Set {key} = {value}")
        except Exception as exc:
            self.formatter.print_error(f"Failed to set {key}: {exc}")

    @staticmethod
    def _get_nested_attr(obj: Any, dotted_key: str) -> Any:
        """Resolve a dotted key like 'model.temperature' on a nested object."""
        parts = dotted_key.split(".")
        current = obj
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    @staticmethod
    def _set_nested_attr(obj: Any, dotted_key: str, raw_value: str) -> None:
        """Set a dotted key, auto-casting the string value to the right type."""
        parts = dotted_key.split(".")
        # Navigate to the parent
        current = obj
        for part in parts[:-1]:
            current = getattr(current, part)

        final_key = parts[-1]
        current_val = getattr(current, final_key)

        # Cast
        if isinstance(current_val, bool):
            new_val = raw_value.lower() in ("true", "1", "yes", "on")
        elif isinstance(current_val, int):
            new_val = int(raw_value)
        elif isinstance(current_val, float):
            new_val = float(raw_value)
        elif isinstance(current_val, str):
            new_val = raw_value
        else:
            new_val = raw_value

        setattr(current, final_key, new_val)

    # -- System prompt --------------------------------------------------

    def _cmd_system(self, args: str, ctx: Dict[str, Any]) -> None:
        """Set the system prompt for the conversation."""
        if not args:
            current = ctx.get("system_prompt", "")
            if current:
                self.formatter.print_info(f"System prompt: {current}")
            else:
                self.formatter.print_info("No system prompt set.")
            return

        ctx["system_prompt"] = args
        # Prepend as a system message in the current conversation
        self.history.add_message("system", args)
        self.formatter.print_success("System prompt updated.")

    # -- Conversation ---------------------------------------------------

    def _cmd_clear(self, args: str, ctx: Dict[str, Any]) -> None:
        """Clear the current conversation."""
        self.history.clear_current()
        ctx.pop("system_prompt", None)
        self.formatter.print_success("Conversation cleared.")

    def _cmd_history(self, args: str, ctx: Dict[str, Any]) -> None:
        """Show saved conversation history."""
        convs = self.history.list_conversations()
        if not convs:
            self.formatter.print_info("No saved conversations.")
            return

        from rich.table import Table
        table = Table(title="Chat History", show_header=True, border_style=self.formatter.theme.border_color)
        table.add_column("File", style=self.formatter.theme.highlight, min_width=22)
        table.add_column("Title", style=self.formatter.theme.assistant_text)
        table.add_column("Messages", style=self.formatter.theme.info_text)
        table.add_column("Model", style=self.formatter.theme.info_text)

        for c in convs:
            table.add_row(c["filename"], c["title"], str(c["message_count"]), c["model"])

        self.formatter.console.print(table)

    def _cmd_save(self, args: str, ctx: Dict[str, Any]) -> None:
        """Save the current conversation."""
        try:
            filename = args.strip() if args else None
            path = self.history.save(filename=filename)
            self.formatter.print_success(f"Conversation saved to {path}")
        except Exception as exc:
            self.formatter.print_error(f"Save failed: {exc}")

    def _cmd_load_chat(self, args: str, ctx: Dict[str, Any]) -> None:
        """Load a saved conversation."""
        if not args:
            self.formatter.print_error("Usage: /load-chat <filename>")
            return
        try:
            conv = self.history.load(args.strip())
            self.formatter.print_success(f"Loaded conversation: {conv.title} ({len(conv.messages)} messages)")
        except FileNotFoundError:
            self.formatter.print_error(f"File not found: {args.strip()}")
        except Exception as exc:
            self.formatter.print_error(f"Load failed: {exc}")

    def _cmd_export(self, args: str, ctx: Dict[str, Any]) -> None:
        """Export conversation as Markdown."""
        try:
            filepath = args.strip() if args else None
            path = self.history.export_markdown(filepath=filepath)
            self.formatter.print_success(f"Exported to {path}")
        except Exception as exc:
            self.formatter.print_error(f"Export failed: {exc}")

    # -- Themes ---------------------------------------------------------

    def _cmd_theme(self, args: str, ctx: Dict[str, Any]) -> None:
        """Change the terminal theme."""
        if not args:
            current = self.formatter.theme.name
            self.formatter.print_info(f"Current theme: {current}")
            return

        theme = get_theme(args.strip())
        self.formatter.set_theme(theme)
        settings = ctx.get("settings")
        if settings:
            settings.terminal.theme = theme.name
        self.formatter.print_success(f"Theme changed to: {theme.name} ({theme.description})")

    def _cmd_themes(self, args: str, ctx: Dict[str, Any]) -> None:
        """List available themes."""
        from rich.table import Table
        themes = list_themes()
        table = Table(title="Themes", show_header=True, border_style=self.formatter.theme.border_color)
        table.add_column("Name", style=self.formatter.theme.highlight)
        table.add_column("Description", style=self.formatter.theme.assistant_text)
        for name, t in themes.items():
            marker = " (active)" if t.name == self.formatter.theme.name else ""
            table.add_row(name + marker, t.description)
        self.formatter.console.print(table)

    # -- Generation parameters ------------------------------------------

    def _cmd_stream(self, args: str, ctx: Dict[str, Any]) -> None:
        """Toggle streaming on/off."""
        settings = ctx.get("settings")
        if settings is None:
            self.formatter.print_error("Settings not available.")
            return
        if not args:
            self.formatter.print_info(f"Streaming: {'on' if settings.terminal.streaming else 'off'}")
            return

        val = args.strip().lower()
        if val in ("on", "true", "yes", "1"):
            settings.terminal.streaming = True
            self.formatter.print_success("Streaming enabled.")
        elif val in ("off", "false", "no", "0"):
            settings.terminal.streaming = False
            self.formatter.print_success("Streaming disabled.")
        else:
            self.formatter.print_error("Usage: /stream [on|off]")

    def _cmd_temp(self, args: str, ctx: Dict[str, Any]) -> None:
        """Set temperature."""
        settings = ctx.get("settings")
        if settings is None:
            self.formatter.print_error("Settings not available.")
            return
        if not args:
            self.formatter.print_info(f"Temperature: {settings.model.temperature}")
            return
        try:
            val = float(args.strip())
            settings.model.temperature = val
            self.formatter.print_success(f"Temperature set to {val}")
        except ValueError:
            self.formatter.print_error("Invalid value. Usage: /temp <float>")

    def _cmd_topp(self, args: str, ctx: Dict[str, Any]) -> None:
        """Set top_p."""
        settings = ctx.get("settings")
        if settings is None:
            self.formatter.print_error("Settings not available.")
            return
        if not args:
            self.formatter.print_info(f"top_p: {settings.model.top_p}")
            return
        try:
            val = float(args.strip())
            settings.model.top_p = val
            self.formatter.print_success(f"top_p set to {val}")
        except ValueError:
            self.formatter.print_error("Invalid value. Usage: /topp <float>")

    def _cmd_topk(self, args: str, ctx: Dict[str, Any]) -> None:
        """Set top_k."""
        settings = ctx.get("settings")
        if settings is None:
            self.formatter.print_error("Settings not available.")
            return
        if not args:
            self.formatter.print_info(f"top_k: {settings.model.top_k}")
            return
        try:
            val = int(args.strip())
            settings.model.top_k = val
            self.formatter.print_success(f"top_k set to {val}")
        except ValueError:
            self.formatter.print_error("Invalid value. Usage: /topk <int>")

    def _cmd_maxlen(self, args: str, ctx: Dict[str, Any]) -> None:
        """Set max generation length."""
        settings = ctx.get("settings")
        if settings is None:
            self.formatter.print_error("Settings not available.")
            return
        if not args:
            self.formatter.print_info(f"max_length: {settings.model.max_length}")
            return
        try:
            val = int(args.strip())
            settings.model.max_length = val
            self.formatter.print_success(f"max_length set to {val}")
        except ValueError:
            self.formatter.print_error("Invalid value. Usage: /maxlen <int>")

    def _cmd_seed(self, args: str, ctx: Dict[str, Any]) -> None:
        """Set random seed."""
        if not args:
            self.formatter.print_info("No seed set (random).")
            return
        try:
            import random
            seed = int(args.strip())
            random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass
            ctx["seed"] = seed
            self.formatter.print_success(f"Random seed set to {seed}")
        except ValueError:
            self.formatter.print_error("Invalid value. Usage: /seed <int>")

    def _cmd_reset(self, args: str, ctx: Dict[str, Any]) -> None:
        """Reset all settings to defaults."""
        from nexus_llm.core.config import Settings
        settings = Settings()
        ctx["settings"] = settings
        self.formatter.set_theme(get_theme(settings.terminal.theme))
        self.formatter.print_success("All settings reset to defaults.")

    # -- Stats ----------------------------------------------------------

    def _cmd_stats(self, args: str, ctx: Dict[str, Any]) -> None:
        """Show generation statistics."""
        engine = ctx.get("engine")
        if engine is None:
            self.formatter.print_error("Engine not available.")
            return
        stats = engine.get_stats()
        self.formatter.print_config_table(stats, title="Generation Statistics")

    # -- Training -------------------------------------------------------

    def _cmd_train(self, args: str, ctx: Dict[str, Any]) -> None:
        """Start training on a dataset (placeholder)."""
        if not args:
            self.formatter.print_error("Usage: /train <dataset_path>")
            return
        self.formatter.print_info(f"Training on '{args}' is not yet implemented in this version.")
        self.formatter.print_info("Use the Python API or server endpoints for training.")

    # -- Server ---------------------------------------------------------

    def _cmd_server(self, args: str, ctx: Dict[str, Any]) -> None:
        """Manage the API server."""
        action = args.strip().lower() if args else ""
        if action == "start":
            settings = ctx.get("settings")
            host = settings.server.host if settings else "127.0.0.1"
            port = settings.server.port if settings else 8000
            self.formatter.print_info(f"Starting server on {host}:{port} ...")
            try:
                from nexus_llm.backend.server import LLMServer
                engine = ctx.get("engine")
                server = LLMServer(host=host, port=port)
                # Note: this blocks. In production you'd run in a thread.
                self.formatter.print_info("Server would start here (blocking call omitted in terminal mode).")
                self.formatter.print_info("Run: python -m nexus_llm serve  to start the server.")
            except Exception as exc:
                self.formatter.print_error(f"Server start failed: {exc}")
        elif action == "stop":
            self.formatter.print_info("Server stop requested (no server running from this terminal).")
        else:
            self.formatter.print_error("Usage: /server [start|stop]")

    # -- Download -------------------------------------------------------

    def _cmd_download(self, args: str, ctx: Dict[str, Any]) -> None:
        """Download a model from HuggingFace."""
        if not args:
            self.formatter.print_error("Usage: /download <model_id>")
            return

        model_id = args.strip()
        from nexus_llm.core.model_catalog import MODEL_CATALOG
        if model_id not in MODEL_CATALOG:
            self.formatter.print_error(f"Model '{model_id}' not found in catalog. Use /models to see available models.")
            return

        info = MODEL_CATALOG[model_id]
        self.formatter.print_info(f"Downloading {info.name} ({info.hf_id}) ...")
        self.formatter.print_info("This will cache the model on first load. Use /load to load it after download.")

        settings = ctx.get("settings")
        engine = ctx.get("engine")
        if engine:
            try:
                device = settings.model.device if settings else "auto"
                precision = settings.model.precision if settings else "fp32"
                cache_dir = settings.model.cache_dir if settings else None
                engine.load_model(model_id, device=device, precision=precision, cache_dir=cache_dir)
                self.formatter.print_success(f"Model '{model_id}' downloaded and loaded.")
                self.history.new_conversation(model=model_id)
            except Exception as exc:
                self.formatter.print_error(f"Download/load failed: {exc}")

    # -- Quit -----------------------------------------------------------

    def _cmd_quit(self, args: str, ctx: Dict[str, Any]) -> None:
        """Exit the application."""
        raise SystemExit(0)
