"""Main TerminalChat class for Nexus-LLM — the Claude-like terminal experience.

Provides an interactive chat loop with streaming responses, slash-command
handling, auto-loading of the default model, a beautiful startup banner,
graceful interrupt handling, multi-line input, and auto-suggest for commands.
"""

import sys
import time
from typing import Any, Dict, List, Optional

from rich.console import Console

from nexus_llm.terminal.formatter import OutputFormatter
from nexus_llm.terminal.commands import CommandHandler
from nexus_llm.terminal.history import ChatHistory
from nexus_llm.terminal.themes import get_theme

from nexus_llm.core.config import Settings, get_settings
from nexus_llm.core.model_catalog import MODEL_CATALOG


class TerminalChat:
    """Interactive terminal chat interface for Nexus-LLM.

    Usage::

        chat = TerminalChat()
        chat.run()
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        engine: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine

        # Rich console & output
        self.console = Console()
        theme = get_theme(self.settings.terminal.theme)
        self.formatter = OutputFormatter(console=self.console, theme=theme)

        # History & commands
        self.history = ChatHistory(
            history_dir=self.settings.terminal.history_file,
            max_history=self.settings.terminal.max_history,
        )
        self.commands = CommandHandler(formatter=self.formatter, history=self.history)

        # Shared context passed to every command handler
        self.ctx: Dict[str, Any] = {
            "settings": self.settings,
            "engine": self.engine,
            "system_prompt": "",
            "seed": None,
        }

        self._running = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _auto_load_model(self) -> None:
        """Attempt to load the default model on startup."""
        if self.engine is None:
            try:
                from nexus_llm.backend.inference import InferenceEngine
                self.engine = InferenceEngine()
                self.ctx["engine"] = self.engine
            except Exception:
                self.formatter.print_error("Could not initialise inference engine.")
                self.formatter.print_info("You can load a model later with /load <model_id>.")
                return

        if self.engine.model_manager.is_loaded:
            model_id = self.engine.model_manager.model_id or self.settings.model.name
            self.formatter.print_welcome(model_name=model_id)
            return

        model_id = self.settings.model.name
        if model_id not in MODEL_CATALOG:
            self.formatter.print_info(f"Default model '{model_id}' not found in catalog.")
            self.formatter.print_info("Use /models to see available models, then /load <model_id>.")
            self.formatter.print_welcome()
            return

        self.formatter.print_info(f"Loading default model: {model_id} ...")
        try:
            self.engine.load_model(
                model_id=model_id,
                device=self.settings.model.device,
                precision=self.settings.model.precision,
                cache_dir=self.settings.model.cache_dir,
            )
            self.formatter.print_welcome(model_name=model_id)
            self.history.new_conversation(model=model_id)
        except Exception as exc:
            self.formatter.print_error(f"Failed to load default model: {exc}")
            self.formatter.print_info("Use /load <model_id> to load a different model.")
            self.formatter.print_welcome()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _get_input(self) -> Optional[str]:
        """Read user input with optional prompt_toolkit support.

        Falls back to ``input()`` if prompt_toolkit is not installed.
        Returns None on empty input.
        """
        try:
            return self._get_input_prompt_toolkit()
        except ImportError:
            return self._get_input_builtin()

    def _get_input_prompt_toolkit(self) -> Optional[str]:
        """Read input using prompt_toolkit with auto-suggest & multi-line."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import WordCompleter

        if not hasattr(self, "_prompt_session"):
            command_completer = WordCompleter(
                self.commands.command_names,
                ignore_case=True,
                sentence=True,
            )
            self._prompt_session = PromptSession(
                auto_suggest=AutoSuggestFromHistory(),
                completer=command_completer,
                complete_while_typing=True,
            )

        try:
            text = self._prompt_session.prompt(
                "\U0001f916 > ",
                multiline=False,
            )
        except EOFError:
            raise SystemExit(0)

        return text.strip() if text else None

    def _get_input_builtin(self) -> Optional[str]:
        """Fallback input using the built-in ``input()``."""
        try:
            text = input("\U0001f916 > ")
        except EOFError:
            raise SystemExit(0)

        return text.strip() if text else None

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def _generate_response(self, user_text: str) -> None:
        """Generate and display an assistant response.

        Chooses between streaming and single-shot based on settings.
        """
        if self.engine is None or not self.engine.is_ready:
            self.formatter.print_error("No model loaded. Use /load <model_id> first.")
            return

        # Build the message list
        messages = self._build_messages(user_text)

        # Add the user message to history
        self.history.add_message("user", user_text)

        try:
            if self.settings.terminal.streaming:
                self._generate_streaming(messages)
            else:
                self._generate_single(messages)
        except KeyboardInterrupt:
            self.formatter.print_info("\nGeneration interrupted.")
            if self.engine:
                self.engine.stop_generation()
        except Exception as exc:
            self.formatter.print_error(f"Generation failed: {exc}")

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        """Build the full message list for the engine."""
        messages: List[Dict[str, str]] = []

        # System prompt
        system_prompt = self.ctx.get("system_prompt", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(self.history.get_message_dicts())

        # Current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    def _generate_streaming(self, messages: List[Dict[str, str]]) -> None:
        """Generate a streaming response and display it token by token."""
        self.formatter.start_stream()

        stream = self.engine.chat_stream(
            messages=messages,
            max_new_tokens=self.settings.model.max_length,
            temperature=self.settings.model.temperature,
            top_p=self.settings.model.top_p,
            top_k=self.settings.model.top_k,
        )

        for token in stream:
            self.formatter.append_stream(token)

        stats = self.formatter.end_stream()
        full_text = self.formatter.stream_buffer

        # Record assistant message in history
        self.history.add_message("assistant", full_text, metadata=stats)

        # Trim history if needed
        self.history.auto_trim()

    def _generate_single(self, messages: List[Dict[str, str]]) -> None:
        """Generate a single-shot (non-streaming) response."""
        start = time.monotonic()
        result = self.engine.chat(
            messages=messages,
            max_new_tokens=self.settings.model.max_length,
            temperature=self.settings.model.temperature,
            top_p=self.settings.model.top_p,
            top_k=self.settings.model.top_k,
        )
        elapsed = time.monotonic() - start

        text = result.get("text", "")
        self.formatter.print_assistant_message(text)

        if self.settings.terminal.show_timing or self.settings.terminal.show_tokens:
            self.formatter.print_generation_stats(result)

        # Record in history
        metadata = {
            "tokens": result.get("generated_tokens", 0),
            "elapsed_s": round(elapsed, 3),
        }
        self.history.add_message("assistant", text, metadata=metadata)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive terminal chat loop."""
        self._running = True

        # Banner
        self.formatter.print_banner(version="2.0.0")

        # Auto-load default model
        self._auto_load_model()

        # Main REPL
        while self._running:
            try:
                user_input = self._get_input()
                if user_input is None:
                    continue

                # Slash command
                if user_input.startswith("/"):
                    try:
                        self.commands.handle(user_input, self.ctx)
                    except SystemExit:
                        self._running = False
                        break
                    continue

                # Empty input
                if not user_input.strip():
                    continue

                # Normal chat
                self._generate_response(user_input)

            except KeyboardInterrupt:
                # Ctrl-C during input: confirm or exit
                self.console.print()
                try:
                    confirm = input("  Exit? (y/N) ").strip().lower()
                    if confirm in ("y", "yes"):
                        self._running = False
                        break
                    else:
                        continue
                except (EOFError, KeyboardInterrupt):
                    self._running = False
                    break

            except SystemExit:
                self._running = False
                break

            except Exception as exc:
                self.formatter.print_error(f"Unexpected error: {exc}")
                if self.settings.debug:
                    import traceback
                    traceback.print_exc()

        # Cleanup
        self._shutdown()

    def _shutdown(self) -> None:
        """Graceful shutdown: save history, unload model, print goodbye."""
        try:
            if self.history.current and self.history.current.messages:
                self.history.save()
        except Exception:
            pass

        try:
            if self.engine and self.engine.model_manager.is_loaded:
                self.engine.unload_model()
        except Exception:
            pass

        self.formatter.print_goodbye()

    def stop(self) -> None:
        """Signal the chat loop to exit."""
        self._running = False


# ======================================================================
# Convenience entry-point
# ======================================================================

def run_terminal(settings: Optional[Settings] = None) -> None:
    """Create and run a TerminalChat session.

    This is the primary entry-point used by the CLI.
    """
    chat = TerminalChat(settings=settings)
    chat.run()
