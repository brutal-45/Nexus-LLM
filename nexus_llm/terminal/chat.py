"""
Nexus-LLM Interactive Chat Module

Provides a full-featured interactive chat loop with streaming responses,
multi-turn conversation management, system prompts, and raw string regex
pattern extraction from model outputs.
"""

from __future__ import annotations

import re
import sys
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from nexus_llm.terminal.formatter import RichFormatter
from nexus_llm.terminal.history import ChatHistory, HistoryEntry
from nexus_llm.terminal.prompts import PromptSession
from nexus_llm.terminal.status import StatusBar, StatusField
from nexus_llm.terminal.spinner import Spinner, SpinnerStyle


class MessageRole(str, Enum):
    """Enumeration of message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message to a dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        """Deserialize a message from a dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


# Raw string regex patterns for extracting structured data from model outputs
RAW_PATTERNS: dict[str, re.Pattern[str]] = {
    "code_block": re.compile(r"```(\w*)\n(.*?)```", re.DOTALL),
    "inline_code": re.compile(r"`([^`]+)`"),
    "bold": re.compile(r"\*\*(.+?)\*\*"),
    "italic": re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"),
    "link": re.compile(r"\[([^\]]+)\]\(([^)]+)\)"),
    "image": re.compile(r"!\[([^\]]*)\]\(([^)]+)\)"),
    "heading": re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),
    "list_item": re.compile(r"^[\s]*[-*+]\s+(.+)$", re.MULTILINE),
    "ordered_list": re.compile(r"^[\s]*\d+\.\s+(.+)$", re.MULTILINE),
    "blockquote": re.compile(r"^>\s+(.+)$", re.MULTILINE),
    "horizontal_rule": re.compile(r"^(-{3,}|\*{3,}|_{3,})$", re.MULTILINE),
    "table_row": re.compile(r"^\|(.+)\|$", re.MULTILINE),
    "table_separator": re.compile(r"^\|[\s:-]+\|$", re.MULTILINE),
    "json_object": re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL),
    "url": re.compile(r"https?://[^\s<>\"')\]]+"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "number": re.compile(r"-?\d+\.?\d*"),
    "tag": re.compile(r"<(\w+)[^>]*>(.*?)</\1>", re.DOTALL),
    "think_block": re.compile(r"<think\s*>(.*?)</think\s*>", re.DOTALL),
    "function_call": re.compile(r"(\w+)\(([^)]*)\)"),
}


def extract_raw_patterns(text: str) -> dict[str, list[str]]:
    """Extract all raw pattern matches from text.

    Args:
        text: The text to scan for patterns.

    Returns:
        A dictionary mapping pattern names to lists of matched strings.
    """
    results: dict[str, list[str]] = {}
    for name, pattern in RAW_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            results[name] = matches if isinstance(matches[0], str) else [m[0] for m in matches]
    return results


def strip_raw_patterns(text: str) -> str:
    """Strip markdown formatting from text, returning plain content.

    Args:
        text: Text containing markdown/raw patterns.

    Returns:
        Plain text with formatting removed.
    """
    result = text
    result = RAW_PATTERNS["code_block"].sub(r"\2", result)
    result = RAW_PATTERNS["inline_code"].sub(r"\1", result)
    result = RAW_PATTERNS["bold"].sub(r"\1", result)
    result = RAW_PATTERNS["italic"].sub(r"\1", result)
    result = RAW_PATTERNS["link"].sub(r"\1", result)
    result = RAW_PATTERNS["image"].sub(r"\1", result)
    result = RAW_PATTERNS["heading"].sub(r"\2", result)
    result = RAW_PATTERNS["list_item"].sub(r"\1", result)
    result = RAW_PATTERNS["ordered_list"].sub(r"\1", result)
    result = RAW_PATTERNS["blockquote"].sub(r"\1", result)
    result = RAW_PATTERNS["horizontal_rule"].sub("", result)
    result = RAW_PATTERNS["think_block"].sub("", result)
    return result.strip()


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: str = "gpt2-medium"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    stream: bool = True
    stop_sequences: list[str] = field(default_factory=lambda: ["\n\n\n", "<|endoftext|>"])
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class ChatSessionConfig:
    """Configuration for a chat session."""
    system_prompt: str = "You are a helpful AI assistant."
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    max_history: int = 100
    auto_save: bool = True
    save_interval: float = 30.0
    context_window: int = 4096
    truncate_strategy: str = "sliding_window"


class ChatSession:
    """Manages a multi-turn chat conversation with context management."""

    def __init__(self, config: ChatSessionConfig | None = None) -> None:
        self.config = config or ChatSessionConfig()
        self.messages: list[ChatMessage] = []
        self.history = ChatHistory()
        self._lock = threading.Lock()
        self._token_count = 0
        self._last_save = time.time()
        self._set_system_prompt(self.config.system_prompt)

    def _set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt."""
        with self._lock:
            if self.messages and self.messages[0].role == MessageRole.SYSTEM:
                self.messages[0] = ChatMessage(role=MessageRole.SYSTEM, content=prompt)
            else:
                self.messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=prompt))

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        with self._lock:
            for msg in self.messages:
                if msg.role == MessageRole.SYSTEM:
                    return msg.content
        return ""

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set the system prompt."""
        self._set_system_prompt(value)

    def add_message(self, role: MessageRole, content: str, metadata: dict[str, Any] | None = None) -> ChatMessage:
        """Add a message to the conversation.

        Args:
            role: The role of the message author.
            content: The message content.
            metadata: Optional metadata dictionary.

        Returns:
            The created ChatMessage instance.
        """
        with self._lock:
            msg = ChatMessage(
                role=role,
                content=content,
                metadata=metadata or {},
            )
            self.messages.append(msg)
            entry = HistoryEntry(
                role=role.value,
                content=content,
                timestamp=msg.timestamp,
                metadata=msg.metadata,
            )
            self.history.add_entry(entry)
            self._enforce_max_history()
            if self.config.auto_save and (time.time() - self._last_save) > self.config.save_interval:
                self._auto_save()
            return msg

    def _enforce_max_history(self) -> None:
        """Enforce the maximum number of non-system messages."""
        system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        other_msgs = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        if len(other_msgs) > self.config.max_history:
            excess = len(other_msgs) - self.config.max_history
            other_msgs = other_msgs[excess:]
        self.messages = system_msgs + other_msgs

    def get_context(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """Get the conversation context formatted for model input.

        Args:
            max_tokens: Optional token limit for context windowing.

        Returns:
            List of message dictionaries suitable for the model.
        """
        with self._lock:
            max_tokens = max_tokens or self.config.context_window
            context: list[dict[str, str]] = []
            estimated_tokens = 0
            # Always include system prompt
            for msg in reversed(self.messages):
                msg_tokens = len(msg.content.split()) * 2  # rough estimate
                if estimated_tokens + msg_tokens > max_tokens and context:
                    if self.config.truncate_strategy == "sliding_window":
                        break
                    elif self.config.truncate_strategy == "summarize":
                        # Placeholder for summarization strategy
                        break
                context.insert(0, {"role": msg.role.value, "content": msg.content})
                estimated_tokens += msg_tokens
            return context

    def clear(self) -> None:
        """Clear all messages except the system prompt."""
        with self._lock:
            system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            self.messages = system_msgs

    def undo(self) -> ChatMessage | None:
        """Remove the last user-assistant exchange.

        Returns:
            The removed assistant message, or None if nothing to undo.
        """
        with self._lock:
            if len(self.messages) < 2:
                return None
            # Remove last assistant message
            removed = None
            if self.messages[-1].role == MessageRole.ASSISTANT:
                removed = self.messages.pop()
            # Remove last user message
            if self.messages and self.messages[-1].role == MessageRole.USER:
                self.messages.pop()
            return removed

    def _auto_save(self) -> None:
        """Auto-save the conversation history."""
        try:
            self._last_save = time.time()
        except OSError:
            pass

    def export_messages(self, format: str = "json") -> str:
        """Export all messages in the specified format.

        Args:
            format: Export format - 'json', 'markdown', or 'text'.

        Returns:
            The exported conversation as a string.
        """
        import json

        with self._lock:
            if format == "json":
                return json.dumps([m.to_dict() for m in self.messages], indent=2, ensure_ascii=False)
            elif format == "markdown":
                lines: list[str] = []
                for msg in self.messages:
                    if msg.role == MessageRole.SYSTEM:
                        lines.append(f"**System**: {msg.content}\n")
                    elif msg.role == MessageRole.USER:
                        lines.append(f"**User**: {msg.content}\n")
                    elif msg.role == MessageRole.ASSISTANT:
                        lines.append(f"**Assistant**: {msg.content}\n")
                return "\n".join(lines)
            else:
                return "\n\n".join(f"[{m.role.value}] {m.content}" for m in self.messages)


class ChatLoop:
    """Main interactive chat loop with streaming, commands, and rich output.

    This is the primary entry point for the terminal-based chat experience.
    It handles user input, command dispatching, model invocation with
    streaming, and rich output formatting.
    """

    def __init__(
        self,
        session: ChatSession | None = None,
        model_client: Any = None,
        formatter: RichFormatter | None = None,
        status_bar: StatusBar | None = None,
    ) -> None:
        self.session = session or ChatSession()
        self.model_client = model_client
        self.formatter = formatter or RichFormatter()
        self.status_bar = status_bar or StatusBar()
        self.prompt_session = PromptSession()
        self._running = False
        self._interrupted = False
        self._command_handlers: dict[str, Callable[..., Any]] = {}
        self._stream_lock = threading.Lock()
        self._on_response_start: list[Callable[[], None]] = []
        self._on_response_chunk: list[Callable[[str], None]] = []
        self._on_response_end: list[Callable[[str], None]] = []
        self._on_error: list[Callable[[Exception], None]] = []

    def on_response_start(self, callback: Callable[[], None]) -> None:
        """Register a callback for when a response starts streaming."""
        self._on_response_start.append(callback)

    def on_response_chunk(self, callback: Callable[[str], None]) -> None:
        """Register a callback for each streamed chunk."""
        self._on_response_chunk.append(callback)

    def on_response_end(self, callback: Callable[[str], None]) -> None:
        """Register a callback for when a response finishes."""
        self._on_response_end.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback for when an error occurs."""
        self._on_error.append(callback)

    def register_command(self, name: str, handler: Callable[..., Any]) -> None:
        """Register a custom command handler.

        Args:
            name: The command name (without the leading /).
            handler: The callable to invoke when the command is entered.
        """
        self._command_handlers[name] = handler

    def _dispatch_command(self, line: str) -> bool:
        """Try to dispatch a slash command.

        Args:
            line: The raw input line.

        Returns:
            True if the line was a command and was handled.
        """
        if not line.startswith("/"):
            return False
        parts = line[1:].split(maxsplit=1)
        if not parts:
            return False
        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""
        handler = self._command_handlers.get(cmd_name)
        if handler:
            handler(cmd_args)
            return True
        self.formatter.print_error(f"Unknown command: /{cmd_name}. Type /help for available commands.")
        return True

    def _stream_response(self, prompt: str) -> str:
        """Stream a model response, yielding chunks in real time.

        If no model client is configured, returns a placeholder response.

        Args:
            prompt: The user prompt to send to the model.

        Returns:
            The full response text.
        """
        if self.model_client is None:
            placeholder = (
                f"[No model client configured] Echo: {prompt}\n\n"
                "Configure a model client to enable real responses. "
                "Use /model to set the model path."
            )
            self.formatter.print_assistant(placeholder)
            return placeholder

        for cb in self._on_response_start:
            cb()

        context = self.session.get_context()
        full_response = ""

        try:
            if self.session.config.generation.stream:
                with Spinner("Thinking", style=SpinnerStyle.DOTS):
                    chunk_iter = self._call_model_stream(context)
                    for chunk in chunk_iter:
                        if self._interrupted:
                            break
                        full_response += chunk
                        self.formatter.print_stream_chunk(chunk)
                        for cb in self._on_response_chunk:
                            cb(chunk)
            else:
                with Spinner("Generating", style=SpinnerStyle.LINE):
                    full_response = self._call_model(context)
                self.formatter.print_assistant(full_response)

        except Exception as exc:
            for cb in self._on_error:
                cb(exc)
            self.formatter.print_error(f"Generation error: {exc}")
            return ""

        if full_response:
            self.session.add_message(MessageRole.ASSISTANT, full_response)

        for cb in self._on_response_end:
            cb(full_response)

        return full_response

    def _call_model(self, context: list[dict[str, str]]) -> str:
        """Make a non-streaming call to the model.

        Args:
            context: The conversation context.

        Returns:
            The model's response text.
        """
        gen_config = self.session.config.generation
        try:
            response = self.model_client.generate(
                context=context,
                max_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                stop=gen_config.stop_sequences,
                repetition_penalty=gen_config.repetition_penalty,
            )
            return response
        except AttributeError:
            return "[Model client does not support generate()]"

    def _call_model_stream(self, context: list[dict[str, str]]) -> Iterator[str]:
        """Make a streaming call to the model.

        Args:
            context: The conversation context.

        Yields:
            Individual text chunks from the model.
        """
        gen_config = self.session.config.generation
        try:
            for chunk in self.model_client.generate_stream(
                context=context,
                max_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                stop=gen_config.stop_sequences,
            ):
                yield chunk
        except AttributeError:
            yield "[Model client does not support streaming]"

    def _handle_input(self, user_input: str) -> None:
        """Process a single line of user input.

        Args:
            user_input: The raw text entered by the user.
        """
        stripped = user_input.strip()
        if not stripped:
            return

        if self._dispatch_command(stripped):
            return

        self.session.add_message(MessageRole.USER, stripped)

        start_time = time.time()
        response = self._stream_response(stripped)
        latency = time.time() - start_time

        self.status_bar.update_field("latency", f"{latency:.2f}s")
        if response:
            token_count = len(response.split())
            self.status_bar.update_field("tokens", str(token_count))

    def run(self) -> None:
        """Start the interactive chat loop.

        This method blocks until the user exits (via /quit or Ctrl+C).
        It displays a welcome banner, initializes the prompt session,
        and enters the main input-processing loop.
        """
        self._running = True
        self._print_welcome()

        self.status_bar.add_field(StatusField(name="model", value=self.session.config.generation.model))
        self.status_bar.add_field(StatusField(name="tokens", value="0"))
        self.status_bar.add_field(StatusField(name="latency", value="-"))
        self.status_bar.add_field(StatusField(name="mode", value="chat"))
        self.status_bar.render()

        while self._running:
            try:
                user_input = self.prompt_session.get_input(
                    prompt="You> ",
                    multiline=False,
                )
                self._handle_input(user_input)
                self._interrupted = False
            except KeyboardInterrupt:
                if self._stream_lock.locked():
                    self._interrupted = True
                    self.formatter.print_warning("\nGeneration interrupted.")
                else:
                    self.formatter.print_warning("\nPress Ctrl+C again or type /quit to exit.")
                    try:
                        second_input = self.prompt_session.get_input(
                            prompt="Exit? (y/N): ",
                            multiline=False,
                        )
                        if second_input.strip().lower() in ("y", "yes"):
                            self.stop()
                    except (KeyboardInterrupt, EOFError):
                        self.stop()
            except EOFError:
                self.stop()

    def stop(self) -> None:
        """Stop the chat loop gracefully."""
        self._running = False
        self.formatter.print_info("Goodbye!")

    def _print_welcome(self) -> None:
        """Print the welcome banner."""
        banner = (
            "[bold cyan]Nexus-LLM[/bold cyan] [dim]v0.1.0[/dim]\n"
            "Interactive Chat Terminal\n"
            "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit."
        )
        self.formatter.print_panel(banner, title="Welcome", style="cyan")
