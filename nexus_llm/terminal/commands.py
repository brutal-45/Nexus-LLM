"""
Nexus-LLM Slash Commands Module

Provides 25+ slash commands for the interactive chat terminal,
including model management, configuration, session control,
history, and utility commands.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from nexus_llm.terminal.chat import ChatSession, ChatSessionConfig, GenerationConfig, MessageRole
from nexus_llm.terminal.formatter import RichFormatter


@dataclass
class CommandContext:
    """Context passed to command handlers providing access to session state."""
    session: ChatSession
    formatter: RichFormatter
    args: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    command_history: list[str] = field(default_factory=list)
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)

    def push_undo(self, state: dict[str, Any]) -> None:
        """Push state onto the undo stack and clear redo stack."""
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def pop_undo(self) -> dict[str, Any] | None:
        """Pop state from the undo stack."""
        return self.undo_stack.pop() if self.undo_stack else None

    def push_redo(self, state: dict[str, Any]) -> None:
        """Push state onto the redo stack."""
        self.redo_stack.append(state)

    def pop_redo(self) -> dict[str, Any] | None:
        """Pop state from the redo stack."""
        return self.redo_stack.pop() if self.redo_stack else None


class CommandRegistry:
    """Registry and dispatcher for slash commands.

    Each command is registered with a name, handler function, description,
    and optional aliases. The registry also provides help text generation
    and argument parsing utilities.
    """

    def __init__(self) -> None:
        self._commands: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: Callable[[CommandContext], None],
        description: str = "",
        aliases: list[str] | None = None,
        usage: str = "",
    ) -> None:
        """Register a command.

        Args:
            name: Primary command name (without leading /).
            handler: Function that takes a CommandContext.
            description: Short description for help text.
            aliases: Alternative names for the command.
            usage: Usage string for help text.
        """
        self._commands[name] = {
            "handler": handler,
            "description": description,
            "usage": usage or f"/{name}",
            "name": name,
        }
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def dispatch(self, name: str, ctx: CommandContext) -> bool:
        """Dispatch a command by name.

        Args:
            name: Command name (without leading /).
            ctx: The command context.

        Returns:
            True if the command was found and executed.
        """
        resolved = self._aliases.get(name, name)
        cmd = self._commands.get(resolved)
        if cmd:
            cmd["handler"](ctx)
            return True
        return False

    def get_commands(self) -> dict[str, dict[str, Any]]:
        """Get all registered commands."""
        return dict(self._commands)

    def get_help_text(self) -> str:
        """Generate formatted help text for all commands."""
        lines = ["Available Commands:", ""]
        max_name_len = max(len(info["name"]) for info in self._commands.values()) if self._commands else 0
        for name in sorted(self._commands):
            info = self._commands[name]
            alias_str = ""
            aliases_for = [a for a, target in self._aliases.items() if target == name]
            if aliases_for:
                alias_str = f" ({', '.join('/' + a for a in aliases_for)})"
            padded = name.ljust(max_name_len)
            lines.append(f"  /{padded}{alias_str}  -  {info['description']}")
            if info["usage"] and info["usage"] != f"/{name}":
                lines.append(f"    Usage: {info['usage']}")
        return "\n".join(lines)


def _cmd_help(ctx: CommandContext) -> None:
    """Display help information for available commands."""
    registry = ctx.config.get("_registry")
    if registry and isinstance(registry, CommandRegistry):
        ctx.formatter.print_panel(registry.get_help_text(), title="Help", style="cyan")
    else:
        ctx.formatter.print_info("Type /help to see available commands.")


def _cmd_model(ctx: CommandContext) -> None:
    """Show or set the current model."""
    args = ctx.args.strip()
    gen_config = ctx.session.config.generation
    if not args:
        ctx.formatter.print_key_value({
            "Current Model": gen_config.model,
            "Temperature": gen_config.temperature,
            "Top-p": gen_config.top_p,
            "Top-k": gen_config.top_k,
            "Max Tokens": gen_config.max_tokens,
            "Stream": gen_config.stream,
        })
    else:
        old_model = gen_config.model
        ctx.push_undo({"action": "model", "old": old_model})
        gen_config.model = args
        ctx.formatter.print_success(f"Model changed: {old_model} → {args}")


def _cmd_switch(ctx: CommandContext) -> None:
    """Switch to a different conversation or model profile."""
    args = ctx.args.strip()
    if not args:
        ctx.formatter.print_info("Usage: /switch <profile_name>")
        return
    gen_config = ctx.session.config.generation
    old_model = gen_config.model
    ctx.push_undo({"action": "switch", "old": old_model})
    gen_config.model = args
    ctx.formatter.print_success(f"Switched to: {args}")


def _cmd_config(ctx: CommandContext) -> None:
    """Show or modify configuration settings."""
    args = ctx.args.strip()
    if not args:
        config_data = {
            "Model": ctx.session.config.generation.model,
            "Temperature": ctx.session.config.generation.temperature,
            "Top-p": ctx.session.config.generation.top_p,
            "Top-k": ctx.session.config.generation.top_k,
            "Max Tokens": ctx.session.config.generation.max_tokens,
            "Stream": ctx.session.config.generation.stream,
            "System Prompt": ctx.session.config.system_prompt[:60] + "..."
            if len(ctx.session.config.system_prompt) > 60
            else ctx.session.config.system_prompt,
            "Max History": ctx.session.config.max_history,
            "Context Window": ctx.session.config.context_window,
        }
        ctx.formatter.print_table(
            headers=["Setting", "Value"],
            rows=[[k, str(v)] for k, v in config_data.items()],
            title="Configuration",
        )
        return

    parts = args.split("=", 1)
    if len(parts) != 2:
        ctx.formatter.print_info("Usage: /config <key>=<value>")
        return

    key, value = parts[0].strip(), parts[1].strip()
    gen_config = ctx.session.config.generation
    config_map: dict[str, tuple[Any, Callable[[str], Any]]]] = {
        "temperature": (gen_config, lambda v: float(v)),
        "top_p": (gen_config, lambda v: float(v)),
        "top_k": (gen_config, lambda v: int(v)),
        "max_tokens": (gen_config, lambda v: int(v)),
        "repetition_penalty": (gen_config, lambda v: float(v)),
        "frequency_penalty": (gen_config, lambda v: float(v)),
        "presence_penalty": (gen_config, lambda v: float(v)),
        "stream": (gen_config, lambda v: v.lower() in ("true", "1", "yes")),
        "max_history": (ctx.session.config, lambda v: int(v)),
        "context_window": (ctx.session.config, lambda v: int(v)),
    }

    if key not in config_map:
        ctx.formatter.print_error(f"Unknown config key: {key}")
        valid = ", ".join(sorted(config_map.keys()))
        ctx.formatter.print_info(f"Valid keys: {valid}")
        return

    obj, parser = config_map[key]
    try:
        old_val = getattr(obj, key)
        new_val = parser(value)
        ctx.push_undo({"action": "config", "key": key, "old": old_val})
        setattr(obj, key, new_val)
        ctx.formatter.print_success(f"{key}: {old_val} → {new_val}")
    except (ValueError, TypeError) as exc:
        ctx.formatter.print_error(f"Invalid value for {key}: {exc}")


def _cmd_save(ctx: CommandContext) -> None:
    """Save the current conversation to a file."""
    args = ctx.args.strip()
    filename = args or f"chat_{int(time.time())}.json"
    if not filename.endswith(".json"):
        filename += ".json"
    try:
        data = ctx.session.export_messages(format="json")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)
        ctx.formatter.print_success(f"Conversation saved to {filename}")
    except OSError as exc:
        ctx.formatter.print_error(f"Failed to save: {exc}")


def _cmd_load(ctx: CommandContext) -> None:
    """Load a conversation from a file."""
    args = ctx.args.strip()
    if not args:
        ctx.formatter.print_info("Usage: /load <filename>")
        return
    if not os.path.exists(args):
        ctx.formatter.print_error(f"File not found: {args}")
        return
    try:
        with open(args, "r", encoding="utf-8") as f:
            data = json.load(f)
        ctx.session.clear()
        from nexus_llm.terminal.chat import ChatMessage, MessageRole
        for msg_data in data:
            msg = ChatMessage.from_dict(msg_data)
            if msg.role != MessageRole.SYSTEM:
                ctx.session.messages.append(msg)
        ctx.formatter.print_success(f"Loaded {len(data)} messages from {args}")
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        ctx.formatter.print_error(f"Failed to load: {exc}")


def _cmd_clear(ctx: CommandContext) -> None:
    """Clear the conversation history."""
    msg_count = len(ctx.session.messages)
    ctx.push_undo({"action": "clear", "messages": list(ctx.session.messages)})
    ctx.session.clear()
    ctx.formatter.print_success(f"Cleared {msg_count} messages")


def _cmd_history(ctx: CommandContext) -> None:
    """Display conversation history."""
    args = ctx.args.strip()
    limit = 20
    if args and args.isdigit():
        limit = int(args)
    messages = ctx.session.messages
    display = messages[-limit:] if len(messages) > limit else messages
    if not display:
        ctx.formatter.print_info("No conversation history.")
        return
    rows = []
    for msg in display:
        role_label = msg.role.value.capitalize()
        content_preview = msg.content[:80].replace("\n", " ")
        if len(msg.content) > 80:
            content_preview += "..."
        rows.append([role_label, content_preview, f"{msg.timestamp:.0f}"])
    ctx.formatter.print_table(
        headers=["Role", "Content", "Timestamp"],
        rows=rows,
        title=f"Conversation History (last {limit})",
    )


def _cmd_export(ctx: CommandContext) -> None:
    """Export the conversation in various formats."""
    args = ctx.args.strip()
    parts = args.split(maxsplit=1)
    if not parts:
        ctx.formatter.print_info("Usage: /export <format> [filename]\nFormats: json, markdown, text")
        return
    fmt = parts[0].lower()
    filename = parts[1] if len(parts) > 1 else f"export_{int(time.time())}"
    ext_map = {"json": ".json", "markdown": ".md", "text": ".txt"}
    filename += ext_map.get(fmt, ".txt")
    try:
        data = ctx.session.export_messages(format=fmt)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)
        ctx.formatter.print_success(f"Exported as {fmt} to {filename}")
    except OSError as exc:
        ctx.formatter.print_error(f"Export failed: {exc}")


def _cmd_import(ctx: CommandContext) -> None:
    """Import a conversation from a file."""
    args = ctx.args.strip()
    if not args:
        ctx.formatter.print_info("Usage: /import <filename>")
        return
    if not os.path.exists(args):
        ctx.formatter.print_error(f"File not found: {args}")
        return
    _cmd_load(ctx)


def _cmd_system(ctx: CommandContext) -> None:
    """Show or set the system prompt."""
    args = ctx.args.strip()
    if not args:
        ctx.formatter.print_panel(
            ctx.session.system_prompt or "(no system prompt set)",
            title="System Prompt",
            style="yellow",
        )
    else:
        old_prompt = ctx.session.system_prompt
        ctx.push_undo({"action": "system", "old": old_prompt})
        ctx.session.system_prompt = args
        ctx.formatter.print_success("System prompt updated.")


def _cmd_temperature(ctx: CommandContext) -> None:
    """Show or set the generation temperature."""
    args = ctx.args.strip()
    gen_config = ctx.session.config.generation
    if not args:
        ctx.formatter.print_info(f"Temperature: {gen_config.temperature}")
        return
    try:
        new_temp = float(args)
        if not 0.0 <= new_temp <= 2.0:
            ctx.formatter.print_error("Temperature must be between 0.0 and 2.0")
            return
        old_temp = gen_config.temperature
        ctx.push_undo({"action": "temperature", "old": old_temp})
        gen_config.temperature = new_temp
        ctx.formatter.print_success(f"Temperature: {old_temp} → {new_temp}")
    except ValueError:
        ctx.formatter.print_error("Invalid temperature value. Must be a number.")


def _cmd_topp(ctx: CommandContext) -> None:
    """Show or set the top-p (nucleus sampling) parameter."""
    args = ctx.args.strip()
    gen_config = ctx.session.config.generation
    if not args:
        ctx.formatter.print_info(f"Top-p: {gen_config.top_p}")
        return
    try:
        new_val = float(args)
        if not 0.0 <= new_val <= 1.0:
            ctx.formatter.print_error("Top-p must be between 0.0 and 1.0")
            return
        old_val = gen_config.top_p
        ctx.push_undo({"action": "topp", "old": old_val})
        gen_config.top_p = new_val
        ctx.formatter.print_success(f"Top-p: {old_val} → {new_val}")
    except ValueError:
        ctx.formatter.print_error("Invalid top-p value. Must be a number.")


def _cmd_topk(ctx: CommandContext) -> None:
    """Show or set the top-k parameter."""
    args = ctx.args.strip()
    gen_config = ctx.session.config.generation
    if not args:
        ctx.formatter.print_info(f"Top-k: {gen_config.top_k}")
        return
    try:
        new_val = int(args)
        if new_val < 1:
            ctx.formatter.print_error("Top-k must be at least 1")
            return
        old_val = gen_config.top_k
        ctx.push_undo({"action": "topk", "old": old_val})
        gen_config.top_k = new_val
        ctx.formatter.print_success(f"Top-k: {old_val} → {new_val}")
    except ValueError:
        ctx.formatter.print_error("Invalid top-k value. Must be an integer.")


def _cmd_maxtokens(ctx: CommandContext) -> None:
    """Show or set the maximum token count for generation."""
    args = ctx.args.strip()
    gen_config = ctx.session.config.generation
    if not args:
        ctx.formatter.print_info(f"Max tokens: {gen_config.max_tokens}")
        return
    try:
        new_val = int(args)
        if new_val < 1:
            ctx.formatter.print_error("Max tokens must be at least 1")
            return
        old_val = gen_config.max_tokens
        ctx.push_undo({"action": "maxtokens", "old": old_val})
        gen_config.max_tokens = new_val
        ctx.formatter.print_success(f"Max tokens: {old_val} → {new_val}")
    except ValueError:
        ctx.formatter.print_error("Invalid max tokens value. Must be an integer.")


def _cmd_stream(ctx: CommandContext) -> None:
    """Toggle streaming mode on/off."""
    gen_config = ctx.session.config.generation
    old_val = gen_config.stream
    ctx.push_undo({"action": "stream", "old": old_val})
    gen_config.stream = not old_val
    state = "ON" if gen_config.stream else "OFF"
    ctx.formatter.print_success(f"Streaming: {state}")


def _cmd_bench(ctx: CommandContext) -> None:
    """Run a simple generation benchmark."""
    import time as _time

    gen_config = ctx.session.config.generation
    ctx.formatter.print_info(f"Benchmarking model: {gen_config.model}")

    # Simulated benchmark since we may not have a real model
    prompts = [
        "The quick brown fox",
        "In a world where",
        "def fibonacci(n):",
    ]
    results = []
    for prompt in prompts:
        start = _time.time()
        # In real usage, this would call model_client.generate()
        elapsed = _time.time() - start
        results.append([prompt, f"{elapsed * 1000:.1f}ms", "N/A"])

    ctx.formatter.print_table(
        headers=["Prompt", "Latency", "Tokens/s"],
        rows=results,
        title="Benchmark Results",
    )
    ctx.formatter.print_warning("Note: Benchmark requires an active model client for real measurements.")


def _cmd_info(ctx: CommandContext) -> None:
    """Display system and session information."""
    gen_config = ctx.session.config.generation
    info_data = {
        "Model": gen_config.model,
        "Temperature": gen_config.temperature,
        "Top-p": gen_config.top_p,
        "Top-k": gen_config.top_k,
        "Max Tokens": gen_config.max_tokens,
        "Streaming": gen_config.stream,
        "Messages": str(len(ctx.session.messages)),
        "System Prompt Length": str(len(ctx.session.system_prompt)),
        "Python": sys.version.split()[0],
        "Platform": sys.platform,
        "Max History": ctx.session.config.max_history,
        "Context Window": ctx.session.config.context_window,
    }
    ctx.formatter.print_table(
        headers=["Property", "Value"],
        rows=[[k, v] for k, v in info_data.items()],
        title="Session Info",
    )


def _cmd_quit(ctx: CommandContext) -> None:
    """Exit the chat session."""
    ctx.formatter.print_info("Goodbye! Use Ctrl+D or /quit to exit.")
    ctx.config["should_quit"] = True


def _cmd_reset(ctx: CommandContext) -> None:
    """Reset the session to default settings."""
    ctx.push_undo({"action": "reset", "config": ctx.session.config})
    ctx.session.config = ChatSessionConfig()
    ctx.session.clear()
    ctx.formatter.print_success("Session reset to defaults.")


def _cmd_copy(ctx: CommandContext) -> None:
    """Copy the last assistant response to clipboard."""
    last_assistant = None
    for msg in reversed(ctx.session.messages):
        if msg.role == MessageRole.ASSISTANT:
            last_assistant = msg.content
            break
    if not last_assistant:
        ctx.formatter.print_warning("No assistant response to copy.")
        return
    try:
        import subprocess
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            text=True,
        )
        process.communicate(input=last_assistant)
        if process.returncode == 0:
            ctx.formatter.print_success("Copied to clipboard (xclip).")
            return
    except (ImportError, FileNotFoundError, OSError):
        pass
    try:
        import subprocess
        process = subprocess.Popen(
            ["pbcopy"],
            stdin=subprocess.PIPE,
            text=True,
        )
        process.communicate(input=last_assistant)
        if process.returncode == 0:
            ctx.formatter.print_success("Copied to clipboard (pbcopy).")
            return
    except (ImportError, FileNotFoundError, OSError):
        pass
    # Fallback: write to temp file
    try:
        tmp_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nexus_llm_clipboard.txt")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(last_assistant)
        ctx.formatter.print_info(f"Clipboard unavailable. Saved to {tmp_path}")
    except OSError as exc:
        ctx.formatter.print_error(f"Failed to copy: {exc}")


def _cmd_edit(ctx: CommandContext) -> None:
    """Edit the last user message and regenerate."""
    last_user_idx = None
    for i in range(len(ctx.session.messages) - 1, -1, -1):
        if ctx.session.messages[i].role == MessageRole.USER:
            last_user_idx = i
            break
    if last_user_idx is None:
        ctx.formatter.print_warning("No user message to edit.")
        return
    old_content = ctx.session.messages[last_user_idx].content
    new_content = ctx.args.strip() if ctx.args.strip() else old_content
    if not new_content:
        ctx.formatter.print_info("Usage: /edit <new message>")
        return
    ctx.push_undo({"action": "edit", "index": last_user_idx, "old": old_content})
    ctx.session.messages[last_user_idx] = type(ctx.session.messages[last_user_idx])(
        role=MessageRole.USER,
        content=new_content,
    )
    ctx.formatter.print_success(f"Edited message: {old_content[:50]}... → {new_content[:50]}...")


def _cmd_undo(ctx: CommandContext) -> None:
    """Undo the last action."""
    state = ctx.pop_undo()
    if not state:
        ctx.formatter.print_warning("Nothing to undo.")
        return
    action = state.get("action")
    ctx.push_redo(state)
    if action == "model":
        ctx.session.config.generation.model = state["old"]
        ctx.formatter.print_success(f"Undone: model → {state['old']}")
    elif action == "config":
        key = state["key"]
        setattr(ctx.session.config.generation, key, state["old"])
        ctx.formatter.print_success(f"Undone: {key} → {state['old']}")
    elif action == "system":
        ctx.session.system_prompt = state["old"]
        ctx.formatter.print_success("Undone: system prompt restored.")
    elif action == "temperature":
        ctx.session.config.generation.temperature = state["old"]
        ctx.formatter.print_success(f"Undone: temperature → {state['old']}")
    elif action == "clear":
        ctx.session.messages = state["messages"]
        ctx.formatter.print_success("Undone: messages restored.")
    else:
        ctx.formatter.print_info(f"Undo for '{action}' not fully implemented.")


def _cmd_redo(ctx: CommandContext) -> None:
    """Redo the last undone action."""
    state = ctx.pop_redo()
    if not state:
        ctx.formatter.print_warning("Nothing to redo.")
        return
    ctx.push_undo(state)
    action = state.get("action")
    if action == "model":
        ctx.session.config.generation.model = state.get("new", ctx.session.config.generation.model)
    elif action == "temperature":
        ctx.session.config.generation.temperature = state.get("new", ctx.session.config.generation.temperature)
    ctx.formatter.print_success(f"Redone: {action}")


def _cmd_search(ctx: CommandContext) -> None:
    """Search through conversation history."""
    args = ctx.args.strip()
    if not args:
        ctx.formatter.print_info("Usage: /search <query>")
        return
    query = args.lower()
    results = []
    for i, msg in enumerate(ctx.session.messages):
        if query in msg.content.lower():
            preview = msg.content[:100].replace("\n", " ")
            if len(msg.content) > 100:
                preview += "..."
            results.append([str(i), msg.role.value.capitalize(), preview])
    if not results:
        ctx.formatter.print_info(f"No results for '{args}'.")
    else:
        ctx.formatter.print_table(
            headers=["#", "Role", "Content"],
            rows=results,
            title=f"Search Results: '{args}'",
        )


def create_default_registry() -> CommandRegistry:
    """Create and populate the default command registry with all built-in commands.

    Returns:
        A fully populated CommandRegistry instance.
    """
    registry = CommandRegistry()

    registry.register("help", _cmd_help, "Show available commands", aliases=["h", "?"], usage="/help")
    registry.register("model", _cmd_model, "Show or set model", usage="/model [name]")
    registry.register("switch", _cmd_switch, "Switch model profile", usage="/switch <profile>")
    registry.register("config", _cmd_config, "Show or modify config", usage="/config [key=value]")
    registry.register("save", _cmd_save, "Save conversation", aliases=["s"], usage="/save [filename]")
    registry.register("load", _cmd_load, "Load conversation", usage="/load <filename>")
    registry.register("clear", _cmd_clear, "Clear conversation", aliases=["cls"], usage="/clear")
    registry.register("history", _cmd_history, "Show conversation history", aliases=["hist"], usage="/history [limit]")
    registry.register("export", _cmd_export, "Export conversation", usage="/export <format> [filename]")
    registry.register("import", _cmd_import, "Import conversation", usage="/import <filename>")
    registry.register("system", _cmd_system, "Show/set system prompt", aliases=["sys"], usage="/system [prompt]")
    registry.register("temperature", _cmd_temperature, "Show/set temperature", aliases=["temp"], usage="/temperature [value]")
    registry.register("topp", _cmd_topp, "Show/set top-p", usage="/topp [value]")
    registry.register("topk", _cmd_topk, "Show/set top-k", usage="/topk [value]")
    registry.register("maxtokens", _cmd_maxtokens, "Show/set max tokens", aliases=["mt"], usage="/maxtokens [value]")
    registry.register("stream", _cmd_stream, "Toggle streaming", usage="/stream")
    registry.register("bench", _cmd_bench, "Run benchmark", usage="/bench")
    registry.register("info", _cmd_info, "Show session info", usage="/info")
    registry.register("quit", _cmd_quit, "Exit chat", aliases=["q", "exit"], usage="/quit")
    registry.register("reset", _cmd_reset, "Reset session", usage="/reset")
    registry.register("copy", _cmd_copy, "Copy last response", aliases=["cp"], usage="/copy")
    registry.register("edit", _cmd_edit, "Edit last message", usage="/edit [new message]")
    registry.register("undo", _cmd_undo, "Undo last action", aliases=["u"], usage="/undo")
    registry.register("redo", _cmd_redo, "Redo last undone action", aliases=["r"], usage="/redo")
    registry.register("search", _cmd_search, "Search history", aliases=["find"], usage="/search <query>")

    return registry
