"""Output Formatter - Rich terminal formatting like Claude AI."""

import re
import logging
from typing import Optional, Dict, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table
from rich.columns import Columns
from rich.theme import Theme

logger = logging.getLogger(__name__)


# Custom theme for Claude-like appearance
CLAUDE_THEME = Theme({
    "user": "bold cyan",
    "assistant": "bold magenta",
    "system": "bold yellow",
    "info": "dim",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "stats": "dim italic",
    "divider": "dim",
    "prompt": "bold green",
})


class OutputFormatter:
    """
    Handles rich terminal output formatting to create a Claude AI-like
    experience with markdown rendering, syntax highlighting, and
    professional styling.
    """

    def __init__(self):
        self.console = Console(theme=CLAUDE_THEME)
        self._thinking_shown = False

    def print_welcome(self) -> None:
        """Print the welcome banner."""
        banner = Text()
        banner.append("╔══════════════════════════════════════════════════╗\n", style="bold cyan")
        banner.append("║                                                  ║\n", style="bold cyan")
        banner.append("║         ", style="bold cyan")
        banner.append("Nexus-LLM", style="bold magenta")
        banner.append("  ─  Your Own AI Backend          ║\n", style="bold cyan")
        banner.append("║                                                  ║\n", style="bold cyan")
        banner.append("║  Type your message and press Enter to chat       ║\n", style="bold cyan")
        banner.append("║  Type ", style="bold cyan")
        banner.append("/help", style="bold yellow")
        banner.append(" for commands  ", style="bold cyan")
        banner.append("/quit", style="bold yellow")
        banner.append(" to exit        ║\n", style="bold cyan")
        banner.append("║                                                  ║\n", style="bold cyan")
        banner.append("╚══════════════════════════════════════════════════╝\n", style="bold cyan")

        self.console.print(banner)

    def print_user_message(self, content: str) -> None:
        """Display the user's message with formatting."""
        self.console.print()
        user_panel = Panel(
            Text(content, style="white"),
            title="[bold cyan]You[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
        self.console.print(user_panel)

    def print_assistant_response(self, content: str) -> None:
        """Display the assistant's response with markdown rendering."""
        self.console.print()

        # Render as markdown for rich formatting
        try:
            md = Markdown(content)
            self.console.print(
                Panel(
                    md,
                    title="[bold magenta]Assistant[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                )
            )
        except Exception:
            # Fallback to plain text
            self.console.print(
                Panel(
                    content,
                    title="[bold magenta]Assistant[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                )
            )

    def print_streaming_token(self, token: str) -> None:
        """Print a single streaming token (no newlines between tokens)."""
        self.console.print(token, end="", highlight=False)

    def start_assistant_response(self) -> None:
        """Start a streaming assistant response."""
        self.console.print()
        self.console.print(
            Panel(
                "",
                title="[bold magenta]Assistant[/bold magenta]",
                border_style="magenta",
                padding=(0, 1),
            ),
            end="",
        )
        self._thinking_shown = True

    def print_thinking(self) -> None:
        """Show a thinking indicator while generating."""
        self.console.print("[dim italic]Thinking...[/dim italic]")

    def print_stats(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        generation_time: float = 0.0,
        tokens_per_second: float = 0.0,
    ) -> None:
        """Display generation statistics."""
        stats_text = (
            f"[stats]Tokens: {input_tokens} in → {output_tokens} out │ "
            f"Time: {generation_time:.2f}s │ "
            f"Speed: {tokens_per_second:.1f} tok/s[/stats]"
        )
        self.console.print(stats_text)

    def print_divider(self) -> None:
        """Print a visual divider between messages."""
        self.console.print("[divider]─" * 50 + "[/divider]")

    def print_error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"[error]Error:[/error] {message}")

    def print_warning(self, message: str) -> None:
        """Display a warning message."""
        self.console.print(f"[warning]Warning:[/warning] {message}")

    def print_success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f"[success]Success:[/success] {message}")

    def print_info(self, message: str) -> None:
        """Display an info message."""
        self.console.print(f"[info]{message}[/info]")

    def print_model_info(self, info: Dict[str, Any]) -> None:
        """Display model information in a formatted table."""
        table = Table(title="Model Information", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        for key, value in info.items():
            key_display = key.replace("_", " ").title()
            table.add_row(key_display, str(value))

        self.console.print(table)

    def print_sessions_list(self, sessions: list) -> None:
        """Display a list of saved sessions."""
        if not sessions:
            self.console.print("[info]No saved sessions found.[/info]")
            return

        table = Table(title="Chat Sessions", show_header=True, header_style="bold cyan")
        table.add_column("Session ID", style="cyan")
        table.add_column("Messages", style="white")
        table.add_column("Last Updated", style="dim")

        for session in sessions:
            sid = session.get("session_id", "unknown")
            count = str(session.get("message_count", 0))
            updated = session.get("updated_at", "N/A")
            if isinstance(updated, (int, float)):
                import time
                updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(updated))
            table.add_row(sid, count, str(updated))

        self.console.print(table)

    def print_help(self) -> None:
        """Display help information with all available commands."""
        help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/quit` or `/exit` | Exit the application |
| `/clear` | Clear the current conversation |
| `/save` | Save the current conversation |
| `/history` | Show saved conversation sessions |
| `/load <session_id>` | Load a previous session |
| `/search <query>` | Search through conversation history |
| `/export [json|md]` | Export current session |
| `/model` | Show current model information |
| `/models` | List available models to switch |
| `/switch <model>` | Switch to a different model |
| `/config` | Show current configuration |
| `/set <key> <value>` | Update a configuration setting |
| `/stats` | Show inference statistics |
| `/stop` | Stop current generation |
| `/system <prompt>` | Set the system prompt |
| `/copy` | Copy last response to clipboard |
| `/tokenize <text>` | Count tokens in text |
| `/reset` | Reset everything to defaults |
"""
        self.console.print(Markdown(help_text))

    def print_code_block(self, code: str, language: str = "python") -> None:
        """Display a code block with syntax highlighting."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)
