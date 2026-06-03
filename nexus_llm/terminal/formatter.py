"""Output formatter for Nexus-LLM using Rich.

Renders markdown, syntax-highlighted code blocks, streaming text,
generation stats, and model-info tables in the terminal.
"""

import re
import time
from typing import Dict, List, Optional, Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.padding import Padding

from nexus_llm.terminal.themes import Theme, get_theme


# Regex to detect fenced code blocks
_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


class OutputFormatter:
    """Beautiful terminal output using Rich.

    Handles:
    * Markdown rendering (headers, bold, italic, code blocks, lists)
    * Syntax highlighting via Pygments / Rich Syntax
    * Code block detection and formatting
    * Streaming text display
    * Token count and timing display
    * Table formatting for model info
    """

    def __init__(self, console: Optional[Console] = None, theme: Optional[Theme] = None) -> None:
        self.console = console or Console()
        self.theme = theme or get_theme()
        self._stream_buffer: str = ""
        self._stream_start: float = 0.0
        self._stream_tokens: int = 0

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        """Switch to a different theme."""
        self.theme = theme

    # ------------------------------------------------------------------
    # Markdown / text rendering
    # ------------------------------------------------------------------

    def print(self, text: str, style: Optional[str] = None, **kwargs: Any) -> None:
        """Print text with optional Rich style."""
        self.console.print(text, style=style, **kwargs)

    def print_markdown(self, text: str) -> None:
        """Render text as Markdown with syntax highlighting for code blocks."""
        if not text.strip():
            return

        # Split text into code blocks and non-code sections, render separately
        # so that we get proper syntax highlighting for code blocks.
        parts = self._split_code_blocks(text)
        for is_code, content, lang in parts:
            if is_code:
                self._render_code_block(content, lang)
            else:
                if content.strip():
                    md = Markdown(content)
                    self.console.print(md)

    def _split_code_blocks(self, text: str) -> List[tuple]:
        """Split text into code / non-code segments.

        Returns a list of (is_code: bool, content: str, lang: str) tuples.
        """
        parts: List[tuple] = []
        last_end = 0

        for match in _CODE_BLOCK_RE.finditer(text):
            # Non-code before this block
            before = text[last_end:match.start()]
            if before:
                parts.append((False, before, ""))

            lang = match.group(1).strip() or "text"
            code = match.group(2)
            parts.append((True, code, lang))

            last_end = match.end()

        # Trailing non-code
        trailing = text[last_end:]
        if trailing:
            parts.append((False, trailing, ""))

        # If no code blocks were found, return the whole text as non-code
        if not parts:
            parts.append((False, text, ""))

        return parts

    def _render_code_block(self, code: str, language: str) -> None:
        """Render a single code block with syntax highlighting."""
        try:
            syntax = Syntax(
                code.strip(),
                lexer=language,
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
            )
            self.console.print(syntax)
        except Exception:
            # Fall back to a plain panel if syntax highlighting fails
            self.console.print(Panel(code.strip(), border_style=self.theme.code_text))

    # ------------------------------------------------------------------
    # Role-based messages
    # ------------------------------------------------------------------

    def print_user_message(self, content: str) -> None:
        """Print a user message with appropriate styling."""
        styled = Text(content, style=self.theme.user_prompt)
        self.console.print()
        self.console.print(styled)

    def print_assistant_message(self, content: str) -> None:
        """Print an assistant response with markdown rendering."""
        self.console.print()
        self.print_markdown(content)

    def print_system_message(self, content: str) -> None:
        """Print a system message."""
        self.console.print()
        panel = Panel(
            Text(content, style=self.theme.system_text),
            title="[system]",
            border_style=self.theme.system_text,
        )
        self.console.print(panel)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print()
        self.console.print(f"  Error: {message}", style=self.theme.error_text)

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"  {message}", style=self.theme.success_text)

    def print_info(self, message: str) -> None:
        """Print an info / dim message."""
        self.console.print(f"  {message}", style=self.theme.info_text)

    # ------------------------------------------------------------------
    # Streaming display
    # ------------------------------------------------------------------

    def start_stream(self) -> None:
        """Prepare for a new streaming response."""
        self._stream_buffer = ""
        self._stream_start = time.monotonic()
        self._stream_tokens = 0

    def append_stream(self, token: str) -> None:
        """Append a token to the streaming buffer and print it."""
        self._stream_buffer += token
        self._stream_tokens += 1
        self.console.print(token, end="", highlight=False)

    def end_stream(self) -> Dict[str, Any]:
        """Finalize the stream and return timing / token stats.

        Also prints a trailing newline and the stats footer.
        """
        self.console.print()  # trailing newline
        elapsed = time.monotonic() - self._stream_start
        stats = {
            "tokens": self._stream_tokens,
            "elapsed_s": round(elapsed, 3),
            "tokens_per_second": round(self._stream_tokens / elapsed, 2) if elapsed > 0 else 0,
        }
        self.print_stream_stats(stats)
        return stats

    def print_stream_stats(self, stats: Dict[str, Any]) -> None:
        """Print a small stats line after a streaming response."""
        parts = []
        if "tokens" in stats:
            parts.append(f"{stats['tokens']} tokens")
        if "elapsed_s" in stats:
            parts.append(f"{stats['elapsed_s']}s")
        if "tokens_per_second" in stats:
            parts.append(f"{stats['tokens_per_second']} tok/s")
        if parts:
            self.console.print()
            self.console.print("  " + " | ".join(parts), style=self.theme.info_text)

    @property
    def stream_buffer(self) -> str:
        """The accumulated text from the current / last stream."""
        return self._stream_buffer

    # ------------------------------------------------------------------
    # Token count & timing display
    # ------------------------------------------------------------------

    def print_generation_stats(self, result: Dict[str, Any]) -> None:
        """Print generation statistics from an InferenceEngine result."""
        table = Table(title="Generation Stats", show_header=True, border_style=self.theme.border_color)
        table.add_column("Metric", style=self.theme.highlight)
        table.add_column("Value", style=self.theme.assistant_text)

        for key, label in [
            ("prompt_tokens", "Prompt Tokens"),
            ("generated_tokens", "Generated Tokens"),
            ("total_tokens", "Total Tokens"),
            ("generation_time_s", "Generation Time (s)"),
        ]:
            if key in result:
                table.add_row(label, str(result[key]))

        if "generated_tokens" in result and "generation_time_s" in result:
            gen = result["generated_tokens"]
            t = result["generation_time_s"]
            if t > 0:
                table.add_row("Tokens/sec", f"{gen / t:.2f}")

        self.console.print(table)

    # ------------------------------------------------------------------
    # Table formatting
    # ------------------------------------------------------------------

    def print_model_info_table(self, info: Dict[str, Any]) -> None:
        """Print model information as a styled table."""
        table = Table(
            title="Model Information",
            show_header=False,
            border_style=self.theme.border_color,
        )
        table.add_column("Property", style=self.theme.highlight, min_width=20)
        table.add_column("Value", style=self.theme.assistant_text)

        display_map = {
            "model_id": "Model ID",
            "name": "Name",
            "hf_id": "HuggingFace ID",
            "category": "Category",
            "size": "Size",
            "params": "Parameters",
            "model_type": "Type",
            "device": "Device",
            "precision": "Precision",
            "state": "State",
            "recommended": "Recommended",
            "min_ram_gb": "Min RAM (GB)",
            "total_parameters": "Total Parameters",
            "load_time_s": "Load Time (s)",
        }

        for key, label in display_map.items():
            if key in info:
                val = info[key]
                if isinstance(val, bool):
                    val = "Yes" if val else "No"
                elif isinstance(val, float):
                    val = f"{val:.2f}"
                table.add_row(label, str(val))

        # Nested model_info
        nested = info.get("model_info")
        if isinstance(nested, dict):
            for key, label in display_map.items():
                if key in nested and key not in info:
                    val = nested[key]
                    if isinstance(val, bool):
                        val = "Yes" if val else "No"
                    elif isinstance(val, float):
                        val = f"{val:.2f}"
                    table.add_row(label, str(val))

        self.console.print(table)

    def print_models_table(self, models: List[Dict[str, Any]]) -> None:
        """Print a table of available models."""
        table = Table(
            title="Available Models",
            show_header=True,
            border_style=self.theme.border_color,
        )
        table.add_column("ID", style=self.theme.highlight, min_width=18)
        table.add_column("Name", style=self.theme.assistant_text, min_width=22)
        table.add_column("Size", style=self.theme.info_text, min_width=8)
        table.add_column("Category", style=self.theme.info_text, min_width=10)
        table.add_column("Rec.", style=self.theme.success_text, min_width=5)

        for m in models:
            recommended = "★" if m.get("recommended") else ""
            table.add_row(
                m.get("id", ""),
                m.get("name", ""),
                m.get("size", ""),
                m.get("category", ""),
                recommended,
            )

        self.console.print(table)

    def print_config_table(self, config: Dict[str, Any], title: str = "Configuration") -> None:
        """Print a configuration dictionary as a table."""
        table = Table(title=title, show_header=False, border_style=self.theme.border_color)
        table.add_column("Key", style=self.theme.highlight, min_width=25)
        table.add_column("Value", style=self.theme.assistant_text)

        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                for sub_key, sub_val in sorted(value.items()):
                    table.add_row(f"{key}.{sub_key}", str(sub_val))
            else:
                table.add_row(key, str(value))

        self.console.print(table)

    # ------------------------------------------------------------------
    # Banners / separators
    # ------------------------------------------------------------------

    def print_banner(self, version: str = "2.0.0") -> None:
        """Print the Nexus-LLM startup banner."""
        banner = Text()
        banner.append("╔══════════════════════════════════════╗\n", style=self.theme.border_color)
        banner.append("║", style=self.theme.border_color)
        banner.append("   Nexus-LLM", style=self.theme.title_color)
        banner.append(f"  v{version}            ║\n", style=self.theme.info_text)
        banner.append("║", style=self.theme.border_color)
        banner.append("   Terminal LLM Chat Interface        ║\n", style=self.theme.info_text)
        banner.append("╚══════════════════════════════════════╝", style=self.theme.border_color)
        self.console.print(banner)

    def print_separator(self, char: str = "─") -> None:
        """Print a horizontal separator line."""
        self.console.print(Rule(style=self.theme.border_color))

    def print_welcome(self, model_name: Optional[str] = None) -> None:
        """Print welcome message after startup."""
        self.console.print()
        if model_name:
            self.print_success(f"Model loaded: {model_name}")
        self.print_info("Type your message and press Enter to chat.")
        self.print_info("Type /help for a list of commands.")
        self.console.print()

    def print_goodbye(self) -> None:
        """Print a goodbye message on exit."""
        self.console.print()
        self.console.print("  Goodbye from Nexus-LLM!", style=self.theme.title_color)
        self.console.print()
