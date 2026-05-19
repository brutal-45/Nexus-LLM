"""
Nexus-LLM Multi-line Input Module

Provides advanced multi-line input handling with bracket matching,
auto-indentation, and continuation line support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.validation import Validator, ValidationError

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


# Bracket pairs for matching
BRACKET_PAIRS: dict[str, str] = {
    "(": ")",
    "[": "]",
    "{": "}",
    '"': '"',
    "'": "'",
    "`": "`",
}

OPEN_BRACKETS = set(k for k in BRACKET_PAIRS if k in "([{")
CLOSE_BRACKETS = set(k for k in BRACKET_PAIRS if k in ")]}")

# Patterns that indicate continuation
CONTINUATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r":\s*$"),                # Ends with colon (Python blocks)
    re.compile(r"\\\s*$"),               # Line continuation
    re.compile(r",\s*$"),                # Ends with comma
    re.compile(r"\(\s*$"),               # Open paren at end
    re.compile(r"\[\s*$"),              # Open bracket at end
    re.compile(r"\{\s*$"),              # Open brace at end
    re.compile(r"(?:if|else|elif|for|while|with|try|except|finally|def|class)\s*.*:\s*$"),  # Python blocks
    re.compile(r"\|\s*$"),              # Pipe at end (shell)
    re.compile(r"&&\s*$"),              # && at end (shell)
    re.compile(r"\|\|\s*$"),            # || at end (shell)
    re.compile(r"->\s*$"),              # Arrow at end
]

# Patterns that indicate the line is complete (submit on Enter)
COMPLETE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*$"),               # Empty or whitespace-only
    re.compile(r"^\s*#\s*"),            # Comment
    re.compile(r"^\s*(?:exit|quit|help)\s*$", re.IGNORECASE),  # Commands
]


@dataclass
class BracketMatch:
    """Result of bracket matching analysis."""
    is_balanced: bool
    unmatched_open: list[tuple[str, int]]  # (bracket, line_number)
    unmatched_close: list[tuple[str, int]]
    indent_level: int
    last_open_bracket: str | None = None
    last_open_line: int = 0


class MultilineInput:
    """Advanced multi-line input handler with bracket matching and auto-indent.

    Provides intelligent multi-line editing with:
    - Bracket matching and auto-closing
    - Auto-indentation based on context
    - Continuation detection (knows when more input is expected)
    - Smart Enter (submit vs. newline)
    - Indentation tracking and adjustment
    """

    def __init__(
        self,
        indent_str: str = "    ",
        max_lines: int = 1000,
        smart_enter: bool = True,
        auto_indent: bool = True,
        auto_close_brackets: bool = True,
    ) -> None:
        self._indent_str = indent_str
        self._max_lines = max_lines
        self._smart_enter = smart_enter
        self._auto_indent = auto_indent
        self._auto_close_brackets = auto_close_brackets
        self._lines: list[str] = [""]
        self._cursor_line = 0
        self._cursor_col = 0
        self._history: list[str] = []
        self._continuation_detected = False

    @property
    def text(self) -> str:
        """Get the full input text."""
        return "\n".join(self._lines)

    @text.setter
    def text(self, value: str) -> None:
        """Set the full input text."""
        self._lines = value.split("\n") if value else [""]
        self._cursor_line = len(self._lines) - 1
        self._cursor_col = len(self._lines[-1])

    @property
    def line_count(self) -> int:
        """Get the number of input lines."""
        return len(self._lines)

    @property
    def current_line(self) -> str:
        """Get the current line content."""
        return self._lines[self._cursor_line] if self._cursor_line < len(self._lines) else ""

    def analyze_brackets(self) -> BracketMatch:
        """Analyze bracket balance across all lines.

        Returns:
            A BracketMatch with analysis results.
        """
        stack: list[tuple[str, int]] = []
        unmatched_close: list[tuple[str, int]] = []

        for line_num, line in enumerate(self._lines):
            in_string: str | None = None
            i = 0
            while i < len(line):
                char = line[i]

                # Handle string literals
                if char in ('"', "'", "`") and (i == 0 or line[i - 1] != "\\"):
                    if in_string == char:
                        in_string = None
                    elif in_string is None:
                        in_string = char
                        stack.append((char, line_num))
                    i += 1
                    continue

                if in_string:
                    i += 1
                    continue

                if char in OPEN_BRACKETS:
                    stack.append((char, line_num))
                elif char in CLOSE_BRACKETS:
                    # Find matching open bracket
                    expected_open = None
                    for open_b, close_b in BRACKET_PAIRS.items():
                        if close_b == char and open_b in OPEN_BRACKETS:
                            expected_open = open_b
                            break

                    if expected_open:
                        # Pop until we find the matching open
                        found = False
                        for j in range(len(stack) - 1, -1, -1):
                            if stack[j][0] == expected_open:
                                stack.pop(j)
                                found = True
                                break
                        if not found:
                            unmatched_close.append((char, line_num))

                i += 1

        # Filter out string delimiters from unmatched
        unmatched_open = [(b, ln) for b, ln in stack if b in OPEN_BRACKETS]

        # Calculate indent level based on remaining open brackets
        indent_level = sum(1 for b, _ in stack if b in OPEN_BRACKETS)

        last_open = None
        last_open_line = 0
        for b, ln in reversed(stack):
            if b in OPEN_BRACKETS:
                last_open = b
                last_open_line = ln
                break

        return BracketMatch(
            is_balanced=len(unmatched_open) == 0 and len(unmatched_close) == 0,
            unmatched_open=unmatched_open,
            unmatched_close=unmatched_close,
            indent_level=indent_level,
            last_open_bracket=last_open,
            last_open_line=last_open_line,
        )

    def detect_continuation(self) -> bool:
        """Detect whether the current input expects more lines.

        Analyzes the last line for continuation indicators like
        trailing colons, backslashes, open brackets, etc.

        Returns:
            True if more input is expected.
        """
        if not self._lines:
            return False

        last_line = self._lines[-1].rstrip()

        # Check for continuation patterns
        for pattern in CONTINUATION_PATTERNS:
            if pattern.search(last_line):
                self._continuation_detected = True
                return True

        # Check bracket balance
        bracket_match = self.analyze_brackets()
        if not bracket_match.is_balanced:
            self._continuation_detected = True
            return True

        self._continuation_detected = False
        return False

    def calculate_indent(self, line_index: int | None = None) -> str:
        """Calculate the appropriate indentation for a line.

        Args:
            line_index: Line to calculate indent for (defaults to current).

        Returns:
            The indentation string.
        """
        if not self._auto_indent:
            return ""

        idx = line_index if line_index is not None else self._cursor_line
        if idx <= 0:
            return ""

        prev_line = self._lines[idx - 1] if idx > 0 else ""
        prev_indent = self._extract_indent(prev_line)

        # Check if previous line ends with a colon (Python block)
        stripped_prev = prev_line.rstrip()
        if stripped_prev.endswith(":"):
            return prev_indent + self._indent_str

        # Check if previous line opens a bracket
        bracket_match = self.analyze_brackets()
        if bracket_match.last_open_bracket and bracket_match.last_open_line == idx - 1:
            # Indent relative to the opening bracket position
            bracket_pos = prev_line.index(bracket_match.last_open_bracket)
            return " " * (bracket_pos + 1)

        # Check if we're still inside an open bracket
        if bracket_match.indent_level > 0:
            # Maintain indent level of previous line with brackets
            return prev_indent

        # Check if previous line was a continuation
        for pattern in CONTINUATION_PATTERNS:
            if pattern.search(stripped_prev):
                return prev_indent + self._indent_str

        # Default: match previous indent
        return prev_indent

    @staticmethod
    def _extract_indent(line: str) -> str:
        """Extract the leading whitespace from a line.

        Args:
            line: Source line.

        Returns:
            The leading whitespace string.
        """
        indent = ""
        for char in line:
            if char in (" ", "\t"):
                indent += char
            else:
                break
        return indent

    def insert_text(self, text: str) -> None:
        """Insert text at the current cursor position.

        Handles newlines with auto-indentation and bracket auto-closing.

        Args:
            text: Text to insert.
        """
        if "\n" in text:
            for char in text:
                if char == "\n":
                    self._insert_newline()
                else:
                    self._insert_char(char)
        else:
            for char in text:
                self._insert_char(char)

    def _insert_char(self, char: str) -> None:
        """Insert a single character with bracket auto-closing.

        Args:
            char: Character to insert.
        """
        line = self._lines[self._cursor_line]
        self._lines[self._cursor_line] = (
            line[:self._cursor_col] + char + line[self._cursor_col:]
        )
        self._cursor_col += 1

        # Auto-close brackets
        if self._auto_close_brackets and char in OPEN_BRACKETS:
            close_char = BRACKET_PAIRS[char]
            line = self._lines[self._cursor_line]
            self._lines[self._cursor_line] = (
                line[:self._cursor_col] + close_char + line[self._cursor_col:]
            )

    def _insert_newline(self) -> None:
        """Insert a newline with auto-indentation."""
        line = self._lines[self._cursor_line]
        before = line[:self._cursor_col]
        after = line[self._cursor_col:]

        self._lines[self._cursor_line] = before
        self._lines.insert(self._cursor_line + 1, after)

        self._cursor_line += 1

        # Calculate and apply auto-indent
        indent = self.calculate_indent(self._cursor_line)
        self._lines[self._cursor_line] = indent + after
        self._cursor_col = len(indent)

    def handle_enter(self) -> str | None:
        """Handle the Enter key with smart behavior.

        If smart_enter is enabled, submits the input if it appears
        complete (no continuation). Otherwise inserts a newline.

        Returns:
            The complete text if submitted, or None if a newline was inserted.
        """
        if self._smart_enter and not self.detect_continuation():
            # Check for complete patterns that should submit immediately
            if self._is_submit_ready():
                text = self.text
                self._history.append(text)
                self._lines = [""]
                self._cursor_line = 0
                self._cursor_col = 0
                return text

        # Insert newline
        self._insert_newline()
        return None

    def _is_submit_ready(self) -> bool:
        """Check if the current input is ready to be submitted.

        Returns:
            True if the input can be submitted.
        """
        text = self.text.strip()

        # Empty input
        if not text:
            return False

        # Check bracket balance
        bracket_match = self.analyze_brackets()
        if not bracket_match.is_balanced:
            return False

        # Check for continuation
        if self.detect_continuation():
            return False

        return True

    def delete_backwards(self) -> None:
        """Delete the character before the cursor, handling indentation."""
        if self._cursor_col > 0:
            # Check if we're deleting auto-indent whitespace
            line = self._lines[self._cursor_line]
            before_cursor = line[:self._cursor_col]
            if before_cursor.strip() == "" and len(before_cursor) >= len(self._indent_str):
                # Delete one indent level
                indent_len = len(self._indent_str)
                remove_count = min(self._cursor_col, indent_len)
                self._lines[self._cursor_line] = line[remove_count:]
                self._cursor_col -= remove_count
            else:
                self._lines[self._cursor_line] = (
                    line[:self._cursor_col - 1] + line[self._cursor_col:]
                )
                self._cursor_col -= 1
        elif self._cursor_line > 0:
            # Merge with previous line
            prev_line = self._lines[self._cursor_line - 1]
            self._cursor_col = len(prev_line)
            self._lines[self._cursor_line - 1] = prev_line + self._lines[self._cursor_line]
            self._lines.pop(self._cursor_line)
            self._cursor_line -= 1

    def delete_forward(self) -> None:
        """Delete the character at the cursor position."""
        line = self._lines[self._cursor_line]
        if self._cursor_col < len(line):
            self._lines[self._cursor_line] = (
                line[:self._cursor_col] + line[self._cursor_col + 1:]
            )
        elif self._cursor_line < len(self._lines) - 1:
            # Merge with next line
            self._lines[self._cursor_line] += self._lines[self._cursor_line + 1]
            self._lines.pop(self._cursor_line + 1)

    def move_cursor(self, direction: str) -> None:
        """Move the cursor in the specified direction.

        Args:
            direction: One of 'left', 'right', 'up', 'down', 'home', 'end'.
        """
        if direction == "left":
            if self._cursor_col > 0:
                self._cursor_col -= 1
            elif self._cursor_line > 0:
                self._cursor_line -= 1
                self._cursor_col = len(self._lines[self._cursor_line])
        elif direction == "right":
            if self._cursor_col < len(self._lines[self._cursor_line]):
                self._cursor_col += 1
            elif self._cursor_line < len(self._lines) - 1:
                self._cursor_line += 1
                self._cursor_col = 0
        elif direction == "up":
            if self._cursor_line > 0:
                self._cursor_line -= 1
                self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_line]))
        elif direction == "down":
            if self._cursor_line < len(self._lines) - 1:
                self._cursor_line += 1
                self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_line]))
        elif direction == "home":
            # Go to start of content (skip indent)
            line = self._lines[self._cursor_line]
            content_start = len(line) - len(line.lstrip())
            self._cursor_col = content_start if self._cursor_col != content_start else 0
        elif direction == "end":
            self._cursor_col = len(self._lines[self._cursor_line])

    def get_continuation_prompt(self, line_num: int) -> str:
        """Get the prompt string for a continuation line.

        Args:
            line_num: The line number (0-indexed).

        Returns:
            Prompt string - '... ' for continuations, '>>> ' for first line.
        """
        if line_num == 0:
            return ">>> "
        return "... "

    def render(self) -> str:
        """Render the multi-line input with prompts.

        Returns:
            Formatted string showing all lines with prompts and cursor.
        """
        lines = []
        for i, line in enumerate(self._lines):
            prompt = self.get_continuation_prompt(i)
            if i == self._cursor_line:
                # Show cursor position
                before = line[:self._cursor_col]
                after = line[self._cursor_col:]
                cursor_line = f"{prompt}{before}▏{after}"
            else:
                cursor_line = f"{prompt}{line}"
            lines.append(cursor_line)
        return "\n".join(lines)
