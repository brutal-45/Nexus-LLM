"""
Nexus-LLM Syntax Highlighting Module

Provides syntax highlighting for 11+ programming languages:
Python, JavaScript, HTML, CSS, JSON, YAML, Markdown, SQL, Bash, Rust, Go.

Uses pygments when available, with a custom tokenizer fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import pygments
    from pygments.lexers import (
        get_lexer_by_name,
        guess_lexer,
        PythonLexer,
        JavaScriptLexer,
        HtmlLexer,
        CssLexer,
        JsonLexer,
        YamlLexer,
        MarkdownLexer,
        SqlLexer,
        BashLexer,
        RustLexer,
        GoLexer,
    )
    from pygments.token import Token, Keyword, Name, String, Comment, Number, Operator, Punctuation
    from pygments.formatters import TerminalTrueColorFormatter, TerminalFormatter

    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    SQL = "sql"
    BASH = "bash"
    RUST = "rust"
    GO = "go"
    TEXT = "text"
    AUTO = "auto"


@dataclass
class TokenMatch:
    """A highlighted token with type and position."""
    type: str
    value: str
    start: int
    end: int


# Fallback token patterns for each language when pygments is unavailable
FALLBACK_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    Language.PYTHON: [
        ("keyword", re.compile(r"\b(and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield|True|False|None)\b")),
        ("string", re.compile(r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|f\"[^\"]*\"|f\'[^\']*\'|\"[^\"]*\"|\'[^\']*\')')),
        ("comment", re.compile(r"(#[^\n]*)")),
        ("decorator", re.compile(r"(@\w+)")),
        ("number", re.compile(r"\b(\d+\.?\d*(?:e[+-]?\d+)?)\b")),
        ("builtin", re.compile(r"\b(print|len|range|int|str|float|list|dict|tuple|set|type|isinstance|hasattr|getattr|setattr|super|property|staticmethod|classmethod|abs|all|any|bin|bool|chr|dir|enumerate|eval|exec|filter|format|hex|id|input|map|max|min|next|oct|open|ord|pow|repr|reversed|round|sorted|sum|vars|zip)\b")),
        ("function", re.compile(r"\b(\w+)\s*\(")),
    ],
    Language.JAVASCRIPT: [
        ("keyword", re.compile(r"\b(async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|export|extends|finally|for|from|function|if|import|in|instanceof|let|new|of|return|switch|this|throw|try|typeof|var|void|while|with|yield|true|false|null|undefined)\b")),
        ("string", re.compile(r'(`[\s\S]*?`|\"[^\"]*\"|\'[^\']*\')')),
        ("comment", re.compile(r"(//[^\n]*|/\*[\s\S]*?\*/)")),
        ("number", re.compile(r"\b(\d+\.?\d*(?:e[+-]?\d+)?)\b")),
        ("function", re.compile(r"\b(\w+)\s*\(")),
        ("arrow", re.compile(r"(=>)")),
    ],
    Language.HTML: [
        ("tag", re.compile(r"(<\/?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?>)")),
        ("attribute", re.compile(r'\b([a-zA-Z-]+)=["\']')),
        ("string", re.compile(r'(\"[^\"]*\"|\'[^\']*\')')),
        ("comment", re.compile(r"(<!--[\s\S]*?-->)")),
        ("entity", re.compile(r"(&[a-zA-Z]+;|&#\d+;)")),
    ],
    Language.CSS: [
        ("selector", re.compile(r"([\w.#\[\]:,>+~\s-]+)\s*\{")),
        ("property", re.compile(r"\b([a-zA-Z-]+)\s*:")),
        ("value", re.compile(r":\s*([^;]+)")),
        ("string", re.compile(r'(\"[^\"]*\"|\'[^\']*\')')),
        ("comment", re.compile(r"(/\*[\s\S]*?\*/)")),
        ("number", re.compile(r"\b(\d+\.?\d*)(px|em|rem|%|vh|vw|s|ms|deg|fr)?\b")),
        ("at_rule", re.compile(r"(@[a-zA-Z-]+)")),
    ],
    Language.JSON: [
        ("key", re.compile(r'(\"[^\"]+\")\s*:')),
        ("string", re.compile(r':\s*(\"[^\"]*\")')),
        ("number", re.compile(r":\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)")),
        ("boolean", re.compile(r"\b(true|false|null)\b")),
    ],
    Language.YAML: [
        ("key", re.compile(r"^(\s*\w[\w\s-]*):", re.MULTILINE)),
        ("string", re.compile(r'(\"[^\"]*\"|\'[^\']*\'|[|>]\s*$)')),
        ("comment", re.compile(r"(#[^\n]*)")),
        ("number", re.compile(r"\b(\d+\.?\d*)\b")),
        ("boolean", re.compile(r"\b(true|false|null|yes|no|on|off)\b")),
        ("anchor", re.compile(r"(&\w+|\\*\w+)")),
    ],
    Language.SQL: [
        ("keyword", re.compile(r"\b(SELECT|FROM|WHERE|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|ALTER|DROP|INDEX|JOIN|INNER|OUTER|LEFT|RIGHT|ON|AND|OR|NOT|IN|LIKE|BETWEEN|IS|NULL|AS|ORDER|BY|GROUP|HAVING|LIMIT|OFFSET|UNION|ALL|DISTINCT|COUNT|SUM|AVG|MIN|MAX|EXISTS|CASE|WHEN|THEN|ELSE|END|PRIMARY|KEY|FOREIGN|REFERENCES|CONSTRAINT|DEFAULT|AUTO_INCREMENT|UNIQUE|CHECK|VIEW|TRIGGER|PROCEDURE|FUNCTION|BEGIN|COMMIT|ROLLBACK|GRANT|REVOKE)\b", re.IGNORECASE)),
        ("string", re.compile(r'(\"[^\"]*\"|\'[^\']*\')')),
        ("comment", re.compile(r"(--[^\n]*|/\*[\s\S]*?\*/)")),
        ("number", re.compile(r"\b(\d+\.?\d*)\b")),
        ("function", re.compile(r"\b(\w+)\s*\(")),
        ("operator", re.compile(r"(<>|!=|>=|<=|==|<|>|\+|-|\*|/)")),
    ],
    Language.BASH: [
        ("keyword", re.compile(r"\b(if|then|else|elif|fi|for|while|do|done|case|esac|in|function|select|until|time|coproc|return|exit|break|continue|declare|export|local|readonly|typeset|unset|source|alias|echo|cd|pwd|ls|mkdir|rm|cp|mv|cat|grep|sed|awk|find|sort|uniq|wc|head|tail|chmod|chown|sudo|apt|yum|pip|npm|git|docker)\b")),
        ("string", re.compile(r'(\"[^\"]*\"|\'[^\']*\'|`[^`]*`)')),
        ("comment", re.compile(r"(#[^\n]*)")),
        ("variable", re.compile(r"(\$\{?\w+\}?)")),
        ("number", re.compile(r"\b(\d+\.?\d*)\b")),
    ],
    Language.RUST: [
        ("keyword", re.compile(r"\b(as|async|await|break|const|continue|crate|dyn|else|enum|extern|fn|for|if|impl|in|let|loop|match|mod|move|mut|pub|ref|return|self|Self|static|struct|super|trait|type|unsafe|use|where|while|yield|true|false|Some|None|Ok|Err)\b")),
        ("string", re.compile(r'(\"[^\"]*\"|r#\"[\s\S]*?\"#|r\"[\s\S]*?\"|b\"[^\"]*\")')),
        ("comment", re.compile(r"(//[^\n]*|/\*[\s\S]*?\*/)")),
        ("attribute", re.compile(r"(#!?\[[^\]]*\])")),
        ("lifetime", re.compile(r"('\w+)")),
        ("number", re.compile(r"\b(\d+\.?\d*(?:_\d+)*(?:f32|f64|u\d+|i\d+|usize|isize)?)\b")),
        ("function", re.compile(r"\b(\w+)\s*[<(]")),
        ("macro", re.compile(r"(\w+!)")),
    ],
    Language.GO: [
        ("keyword", re.compile(r"\b(break|case|chan|const|continue|default|defer|else|fallthrough|for|func|go|goto|if|import|interface|map|package|range|return|select|struct|switch|type|var|true|false|nil|iota|append|cap|close|copy|delete|len|make|new|panic|print|println|recover)\b")),
        ("string", re.compile(r'(\"[^\"]*\"|`[\s\S]*?`)')),
        ("comment", re.compile(r"(//[^\n]*|/\*[\s\S]*?\*/)")),
        ("number", re.compile(r"\b(\d+\.?\d*(?:e[+-]?\d+)?)\b")),
        ("type", re.compile(r"\b(bool|byte|complex64|complex128|error|float32|float64|int|int8|int16|int32|int64|rune|string|uint|uint8|uint16|uint32|uint64|uintptr)\b")),
        ("function", re.compile(r"\b(\w+)\s*\(")),
    ],
    Language.MARKDOWN: [
        ("heading", re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)),
        ("code_block", re.compile(r"```[\s\S]*?```")),
        ("inline_code", re.compile(r"`([^`]+)`")),
        ("bold", re.compile(r"\*\*(.+?)\*\*")),
        ("italic", re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")),
        ("link", re.compile(r"\[([^\]]+)\]\(([^)]+)\)")),
        ("image", re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")),
        ("list_item", re.compile(r"^[\s]*[-*+]\s+(.+)$", re.MULTILINE)),
        ("blockquote", re.compile(r"^>\s+(.+)$", re.MULTILINE)),
    ],
}

# ANSI color codes for token types
TOKEN_COLORS: dict[str, str] = {
    "keyword": "\033[38;5;198m",       # pink/magenta
    "string": "\033[38;5;186m",        # yellow
    "comment": "\033[38;5;67m",        # blue-gray
    "number": "\033[38;5;141m",        # purple
    "function": "\033[38;5;84m",       # green
    "class_name": "\033[38;5;117m",    # cyan
    "decorator": "\033[38;5;215m",     # orange
    "attribute": "\033[38;5;215m",     # orange
    "operator": "\033[38;5;198m",      # pink
    "variable": "\033[38;5;231m",      # white
    "constant": "\033[38;5;141m",      # purple
    "builtin": "\033[38;5;84m",        # green
    "tag": "\033[38;5;117m",           # cyan
    "selector": "\033[38;5;84m",       # green
    "property": "\033[38;5;117m",      # cyan
    "value": "\033[38;5;186m",         # yellow
    "key": "\033[38;5;117m",           # cyan
    "boolean": "\033[38;5;141m",       # purple
    "type": "\033[38;5;117m",          # cyan
    "lifetime": "\033[38;5;215m",      # orange
    "macro": "\033[38;5;84m",          # green
    "arrow": "\033[38;5;198m",         # pink
    "at_rule": "\033[38;5;198m",       # pink
    "entity": "\033[38;5;141m",        # purple
    "anchor": "\033[38;5;215m",        # orange
    "heading": "\033[38;5;117m\033[1m",# bold cyan
    "code_block": "\033[38;5;186m",    # yellow
    "inline_code": "\033[38;5;186m",   # yellow
    "bold": "\033[1m",                 # bold
    "italic": "\033[3m",               # italic
    "link": "\033[38;5;117m",          # cyan
    "image": "\033[38;5;141m",         # purple
    "list_item": "\033[38;5;84m",      # green
    "blockquote": "\033[38;5;67m",     # blue-gray
}

RESET = "\033[0m"


class SyntaxHighlighter:
    """Syntax highlighting engine for terminal output.

    Supports 11+ programming languages with pygments as the primary
    engine and a regex-based fallback when pygments is unavailable.
    """

    # Mapping from our Language enum to pygments lexer names
    LEXER_MAP: dict[str, str] = {
        Language.PYTHON: "python",
        Language.JAVASCRIPT: "javascript",
        Language.HTML: "html",
        Language.CSS: "css",
        Language.JSON: "json",
        Language.YAML: "yaml",
        Language.MARKDOWN: "markdown",
        Language.SQL: "sql",
        Language.BASH: "bash",
        Language.RUST: "rust",
        Language.GO: "go",
    }

    def __init__(self, theme: str = "monokai", style: str = "terminal256") -> None:
        self._theme = theme
        self._style = style

    def highlight(
        self,
        code: str,
        language: str | Language = Language.AUTO,
        line_numbers: bool = False,
    ) -> str:
        """Highlight source code with syntax coloring.

        Args:
            code: Source code string to highlight.
            language: Programming language or 'auto' for detection.
            line_numbers: Whether to prepend line numbers.

        Returns:
            ANSI-colored code string.
        """
        if HAS_PYGMENTS:
            return self._highlight_pygments(code, language, line_numbers)
        return self._highlight_fallback(code, language, line_numbers)

    def _highlight_pygments(
        self,
        code: str,
        language: str | Language,
        line_numbers: bool,
    ) -> str:
        """Highlight using pygments."""
        lang_str = language.value if isinstance(language, Language) else language
        try:
            if lang_str == Language.AUTO:
                lexer = guess_lexer(code)
            else:
                lexer_name = self.LEXER_MAP.get(lang_str, lang_str)
                lexer = get_lexer_by_name(lexer_name)
        except Exception:
            lexer = PythonLexer()

        try:
            formatter = TerminalTrueColorFormatter(style=self._theme)
        except Exception:
            formatter = TerminalFormatter(style=self._theme)

        result = pygments.highlight(code, lexer, formatter)

        if line_numbers:
            lines = result.rstrip().split("\n")
            width = len(str(len(lines)))
            numbered = []
            for i, line in enumerate(lines, 1):
                numbered.append(f"\033[38;5;67m{i:>{width}} │\033[0m {line}")
            result = "\n".join(numbered)

        return result

    def _highlight_fallback(
        self,
        code: str,
        language: str | Language,
        line_numbers: bool,
    ) -> str:
        """Highlight using regex-based fallback."""
        lang_str = language.value if isinstance(language, Language) else language

        if lang_str == Language.AUTO:
            lang_str = self._detect_language(code)

        patterns = FALLBACK_PATTERNS.get(lang_str, [])
        if not patterns:
            if line_numbers:
                return self._add_line_numbers(code)
            return code

        # Apply patterns in order, building highlighted output
        result = code
        # Use a marker approach to avoid double-highlighting
        markers: list[tuple[int, int, str]] = []

        for token_type, pattern in patterns:
            for match in pattern.finditer(code):
                start, end = match.start(), match.end()
                # Check if this range is already marked
                overlaps = any(s <= start < e or s < end <= e for s, e, _ in markers)
                if not overlaps:
                    markers.append((start, end, token_type))

        # Sort markers by position
        markers.sort(key=lambda m: m[0])

        # Build the output
        output_parts = []
        last_end = 0
        for start, end, token_type in markers:
            if start > last_end:
                output_parts.append(code[last_end:start])
            color = TOKEN_COLORS.get(token_type, "")
            output_parts.append(f"{color}{code[start:end]}{RESET}")
            last_end = end
        if last_end < len(code):
            output_parts.append(code[last_end:])

        result = "".join(output_parts)

        if line_numbers:
            result = self._add_line_numbers(result)

        return result

    def _add_line_numbers(self, code: str) -> str:
        """Add line numbers to code.

        Args:
            code: Source code.

        Returns:
            Code with line numbers prepended.
        """
        lines = code.split("\n")
        width = len(str(len(lines)))
        numbered = []
        for i, line in enumerate(lines, 1):
            numbered.append(f"\033[38;5;67m{i:>{width}} │\033[0m {line}")
        return "\n".join(numbered)

    def _detect_language(self, code: str) -> str:
        """Attempt to detect the programming language of code.

        Uses simple heuristic patterns for language detection.

        Args:
            code: Source code to analyze.

        Returns:
            Detected language string.
        """
        indicators: dict[str, list[str]] = {
            Language.PYTHON: ["def ", "import ", "from ", "class ", "if __name__", "self.", "print("],
            Language.JAVASCRIPT: ["const ", "let ", "var ", "function ", "=>", "console.log", "require("],
            Language.HTML: ["<!DOCTYPE", "<html", "<div", "<span", "<head>", "<body>"],
            Language.CSS: ["{", "}", ":", ";", "@media", "@import", "font-family"],
            Language.JSON: ['{', '}', '"', ':', '[', ']'],
            Language.YAML: ["---", "- ", ": ", "true:", "false:"],
            Language.SQL: ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE TABLE", "FROM"],
            Language.BASH: ["#!/bin/bash", "#!/bin/sh", "sudo ", "apt ", "chmod "],
            Language.RUST: ["fn ", "let ", "mut ", "impl ", "pub ", "use ", "::", "pub fn"],
            Language.GO: ["func ", "package ", "import (", "fmt.", "go func"],
        }

        scores: dict[str, int] = {}
        for lang, patterns in indicators.items():
            score = 0
            for pattern in patterns:
                if pattern in code:
                    score += 1
            scores[lang] = score

        if not scores or max(scores.values()) == 0:
            return Language.TEXT

        return max(scores, key=lambda k: scores[k])

    def tokenize(self, code: str, language: str | Language = Language.PYTHON) -> list[TokenMatch]:
        """Tokenize code and return structured token information.

        Args:
            code: Source code to tokenize.
            language: Programming language.

        Returns:
            List of TokenMatch objects.
        """
        lang_str = language.value if isinstance(language, Language) else language
        patterns = FALLBACK_PATTERNS.get(lang_str, [])

        tokens: list[TokenMatch] = []
        seen_ranges: set[tuple[int, int]] = set()

        for token_type, pattern in patterns:
            for match in pattern.finditer(code):
                start, end = match.start(), match.end()
                if (start, end) not in seen_ranges:
                    seen_ranges.add((start, end))
                    tokens.append(TokenMatch(
                        type=token_type,
                        value=match.group(),
                        start=start,
                        end=end,
                    ))

        tokens.sort(key=lambda t: t.start)
        return tokens

    def get_supported_languages(self) -> list[str]:
        """Get a list of supported language names.

        Returns:
            List of language name strings.
        """
        return [lang.value for lang in Language if lang != Language.AUTO]
