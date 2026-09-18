"""Tests for the terminal rendering layer.

Covers the pieces the chat UI is built from: text wrapping/truncation, panels,
tables, themes, syntax highlighting, markdown rendering and the Rich-based
OutputFormatter used by ``nexus-llm chat``.
"""

from __future__ import annotations

import re

import pytest

from nexus_llm.terminal.formatter import OutputFormatter
from nexus_llm.terminal.markdown_ext import MarkdownRenderer
from nexus_llm.terminal.panel import CollapsiblePanel, PanelRenderer, PanelStyle
from nexus_llm.terminal.renderer import Alignment, RenderOptions, TextRenderer
from nexus_llm.terminal.syntax import Language, SyntaxHighlighter
from nexus_llm.terminal.table import FilterOp, SortOrder, TableBuilder, TableFilter
from nexus_llm.terminal.themes import THEMES, get_theme, list_themes


# ---------------------------------------------------------------------------
# TextRenderer
# ---------------------------------------------------------------------------


class TestTextRenderer:
    def test_default_width_is_positive(self):
        assert TextRenderer().default_width > 0

    def test_truncate_shortens_and_adds_suffix(self):
        renderer = TextRenderer()
        out = renderer.truncate("a" * 100, 20)
        assert len(out) <= 23  # 20 chars + "..."
        assert out.endswith("...")

    def test_truncate_leaves_short_text_alone(self):
        assert TextRenderer().truncate("short", 40) == "short"

    def test_wrap_respects_width(self):
        renderer = TextRenderer()
        wrapped = renderer.wrap("word " * 200, RenderOptions(width=40))
        assert all(len(line) <= 40 for line in wrapped.splitlines())

    def test_indent_adds_prefix(self):
        renderer = TextRenderer()
        out = renderer.indent("line1\nline2", indent_str="  ")
        assert all(line.startswith("  ") for line in out.splitlines())

    def test_indent_skip_first(self):
        renderer = TextRenderer()
        out = renderer.indent("line1\nline2", indent_str="  ", skip_first=True)
        first, second = out.splitlines()
        assert not first.startswith("  ")
        assert second.startswith("  ")

    def test_dedent_is_inverse_of_indent(self):
        renderer = TextRenderer()
        text = "alpha\nbeta"
        assert renderer.dedent(renderer.indent(text, indent_str="    ")) == text

    def test_pad_left_right_center(self):
        renderer = TextRenderer()
        assert renderer.pad("ab", 6, Alignment.LEFT).startswith("ab")
        assert renderer.pad("ab", 6, Alignment.RIGHT).endswith("ab")
        assert renderer.pad("ab", 6, Alignment.CENTER).count(" ") == 4

    def test_visible_length_ignores_markup(self):
        assert TextRenderer().visible_length("\x1b[31mred\x1b[0m") == 3

    def test_ruler_spans_width(self):
        renderer = TextRenderer()
        assert len(renderer.ruler("=", width=20)) == 20

    def test_ruler_with_title_includes_title(self):
        out = TextRenderer().ruler(width=30, title="Models")
        assert "Models" in out and len(out) <= 30

    def test_box_contains_text(self):
        out = TextRenderer().box("hello", width=20)
        assert "hello" in out
        assert len(out.splitlines()) >= 3  # top border, content, bottom border

    def test_columnize_lays_out_items(self):
        out = TextRenderer().columnize(["one", "two", "three", "four"], columns=2)
        assert "one" in out and "four" in out
        assert len(out.splitlines()) == 2

    def test_max_lines_truncates_paragraph_count(self):
        renderer = TextRenderer()
        text = "\n\n".join(f"paragraph {i} body" for i in range(10))
        out = renderer.wrap(text, RenderOptions(width=60, max_lines=3, placeholder="…"))
        assert len(out.splitlines()) <= 3
        assert "…" in out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class TestPanels:
    def test_render_contains_content_and_title(self):
        out = PanelRenderer().render("body text", title="Header")
        assert "body text" in out and "Header" in out

    def test_render_with_custom_style_and_width(self):
        style = PanelStyle(border_color="red", border_style="heavy")
        out = PanelRenderer().render("x", style=style, width=30)
        assert "x" in out

    def test_subtitle_appears(self):
        out = PanelRenderer().render("content", subtitle="footer note")
        assert "footer note" in out

    def test_collapsible_toggle(self):
        panel = CollapsiblePanel("Details", content="hidden stuff")
        assert panel.collapsed is False
        panel.collapse()
        assert panel.collapsed is True
        panel.toggle()
        assert panel.collapsed is False

    def test_collapsed_render_hides_content(self):
        panel = CollapsiblePanel("Secrets", content="token=abc123", collapsed=True)
        assert "token=abc123" not in panel.render(width=60)
        panel.expand()
        assert "token=abc123" in panel.render(width=60)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestTables:
    def test_builder_renders_rows_and_headers(self):
        builder = (
            TableBuilder(title="Models")
            .add_column("name", header="Name")
            .add_column("size", header="Size")
            .add_row(name="gpt2", size=124)
            .add_row(name="llama", size=7000)
        )
        assert builder.row_count == 2
        rendered = builder.render()
        assert "gpt2" in rendered and "llama" in rendered
        assert "Name" in rendered and "Size" in rendered

    def test_add_rows_and_sorting(self):
        builder = (
            TableBuilder()
            .add_column("name", header="Name")
            .add_column("size", header="Size")
            .add_rows([{"name": "gpt2", "size": 124}, {"name": "llama", "size": 7000}])
        )
        assert builder.row_count == 2
        builder.sort_by("size", order=SortOrder.DESCENDING)
        rows = builder.get_processed_rows()
        assert [r["name"] for r in rows] == ["llama", "gpt2"]

    def test_filter_narrows_rows(self):
        builder = (
            TableBuilder()
            .add_column("name")
            .add_rows([{"name": "gpt2"}, {"name": "llama"}, {"name": "gpt2-medium"}])
            .filter("name", FilterOp.CONTAINS, "gpt2")
        )
        assert builder.filtered_count == 2

    def test_table_filter_contains(self):
        assert TableFilter("name", FilterOp.CONTAINS, "gpt").matches("gpt2-medium")
        assert not TableFilter("name", FilterOp.CONTAINS, "zzz").matches("gpt2")

    def test_table_filter_equality_and_numeric(self):
        assert TableFilter("n", FilterOp.EQ, 5).matches(5)
        assert TableFilter("n", FilterOp.NE, 5).matches(6)
        assert TableFilter("n", FilterOp.GT, 5).matches(9)
        assert not TableFilter("n", FilterOp.GT, 5).matches(1)
        assert TableFilter("n", FilterOp.GTE, 5).matches(5)
        assert TableFilter("n", FilterOp.LT, 5).matches(1)


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


class TestThemes:
    def test_list_themes_returns_mapping(self):
        themes = list_themes()
        assert isinstance(themes, dict) and "dark" in themes

    def test_get_theme_defaults_to_dark(self):
        assert get_theme("does-not-exist").name == "dark"

    def test_every_theme_has_required_styles(self):
        required = {"user_prompt", "assistant_text", "error_text", "border_color"}
        for name, theme in THEMES.items():
            missing = {a for a in required if not getattr(theme, a, "")}
            assert not missing, f"theme {name} missing {missing}"

    def test_theme_names_match_keys(self):
        for name, theme in THEMES.items():
            assert theme.name == name


# ---------------------------------------------------------------------------
# Syntax highlighting & markdown
# ---------------------------------------------------------------------------


class TestSyntaxHighlighting:
    def test_python_highlight_emits_escape_codes(self):
        highlighter = SyntaxHighlighter()
        code = "def f():\n    return 1"
        out = highlighter.highlight(code, language=Language.PYTHON)
        # Highlighting adds ANSI sequences but must not lose the source text.
        assert "\x1b[" in out
        assert "def" in out and "return" in out

    def test_unknown_language_falls_back_to_plain(self):
        highlighter = SyntaxHighlighter()
        code = "not real code @@@ ###"
        assert code in highlighter.highlight(code, language="not-a-language")

    def test_supported_languages_include_python(self):
        assert "python" in SyntaxHighlighter().get_supported_languages()

    def test_tokenize_finds_keywords(self):
        tokens = SyntaxHighlighter().tokenize("x = 1", language=Language.PYTHON)
        assert tokens
        assert all(hasattr(t, "start") and hasattr(t, "end") for t in tokens)
        assert all(t.end >= t.start for t in tokens)

    def test_line_numbers_add_prefix(self):
        highlighter = SyntaxHighlighter()
        numbered = highlighter.highlight("a = 1\nb = 2", language=Language.PYTHON, line_numbers=True)
        assert "1" in numbered and "2" in numbered


class TestMarkdown:
    def test_render_headings(self):
        out = MarkdownRenderer().render("# Title\n\ntext")
        assert "Title" in out and "text" in out

    def test_render_code_fence(self):
        out = MarkdownRenderer().render("```python\nx = 1\n```")
        plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert "x =" in plain and "1" in plain
        # No stray empty numbered line after the snippet.
        assert not [ln for ln in plain.splitlines() if ln.strip().endswith("│") and not ln.strip().rstrip("│").strip()]

    def test_render_multiline_document(self):
        text = "# Title\n\nfirst paragraph\n\n## Sub\n\nsecond paragraph"
        out = MarkdownRenderer().render(text)
        for fragment in ("Title", "first paragraph", "Sub", "second paragraph"):
            assert fragment in out

    def test_render_empty_input(self):
        assert MarkdownRenderer().render("") == ""


# ---------------------------------------------------------------------------
# OutputFormatter (the Rich wrapper used by chat)
# ---------------------------------------------------------------------------


@pytest.fixture
def formatter() -> OutputFormatter:
    from rich.console import Console

    return OutputFormatter(console=Console(record=True, width=80, force_terminal=False))


class TestOutputFormatter:
    def test_print_writes_to_console(self, formatter):
        formatter.print("hello world")
        assert "hello world" in formatter.console.export_text()

    def test_print_error_and_success_are_distinguishable(self, formatter):
        formatter.print_error("bad thing")
        formatter.print_success("good thing")
        text = formatter.console.export_text()
        assert "bad thing" in text and "good thing" in text

    def test_stream_collects_tokens(self, formatter):
        formatter.start_stream()
        for token in ["Hello", ", ", "world"]:
            formatter.append_stream(token)
        stats = formatter.end_stream()
        # The buffer keeps the full streamed text (documented as "current / last
        # stream") and the stats report how much arrived.
        assert formatter.stream_buffer == "Hello, world"
        assert stats["tokens"] == 3

    def test_set_theme_switches_colours(self, formatter):
        before = formatter.theme.name
        formatter.set_theme(get_theme("ocean"))
        assert formatter.theme.name != before or before == "ocean"

    def test_print_models_table(self, formatter):
        formatter.print_models_table([{"name": "gpt2", "size": "124M"}])
        assert "gpt2" in formatter.console.export_text()

    def test_print_banner_includes_version(self, formatter):
        formatter.print_banner("9.9.9")
        assert "9.9.9" in formatter.console.export_text()
