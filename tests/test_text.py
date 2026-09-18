"""Test text processing utilities for Nexus-LLM."""
import re
import pytest
from typing import List, Optional


# --- Text processing implementations to test ---

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    if max_length <= len(suffix):
        return suffix[:max_length]
    return text[: max_length - len(suffix)] + suffix


def count_words(text: str) -> int:
    return len(text.split())


def count_characters(text: str) -> int:
    return len(text)


def count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def remove_urls(text: str) -> str:
    return re.sub(r'https?://\S+', '', text)


def remove_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


def extract_emails(text: str) -> List[str]:
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')


def capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def wrap_text(text: str, width: int = 80) -> str:
    if width < 1:
        raise ValueError("Width must be positive")
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if current_line and len(current_line) + 1 + len(word) > width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = current_line + " " + word if current_line else word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def contains_cjk(text: str) -> bool:
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
    return bool(cjk_pattern.search(text))


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("hello", 10) == "hello"

    def test_long_text_truncated(self):
        result = truncate_text("hello world this is long", 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_custom_suffix(self):
        result = truncate_text("hello world this is long", 10, suffix="…")
        assert result.endswith("…")

    def test_exact_length(self):
        assert truncate_text("hello", 5) == "hello"

    def test_very_short_max_length(self):
        result = truncate_text("hello", 2)
        assert len(result) == 2


class TestCounting:
    def test_count_words(self):
        assert count_words("hello world") == 2

    def test_count_words_empty(self):
        assert count_words("") == 0

    def test_count_words_multiple_spaces(self):
        assert count_words("hello   world") == 2

    def test_count_characters(self):
        assert count_characters("hello") == 5

    def test_count_sentences(self):
        assert count_sentences("Hello. World! How are you?") == 3

    def test_count_sentences_single(self):
        assert count_sentences("Just one sentence.") == 1

    def test_count_sentences_empty(self):
        assert count_sentences("") == 0


class TestNormalization:
    def test_normalize_whitespace(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_normalize_tabs_and_newlines(self):
        assert normalize_whitespace("hello\t\nworld") == "hello world"

    def test_normalize_leading_trailing(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_normalize_single_space(self):
        assert normalize_whitespace("hello world") == "hello world"


class TestRemovePatterns:
    def test_remove_urls(self):
        text = "Visit https://example.com for more"
        assert remove_urls(text) == "Visit  for more"

    def test_remove_urls_http(self):
        text = "Go to http://test.org/page"
        assert "http://test.org" not in remove_urls(text)

    def test_remove_html_tags(self):
        text = "<p>Hello <b>world</b></p>"
        assert remove_html_tags(text) == "Hello world"

    def test_remove_html_self_closing(self):
        text = "Line 1<br/>Line 2"
        result = remove_html_tags(text)
        assert "<br/>" not in result


class TestExtractEmails:
    def test_extract_single_email(self):
        text = "Contact us at test@example.com"
        assert extract_emails(text) == ["test@example.com"]

    def test_extract_multiple_emails(self):
        text = "Email a@b.com and c@d.org"
        assert len(extract_emails(text)) == 2

    def test_no_emails(self):
        assert extract_emails("no emails here") == []

    def test_complex_email(self):
        text = "Send to user.name+tag@sub.domain.com"
        result = extract_emails(text)
        assert len(result) >= 1


class TestSlugify:
    def test_simple(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars_removed(self):
        result = slugify("Hello, World! (2024)")
        assert "!" not in result
        assert "," not in result

    def test_multiple_hyphens(self):
        assert slugify("hello   world") == "hello-world"

    def test_leading_trailing_hyphens(self):
        assert slugify("--hello--") == "hello"


class TestCapitalize:
    def test_capitalize_first(self):
        assert capitalize_first("hello") == "Hello"

    def test_already_capitalized(self):
        assert capitalize_first("Hello") == "Hello"

    def test_empty_string(self):
        assert capitalize_first("") == ""

    def test_single_char(self):
        assert capitalize_first("a") == "A"


class TestWrapText:
    def test_short_text_unchanged(self):
        assert wrap_text("hello", 80) == "hello"

    def test_wrap_long_text(self):
        text = " ".join(["word"] * 20)
        result = wrap_text(text, 40)
        for line in result.split("\n"):
            assert len(line) <= 40 or " " not in line

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="positive"):
            wrap_text("hello", 0)


class TestAnsiStripping:
    def test_strip_ansi(self):
        text = "\x1b[31mError\x1b[0m: something failed"
        result = strip_ansi(text)
        assert "\x1b" not in result
        assert "Error" in result

    def test_no_ansi(self):
        text = "plain text"
        assert strip_ansi(text) == "plain text"


class TestCJKDetection:
    def test_contains_chinese(self):
        assert contains_cjk("你好世界") is True

    def test_contains_japanese(self):
        assert contains_cjk("こんにちは") is True

    def test_contains_korean(self):
        assert contains_cjk("안녕하세요") is True

    def test_no_cjk(self):
        assert contains_cjk("hello world") is False

    def test_mixed(self):
        assert contains_cjk("Hello 你好") is True
