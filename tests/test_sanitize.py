"""Test input sanitization utilities for Nexus-LLM."""
import re
import html
import pytest
from typing import Optional


def sanitize_string(value: str) -> str:
    return value.strip()


def sanitize_html(value: str) -> str:
    return html.escape(value)


def remove_sql_injection(value: str) -> str:
    dangerous_patterns = [
        r";\s*DROP\s+TABLE",
        r";\s*DELETE\s+FROM",
        r";\s*INSERT\s+INTO",
        r";\s*UPDATE\s+\w+\s+SET",
        r"--\s*$",
        r"/\*.*\*/",
        r"UNION\s+SELECT",
        r"OR\s+1\s*=\s*1",
    ]
    result = value
    for pattern in dangerous_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    filename = filename.strip('. ')
    if not filename:
        filename = "unnamed"
    return filename


def sanitize_path(path: str) -> str:
    path = path.replace("..", "")
    path = re.sub(r'[<>|?*]', '', path)
    return path


def sanitize_prompt(text: str) -> str:
    text = text.strip()
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\x00', '', text)
    return text


def remove_control_chars(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def truncate_safe(text: str, max_length: int = 10000) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length]


def sanitize_headers(headers: dict) -> dict:
    sanitized = {}
    for key, value in headers.items():
        clean_key = re.sub(r'[\r\n]', '', str(key))
        clean_value = re.sub(r'[\r\n]', '', str(value))
        sanitized[clean_key] = clean_value
    return sanitized


def normalize_unicode(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", text)


class TestSanitizeString:
    def test_strips_whitespace(self):
        assert sanitize_string("  hello  ") == "hello"

    def test_tabs_and_newlines(self):
        assert sanitize_string("\thello\n") == "hello"

    def test_empty_string(self):
        assert sanitize_string("") == ""


class TestSanitizeHtml:
    def test_escapes_angle_brackets(self):
        result = sanitize_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;" in result

    def test_escapes_ampersand(self):
        assert sanitize_html("a & b") == "a &amp; b"

    def test_escapes_quotes(self):
        result = sanitize_html('value="test"')
        assert "&quot;" in result

    def test_plain_text_unchanged(self):
        assert sanitize_html("hello world") == "hello world"


class TestRemoveSqlInjection:
    def test_removes_drop_table(self):
        result = remove_sql_injection("'; DROP TABLE users; --")
        assert "DROP TABLE" not in result.upper() or ";" not in result

    def test_removes_union_select(self):
        result = remove_sql_injection("' UNION SELECT * FROM users")
        assert "UNION SELECT" not in result.upper()

    def test_removes_or_1_equals_1(self):
        result = remove_sql_injection("' OR 1=1 --")
        assert "1=1" not in result and "1 = 1" not in result

    def test_normal_text_unchanged(self):
        text = "This is a normal query"
        assert remove_sql_injection(text) == text

    def test_removes_comment(self):
        result = remove_sql_injection("value -- comment")
        assert "--" not in result


class TestSanitizeFilename:
    def test_removes_special_chars(self):
        result = sanitize_filename('file<>:"/\\|?*name.txt')
        assert "<" not in result
        assert ">" not in result

    def test_removes_leading_dots(self):
        result = sanitize_filename("..hidden")
        assert not result.startswith(".")

    def test_empty_becomes_unnamed(self):
        assert sanitize_filename("") == "unnamed"

    def test_normal_filename(self):
        assert sanitize_filename("document.pdf") == "document.pdf"

    def test_removes_control_chars(self):
        result = sanitize_filename("file\x00name")
        assert "\x00" not in result


class TestSanitizePath:
    def test_removes_parent_dir(self):
        result = sanitize_path("../../etc/passwd")
        assert ".." not in result

    def test_removes_special_chars(self):
        result = sanitize_path("/path/<>file")
        assert "<" not in result

    def test_normal_path(self):
        result = sanitize_path("/home/user/file.txt")
        assert "home" in result


class TestSanitizePrompt:
    def test_removes_special_tokens(self):
        result = sanitize_prompt("Hello <|endoftext|> world")
        assert "<|endoftext|>" not in result

    def test_removes_null_bytes(self):
        result = sanitize_prompt("Hello\x00World")
        assert "\x00" not in result

    def test_strips_whitespace(self):
        assert sanitize_prompt("  hello  ") == "hello"

    def test_normal_text(self):
        text = "What is machine learning?"
        assert sanitize_prompt(text) == text


class TestRemoveControlChars:
    def test_removes_control_chars(self):
        result = remove_control_chars("hello\x00\x01\x02world")
        assert result == "helloworld"

    def test_preserves_newlines(self):
        text = "line1\nline2"
        assert remove_control_chars(text) == "line1\nline2"

    def test_preserves_tabs(self):
        text = "hello\tworld"
        assert remove_control_chars(text) == "hello\tworld"

    def test_no_control_chars(self):
        text = "clean text"
        assert remove_control_chars(text) == "clean text"


class TestTruncateSafe:
    def test_short_text_unchanged(self):
        assert truncate_safe("hello", 100) == "hello"

    def test_long_text_truncated(self):
        text = "a" * 20000
        result = truncate_safe(text, 10000)
        assert len(result) == 10000

    def test_exact_length(self):
        text = "a" * 100
        assert truncate_safe(text, 100) == text


class TestSanitizeHeaders:
    def test_removes_crlf(self):
        headers = {"X-Custom": "value\r\nX-Injected: malicious"}
        result = sanitize_headers(headers)
        assert "\r\n" not in result["X-Custom"]

    def test_cleans_keys(self):
        headers = {"X-Key\r\n": "value"}
        result = sanitize_headers(headers)
        assert "\r" not in list(result.keys())[0]

    def test_normal_headers(self):
        headers = {"Content-Type": "application/json"}
        result = sanitize_headers(headers)
        assert result == headers


class TestNormalizeUnicode:
    def test_normalize_nfc(self):
        text = "café"
        result = normalize_unicode(text)
        assert isinstance(result, str)
        assert "café" in result

    def test_already_normalized(self):
        text = "hello"
        assert normalize_unicode(text) == text
