"""Tests for nexus_llm.security.input_sanitizer module."""

import pytest
from nexus_llm.security.input_sanitizer import InputSanitizer


class TestInputSanitizer:
    """Tests for the InputSanitizer class."""

    def test_init(self):
        sanitizer = InputSanitizer()
        assert sanitizer is not None

    def test_sanitize_html(self):
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("<script>alert('xss')</script>hello")
        assert "<script>" not in result
        assert "hello" in result

    def test_sanitize_sql_injection(self):
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("SELECT * FROM users; DROP TABLE users;")
        assert isinstance(result, str)

    def test_sanitize_path_traversal(self):
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("../../etc/passwd")
        assert ".." not in result

    def test_sanitize_empty(self):
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("")
        assert result == ""

    def test_sanitize_clean_text(self):
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("Hello, this is clean text.")
        assert result == "Hello, this is clean text."

    def test_validate_input(self):
        sanitizer = InputSanitizer()
        is_valid = sanitizer.validate("normal text")
        assert is_valid is True

    def test_validate_malicious_input(self):
        sanitizer = InputSanitizer()
        is_valid = sanitizer.validate("<script>alert('xss')</script>")
        assert is_valid is False
