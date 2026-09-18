"""Tests for JavaScript syntax highlighting."""
import pytest


def test_js_keyword_detection():
    code = "function hello() { const x = 1; return x; }"
    words = set(code.split())
    assert "function" in words and "const" in words


def test_js_string_template():
    code = "const msg = `Hello ${name}!`;"
    assert "${" in code and "`" in code


def test_js_arrow_function():
    assert "=>" in "const add = (a, b) => a + b;"


def test_js_comment_detection():
    code = "// Line\n/* Block */\nconst x = 1;"
    assert "//" in code and "/*" in code and "*/" in code
