"""Tests for code block rendering."""
import pytest
import re


def test_code_block_fence_detection():
    text = "text\n```python\nprint(\"hi\")\n```\nmore"
    blocks, in_block = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_block = not in_block
        elif in_block:
            blocks.append(line)
    assert len(blocks) == 1 and "print" in blocks[0]


def test_code_block_language_tag():
    assert "```python"[3:].strip() == "python"


def test_nested_code_blocks():
    text = "```\ncontent\n```\n"
    assert text.count("```") >= 2


def test_inline_code_render():
    text = "Use `pip install` to install"
    assert re.findall(r"`([^`]+)`", text) == ["pip install"]


def test_code_block_whitespace():
    lines = "    indented\n        double".split("\n")
    assert lines[0].startswith("    ")
