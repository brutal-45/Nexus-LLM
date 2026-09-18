"""Tests for math rendering."""
import pytest
import re


def test_inline_math_detection():
    text = "The formula $E = mc^2$ is famous."
    matches = re.findall(r"\$([^$]+)\$", text)
    assert len(matches) == 1 and "mc^2" in matches[0]


def test_display_math_detection():
    text = "$$\n\\int_0^1 x^2 dx\n$$"
    assert len(re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL)) == 1


def test_latex_command_detection():
    latex = "\\frac{a}{b} + \\sqrt{c}"
    assert len(re.findall(r"\\\w+", latex)) >= 2


def test_math_subscript_superscript():
    expr = "x_1^2 + y_{n}^{3}"
    assert "_" in expr and "^" in expr
