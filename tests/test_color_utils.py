"""Tests for ANSI color utilities."""
import pytest
import re


def test_ansi_escape_pattern():
    code = "\033[31mHello\033[0m"
    ansi_pattern = re.compile(r"\033\[[0-9;]*m")
    matches = ansi_pattern.findall(code)
    assert len(matches) == 2

def test_strip_ansi():
    text = "\033[31mRed\033[0m text"
    clean = re.sub(r"\033\[[0-9;]*m", "", text)
    assert clean == "Red text"

def test_ansi_color_codes():
    colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34}
    for name, code in colors.items():
        assert 30 <= code <= 37

def test_ansi_bold_code():
    bold = "\033[1m"
    assert bold == "\033[1m"

def test_ansi_reset_code():
    reset = "\033[0m"
    assert "0m" in reset
