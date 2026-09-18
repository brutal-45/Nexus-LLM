"""Tests for ANSI stripping."""
import pytest
import re


def test_ansi_strip_basic():
    text = "\033[31mRed\033[0m text"
    clean = re.sub(r"\033\[[0-9;]*m", "", text)
    assert clean == "Red text"

def test_ansi_strip_multiple():
    text = "\033[1m\033[31mBold Red\033[0m"
    clean = re.sub(r"\033\[[0-9;]*m", "", text)
    assert clean == "Bold Red"

def test_ansi_strip_no_codes():
    text = "Plain text"
    clean = re.sub(r"\033\[[0-9;]*m", "", text)
    assert clean == "Plain text"

def test_ansi_strip_complex():
    text = "\033[38;5;196mCustom\033[0m"
    clean = re.sub(r"\033\[[0-9;]*m", "", text)
    assert clean == "Custom"
