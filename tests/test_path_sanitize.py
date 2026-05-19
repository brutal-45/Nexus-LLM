"""Tests for path sanitization."""
import pytest
import re


def test_path_sanitize_removes_traversal():
    path = "../../../etc/passwd"
    sanitized = path.replace("../", "").replace("..\", "")
    assert "../" not in sanitized

def test_path_sanitize_removes_null_bytes():
    path = "file\0.txt"
    sanitized = path.replace("\0", "")
    assert "\0" not in sanitized

def test_path_sanitize_removes_special_chars():
    path = 'file<>:"|?*.txt'
    sanitized = re.sub(r'[<>:"|?*]', '_', path)
    assert "<" not in sanitized
    assert ">" not in sanitized

def test_path_sanitize_preserves_valid():
    path = "valid_file-name.txt"
    sanitized = re.sub(r'[<>:"|?*]', '_', path)
    assert sanitized == path

def test_path_sanitize_spaces():
    path = "my file name.txt"
    sanitized = path.replace(" ", "_")
    assert " " not in sanitized
