"""Tests for file I/O operations."""
import pytest
from pathlib import Path


def test_file_write_read(tmp_dir):
    p = tmp_dir / "test.txt"
    p.write_text("Hello, World!")
    assert p.read_text() == "Hello, World!"

def test_file_exists(tmp_dir):
    p = tmp_dir / "exists.txt"
    assert not p.exists()
    p.write_text("")
    assert p.exists()

def test_file_append(tmp_dir):
    p = tmp_dir / "append.txt"
    p.write_text("line1\n")
    with open(p, "a") as f:
        f.write("line2\n")
    assert p.read_text() == "line1\nline2\n"

def test_file_binary_io(tmp_dir):
    p = tmp_dir / "binary.bin"
    data = bytes(range(256))
    p.write_bytes(data)
    assert p.read_bytes() == data

def test_file_size(tmp_dir):
    p = tmp_dir / "size.txt"
    p.write_text("A" * 1000)
    assert p.stat().st_size == 1000
