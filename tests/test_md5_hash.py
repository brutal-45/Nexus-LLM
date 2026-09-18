"""Tests for MD5 hashing."""
import pytest
import hashlib


def test_md5_consistency():
    data = b"Hello, World!"
    h1 = hashlib.md5(data).hexdigest()
    h2 = hashlib.md5(data).hexdigest()
    assert h1 == h2

def test_md5_different_inputs():
    h1 = hashlib.md5(b"hello").hexdigest()
    h2 = hashlib.md5(b"world").hexdigest()
    assert h1 != h2

def test_md5_length():
    h = hashlib.md5(b"test").hexdigest()
    assert len(h) == 32  # MD5 hex digest is 32 chars

def test_md5_empty_input():
    h = hashlib.md5(b"").hexdigest()
    assert len(h) == 32

def test_md5_large_input():
    data = b"x" * 10_000_000
    h = hashlib.md5(data).hexdigest()
    assert len(h) == 32
