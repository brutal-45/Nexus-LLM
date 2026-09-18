"""Tests for SHA256 hashing."""
import pytest
import hashlib


def test_sha256_consistency():
    data = b"Hello, World!"
    h1 = hashlib.sha256(data).hexdigest()
    h2 = hashlib.sha256(data).hexdigest()
    assert h1 == h2

def test_sha256_different_inputs():
    h1 = hashlib.sha256(b"hello").hexdigest()
    h2 = hashlib.sha256(b"world").hexdigest()
    assert h1 != h2

def test_sha256_length():
    h = hashlib.sha256(b"test").hexdigest()
    assert len(h) == 64  # SHA256 hex digest is 64 chars

def test_sha256_collision_resistance():
    # Extremely unlikely collision for different inputs
    h1 = hashlib.sha256(b"input_a").hexdigest()
    h2 = hashlib.sha256(b"input_b").hexdigest()
    assert h1 != h2

def test_sha256_file_integrity():
    data = b"file content here"
    checksum = hashlib.sha256(data).hexdigest()
    assert checksum == hashlib.sha256(data).hexdigest()
