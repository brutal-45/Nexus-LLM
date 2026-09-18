"""Test hashing utilities for Nexus-LLM."""
import hashlib
import pytest
from typing import Optional


def hash_string(text: str, algorithm: str = "sha256") -> str:
    if algorithm not in ("md5", "sha1", "sha256", "sha512"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def hash_bytes(data: bytes, algorithm: str = "sha256") -> str:
    if algorithm not in ("md5", "sha1", "sha256", "sha512"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


def hash_file(path: str, algorithm: str = "sha256") -> str:
    if algorithm not in ("md5", "sha1", "sha256", "sha512"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def consistent_hash(text: str, max_value: int = 2**32) -> int:
    h = hashlib.md5(text.encode("utf-8"))
    return int(h.hexdigest()[:8], 16) % max_value


def fingerprint_dict(data: dict) -> str:
    import json
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hash_string(serialized)


class TestHashString:
    def test_sha256(self):
        result = hash_string("hello")
        assert len(result) == 64
        assert result == hashlib.sha256(b"hello").hexdigest()

    def test_md5(self):
        result = hash_string("hello", algorithm="md5")
        assert len(result) == 32

    def test_sha1(self):
        result = hash_string("hello", algorithm="sha1")
        assert len(result) == 40

    def test_sha512(self):
        result = hash_string("hello", algorithm="sha512")
        assert len(result) == 128

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported"):
            hash_string("hello", algorithm="blake2")

    def test_deterministic(self):
        assert hash_string("test") == hash_string("test")

    def test_different_inputs(self):
        assert hash_string("a") != hash_string("b")

    def test_empty_string(self):
        result = hash_string("")
        assert isinstance(result, str)
        assert len(result) == 64


class TestHashBytes:
    def test_sha256_bytes(self):
        result = hash_bytes(b"hello")
        assert result == hash_string("hello")

    def test_deterministic(self):
        assert hash_bytes(b"data") == hash_bytes(b"data")

    def test_binary_data(self):
        result = hash_bytes(b"\x00\x01\x02\xff")
        assert isinstance(result, str)


class TestHashFile:
    def test_hash_file(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("hello world")
        result = hash_file(str(f))
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert result == expected

    def test_hash_empty_file(self, tmp_dir):
        f = tmp_dir / "empty.txt"
        f.write_text("")
        result = hash_file(str(f))
        assert len(result) == 64

    def test_hash_large_file(self, tmp_dir):
        f = tmp_dir / "large.bin"
        f.write_bytes(b"x" * 100000)
        result = hash_file(str(f))
        assert len(result) == 64

    def test_hash_file_md5(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("test")
        result = hash_file(str(f), algorithm="md5")
        assert len(result) == 32


class TestConsistentHash:
    def test_returns_int(self):
        result = consistent_hash("test")
        assert isinstance(result, int)

    def test_within_range(self):
        result = consistent_hash("test", max_value=100)
        assert 0 <= result < 100

    def test_deterministic(self):
        assert consistent_hash("key1") == consistent_hash("key1")

    def test_different_keys(self):
        assert consistent_hash("key1") != consistent_hash("key2")

    def test_distribution(self):
        buckets = [0] * 10
        for i in range(1000):
            idx = consistent_hash(f"item_{i}", max_value=10)
            buckets[idx] += 1
        for count in buckets:
            assert count > 0


class TestFingerprintDict:
    def test_deterministic(self):
        data = {"a": 1, "b": 2}
        assert fingerprint_dict(data) == fingerprint_dict(data)

    def test_key_order_independent(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert fingerprint_dict(d1) == fingerprint_dict(d2)

    def test_different_values(self):
        d1 = {"a": 1}
        d2 = {"a": 2}
        assert fingerprint_dict(d1) != fingerprint_dict(d2)

    def test_nested_dict(self):
        data = {"outer": {"inner": "value"}}
        result = fingerprint_dict(data)
        assert isinstance(result, str)
        assert len(result) == 64
