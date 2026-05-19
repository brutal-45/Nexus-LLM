"""Test crypto utilities for Nexus-LLM."""
import hashlib
import hmac
import os
import base64
import pytest


# --- Crypto utility implementations to test ---

def generate_random_bytes(length: int = 32) -> bytes:
    if length < 1:
        raise ValueError("Length must be positive")
    return os.urandom(length)


def generate_token(length: int = 32) -> str:
    raw = generate_random_bytes(length)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def hash_sha256(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
    return hashlib.sha256(data).hexdigest()


def hash_sha512(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
    return hashlib.sha512(data).hexdigest()


def hash_md5(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
    return hashlib.md5(data).hexdigest()


def hmac_sha256(key: bytes, message: bytes) -> str:
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def derive_key(password: str, salt: bytes, iterations: int = 100000, key_length: int = 32) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen=key_length)


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_base64(data: str) -> bytes:
    return base64.b64decode(data)


def generate_api_key(prefix: str = "nx", length: int = 32) -> str:
    token = generate_token(length)
    return f"{prefix}_{token}"


class TestRandomBytes:
    def test_generates_correct_length(self):
        result = generate_random_bytes(16)
        assert len(result) == 16

    def test_default_length(self):
        result = generate_random_bytes()
        assert len(result) == 32

    def test_different_each_call(self):
        a = generate_random_bytes(16)
        b = generate_random_bytes(16)
        assert a != b

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="positive"):
            generate_random_bytes(0)

    def test_returns_bytes(self):
        result = generate_random_bytes()
        assert isinstance(result, bytes)


class TestTokenGeneration:
    def test_default_length(self):
        token = generate_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_custom_length(self):
        token = generate_token(16)
        assert isinstance(token, str)

    def test_url_safe(self):
        token = generate_token()
        safe_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
        assert all(c in safe_chars for c in token)

    def test_uniqueness(self):
        tokens = {generate_token() for _ in range(100)}
        assert len(tokens) == 100


class TestHashing:
    def test_sha256(self):
        result = hash_sha256(b"hello")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_deterministic(self):
        assert hash_sha256(b"hello") == hash_sha256(b"hello")

    def test_sha256_different_inputs(self):
        assert hash_sha256(b"hello") != hash_sha256(b"world")

    def test_sha512(self):
        result = hash_sha512(b"hello")
        assert len(result) == 128

    def test_md5(self):
        result = hash_md5(b"hello")
        assert len(result) == 32

    def test_md5_known_value(self):
        assert hash_md5(b"hello") == "5d41402abc4b2a76b9719d911017c592"

    def test_non_bytes_raises(self):
        with pytest.raises(TypeError, match="bytes"):
            hash_sha256("hello")


class TestHMAC:
    def test_hmac_sha256(self):
        result = hmac_sha256(b"key", b"message")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hmac_deterministic(self):
        assert hmac_sha256(b"key", b"msg") == hmac_sha256(b"key", b"msg")

    def test_different_keys_different_hmac(self):
        assert hmac_sha256(b"key1", b"msg") != hmac_sha256(b"key2", b"msg")

    def test_different_messages_different_hmac(self):
        assert hmac_sha256(b"key", b"msg1") != hmac_sha256(b"key", b"msg2")


class TestConstantTimeCompare:
    def test_equal_strings(self):
        assert constant_time_compare("abc", "abc") is True

    def test_different_strings(self):
        assert constant_time_compare("abc", "def") is False

    def test_different_lengths(self):
        assert constant_time_compare("abc", "abcd") is False

    def test_empty_strings(self):
        assert constant_time_compare("", "") is True


class TestKeyDerivation:
    def test_derive_key_length(self):
        salt = os.urandom(16)
        key = derive_key("password", salt, key_length=32)
        assert len(key) == 32

    def test_derive_key_custom_length(self):
        salt = os.urandom(16)
        key = derive_key("password", salt, key_length=64)
        assert len(key) == 64

    def test_derive_key_deterministic(self):
        salt = b"fixed_salt_12345"
        key1 = derive_key("password", salt, iterations=100)
        key2 = derive_key("password", salt, iterations=100)
        assert key1 == key2

    def test_different_salt_different_key(self):
        key1 = derive_key("password", os.urandom(16), iterations=100)
        key2 = derive_key("password", os.urandom(16), iterations=100)
        assert key1 != key2


class TestBase64:
    def test_encode_decode_roundtrip(self):
        original = b"hello world"
        encoded = encode_base64(original)
        decoded = decode_base64(encoded)
        assert decoded == original

    def test_encode_returns_string(self):
        result = encode_base64(b"data")
        assert isinstance(result, str)

    def test_decode_returns_bytes(self):
        result = decode_base64("aGVsbG8=")
        assert isinstance(result, bytes)
        assert result == b"hello"


class TestAPIKey:
    def test_has_prefix(self):
        key = generate_api_key("nx")
        assert key.startswith("nx_")

    def test_custom_prefix(self):
        key = generate_api_key("test")
        assert key.startswith("test_")

    def test_uniqueness(self):
        keys = {generate_api_key() for _ in range(50)}
        assert len(keys) == 50

    def test_default_prefix(self):
        key = generate_api_key()
        assert key.startswith("nx_")
