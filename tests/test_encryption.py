"""Tests for nexus_llm.security.encryption module."""

import pytest
from nexus_llm.security.encryption import EncryptionManager, encrypt_data, decrypt_data


class TestEncryptionManager:
    """Tests for the EncryptionManager class."""

    def test_init_default(self):
        mgr = EncryptionManager()
        assert mgr.algorithm is not None

    def test_init_with_key(self):
        key = EncryptionManager.generate_key(EncryptionManager())
        mgr = EncryptionManager(key=key)
        assert mgr.algorithm is not None

    def test_encrypt_decrypt(self):
        mgr = EncryptionManager()
        plaintext = "Hello, World!"
        ciphertext = mgr.encrypt(plaintext)
        assert ciphertext != plaintext
        decrypted = mgr.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_encrypt_empty(self):
        mgr = EncryptionManager()
        ciphertext = mgr.encrypt("")
        decrypted = mgr.decrypt(ciphertext)
        assert decrypted == ""

    def test_hash_sha256(self):
        mgr = EncryptionManager()
        h = mgr.hash("test data")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_sha512(self):
        mgr = EncryptionManager()
        h = mgr.hash("test data", algorithm="sha512")
        assert len(h) == 128

    def test_generate_key(self):
        mgr = EncryptionManager()
        key = mgr.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_export_key(self):
        mgr = EncryptionManager()
        exported = mgr.export_key()
        assert isinstance(exported, str)

    def test_from_key_string(self):
        mgr1 = EncryptionManager()
        key_str = mgr1.export_key()
        mgr2 = EncryptionManager.from_key_string(key_str)
        text = "round trip"
        assert mgr2.decrypt(mgr1.encrypt(text)) == text


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_encrypt_data(self):
        result = encrypt_data("secret")
        assert "ciphertext" in result
        assert "key" in result
        assert "algorithm" in result

    def test_decrypt_data(self):
        result = encrypt_data("secret")
        decrypted = decrypt_data(result["ciphertext"], result["key"])
        assert decrypted == "secret"
