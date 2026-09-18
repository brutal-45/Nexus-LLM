"""Tests for nexus_llm.security.key_manager module."""

import pytest
from nexus_llm.security.key_manager import KeyManager


class TestKeyManager:
    """Tests for the KeyManager class."""

    def test_init(self):
        km = KeyManager()
        assert km is not None

    def test_generate_key(self):
        km = KeyManager()
        key = km.generate_key("test_key")
        assert isinstance(key, str)
        assert len(key) > 0

    def test_get_key(self):
        km = KeyManager()
        km.generate_key("test_key")
        key = km.get_key("test_key")
        assert key is not None

    def test_get_missing_key(self):
        km = KeyManager()
        key = km.get_key("nonexistent")
        assert key is None

    def test_rotate_key(self):
        km = KeyManager()
        old_key = km.generate_key("test_key")
        new_key = km.rotate_key("test_key")
        assert new_key != old_key

    def test_delete_key(self):
        km = KeyManager()
        km.generate_key("test_key")
        km.delete_key("test_key")
        assert km.get_key("test_key") is None

    def test_list_keys(self):
        km = KeyManager()
        km.generate_key("key1")
        km.generate_key("key2")
        keys = km.list_keys()
        assert "key1" in keys
        assert "key2" in keys
