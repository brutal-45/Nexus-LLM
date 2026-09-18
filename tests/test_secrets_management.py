"""Tests for nexus_llm.security.secrets module."""

import os
import pytest
from nexus_llm.security.secrets import SecretsManager


class TestSecretsManager:
    """Tests for the SecretsManager class."""

    def test_init(self):
        sm = SecretsManager()
        assert sm is not None

    def test_set_and_get(self):
        sm = SecretsManager()
        sm.set("api_key", "sk-12345")
        assert sm.get("api_key") == "sk-12345"

    def test_get_missing(self):
        sm = SecretsManager()
        assert sm.get("missing") is None

    def test_get_with_default(self):
        sm = SecretsManager()
        assert sm.get("missing", default="default") == "default"

    def test_has(self):
        sm = SecretsManager()
        sm.set("key1", "value1")
        assert sm.has("key1") is True
        assert sm.has("missing") is False

    def test_delete(self):
        sm = SecretsManager()
        sm.set("key1", "value1")
        assert sm.delete("key1") is True
        assert sm.has("key1") is False

    def test_delete_missing(self):
        sm = SecretsManager()
        assert sm.delete("missing") is False

    def test_list_keys(self):
        sm = SecretsManager()
        sm.set("key1", "value1")
        sm.set("key2", "value2")
        keys = sm.list_keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_require_existing(self):
        sm = SecretsManager()
        sm.set("api_key", "sk-12345")
        value = sm.require("api_key")
        assert value == "sk-12345"

    def test_require_missing(self):
        sm = SecretsManager()
        with pytest.raises(KeyError):
            sm.require("missing_key")

    def test_env_variable_override(self):
        sm = SecretsManager(env_prefix="TEST_NEXUS_")
        os.environ["TEST_NEXUS_API_KEY"] = "env-value"
        try:
            assert sm.get("api_key") == "env-value"
        finally:
            del os.environ["TEST_NEXUS_API_KEY"]

    def test_export_redacted(self):
        sm = SecretsManager()
        sm.set("api_key", "secret-value")
        exported = sm.export_redacted()
        assert exported["api_key"] == "***REDACTED***"
        assert "secret-value" not in str(exported)
