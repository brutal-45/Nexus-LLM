"""Tests for nexus_llm.client.sdk module."""

import pytest
from unittest.mock import MagicMock, patch
from nexus_llm.client.sdk import NexusSDK


class TestNexusSDK:
    """Tests for the NexusSDK class."""

    def test_init(self):
        sdk = NexusSDK(base_url="http://localhost:8000")
        assert sdk is not None

    def test_default_model(self):
        sdk = NexusSDK(default_model="gpt-4")
        assert sdk.default_model == "gpt-4"

    def test_set_default_model(self):
        sdk = NexusSDK()
        sdk.default_model = "gpt-4"
        assert sdk.default_model == "gpt-4"

    def test_chat(self):
        sdk = NexusSDK()
        with patch.object(sdk._client, "chat", return_value={"choices": []}):
            result = sdk.chat("Hello")
            assert "choices" in result

    def test_complete(self):
        sdk = NexusSDK()
        with patch.object(sdk._client, "complete", return_value={"choices": []}):
            result = sdk.complete("Once upon a time")
            assert "choices" in result

    def test_embed(self):
        sdk = NexusSDK()
        with patch.object(sdk._client, "embed", return_value={"data": []}):
            result = sdk.embed(["hello"])
            assert "data" in result

    def test_models(self):
        sdk = NexusSDK()
        with patch.object(sdk._client, "list_models", return_value={"data": []}):
            result = sdk.models()
            assert "data" in result

    def test_health(self):
        sdk = NexusSDK()
        with patch.object(sdk._client, "health", return_value={"status": "ok"}):
            result = sdk.health()
            assert result["status"] == "ok"

    def test_extract_text(self):
        sdk = NexusSDK()
        response = {"choices": [{"message": {"content": "Hello!"}}]}
        text = sdk.extract_text(response)
        assert text == "Hello!"

    def test_extract_text_empty(self):
        sdk = NexusSDK()
        text = sdk.extract_text({})
        assert text == ""

    def test_extract_embedding(self):
        sdk = NexusSDK()
        response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        emb = sdk.extract_embedding(response)
        assert emb == [0.1, 0.2, 0.3]
