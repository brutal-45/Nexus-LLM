"""Tests for the FastAPI server endpoints.

Uses TestClient with a mocked InferenceEngine so no real models are needed.
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

from nexus_llm.backend.server import create_app
from nexus_llm.core.exceptions import ModelNotFoundError, ModelLoadError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    """Create a mock InferenceEngine."""
    engine = MagicMock()

    # Model manager mock
    engine.model_manager = MagicMock()
    engine.model_manager.is_loaded = False
    engine.model_manager.state = MagicMock(value="unloaded")
    engine.model_manager.model_id = None
    engine.model_manager.get_info.return_value = {
        "state": "unloaded",
        "model_id": None,
        "device": None,
        "precision": None,
    }
    engine.model_manager.get_memory_usage.return_value = {
        "model_device": None,
        "cpu": {"note": "psutil not installed"},
        "ram": {"note": "psutil not installed"},
    }

    # Tokenizer manager mock
    engine.tokenizer_manager = MagicMock()
    engine.tokenizer_manager.is_loaded = False

    # is_ready depends on both being loaded
    engine.is_ready = False

    # Stats
    engine.get_stats.return_value = {
        "generation_count": 0,
        "total_tokens": 0,
        "total_time_s": 0.0,
        "avg_tokens_per_second": 0.0,
    }

    return engine


@pytest.fixture
def app_with_mock(mock_engine):
    """Create a FastAPI app with a mocked engine."""
    # Patch InferenceEngine constructor to return our mock
    with patch("nexus_llm.backend.server.InferenceEngine", return_value=mock_engine):
        app = create_app()
    return app, mock_engine


@pytest.fixture
def client(app_with_mock):
    """Create a TestClient with the mocked app."""
    app, engine = app_with_mock
    return TestClient(app), engine


@pytest.fixture
def loaded_client(client):
    """Create a TestClient where the model is 'loaded'."""
    test_client, engine = client
    engine.model_manager.is_loaded = True
    engine.model_manager.state = MagicMock(value="loaded")
    engine.model_manager.model_id = "gpt2-medium"
    engine.model_manager.get_info.return_value = {
        "state": "loaded",
        "model_id": "gpt2-medium",
        "device": "cpu",
        "precision": "fp32",
        "model_info": {
            "name": "GPT-2 Medium",
            "hf_id": "openai-community/gpt2-medium",
            "category": "gpt2",
            "size": "355M",
            "params": "355M",
            "model_type": "causal",
            "recommended": True,
            "min_ram_gb": 4,
        },
    }
    engine.tokenizer_manager.is_loaded = True
    engine.is_ready = True
    return test_client, engine


# ---------------------------------------------------------------------------
# Health / Root
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for the health and root endpoints."""

    def test_root(self, client):
        test_client, _ = client
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "Nexus-LLM" in data["message"]

    def test_health_no_model_loaded(self, client):
        test_client, engine = client
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False

    def test_health_model_loaded(self, loaded_client):
        test_client, engine = loaded_client
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["model_id"] == "gpt2-medium"


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""

    def test_model_info_not_loaded(self, client):
        test_client, _ = client
        response = test_client.get("/model/info")
        assert response.status_code == 400

    def test_model_info_loaded(self, loaded_client):
        test_client, engine = loaded_client
        response = test_client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "gpt2-medium"
        assert data["state"] == "loaded"


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    """Tests for the /generate endpoint."""

    def test_generate_no_model(self, client):
        test_client, engine = client
        response = test_client.post("/generate", json={
            "prompt": "Hello world",
        })
        assert response.status_code == 400

    def test_generate_success(self, loaded_client):
        test_client, engine = loaded_client
        engine.generate.return_value = {
            "text": "Hello! How can I help?",
            "prompt_tokens": 3,
            "generated_tokens": 5,
            "total_tokens": 8,
            "generation_time_s": 0.5,
        }

        # We need to mock asyncio.to_thread to call the function directly
        with patch("nexus_llm.backend.server.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)):
            response = test_client.post("/generate", json={
                "prompt": "Hello world",
                "max_new_tokens": 50,
            })

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "Hello! How can I help?"

    def test_generate_with_all_params(self, loaded_client):
        test_client, engine = loaded_client
        engine.generate.return_value = {
            "text": "Generated text",
            "prompt_tokens": 2,
            "generated_tokens": 3,
            "total_tokens": 5,
            "generation_time_s": 0.2,
        }

        with patch("nexus_llm.backend.server.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)):
            response = test_client.post("/generate", json={
                "prompt": "Test",
                "max_new_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 40,
                "repetition_penalty": 1.2,
                "num_beams": 2,
                "do_sample": True,
            })

        assert response.status_code == 200

    def test_generate_empty_prompt_rejected(self, loaded_client):
        test_client, engine = loaded_client
        response = test_client.post("/generate", json={
            "prompt": "",
        })
        # Pydantic validation should reject empty string (min_length=1)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    def test_chat_no_model(self, client):
        test_client, engine = client
        response = test_client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code == 400

    def test_chat_success(self, loaded_client):
        test_client, engine = loaded_client
        engine.chat.return_value = {
            "text": "Hi there!",
            "prompt_tokens": 5,
            "generated_tokens": 3,
            "total_tokens": 8,
            "generation_time_s": 0.3,
            "messages_in": 1,
        }

        with patch("nexus_llm.backend.server.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)):
            response = test_client.post("/chat", json={
                "messages": [{"role": "user", "content": "Hello"}],
            })

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hi there!"
        assert data["messages_in"] == 1

    def test_chat_empty_messages_rejected(self, loaded_client):
        test_client, engine = loaded_client
        response = test_client.post("/chat", json={
            "messages": [],
        })
        assert response.status_code == 422

    def test_chat_inference_error(self, loaded_client):
        test_client, engine = loaded_client
        from nexus_llm.core.exceptions import InferenceError
        engine.chat.side_effect = InferenceError("OOM")

        with patch("nexus_llm.backend.server.asyncio.to_thread", side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)):
            response = test_client.post("/chat", json={
                "messages": [{"role": "user", "content": "Hello"}],
            })

        assert response.status_code == 500


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

class TestStatsEndpoint:
    """Tests for the /stats endpoint."""

    def test_stats(self, client):
        test_client, engine = client
        response = test_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "generation_count" in data
        assert "total_tokens" in data
        assert "avg_tokens_per_second" in data

    def test_stats_after_generation(self, loaded_client):
        test_client, engine = loaded_client
        engine.get_stats.return_value = {
            "generation_count": 1,
            "total_tokens": 50,
            "total_time_s": 1.5,
            "avg_tokens_per_second": 33.33,
        }
        response = test_client.get("/stats")
        data = response.json()
        assert data["generation_count"] == 1
        assert data["total_tokens"] == 50
