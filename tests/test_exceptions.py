"""Tests for custom exceptions (NexusLLMError, ModelNotFoundError, etc.)."""
import pytest

from nexus_llm.exceptions import (
    NexusLLMError,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    TokenizerError,
    ConfigError,
    TrainingError,
    ServerError,
    ChatError,
    PluginError,
)


class TestNexusLLMError:
    """Test base NexusLLMError."""

    def test_default_message(self):
        err = NexusLLMError()
        assert "An error occurred" in str(err)

    def test_custom_message(self):
        err = NexusLLMError(message="Custom error")
        assert str(err) == "Custom error"

    def test_error_code_in_str(self):
        err = NexusLLMError(message="Test", error_code="ERR001")
        assert "[ERR001]" in str(err)

    def test_details_in_str(self):
        err = NexusLLMError(message="Test", details={"key": "value"})
        assert "key" in str(err)

    def test_to_dict(self):
        err = NexusLLMError(message="Test", error_code="E1", details={"k": "v"})
        d = err.to_dict()
        assert d["error_type"] == "NexusLLMError"
        assert d["message"] == "Test"
        assert d["error_code"] == "E1"
        assert d["details"]["k"] == "v"

    def test_repr(self):
        err = NexusLLMError(message="Test", error_code="E1")
        r = repr(err)
        assert "NexusLLMError" in r
        assert "Test" in r

    def test_inherits_from_exception(self):
        assert issubclass(NexusLLMError, Exception)

    def test_default_details_empty(self):
        err = NexusLLMError()
        assert err.details == {}


class TestModelNotFoundError:
    """Test ModelNotFoundError."""

    def test_with_model_name(self):
        err = ModelNotFoundError("gpt2-medium")
        assert "gpt2-medium" in str(err)
        assert err.model_name == "gpt2-medium"

    def test_default_error_code(self):
        err = ModelNotFoundError("gpt2")
        assert err.error_code == "MODEL_NOT_FOUND"

    def test_details_contain_model_name(self):
        err = ModelNotFoundError("gpt2")
        assert err.details["model_name"] == "gpt2"

    def test_custom_message(self):
        err = ModelNotFoundError("gpt2", message="Not found!")
        assert "Not found!" in str(err)

    def test_inherits_from_nexus_llm_error(self):
        assert issubclass(ModelNotFoundError, NexusLLMError)


class TestModelLoadError:
    """Test ModelLoadError."""

    def test_with_model_name(self):
        err = ModelLoadError("llama-7b")
        assert "llama-7b" in str(err)

    def test_with_original_error(self):
        orig = RuntimeError("OOM")
        err = ModelLoadError("llama-7b", original_error=orig)
        assert "OOM" in str(err)
        assert err.original_error is orig

    def test_default_error_code(self):
        err = ModelLoadError("test")
        assert err.error_code == "MODEL_LOAD_ERROR"


class TestInferenceError:
    """Test InferenceError."""

    def test_with_model_name(self):
        err = InferenceError(model_name="gpt2")
        assert "gpt2" in str(err)

    def test_prompt_truncation(self):
        long_prompt = "x" * 300
        err = InferenceError(model_name="gpt2", prompt=long_prompt)
        assert len(err.prompt) == 200

    def test_default_error_code(self):
        err = InferenceError()
        assert err.error_code == "INFERENCE_ERROR"


class TestTokenizerError:
    """Test TokenizerError."""

    def test_with_tokenizer_name(self):
        err = TokenizerError(tokenizer_name="gpt2")
        assert "gpt2" in str(err)

    def test_default_error_code(self):
        err = TokenizerError()
        assert err.error_code == "TOKENIZER_ERROR"


class TestConfigError:
    """Test ConfigError."""

    def test_with_config_key(self):
        err = ConfigError(message="Bad config", config_key="port")
        assert err.config_key == "port"

    def test_with_config_source(self):
        err = ConfigError(message="Bad config", config_source="config.yaml")
        assert err.config_source == "config.yaml"

    def test_default_error_code(self):
        err = ConfigError()
        assert err.error_code == "CONFIG_ERROR"


class TestTrainingError:
    """Test TrainingError."""

    def test_with_step_and_epoch(self):
        err = TrainingError(message="Loss exploded", step=100, epoch=2)
        assert err.step == 100
        assert err.epoch == 2

    def test_default_error_code(self):
        err = TrainingError()
        assert err.error_code == "TRAINING_ERROR"


class TestServerError:
    """Test ServerError."""

    def test_with_endpoint(self):
        err = ServerError(message="Timeout", endpoint="/generate")
        assert err.endpoint == "/generate"

    def test_with_status_code(self):
        err = ServerError(message="Not found", status_code=404)
        assert err.status_code == 404

    def test_default_error_code(self):
        err = ServerError()
        assert err.error_code == "SERVER_ERROR"


class TestChatError:
    """Test ChatError."""

    def test_with_session_id(self):
        err = ChatError(message="Session error", session_id="abc123")
        assert err.session_id == "abc123"

    def test_default_error_code(self):
        err = ChatError()
        assert err.error_code == "CHAT_ERROR"


class TestPluginError:
    """Test PluginError."""

    def test_with_plugin_name(self):
        err = PluginError(plugin_name="weather")
        assert "weather" in str(err)

    def test_default_error_code(self):
        err = PluginError()
        assert err.error_code == "PLUGIN_ERROR"

    def test_details_contain_plugin_name(self):
        err = PluginError(plugin_name="calc")
        assert err.details["plugin_name"] == "calc"
