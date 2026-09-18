"""Tests for custom exceptions module."""

import pytest

from nexus_llm.core.exceptions import (
    NexusLLMError,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    ConfigurationError,
    TrainingError,
    ServerError,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    """Tests that all custom exceptions inherit from NexusLLMError."""

    def test_model_not_found_is_nexus_error(self):
        assert issubclass(ModelNotFoundError, NexusLLMError)

    def test_model_load_error_is_nexus_error(self):
        assert issubclass(ModelLoadError, NexusLLMError)

    def test_inference_error_is_nexus_error(self):
        assert issubclass(InferenceError, NexusLLMError)

    def test_configuration_error_is_nexus_error(self):
        assert issubclass(ConfigurationError, NexusLLMError)

    def test_training_error_is_nexus_error(self):
        assert issubclass(TrainingError, NexusLLMError)

    def test_server_error_is_nexus_error(self):
        assert issubclass(ServerError, NexusLLMError)

    def test_nexus_error_is_exception(self):
        assert issubclass(NexusLLMError, Exception)

    def test_all_exceptions_are_distinct(self):
        """Each exception class is a distinct type."""
        classes = [
            ModelNotFoundError,
            ModelLoadError,
            InferenceError,
            ConfigurationError,
            TrainingError,
            ServerError,
        ]
        for i, cls_a in enumerate(classes):
            for j, cls_b in enumerate(classes):
                if i != j:
                    assert cls_a is not cls_b


# ---------------------------------------------------------------------------
# Exception messages
# ---------------------------------------------------------------------------

class TestExceptionMessages:
    """Tests that exceptions preserve custom messages."""

    def test_nexus_error_message(self):
        msg = "base error occurred"
        err = NexusLLMError(msg)
        assert str(err) == msg

    def test_model_not_found_message(self):
        msg = "Model 'xyz' not found"
        err = ModelNotFoundError(msg)
        assert str(err) == msg

    def test_model_load_error_message(self):
        msg = "Failed to load model 'gpt2'"
        err = ModelLoadError(msg)
        assert str(err) == msg

    def test_inference_error_message(self):
        msg = "Inference failed: OOM"
        err = InferenceError(msg)
        assert str(err) == msg

    def test_configuration_error_message(self):
        msg = "Invalid config key"
        err = ConfigurationError(msg)
        assert str(err) == msg

    def test_training_error_message(self):
        msg = "Training diverged"
        err = TrainingError(msg)
        assert str(err) == msg

    def test_server_error_message(self):
        msg = "Port already in use"
        err = ServerError(msg)
        assert str(err) == msg

    def test_empty_message(self):
        err = NexusLLMError()
        assert str(err) == ""

    def test_multiline_message(self):
        msg = "Line 1\nLine 2\nLine 3"
        err = ModelLoadError(msg)
        assert str(err) == msg


# ---------------------------------------------------------------------------
# Catching with base class
# ---------------------------------------------------------------------------

class TestCatchingWithBaseClass:
    """Tests that specific exceptions can be caught with NexusLLMError."""

    def test_catch_model_not_found_as_base(self):
        with pytest.raises(NexusLLMError):
            raise ModelNotFoundError("not found")

    def test_catch_model_load_error_as_base(self):
        with pytest.raises(NexusLLMError):
            raise ModelLoadError("load failed")

    def test_catch_inference_error_as_base(self):
        with pytest.raises(NexusLLMError):
            raise InferenceError("inference failed")

    def test_catch_configuration_error_as_base(self):
        with pytest.raises(NexusLLMError):
            raise ConfigurationError("config error")

    def test_catch_training_error_as_base(self):
        with pytest.raises(NexusLLMError):
            raise TrainingError("training failed")

    def test_catch_server_error_as_base(self):
        with pytest.raises(NexusLLMError):
            raise ServerError("server error")

    def test_catch_specific_not_base(self):
        """Catching NexusLLMError does NOT catch builtin exceptions."""
        with pytest.raises(ValueError):
            try:
                raise ValueError("not a nexus error")
            except NexusLLMError:
                pass  # Should not catch

    def test_try_except_specific_then_base(self):
        """Can catch specific, then fall through to base."""
        caught_specific = False
        caught_base = False
        try:
            raise ModelNotFoundError("test")
        except ModelNotFoundError:
            caught_specific = True
        except NexusLLMError:
            caught_base = True
        assert caught_specific is True
        assert caught_base is False

    def test_try_except_base_catches_all_subclasses(self):
        """A single except NexusLLMError catches any subclass."""
        for exc_cls in [ModelNotFoundError, ModelLoadError, InferenceError,
                        ConfigurationError, TrainingError, ServerError]:
            caught = False
            try:
                raise exc_cls("test")
            except NexusLLMError:
                caught = True
            assert caught, f"Failed to catch {exc_cls.__name__} as NexusLLMError"

    def test_exception_chaining(self):
        """Exceptions can be chained with 'from'."""
        original = ValueError("original")
        try:
            try:
                raise original
            except ValueError as e:
                raise ModelLoadError("wrapped") from e
        except ModelLoadError as err:
            assert err.__cause__ is original
