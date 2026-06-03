"""Test error handling for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import Optional


class NexusError(Exception):
    """Base error for Nexus-LLM."""
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(message)


class ModelNotFoundError(NexusError):
    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found", code=404)
        self.model_name = model_name


class InvalidRequestError(NexusError):
    def __init__(self, message: str):
        super().__init__(message, code=400)


class AuthenticationError(NexusError):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code=401)


class RateLimitError(NexusError):
    def __init__(self, retry_after: int = 60):
        super().__init__("Rate limit exceeded", code=429)
        self.retry_after = retry_after


class ModelLoadError(NexusError):
    def __init__(self, model_name: str, reason: str = ""):
        msg = f"Failed to load model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, code=500)
        self.model_name = model_name


class SafetyError(NexusError):
    def __init__(self, message: str = "Content blocked by safety filter"):
        super().__init__(message, code=403)


@dataclass
class ErrorResponse:
    error: str
    code: int
    details: Optional[str] = None

    def to_dict(self):
        d = {"error": self.error, "code": self.code}
        if self.details:
            d["details"] = self.details
        return d


def handle_error(error: Exception) -> ErrorResponse:
    if isinstance(error, NexusError):
        return ErrorResponse(error=error.message, code=error.code)
    return ErrorResponse(error="Internal server error", code=500, details=str(error))


class TestNexusError:
    def test_base_error(self):
        err = NexusError("test error", code=500)
        assert err.message == "test error"
        assert err.code == 500
        assert str(err) == "test error"


class TestModelNotFoundError:
    def test_creation(self):
        err = ModelNotFoundError("gpt2")
        assert err.model_name == "gpt2"
        assert err.code == 404
        assert "gpt2" in err.message

    def test_is_nexus_error(self):
        err = ModelNotFoundError("gpt2")
        assert isinstance(err, NexusError)


class TestInvalidRequestError:
    def test_creation(self):
        err = InvalidRequestError("Missing prompt")
        assert err.code == 400
        assert "Missing prompt" in err.message


class TestAuthenticationError:
    def test_default_message(self):
        err = AuthenticationError()
        assert err.code == 401
        assert "Authentication" in err.message

    def test_custom_message(self):
        err = AuthenticationError("Invalid API key")
        assert "Invalid API key" in err.message


class TestRateLimitError:
    def test_creation(self):
        err = RateLimitError(retry_after=30)
        assert err.code == 429
        assert err.retry_after == 30


class TestModelLoadError:
    def test_creation(self):
        err = ModelLoadError("gpt2", "OOM")
        assert err.code == 500
        assert "gpt2" in err.message
        assert "OOM" in err.message

    def test_without_reason(self):
        err = ModelLoadError("gpt2")
        assert "gpt2" in err.message


class TestSafetyError:
    def test_default_message(self):
        err = SafetyError()
        assert err.code == 403

    def test_custom_message(self):
        err = SafetyError("Toxic content detected")
        assert "Toxic" in err.message


class TestErrorResponse:
    def test_to_dict(self):
        resp = ErrorResponse(error="Not found", code=404)
        d = resp.to_dict()
        assert d["error"] == "Not found"
        assert d["code"] == 404

    def test_to_dict_with_details(self):
        resp = ErrorResponse(error="Error", code=500, details="traceback")
        d = resp.to_dict()
        assert "details" in d


class TestHandleError:
    def test_handle_nexus_error(self):
        err = ModelNotFoundError("gpt2")
        resp = handle_error(err)
        assert resp.code == 404
        assert "gpt2" in resp.error

    def test_handle_generic_error(self):
        err = ValueError("something wrong")
        resp = handle_error(err)
        assert resp.code == 500

    def test_handle_auth_error(self):
        err = AuthenticationError("bad token")
        resp = handle_error(err)
        assert resp.code == 401

    def test_handle_rate_limit(self):
        err = RateLimitError()
        resp = handle_error(err)
        assert resp.code == 429
