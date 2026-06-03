"""Nexus-LLM Custom Exceptions Module.

Provides a comprehensive hierarchy of custom exceptions for the
Nexus-LLM framework, enabling structured and specific error handling
across all components.
"""

from typing import Any, Dict, Optional


class NexusLLMError(Exception):
    """Base exception for all Nexus-LLM errors.

    All custom exceptions in the Nexus-LLM framework inherit from this class,
    allowing for both broad and fine-grained exception handling.

    Attributes:
        message: Human-readable error description.
        error_code: Optional error code for programmatic handling.
        details: Optional dictionary with additional error context.
    """

    def __init__(
        self,
        message: str = "An error occurred in Nexus-LLM",
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, error_code={self.error_code!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary representation.

        Returns:
            Dictionary with error type, message, code, and details.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ModelNotFoundError(NexusLLMError):
    """Raised when a requested model is not found.

    This can occur when a model name is misspelled, the model hasn't been
    downloaded, or the model path doesn't exist.

    Attributes:
        model_name: The model name or path that was not found.
    """

    def __init__(
        self,
        model_name: str,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        msg = message or f"Model not found: {model_name}"
        detail = details or {}
        detail["model_name"] = model_name
        super().__init__(message=msg, error_code=error_code or "MODEL_NOT_FOUND", details=detail)


class ModelLoadError(NexusLLMError):
    """Raised when a model fails to load.

    This can occur due to corrupted files, incompatible formats,
    insufficient memory, or missing dependencies.

    Attributes:
        model_name: The model name or path that failed to load.
        original_error: The original exception that caused the load failure.
    """

    def __init__(
        self,
        model_name: str,
        original_error: Optional[Exception] = None,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.original_error = original_error
        msg = message or f"Failed to load model: {model_name}"
        if original_error:
            msg += f" - {original_error}"
        detail = details or {}
        detail["model_name"] = model_name
        if original_error:
            detail["original_error"] = str(original_error)
        super().__init__(message=msg, error_code=error_code or "MODEL_LOAD_ERROR", details=detail)


class InferenceError(NexusLLMError):
    """Raised during inference failures.

    This can occur due to CUDA out-of-memory, invalid inputs,
    model output errors, or generation failures.

    Attributes:
        model_name: The model being used for inference.
        prompt: The input prompt that caused the error (truncated).
    """

    def __init__(
        self,
        model_name: str = "",
        prompt: str = "",
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt[:200] if prompt else ""
        msg = message or "Inference error occurred"
        if model_name:
            msg = f"Inference error with model {model_name}: {msg}"
        detail = details or {}
        detail["model_name"] = model_name
        if prompt:
            detail["prompt_preview"] = self.prompt
        super().__init__(message=msg, error_code=error_code or "INFERENCE_ERROR", details=detail)


class TokenizerError(NexusLLMError):
    """Raised for tokenizer-related errors.

    This can occur during tokenization, detokenization, or when
    loading a tokenizer fails.

    Attributes:
        tokenizer_name: The tokenizer that caused the error.
    """

    def __init__(
        self,
        tokenizer_name: str = "",
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        msg = message or "Tokenizer error occurred"
        if tokenizer_name:
            msg = f"Tokenizer error for '{tokenizer_name}': {msg}"
        detail = details or {}
        detail["tokenizer_name"] = tokenizer_name
        super().__init__(message=msg, error_code=error_code or "TOKENIZER_ERROR", details=detail)


class ConfigError(NexusLLMError):
    """Raised for configuration errors.

    This can occur when configuration files are malformed, missing,
    contain invalid values, or when conflicting settings are provided.

    Attributes:
        config_key: The configuration key that caused the error.
        config_source: Where the configuration was loaded from.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        config_key: Optional[str] = None,
        config_source: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config_key = config_key
        self.config_source = config_source
        msg = message or "Configuration error occurred"
        detail = details or {}
        if config_key:
            detail["config_key"] = config_key
        if config_source:
            detail["config_source"] = config_source
        super().__init__(message=msg, error_code=error_code or "CONFIG_ERROR", details=detail)


class TrainingError(NexusLLMError):
    """Raised during training failures.

    This can occur due to dataset errors, OOM during training,
    gradient explosions, or checkpoint save/load failures.

    Attributes:
        step: The training step at which the error occurred.
        epoch: The epoch at which the error occurred.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.step = step
        self.epoch = epoch
        msg = message or "Training error occurred"
        detail = details or {}
        if step is not None:
            detail["step"] = step
        if epoch is not None:
            detail["epoch"] = epoch
        super().__init__(message=msg, error_code=error_code or "TRAINING_ERROR", details=detail)


class ServerError(NexusLLMError):
    """Raised for server-related errors.

    This can occur during server startup, request handling,
    or when the server encounters an internal error.

    Attributes:
        endpoint: The API endpoint that caused the error.
        status_code: HTTP status code associated with the error.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        msg = message or "Server error occurred"
        detail = details or {}
        if endpoint:
            detail["endpoint"] = endpoint
        if status_code:
            detail["status_code"] = status_code
        super().__init__(message=msg, error_code=error_code or "SERVER_ERROR", details=detail)


class ChatError(NexusLLMError):
    """Raised during chat operations.

    This can occur during chat session management, history handling,
    or when the chat interface encounters an error.

    Attributes:
        session_id: The chat session ID if available.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        session_id: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        msg = message or "Chat error occurred"
        detail = details or {}
        if session_id:
            detail["session_id"] = session_id
        super().__init__(message=msg, error_code=error_code or "CHAT_ERROR", details=detail)


class PluginError(NexusLLMError):
    """Raised for plugin-related errors.

    This can occur during plugin loading, initialization,
    execution, or when plugin dependencies are missing.

    Attributes:
        plugin_name: The name of the plugin that caused the error.
    """

    def __init__(
        self,
        plugin_name: str = "",
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.plugin_name = plugin_name
        msg = message or "Plugin error occurred"
        if plugin_name:
            msg = f"Plugin '{plugin_name}' error: {msg}"
        detail = details or {}
        detail["plugin_name"] = plugin_name
        super().__init__(message=msg, error_code=error_code or "PLUGIN_ERROR", details=detail)
