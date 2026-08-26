"""Custom exceptions for Nexus-LLM."""


class NexusLLMError(Exception):
    """Base exception for Nexus-LLM."""


class ModelNotFoundError(NexusLLMError):
    """Raised when a model is not found in the catalog."""


class ModelLoadError(NexusLLMError):
    """Raised when a model fails to load."""


class InferenceError(NexusLLMError):
    """Raised when inference fails."""


class ConfigurationError(NexusLLMError):
    """Raised when there's a configuration error."""


class TrainingError(NexusLLMError):
    """Raised when training fails."""


class ServerError(NexusLLMError):
    """Raised when the server encounters an error."""
