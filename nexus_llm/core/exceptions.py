"""Custom exceptions for Nexus-LLM."""


class NexusLLMError(Exception):
    """Base exception for Nexus-LLM."""
    pass


class ModelNotFoundError(NexusLLMError):
    """Raised when a model is not found in the catalog."""
    pass


class ModelLoadError(NexusLLMError):
    """Raised when a model fails to load."""
    pass


class InferenceError(NexusLLMError):
    """Raised when inference fails."""
    pass


class ConfigurationError(NexusLLMError):
    """Raised when there's a configuration error."""
    pass


class TrainingError(NexusLLMError):
    """Raised when training fails."""
    pass


class ServerError(NexusLLMError):
    """Raised when the server encounters an error."""
    pass
