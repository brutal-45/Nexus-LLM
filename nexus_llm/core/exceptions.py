"""Custom exceptions for Nexus-LLM (simple-message variants).

This module intentionally keeps a *minimal* constructor: the many call sites
across ``nexus_llm.backend``, ``nexus_llm.training``, ``nexus_llm.api.client``
and the CLI raise these errors with a single human-readable message, e.g.
``raise InferenceError("Generation failed: ...")``.

The richer variants in :mod:`nexus_llm.exceptions` attach structured fields
(``error_code``, ``details``, ``to_dict()``).  Both families now share the same
root class so that a single ``except NexusLLMError`` — as used by the CLI's
error handler — catches errors regardless of which module raised them.
"""

from __future__ import annotations

from nexus_llm.exceptions import NexusLLMError

__all__ = [
    "ConfigurationError",
    "InferenceError",
    "ModelLoadError",
    "ModelNotFoundError",
    "NexusLLMError",
    "ServerError",
    "TrainingError",
]


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
