"""
Nexus-LLM - A Claude-like terminal LLM chat application with its own backend.

Run with: python -m nexus_llm
Or after install: nexus-llm

Submodules are loaded lazily (PEP 562) so that importing ``nexus_llm`` stays
cheap -- the CLI starts without pulling in torch/transformers.
"""

from __future__ import annotations

from nexus_llm.__version__ import (
    __author__,
    __copyright__,
    __description__,
    __email__,
    __license__,
    __title__,
    __url__,
    __version__,
    get_version_info,
    get_version_string,
)

__all__ = [
    "__author__",
    "__description__",
    "__license__",
    "__url__",
    "__version__",
    "get_version_info",
    "get_version_string",
    "NexusLLMApp",
]

#: Public names that live in submodules, imported on first access.
_LAZY_IMPORTS = {
    "NexusLLMApp": "nexus_llm.app",
}


def __getattr__(name: str):
    """Resolve lazily-imported public attributes (PEP 562)."""
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups are direct
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
