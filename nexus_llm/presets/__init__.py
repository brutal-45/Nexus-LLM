"""
Nexus-LLM Presets Module

Provides pre-configured settings for chat, training, and server modes.
Presets allow users to quickly configure Nexus-LLM for common use cases
without manually specifying every parameter.
"""

from nexus_llm.presets.preset_manager import (
    PresetManager,
    PresetError,
    PresetNotFoundError,
    list_presets,
    load_preset,
    apply_preset,
)

__all__ = [
    "PresetManager",
    "PresetError",
    "PresetNotFoundError",
    "list_presets",
    "load_preset",
    "apply_preset",
]
