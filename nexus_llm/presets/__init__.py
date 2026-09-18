"""Presets module for Nexus-LLM.

Provides pre-configured model settings, generation parameters, and
system prompts for common use cases, along with management and
import/export utilities.
"""

from nexus_llm.presets.library import PresetLibrary
from nexus_llm.presets.manager import PresetManager
from nexus_llm.presets.preset import Preset

__all__ = [
    "Preset",
    "PresetLibrary",
    "PresetManager",
]
