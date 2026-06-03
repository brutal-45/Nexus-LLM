"""Preset categories for Nexus-LLM.

Defines category constants, descriptions, and default generation
parameter overrides for each category.
"""

from enum import Enum
from typing import Any, Dict


class PresetCategory(str, Enum):
    """Preset category identifiers."""

    CHAT = "chat"
    CODE = "code"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    EDUCATION = "education"
    CUSTOM = "custom"


# Human-readable descriptions for each category
CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    PresetCategory.CHAT: "General-purpose conversational assistants",
    PresetCategory.CODE: "Code generation, review, and debugging helpers",
    PresetCategory.CREATIVE: "Creative writing, storytelling, and brainstorming",
    PresetCategory.ANALYSIS: "Data analysis, summarisation, and reasoning",
    PresetCategory.EDUCATION: "Teaching, tutoring, and explanation",
    PresetCategory.CUSTOM: "User-defined custom presets",
}

# Sensible default generation params per category
CATEGORY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    PresetCategory.CHAT: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    PresetCategory.CODE: {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 2048,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    PresetCategory.CREATIVE: {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
    },
    PresetCategory.ANALYSIS: {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 2048,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    PresetCategory.EDUCATION: {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 1024,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    },
    PresetCategory.CUSTOM: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
}
