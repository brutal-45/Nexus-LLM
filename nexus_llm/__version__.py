"""Version information for Nexus-LLM.

This module is the single source of truth for the version: ``pyproject.toml``
reads it dynamically (``[tool.setuptools.dynamic]``) and ``VERSION`` at the
repository root is kept in sync by ``scripts/dev/sync_version.py``.
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Nexus-LLM Team"
__email__ = "nexus-llm@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024-2026, Nexus-LLM Team"
__title__ = "nexus-llm"
__description__ = "A powerful LLM framework for training, serving, and chatting"
__url__ = "https://github.com/brutal-45/Nexus-LLM"

# Version tuple for programmatic comparison
VERSION_MAJOR = 2
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_TUPLE = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

# Release status: "alpha", "beta", "rc", "final"
RELEASE_STATUS = "final"

# Build metadata (optional, e.g., "dev1", "rc1", etc.)
BUILD_METADATA = ""

#: Suffix appended for pre-releases; empty for a final release.
_STATUS_SUFFIX = {"alpha": "a", "beta": "b", "rc": "rc", "final": ""}


def get_version_string() -> str:
    """Get the full version string with release status and build metadata.

    Returns:
        Full version string, e.g. ``"2.0.0"``, ``"2.1.0b1"`` or
        ``"2.1.0rc1.dev1"``.
    """
    version = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
    version += _STATUS_SUFFIX.get(RELEASE_STATUS, "")
    if BUILD_METADATA:
        version += ("" if RELEASE_STATUS in ("alpha", "beta", "rc") else "+")
        version += BUILD_METADATA
    return version


def get_version_info() -> dict:
    """Get detailed version information as a dictionary.

    Returns:
        Dictionary containing all version metadata.
    """
    return {
        "version": __version__,
        "version_string": get_version_string(),
        "version_tuple": VERSION_TUPLE,
        "release_status": RELEASE_STATUS,
        "build_metadata": BUILD_METADATA,
        "author": __author__,
        "license": __license__,
        "url": __url__,
        "description": __description__,
    }
