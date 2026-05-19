"""Version information for Nexus-LLM."""

__version__ = "0.1.0"
__author__ = "Nexus-LLM Team"
__email__ = "nexus-llm@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024, Nexus-LLM Team"
__title__ = "nexus-llm"
__description__ = "A powerful LLM framework for training, serving, and chatting"
__url__ = "https://github.com/nexus-llm/nexus-llm"

# Version tuple for programmatic comparison
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_TUPLE = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

# Release status: "alpha", "beta", "rc", "final"
RELEASE_STATUS = "alpha"

# Build metadata (optional, e.g., "dev1", "rc1", etc.)
BUILD_METADATA = ""


def get_version_string() -> str:
    """Get the full version string with release status and build metadata.

    Returns:
        Full version string, e.g. "0.1.0a1" or "0.1.0".
    """
    version = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
    if RELEASE_STATUS == "alpha":
        version += "a"
    elif RELEASE_STATUS == "beta":
        version += "b"
    elif RELEASE_STATUS == "rc":
        version += "rc"
    if BUILD_METADATA:
        version += BUILD_METADATA
    return version


def get_version_info() -> dict:
    """Get detailed version information as a dictionary.

    Returns:
        Dictionary containing all version metadata.
    """
    return {
        "version": __version__,
        "version_tuple": VERSION_TUPLE,
        "release_status": RELEASE_STATUS,
        "build_metadata": BUILD_METADATA,
        "author": __author__,
        "license": __license__,
        "url": __url__,
    }
