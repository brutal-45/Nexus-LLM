"""Sphinx configuration for the Nexus-LLM documentation.

The prose lives in Markdown (handled by ``myst-parser``); the API reference is
generated from the package docstrings via ``autodoc``/``autodoc-typehints``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import nexus_llm` work when building from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus_llm.__version__ import __version__ as package_version  # noqa: E402

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

project = "Nexus-LLM"
copyright = "2024-2026, Nexus-LLM Team"  # noqa: A001
author = "Nexus-LLM Team"
version = package_version
release = package_version

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "strikethrough",
    "tasklist",
]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_imported_members = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
todo_include_todos = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# ---------------------------------------------------------------------------
# Content layout
# ---------------------------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    "_templates",
    "Thumbs.db",
    ".DS_Store",
    "**/__pycache__",
    "node_modules",
]

# Missing cross-references should not fail a docs build.
nitpicky = False

# The prose is full of illustrative pseudo-JSON / pseudo-HTTP blocks that no
# Pygments lexer can parse; rendering them as text is intentional.
suppress_warnings = [
    "misc.highlighting_failure",
    "ref.missing",
    "myst.xref_missing",
    # Several subsystems legitimately define same-named classes (e.g. WorkerConfig);
    # ambiguous cross-reference targets are informational only.
    "ref.python",
]
highlight_language = "default"

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_title = f"Nexus-LLM {release} documentation"
html_static_path: list[str] = []
html_favicon = None
html_last_updated_fmt = "%Y-%m-%d"
