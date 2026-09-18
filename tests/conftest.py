"""Shared pytest configuration and fixtures for the Nexus-LLM test suite.

This module makes the suite runnable in any environment:

* it puts the repository root on ``sys.path`` so ``import nexus_llm`` works
  without an editable install,
* it points ``HOME`` at a temporary directory so tests never touch the
  developer's real ``~/.cache`` / ``~/.nexus-llm`` directories,
* it provides the ``tmp_dir`` fixture used across the suite (a ``pathlib.Path``
  that also works with ``os.path.join`` and ``str()``),
* it supplies a fresh event loop for synchronous tests that drive coroutines
  through ``asyncio.get_event_loop()`` (removed in Python 3.10+).
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Session-wide isolation of user state
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _isolate_user_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep tests out of the developer's real home/cache directories."""
    sandbox = tmp_path_factory.mktemp("nexus-home")
    saved = {}
    overrides = {
        "HOME": str(sandbox),
        "USERPROFILE": str(sandbox),
        "XDG_CACHE_HOME": str(sandbox / ".cache"),
        "XDG_CONFIG_HOME": str(sandbox / ".config"),
        "XDG_DATA_HOME": str(sandbox / ".local" / "share"),
        "NEXUS_HOME": str(sandbox / ".nexus-llm"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
    }
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a per-test temporary directory.

    ``pathlib.Path`` is file-system-like *and* ``os.PathLike``, so it works
    with ``tmp_dir / "file"``, ``os.path.join(tmp_dir, "file")`` and
    ``str(tmp_dir)`` alike.
    """
    return tmp_path


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root (useful for config/data fixtures)."""
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _fresh_event_loop() -> Iterator[None]:
    """Give every test a usable current event loop in the main thread.

    Several tests are plain (synchronous) functions that call
    ``asyncio.get_event_loop().run_until_complete(...)``.  Since Python 3.10
    ``get_event_loop()`` raises when no loop is set, so we install a fresh loop
    before each test and close it afterwards.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        try:
            if not loop.is_closed():
                loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError:  # pragma: no cover - loop already torn down
            pass
        asyncio.set_event_loop(None)
        loop.close()


# ---------------------------------------------------------------------------
# Collection hooks
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register project-local markers so ``--strict-markers`` stays quiet."""
    config.addinivalue_line("markers", "slow: tests that take a long time to run")
    config.addinivalue_line("markers", "gpu: tests that require a CUDA device")
    config.addinivalue_line("markers", "integration: tests that exercise multiple subsystems")
    config.addinivalue_line("markers", "network: tests that would hit the network (skipped by default)")

