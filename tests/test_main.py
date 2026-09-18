"""Tests for the top-level entry points (main.py, python -m nexus_llm).

The repo ships several launch paths; each must reach the same click group.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestPackageMetadata:
    def test_version_is_pep440(self):
        from nexus_llm import __version__

        parts = __version__.split(".")
        assert len(parts) >= 3
        assert parts[0].isdigit()

    def test_version_helpers_agree(self):
        from nexus_llm import get_version_info, get_version_string

        assert get_version_info()["version"] in get_version_string()

    def test_version_file_matches_package(self):
        version_file = REPO_ROOT / "VERSION"
        if not version_file.exists():
            pytest.skip("VERSION file only present in a source checkout")
        from nexus_llm import __version__

        assert version_file.read_text().strip() == __version__

    def test_lazy_attribute_for_app(self):
        import nexus_llm

        assert nexus_llm.NexusLLMApp.__name__ == "NexusLLMApp"

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = sys.modules["nexus_llm"].definitely_not_here


class TestMainScript:
    def test_main_help(self):
        result = _run(["main.py", "--help"])
        assert result.returncode == 0
        assert "Nexus-LLM" in result.stdout

    def test_main_version(self):
        from nexus_llm import __version__

        result = _run(["main.py", "--version"])
        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_module_entry_point(self):
        """python -m nexus_llm must reach the same CLI."""
        result = _run(["-m", "nexus_llm", "--help"])
        assert result.returncode == 0
        assert "chat" in result.stdout

    def test_console_script_or_fallback(self):
        """nexus-llm is the installed script; fall back to the module if absent."""
        script = Path(sys.executable).parent / "nexus-llm"
        args = [str(script), "--version"] if script.exists() else [sys.executable, "-m", "nexus_llm", "--version"]
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0

    def test_run_server_script_is_importable(self):
        for name in ("run_chat.py", "run_server.py", "run_train.py", "run_eval.py", "run_benchmark.py"):
            path = REPO_ROOT / name
            if not path.exists():
                pytest.skip(f"{name} not present")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


class TestVersionSyncScript:
    def test_check_passes_in_a_clean_tree(self):
        script = REPO_ROOT / "scripts" / "dev" / "sync_version.py"
        result = _run([str(script), "--check"])
        # Non-zero only when metadata drifted; message tells the developer what.
        assert result.returncode == 0, result.stdout + result.stderr
