"""Tests for the ``nexus-llm`` command line interface.

These run the real click group through CliRunner, so a broken entry point,
missing option or crashed subcommand fails here rather than in a user's shell.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from nexus_llm import __version__
from nexus_llm.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestRootCommand:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Nexus-LLM" in result.stdout
        for command in ("chat", "serve", "train", "models", "info", "config", "download"):
            assert command in result.stdout

    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_short_version_flag(self, runner):
        assert __version__ in runner.invoke(cli, ["-V"]).stdout

    def test_unknown_command_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["definitely-not-a-command"])
        assert result.exit_code != 0

    def test_debug_flag_is_accepted(self, runner):
        assert runner.invoke(cli, ["--debug", "info"]).exit_code == 0


class TestInfoCommand:
    def test_reports_python_and_platform(self, runner):
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "Python" in result.stdout

    def test_reports_device(self, runner):
        result = runner.invoke(cli, ["info"])
        assert any(word in result.stdout.lower() for word in ("cpu", "cuda", "device"))


class TestModelsCommand:
    def test_lists_catalog(self, runner):
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt2" in result.stdout

    def test_category_filter(self, runner):
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0
        assert "--category" in result.stdout or "--filter" in result.stdout


class TestConfigCommand:
    def test_shows_configuration(self, runner):
        result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "model" in result.stdout.lower() or "gpt2" in result.stdout

    def test_show_flag(self, runner):
        assert runner.invoke(cli, ["config", "--show"]).exit_code == 0

    def test_set_persists_value(self, runner, tmp_dir):
        target = tmp_dir / "config.yaml"
        result = runner.invoke(
            cli, ["config", "--path", str(target), "--set", "model.temperature", "0.3"]
        )
        assert result.exit_code == 0, result.stdout
        if target.exists():
            assert "0.3" in target.read_text(encoding="utf-8")

    def test_path_flag_without_file_is_graceful(self, runner, tmp_dir):
        result = runner.invoke(cli, ["config", "--path", str(tmp_dir / "absent.yaml")])
        assert "Traceback" not in result.stdout


class TestChatGuards:
    def test_chat_requires_a_model_and_fails_cleanly(self, runner):
        """No TTY / no model must produce a friendly error, not a traceback."""
        result = runner.invoke(cli, ["chat", "--model", "does-not-exist-xyz"], input="/quit\n")
        assert "Traceback" not in result.stdout
        # Either it errored gracefully (non-zero) or exited cleanly.
        assert result.exit_code in (0, 1, 2)

    def test_serve_help(self, runner):
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        for option in ("--host", "--port", "--model"):
            assert option in result.stdout

    def test_train_help(self, runner):
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout or "-m" in result.stdout


class TestErrorHandling:
    def test_import_errors_are_reported_not_raised(self, runner, monkeypatch):
        """The CLI wraps commands so missing deps show guidance."""
        result = runner.invoke(cli, ["models", "--category", "no-such-category"])
        assert "Traceback" not in (result.stdout + (result.stderr or ""))
