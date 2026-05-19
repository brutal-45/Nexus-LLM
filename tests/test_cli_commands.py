"""Tests for CLI commands with Click testing."""
import pytest
from click.testing import CliRunner

from nexus_llm.cli import cli


class TestCLIGroup:
    """Test main CLI group."""

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Nexus-LLM" in result.output

    def test_cli_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_verbose_and_quiet_conflict(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--quiet"])
        # Should error because both flags conflict
        assert result.exit_code != 0 or "Cannot use both" in result.output


class TestChatCommand:
    """Test chat command."""

    def test_chat_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower() or "Chat" in result.output

    def test_chat_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "--help"])
        assert "--model" in result.output
        assert "--temperature" in result.output
        assert "--device" in result.output


class TestServeCommand:
    """Test serve command."""

    def test_serve_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output


class TestTrainCommand:
    """Test train command."""

    def test_train_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--dataset" in result.output
        assert "--epochs" in result.output


class TestModelsCommand:
    """Test models command."""

    def test_models_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0
        assert "--list-available" in result.output


class TestDownloadCommand:
    """Test download command."""

    def test_download_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "--help"])
        assert result.exit_code == 0
        assert "MODEL_NAME" in result.output or "model_name" in result.output.lower()


class TestEvalCommand:
    """Test eval command."""

    def test_eval_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output


class TestBenchmarkCommand:
    """Test benchmark command."""

    def test_benchmark_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output


class TestConfigCommand:
    """Test config command."""

    def test_config_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "--list" in result.output or "list" in result.output.lower()


class TestInfoCommand:
    """Test info command."""

    def test_info_runs(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
