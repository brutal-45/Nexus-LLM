"""Tests for nexus_llm.tools.shell module."""

import pytest
from nexus_llm.tools.shell import ShellTool


class TestShellTool:
    """Tests for the ShellTool class."""

    def test_init(self):
        tool = ShellTool()
        assert tool.name == "shell"

    def test_echo_command(self):
        tool = ShellTool()
        result = tool.run(command="echo hello")
        assert result.success is True
        assert "hello" in result.output

    def test_pwd_command(self):
        tool = ShellTool()
        result = tool.run(command="pwd")
        assert result.success is True

    def test_invalid_command(self):
        tool = ShellTool()
        result = tool.run(command="nonexistent_command_xyz")
        assert result.success is False

    def test_command_with_timeout(self):
        tool = ShellTool()
        result = tool.run(command="echo fast", timeout=5)
        assert result.success is True

    def test_command_missing(self):
        tool = ShellTool()
        result = tool.run()
        assert result.success is False
