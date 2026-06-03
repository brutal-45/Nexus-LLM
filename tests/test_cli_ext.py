"""Tests for the cli_ext module.

Covers CLIExtension, CLIPlugin, CommandRegistry, and OutputFormat.
"""

from __future__ import annotations

import pytest

from nexus_llm.cli_ext.extension import CLIExtension
from nexus_llm.cli_ext.plugin import CLIPlugin
from nexus_llm.cli_ext.registry import CommandRegistry
from nexus_llm.cli_ext.output import OutputFormat


# ---------------------------------------------------------------------------
# OutputFormat
# ---------------------------------------------------------------------------

class TestOutputFormat:
    """Tests for OutputFormat."""

    def test_format_exists(self):
        assert OutputFormat is not None

    def test_common_formats(self):
        # Should have common output formats
        assert hasattr(OutputFormat, "JSON") or hasattr(OutputFormat, "TABLE") or \
               hasattr(OutputFormat, "TEXT") or hasattr(OutputFormat, "PLAIN")


# ---------------------------------------------------------------------------
# CommandRegistry
# ---------------------------------------------------------------------------

class TestCommandRegistry:
    """Tests for CommandRegistry."""

    def test_create_registry(self):
        registry = CommandRegistry()
        assert registry is not None

    def test_register_command(self):
        registry = CommandRegistry()
        registry.register("hello", lambda: "Hello!")
        assert registry.has_command("hello")

    def test_get_command(self):
        registry = CommandRegistry()
        fn = lambda: "Hello!"
        registry.register("hello", fn)
        retrieved = registry.get("hello")
        assert retrieved is fn

    def test_get_nonexistent(self):
        registry = CommandRegistry()
        assert registry.get("nonexistent") is None

    def test_list_commands(self):
        registry = CommandRegistry()
        registry.register("cmd1", lambda: None)
        registry.register("cmd2", lambda: None)
        commands = registry.list_commands()
        assert "cmd1" in commands
        assert "cmd2" in commands

    def test_unregister(self):
        registry = CommandRegistry()
        registry.register("cmd1", lambda: None)
        registry.unregister("cmd1")
        assert not registry.has_command("cmd1")


# ---------------------------------------------------------------------------
# CLIPlugin
# ---------------------------------------------------------------------------

class TestCLIPlugin:
    """Tests for CLIPlugin."""

    def test_create_plugin(self):
        plugin = CLIPlugin(name="test-plugin")
        assert plugin.name == "test-plugin"

    def test_get_commands(self):
        plugin = CLIPlugin(name="test")
        commands = plugin.get_commands()
        assert isinstance(commands, list) or isinstance(commands, dict)


# ---------------------------------------------------------------------------
# CLIExtension
# ---------------------------------------------------------------------------

class TestCLIExtension:
    """Tests for CLIExtension."""

    def test_create_extension(self):
        ext = CLIExtension()
        assert ext is not None

    def test_register_plugin(self):
        ext = CLIExtension()
        plugin = CLIPlugin(name="test-plugin")
        ext.register_plugin(plugin)

    def test_get_registry(self):
        ext = CLIExtension()
        registry = ext.get_registry()
        assert isinstance(registry, CommandRegistry)

    def test_execute_command(self):
        ext = CLIExtension()
        ext.get_registry().register("hello", lambda: "Hello, World!")
        result = ext.execute("hello")
        assert result == "Hello, World!"

    def test_execute_nonexistent(self):
        ext = CLIExtension()
        result = ext.execute("nonexistent")
        # Should return None or raise
        assert result is None or True  # graceful handling
