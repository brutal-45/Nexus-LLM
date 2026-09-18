"""Tests for example plugins."""
import pytest

from nexus_llm.plugins.examples.echo import EchoPlugin
from nexus_llm.plugins.examples.custom_greet import CustomGreetPlugin


class TestEchoPlugin:
    """Test EchoPlugin."""

    @pytest.fixture
    def echo(self):
        return EchoPlugin()

    def test_basic_echo(self, echo):
        result = echo.echo("Hello")
        assert result["success"] is True
        assert result["echo"] == "Hello"
        assert result["original"] == "Hello"

    def test_echo_empty_message(self, echo):
        result = echo.echo("")
        assert result["success"] is False

    def test_echo_upper_transform(self):
        echo = EchoPlugin(transform="upper")
        result = echo.echo("hello")
        assert result["echo"] == "HELLO"

    def test_echo_lower_transform(self):
        echo = EchoPlugin(transform="lower")
        result = echo.echo("HELLO")
        assert result["echo"] == "hello"

    def test_echo_reverse_transform(self):
        echo = EchoPlugin(transform="reverse")
        result = echo.echo("abc")
        assert result["echo"] == "cba"

    def test_echo_title_transform(self):
        echo = EchoPlugin(transform="title")
        result = echo.echo("hello world")
        assert result["echo"] == "Hello World"

    def test_echo_repeat_transform(self):
        echo = EchoPlugin(transform="repeat")
        result = echo.echo("hi")
        assert result["echo"] == "hi hi"

    def test_echo_with_prefix(self):
        echo = EchoPlugin(prefix="[BOT] ")
        result = echo.echo("Hello")
        assert result["echo"] == "[BOT] Hello"

    def test_echo_with_suffix(self):
        echo = EchoPlugin(suffix=" !")
        result = echo.echo("Hello")
        assert result["echo"] == "Hello !"

    def test_echo_truncation(self):
        echo = EchoPlugin(max_length=5)
        result = echo.echo("abcdefghij")
        assert result["truncated"] is True
        assert len(result["echo"]) <= 10  # prefix/suffix may add length

    def test_echo_count(self, echo):
        echo.echo("msg1")
        echo.echo("msg2")
        result = echo.echo("msg3")
        assert result["echo_count"] == 3

    def test_get_available_transforms(self, echo):
        transforms = echo.get_available_transforms()
        assert "upper" in transforms
        assert "lower" in transforms
        assert "reverse" in transforms

    def test_get_stats(self, echo):
        echo.echo("test")
        stats = echo.get_stats()
        assert stats["success"] is True
        assert stats["total_echoes"] == 1
        assert stats["active"] is False

    def test_activate_deactivate(self, echo):
        echo.activate()
        assert echo._active is True
        echo.deactivate()
        assert echo._active is False

    def test_name_and_version(self, echo):
        assert echo.name == "echo"
        assert echo.version == "1.0.0"


class TestCustomGreetPlugin:
    """Test CustomGreetPlugin."""

    @pytest.fixture
    def greet(self):
        return CustomGreetPlugin(default_name="User")

    def test_greet_with_name(self, greet):
        result = greet.greet("Alice")
        assert "Alice" in result

    def test_greet_default_name(self, greet):
        result = greet.greet()
        assert "User" in result

    def test_farewell(self, greet):
        result = greet.farewell("Bob")
        assert "Bob" in result

    def test_greet_increments_count(self, greet):
        greet.greet("A")
        greet.greet("B")
        stats = greet.get_greeting_stats()
        assert stats["total_greetings"] == 2

    def test_custom_greetings(self):
        custom = {"morning": ["Good moaning, {name}!"]}
        plugin = CustomGreetPlugin(custom_greetings=custom)
        # Custom greetings should be merged
        assert len(plugin.greetings["morning"]) > 1

    def test_get_greeting_stats(self, greet):
        stats = greet.get_greeting_stats()
        assert stats["success"] is True
        assert "total_greetings" in stats
        assert "time_of_day" in stats

    def test_activate_deactivate(self, greet):
        greet.activate()
        assert greet._active is True
        greet.deactivate()
        assert greet._active is False

    def test_name_and_version(self, greet):
        assert greet.name == "custom_greet"
        assert greet.version == "1.0.0"
