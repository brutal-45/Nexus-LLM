"""Tests for built-in plugins (weather, calculator, etc.)."""
import math

import pytest

from nexus_llm.plugins.builtin.calculator import CalculatorPlugin
from nexus_llm.plugins.builtin.weather import WeatherPlugin
from nexus_llm.plugins.builtin.code_runner import CodeRunnerPlugin
from nexus_llm.plugins.builtin.file_manager import FileManagerPlugin
from nexus_llm.plugins.builtin.note_taker import NoteTakerPlugin
from nexus_llm.plugins.builtin.system_monitor import SystemMonitorPlugin
from nexus_llm.plugins.builtin.web_search import WebSearchPlugin


class TestCalculatorPlugin:
    """Test CalculatorPlugin."""

    @pytest.fixture
    def calc(self):
        return CalculatorPlugin()

    def test_basic_arithmetic(self, calc):
        result = calc.evaluate("2 + 3")
        assert result["success"] is True
        assert result["result"] == 5

    def test_multiplication(self, calc):
        result = calc.evaluate("4 * 5")
        assert result["success"] is True
        assert result["result"] == 20

    def test_division(self, calc):
        result = calc.evaluate("10 / 3")
        assert result["success"] is True
        assert abs(result["result"] - 3.3333) < 0.01

    def test_power(self, calc):
        result = calc.evaluate("2 ** 10")
        assert result["success"] is True
        assert result["result"] == 1024

    def test_sqrt(self, calc):
        result = calc.evaluate("sqrt(16)")
        assert result["success"] is True
        assert result["result"] == 4.0

    def test_pi_constant(self, calc):
        result = calc.evaluate("pi")
        assert result["success"] is True
        assert abs(result["result"] - math.pi) < 0.001

    def test_trig_functions(self, calc):
        result = calc.evaluate("sin(0)")
        assert result["success"] is True
        assert abs(result["result"]) < 0.001

    def test_division_by_zero(self, calc):
        result = calc.evaluate("1 / 0")
        assert result["success"] is False
        assert "error" in result

    def test_invalid_expression(self, calc):
        result = calc.evaluate("not_valid!!!")
        assert result["success"] is False

    def test_empty_expression(self, calc):
        result = calc.evaluate("")
        assert result["success"] is False

    def test_unsafe_code_rejected(self, calc):
        result = calc.evaluate("__import__('os').system('ls')")
        assert result["success"] is False

    def test_history(self, calc):
        calc.evaluate("2 + 2")
        calc.evaluate("3 * 3")
        history = calc.get_history()
        assert len(history) == 2

    def test_clear_history(self, calc):
        calc.evaluate("1 + 1")
        calc.clear_history()
        assert len(calc.get_history()) == 0

    def test_get_help(self, calc):
        help_text = calc.get_help()
        assert "Arithmetic" in help_text
        assert "sqrt" in help_text

    def test_activate_deactivate(self, calc):
        calc.activate()
        assert calc._active is True
        calc.deactivate()
        assert calc._active is False


class TestWeatherPlugin:
    """Test WeatherPlugin."""

    @pytest.fixture
    def weather(self):
        return WeatherPlugin()

    def test_get_weather_known_city(self, weather):
        result = weather.get_weather("New York")
        assert result["location"] == "New York"
        assert "temp_f" in result
        assert "condition" in result
        assert result.get("simulated") is not True

    def test_get_weather_unknown_city(self, weather):
        result = weather.get_weather("Unknown City")
        assert result["location"] == "Unknown City"
        assert "temp_f" in result
        assert result.get("simulated") is True

    def test_get_weather_case_insensitive(self, weather):
        result = weather.get_weather("LONDON")
        assert result["location"] == "London"

    def test_get_forecast(self, weather):
        forecast = weather.get_forecast("Tokyo", days=3)
        assert len(forecast) == 3
        assert "high_f" in forecast[0]
        assert "condition" in forecast[0]

    def test_forecast_days_clamping(self, weather):
        forecast = weather.get_forecast("Paris", days=10)
        assert len(forecast) == 7  # Max 7 days

    def test_format_weather(self, weather):
        formatted = weather.format_weather("Paris")
        assert "Paris" in formatted
        assert "°F" in formatted

    def test_activate_deactivate(self, weather):
        weather.activate()
        assert weather._active is True
        weather.deactivate()
        assert weather._active is False


class TestCodeRunnerPlugin:
    """Test CodeRunnerPlugin."""

    def test_instantiation(self):
        plugin = CodeRunnerPlugin()
        assert plugin.name == "code_runner"

    def test_activate_deactivate(self):
        plugin = CodeRunnerPlugin()
        plugin.activate()
        assert plugin._active is True
        plugin.deactivate()
        assert plugin._active is False


class TestFileManagerPlugin:
    """Test FileManagerPlugin."""

    def test_instantiation(self):
        plugin = FileManagerPlugin()
        assert plugin.name == "file_manager"

    def test_activate_deactivate(self):
        plugin = FileManagerPlugin()
        plugin.activate()
        assert plugin._active is True
        plugin.deactivate()
        assert plugin._active is False


class TestNoteTakerPlugin:
    """Test NoteTakerPlugin."""

    def test_instantiation(self):
        plugin = NoteTakerPlugin()
        assert plugin.name == "note_taker"

    def test_activate_deactivate(self):
        plugin = NoteTakerPlugin()
        plugin.activate()
        assert plugin._active is True
        plugin.deactivate()
        assert plugin._active is False


class TestSystemMonitorPlugin:
    """Test SystemMonitorPlugin."""

    def test_instantiation(self):
        plugin = SystemMonitorPlugin()
        assert plugin.name == "system_monitor"

    def test_activate_deactivate(self):
        plugin = SystemMonitorPlugin()
        plugin.activate()
        assert plugin._active is True
        plugin.deactivate()
        assert plugin._active is False


class TestWebSearchPlugin:
    """Test WebSearchPlugin."""

    def test_instantiation(self):
        plugin = WebSearchPlugin()
        assert plugin.name == "web_search"

    def test_activate_deactivate(self):
        plugin = WebSearchPlugin()
        plugin.activate()
        assert plugin._active is True
        plugin.deactivate()
        assert plugin._active is False
