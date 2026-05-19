"""Weather plugin providing simulated weather information.

A builtin plugin that provides weather data through simulated
results. In production, this would integrate with a real
weather API service.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class WeatherPlugin:
    """Plugin providing weather information (simulated).

    Simulates weather data retrieval for various locations.
    Supports current conditions, forecasts, and alerts.
    """

    name = "weather"
    version = "1.0.0"
    description = "Get weather information for any location (simulated data)"
    dependencies: List[str] = []
    tags = ["weather", "information", "builtin"]

    # Simulated weather database
    WEATHER_DATABASE = {
        "new york": {
            "temp_f": 72, "temp_c": 22, "condition": "Partly Cloudy",
            "humidity": 65, "wind_mph": 12, "wind_dir": "NW",
            "feels_like_f": 74, "uv_index": 6, "visibility_mi": 10,
            "pressure_in": 30.12, "dewpoint_f": 58,
        },
        "london": {
            "temp_f": 59, "temp_c": 15, "condition": "Overcast",
            "humidity": 80, "wind_mph": 15, "wind_dir": "SW",
            "feels_like_f": 55, "uv_index": 3, "visibility_mi": 7,
            "pressure_in": 29.85, "dewpoint_f": 53,
        },
        "tokyo": {
            "temp_f": 77, "temp_c": 25, "condition": "Clear",
            "humidity": 55, "wind_mph": 8, "wind_dir": "E",
            "feels_like_f": 79, "uv_index": 8, "visibility_mi": 12,
            "pressure_in": 30.05, "dewpoint_f": 60,
        },
        "paris": {
            "temp_f": 64, "temp_c": 18, "condition": "Light Rain",
            "humidity": 75, "wind_mph": 10, "wind_dir": "W",
            "feels_like_f": 61, "uv_index": 2, "visibility_mi": 5,
            "pressure_in": 29.70, "dewpoint_f": 56,
        },
        "sydney": {
            "temp_f": 81, "temp_c": 27, "condition": "Sunny",
            "humidity": 45, "wind_mph": 14, "wind_dir": "SE",
            "feels_like_f": 83, "uv_index": 9, "visibility_mi": 15,
            "pressure_in": 30.20, "dewpoint_f": 57,
        },
        "beijing": {
            "temp_f": 70, "temp_c": 21, "condition": "Hazy",
            "humidity": 60, "wind_mph": 6, "wind_dir": "N",
            "feels_like_f": 71, "uv_index": 4, "visibility_mi": 4,
            "pressure_in": 29.95, "dewpoint_f": 55,
        },
        "mumbai": {
            "temp_f": 88, "temp_c": 31, "condition": "Humid & Warm",
            "humidity": 85, "wind_mph": 9, "wind_dir": "W",
            "feels_like_f": 97, "uv_index": 10, "visibility_mi": 6,
            "pressure_in": 29.60, "dewpoint_f": 78,
        },
        "san francisco": {
            "temp_f": 65, "temp_c": 18, "condition": "Foggy",
            "humidity": 78, "wind_mph": 18, "wind_dir": "W",
            "feels_like_f": 60, "uv_index": 4, "visibility_mi": 2,
            "pressure_in": 30.00, "dewpoint_f": 55,
        },
    }

    CONDITION_EMOJIS = {
        "Clear": "☀️", "Sunny": "☀️", "Partly Cloudy": "⛅",
        "Cloudy": "☁️", "Overcast": "☁️", "Light Rain": "🌧️",
        "Rain": "🌧️", "Thunderstorm": "⛈️", "Snow": "❄️",
        "Foggy": "🌫️", "Hazy": "🌫️", "Windy": "💨",
        "Humid & Warm": "🌤️",
    }

    def __init__(self, hook_manager: Optional[HookManager] = None, **kwargs):
        self.hook_manager = hook_manager
        self._active = False

    def activate(self) -> None:
        """Activate the weather plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "tool_request",
                self._handle_tool_request,
                name="weather_tool_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("Weather plugin activated.")

    def deactivate(self) -> None:
        """Deactivate the weather plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("Weather plugin deactivated.")

    def get_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather for a location.

        Args:
            location: City name.

        Returns:
            Weather data dictionary.
        """
        location_lower = location.lower().strip()

        # Check simulated database
        if location_lower in self.WEATHER_DATABASE:
            data = self.WEATHER_DATABASE[location_lower].copy()
            data["location"] = location.title()
            data["simulated"] = False
            return data

        # Generate deterministic simulated weather for unknown locations
        hash_val = int(hashlib.md5(location_lower.encode()).hexdigest()[:8], 16)
        conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Light Rain", "Clear", "Overcast"]
        condition = conditions[hash_val % len(conditions)]
        temp_f = 50 + (hash_val % 50)
        temp_c = round((temp_f - 32) * 5 / 9)
        humidity = 30 + (hash_val % 60)
        wind_mph = 5 + (hash_val % 25)

        return {
            "location": location.title(),
            "temp_f": temp_f,
            "temp_c": temp_c,
            "condition": condition,
            "humidity": humidity,
            "wind_mph": wind_mph,
            "wind_dir": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][hash_val % 8],
            "feels_like_f": temp_f + (hash_val % 5 - 2),
            "uv_index": 1 + (hash_val % 10),
            "visibility_mi": 3 + (hash_val % 12),
            "simulated": True,
        }

    def get_forecast(self, location: str, days: int = 5) -> List[Dict[str, Any]]:
        """Get a simulated weather forecast.

        Args:
            location: City name.
            days: Number of forecast days (1-7).

        Returns:
            List of daily forecast dictionaries.
        """
        days = max(1, min(7, days))
        current = self.get_weather(location)
        base_temp = current["temp_f"]

        forecast = []
        for i in range(days):
            hash_val = int(hashlib.md5(f"{location}_{i}".encode()).hexdigest()[:6], 16)
            variation = (hash_val % 15) - 5  # -5 to +10 degree variation
            high = base_temp + variation
            low = high - 10 - (hash_val % 10)
            conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Rain", "Clear"]
            condition = conditions[hash_val % len(conditions)]

            forecast.append({
                "day_offset": i + 1,
                "high_f": high,
                "low_f": low,
                "high_c": round((high - 32) * 5 / 9),
                "low_c": round((low - 32) * 5 / 9),
                "condition": condition,
                "precipitation_chance": hash_val % 80,
            })

        return forecast

    def format_weather(self, location: str) -> str:
        """Format weather information as a readable string."""
        data = self.get_weather(location)
        emoji = self.CONDITION_EMOJIS.get(data["condition"], "")
        sim_tag = " (simulated)" if data.get("simulated") else ""

        return (
            f"{emoji} Weather for {data['location']}{sim_tag}:\n"
            f"  Condition: {data['condition']}\n"
            f"  Temperature: {data['temp_f']}°F ({data['temp_c']}°C)\n"
            f"  Feels like: {data['feels_like_f']}°F\n"
            f"  Humidity: {data['humidity']}%\n"
            f"  Wind: {data['wind_mph']} mph {data['wind_dir']}\n"
            f"  UV Index: {data['uv_index']}\n"
            f"  Visibility: {data['visibility_mi']} mi"
        )

    def _handle_tool_request(self, result, *args, **kwargs):
        """Handle tool requests for weather information."""
        tool_name = kwargs.get("tool_name", "")
        if tool_name == "weather":
            location = kwargs.get("location", kwargs.get("query", ""))
            if location:
                return self.format_weather(location)
        return result
