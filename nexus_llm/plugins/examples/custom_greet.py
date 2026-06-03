"""Custom greeting plugin example.

Demonstrates how to create a custom plugin with configurable
greetings, time-aware messages, and hook integration.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class CustomGreetPlugin:
    """Example plugin providing custom greeting messages.

    Demonstrates plugin development patterns including:
    - Configurable parameters
    - Hook registration and handling
    - Time-aware behavior
    - State management
    """

    name = "custom_greet"
    version = "1.0.0"
    description = "Custom greeting plugin with configurable messages and time awareness"
    dependencies: List[str] = []
    tags = ["greeting", "example", "social"]

    # Default greeting templates
    GREETING_TEMPLATES = {
        "morning": [
            "Good morning, {name}! Hope you have a great day!",
            "Rise and shine, {name}! Ready for an amazing day?",
            "Morning, {name}! What can I help you with today?",
        ],
        "afternoon": [
            "Good afternoon, {name}! How's your day going?",
            "Hey {name}! What can I do for you this afternoon?",
            "Afternoon, {name}! Let's get productive!",
        ],
        "evening": [
            "Good evening, {name}! Winding down?",
            "Evening, {name}! How can I help tonight?",
            "Hey {name}! Still going strong this evening?",
        ],
        "night": [
            "Working late, {name}? I'm here to help!",
            "Good night... or is it? Hi {name}!",
            "Burning the midnight oil, {name}? Let me assist!",
        ],
        "default": [
            "Hello, {name}! How can I help you?",
            "Hi there, {name}! What can I do for you?",
            "Greetings, {name}! Ready to assist!",
        ],
    }

    FAREWELL_TEMPLATES = [
        "Goodbye, {name}! Have a great day!",
        "See you later, {name}!",
        "Take care, {name}! Until next time.",
        "Bye, {name}! Come back anytime!",
    ]

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        custom_greetings: Optional[Dict[str, List[str]]] = None,
        default_name: str = "there",
        **kwargs,
    ):
        """Initialize the custom greet plugin.

        Args:
            hook_manager: Optional hook manager.
            custom_greetings: Optional custom greeting templates.
            default_name: Default name for greetings.
        """
        self.hook_manager = hook_manager
        self.default_name = default_name
        self._active = False
        self._greet_count = 0

        # Merge custom greetings with defaults
        self.greetings = dict(self.GREETING_TEMPLATES)
        if custom_greetings:
            for time_of_day, templates in custom_greetings.items():
                if time_of_day in self.greetings:
                    self.greetings[time_of_day].extend(templates)
                else:
                    self.greetings[time_of_day] = templates

    def activate(self) -> None:
        """Activate the custom greet plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "on_greeting",
                self._handle_greeting,
                name="custom_greet_handler",
                priority=HookPriority.HIGH,
                owner=self.name,
            )
            self.hook_manager.register(
                "on_farewell",
                self._handle_farewell,
                name="custom_farewell_handler",
                priority=HookPriority.HIGH,
                owner=self.name,
            )
            self.hook_manager.register(
                "on_user_join",
                self._handle_user_join,
                name="custom_join_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("Custom greet plugin activated.")

    def deactivate(self) -> None:
        """Deactivate the custom greet plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("Custom greet plugin deactivated. Total greetings: %d", self._greet_count)

    def greet(self, name: Optional[str] = None) -> str:
        """Generate a time-aware greeting.

        Args:
            name: Name to include in the greeting.

        Returns:
            Greeting string.
        """
        name = name or self.default_name
        time_of_day = self._get_time_of_day()

        templates = self.greetings.get(time_of_day, self.greetings["default"])

        # Use greet count to cycle through templates
        template = templates[self._greet_count % len(templates)]
        self._greet_count += 1

        return template.format(name=name)

    def farewell(self, name: Optional[str] = None) -> str:
        """Generate a farewell message.

        Args:
            name: Name to include in the farewell.

        Returns:
            Farewell string.
        """
        name = name or self.default_name
        template = self.FAREWELL_TEMPLATES[self._greet_count % len(self.FAREWELL_TEMPLATES)]
        return template.format(name=name)

    def get_greeting_stats(self) -> Dict[str, Any]:
        """Get greeting statistics."""
        return {
            "success": True,
            "total_greetings": self._greet_count,
            "time_of_day": self._get_time_of_day(),
            "available_templates": {k: len(v) for k, v in self.greetings.items()},
        }

    def _get_time_of_day(self) -> str:
        """Determine the current time of day."""
        hour = time.localtime().tm_hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _handle_greeting(self, result, *args, **kwargs):
        """Handle greeting hook events."""
        name = kwargs.get("name", self.default_name)
        return self.greet(name)

    def _handle_farewell(self, result, *args, **kwargs):
        """Handle farewell hook events."""
        name = kwargs.get("name", self.default_name)
        return self.farewell(name)

    def _handle_user_join(self, result, *args, **kwargs):
        """Handle user join events."""
        name = kwargs.get("name", self.default_name)
        return self.greet(name)
