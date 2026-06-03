"""Echo plugin example.

A minimal example plugin that echoes back messages with
optional transformations, demonstrating the basic plugin
interface and hook integration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class EchoPlugin:
    """Example plugin that echoes messages with optional transformations.

    The simplest functional plugin demonstrating:
    - Plugin activation/deactivation lifecycle
    - Hook registration and handling
    - Message transformation
    - State management
    """

    name = "echo"
    version = "1.0.0"
    description = "Echo plugin that repeats messages with optional transformations"
    dependencies: List[str] = []
    tags = ["echo", "example", "utility"]

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        transform: str = "none",
        prefix: str = "",
        suffix: str = "",
        max_length: int = 1000,
        **kwargs,
    ):
        """Initialize the echo plugin.

        Args:
            hook_manager: Optional hook manager.
            transform: Transformation mode ('none', 'upper', 'lower', 'reverse', 'title', 'repeat').
            prefix: String to prepend to echoed messages.
            suffix: String to append to echoed messages.
            max_length: Maximum message length to echo.
        """
        self.hook_manager = hook_manager
        self.transform = transform
        self.prefix = prefix
        self.suffix = suffix
        self.max_length = max_length
        self._active = False
        self._echo_count = 0

    def activate(self) -> None:
        """Activate the echo plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "on_message",
                self._handle_message,
                name="echo_message_handler",
                priority=HookPriority.LOW,  # Low priority so other plugins process first
                owner=self.name,
            )
            self.hook_manager.register(
                "on_echo_request",
                self._handle_echo_request,
                name="echo_request_handler",
                priority=HookPriority.HIGH,
                owner=self.name,
            )
        self._active = True
        logger.info("Echo plugin activated (transform=%s).", self.transform)

    def deactivate(self) -> None:
        """Deactivate the echo plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("Echo plugin deactivated. Total echoes: %d", self._echo_count)

    def echo(self, message: str, transform: Optional[str] = None) -> Dict[str, Any]:
        """Echo a message with optional transformation.

        Args:
            message: The message to echo.
            transform: Optional transform override.

        Returns:
            Dictionary with echoed message and metadata.
        """
        if not message:
            return {"success": False, "error": "Empty message", "original": message}

        # Truncate if needed
        original = message
        truncated = False
        if len(message) > self.max_length:
            message = message[:self.max_length]
            truncated = True

        # Apply transformation
        active_transform = transform or self.transform
        transformed = self._apply_transform(message, active_transform)

        # Apply prefix/suffix
        if self.prefix:
            transformed = f"{self.prefix}{transformed}"
        if self.suffix:
            transformed = f"{transformed}{self.suffix}"

        self._echo_count += 1

        return {
            "success": True,
            "echo": transformed,
            "original": original,
            "transform": active_transform,
            "truncated": truncated,
            "echo_count": self._echo_count,
        }

    def _apply_transform(self, message: str, transform: str) -> str:
        """Apply a text transformation.

        Args:
            message: The message to transform.
            transform: Transformation name.

        Returns:
            Transformed message.
        """
        transforms = {
            "none": lambda m: m,
            "upper": lambda m: m.upper(),
            "lower": lambda m: m.lower(),
            "reverse": lambda m: m[::-1],
            "title": lambda m: m.title(),
            "capitalize": lambda m: m.capitalize(),
            "swapcase": lambda m: m.swapcase(),
            "repeat": lambda m: m + " " + m,
            "word_reverse": lambda m: " ".join(m.split()[::-1]),
            "character_count": lambda m: f"{m} ({len(m)} chars)",
        }

        transform_fn = transforms.get(transform, transforms["none"])
        return transform_fn(message)

    def get_available_transforms(self) -> List[str]:
        """Get list of available transformations."""
        return ["none", "upper", "lower", "reverse", "title",
                "capitalize", "swapcase", "repeat", "word_reverse",
                "character_count"]

    def get_stats(self) -> Dict[str, Any]:
        """Get echo plugin statistics."""
        return {
            "success": True,
            "total_echoes": self._echo_count,
            "transform": self.transform,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "max_length": self.max_length,
            "active": self._active,
        }

    def _handle_message(self, result, *args, **kwargs):
        """Handle message hook events."""
        message = kwargs.get("message", "")
        if message:
            echo_result = self.echo(message)
            return echo_result.get("echo", message)
        return result

    def _handle_echo_request(self, result, *args, **kwargs):
        """Handle explicit echo request events."""
        message = kwargs.get("message", kwargs.get("text", ""))
        transform = kwargs.get("transform", self.transform)
        if message:
            echo_result = self.echo(message, transform=transform)
            return echo_result.get("echo", message)
        return result
