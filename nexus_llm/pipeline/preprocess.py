"""Nexus-LLM Preprocessing Pipeline.

Provides the Preprocessor class for applying sequential preprocessing
steps to text before it is sent to an LLM.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreprocessStep:
    """A single preprocessing step.

    Attributes:
        name: Step name.
        fn: Callable that transforms text.
        enabled: Whether the step is active.
        order: Execution order (lower runs first).
    """

    name: str
    fn: Callable[[str], str]
    enabled: bool = True
    order: int = 0


class Preprocessor:
    """Sequential preprocessing pipeline for LLM inputs.

    The Preprocessor applies a configurable chain of text transformations
    to clean and normalize input before model inference.

    Built-in steps include: trim_whitespace, normalize_unicode,
    remove_control_chars, truncate, deduplicate_lines.

    Example::

        pp = Preprocessor()
        pp.add_step("custom_lower", lambda t: t.lower(), order=10)
        result = pp.process("  Hello   World  ")
    """

    def __init__(self, max_length: int = 100000) -> None:
        self._steps: Dict[str, PreprocessStep] = {}
        self._max_length = max_length
        self._register_builtin_steps()
        logger.debug("Preprocessor initialized with max_length=%d", max_length)

    def add_step(self, name: str, fn: Callable[[str], str], enabled: bool = True, order: int = 0) -> None:
        """Add a preprocessing step.

        Args:
            name: Unique step name.
            fn: Transformation function.
            enabled: Whether the step is active.
            order: Execution order.
        """
        self._steps[name] = PreprocessStep(name=name, fn=fn, enabled=enabled, order=order)
        logger.debug("Added preprocess step: %s (order=%d)", name, order)

    def remove_step(self, name: str) -> bool:
        """Remove a preprocessing step.

        Args:
            name: Step name.

        Returns:
            True if the step was found and removed.
        """
        return self._steps.pop(name, None) is not None

    def enable_step(self, name: str) -> None:
        """Enable a preprocessing step."""
        if name in self._steps:
            self._steps[name].enabled = True

    def disable_step(self, name: str) -> None:
        """Disable a preprocessing step."""
        if name in self._steps:
            self._steps[name].enabled = False

    def process(self, text: str) -> str:
        """Apply all enabled preprocessing steps in order.

        Args:
            text: Input text.

        Returns:
            Preprocessed text.
        """
        current = text
        for step in self._ordered_steps():
            if not step.enabled:
                continue
            try:
                current = step.fn(current)
            except Exception as exc:
                logger.warning("Preprocess step '%s' failed: %s", step.name, exc)
        return current

    def list_steps(self) -> List[str]:
        """Return ordered list of step names."""
        return [s.name for s in self._ordered_steps()]

    def _ordered_steps(self) -> List[PreprocessStep]:
        """Return steps sorted by order."""
        return sorted(self._steps.values(), key=lambda s: s.order)

    def _register_builtin_steps(self) -> None:
        """Register built-in preprocessing steps."""
        self.add_step(
            "normalize_unicode",
            lambda t: unicodedata.normalize("NFKC", t),
            order=0,
        )
        self.add_step(
            "remove_control_chars",
            lambda t: "".join(ch for ch in t if ch.isprintable() or ch in "\n\r\t"),
            order=1,
        )
        self.add_step(
            "trim_whitespace",
            lambda t: re.sub(r'[ \t]+', ' ', t).strip(),
            order=2,
        )
        self.add_step(
            "normalize_newlines",
            lambda t: re.sub(r'\r\n?', '\n', t),
            order=3,
        )
        self.add_step(
            "collapse_blank_lines",
            lambda t: re.sub(r'\n{3,}', '\n\n', t),
            order=4,
        )
        self.add_step(
            "truncate",
            self._truncate,
            enabled=False,
            order=100,
        )

    def _truncate(self, text: str) -> str:
        """Truncate text to max_length."""
        if len(text) <= self._max_length:
            return text
        return text[:self._max_length - 3] + "..."
