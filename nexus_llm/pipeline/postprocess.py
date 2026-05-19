"""Nexus-LLM Postprocessing Pipeline.

Provides the Postprocessor class for applying transformations to
LLM output text before returning it to the user.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PostprocessStep:
    """A single postprocessing step.

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


class Postprocessor:
    """Sequential postprocessing pipeline for LLM outputs.

    The Postprocessor applies a configurable chain of transformations
    to clean and format model output before returning to the user.

    Built-in steps include: trim_whitespace, remove_incomplete_sentences,
    format_code_blocks, escape_markdown.

    Example::

        pp = Postprocessor()
        result = pp.process("  Hello world. This is a test  ")
    """

    def __init__(self) -> None:
        self._steps: Dict[str, PostprocessStep] = {}
        self._register_builtin_steps()
        logger.debug("Postprocessor initialized")

    def add_step(self, name: str, fn: Callable[[str], str], enabled: bool = True, order: int = 0) -> None:
        """Add a postprocessing step.

        Args:
            name: Unique step name.
            fn: Transformation function.
            enabled: Whether the step is active.
            order: Execution order.
        """
        self._steps[name] = PostprocessStep(name=name, fn=fn, enabled=enabled, order=order)
        logger.debug("Added postprocess step: %s (order=%d)", name, order)

    def remove_step(self, name: str) -> bool:
        """Remove a postprocessing step."""
        return self._steps.pop(name, None) is not None

    def enable_step(self, name: str) -> None:
        """Enable a postprocessing step."""
        if name in self._steps:
            self._steps[name].enabled = True

    def disable_step(self, name: str) -> None:
        """Disable a postprocessing step."""
        if name in self._steps:
            self._steps[name].enabled = False

    def process(self, text: str) -> str:
        """Apply all enabled postprocessing steps in order.

        Args:
            text: LLM output text.

        Returns:
            Postprocessed text.
        """
        current = text
        for step in self._ordered_steps():
            if not step.enabled:
                continue
            try:
                current = step.fn(current)
            except Exception as exc:
                logger.warning("Postprocess step '%s' failed: %s", step.name, exc)
        return current

    def list_steps(self) -> List[str]:
        """Return ordered list of step names."""
        return [s.name for s in self._ordered_steps()]

    def _ordered_steps(self) -> List[PostprocessStep]:
        """Return steps sorted by order."""
        return sorted(self._steps.values(), key=lambda s: s.order)

    def _register_builtin_steps(self) -> None:
        """Register built-in postprocessing steps."""
        self.add_step(
            "trim_whitespace",
            lambda t: t.strip(),
            order=0,
        )
        self.add_step(
            "normalize_whitespace",
            lambda t: re.sub(r'[ \t]+', ' ', t),
            order=1,
        )
        self.add_step(
            "remove_incomplete_sentences",
            self._remove_incomplete_sentences,
            enabled=False,
            order=10,
        )
        self.add_step(
            "collapse_blank_lines",
            lambda t: re.sub(r'\n{3,}', '\n\n', t),
            order=20,
        )
        self.add_step(
            "ensure_trailing_newline",
            lambda t: t if t.endswith('\n') else t + '\n',
            enabled=False,
            order=50,
        )

    @staticmethod
    def _remove_incomplete_sentences(text: str) -> str:
        """Remove the last sentence if it appears incomplete."""
        # Find the last sentence-ending punctuation
        match = None
        for m in re.finditer(r'[.!?]\s', text):
            match = m
        if match:
            return text[:match.end()].strip()
        return text
