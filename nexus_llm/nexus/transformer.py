"""Nexus-LLM Text Transformer.

Provides the Transformer class for applying text transformations such as
cleaning, normalization, formatting, template expansion, and encoding
conversions.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result from a text transformation.

    Attributes:
        original: Original input text.
        transformed: Transformed output text.
        transform_name: Name of the transform applied.
        changes_count: Number of changes made.
        metadata: Additional transform metadata.
    """

    original: str = ""
    transformed: str = ""
    transform_name: str = ""
    changes_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Transformer:
    """Text transformer for applying sequential transformations.

    The Transformer supports built-in transformations and allows
    registering custom transforms. Transforms can be chained together
    in a pipeline.

    Example::

        t = Transformer()
        result = t.apply("  Hello   World  ", "normalize_whitespace")
        assert result.transformed == "Hello World"

        pipeline = ["lowercase", "normalize_whitespace", "strip"]
        result = t.apply_pipeline("  Hello   World  ", pipeline)
    """

    def __init__(self) -> None:
        self._transforms: Dict[str, Callable[[str], str]] = {}
        self._register_builtin_transforms()
        logger.debug("Transformer initialized with %d built-in transforms", len(self._transforms))

    def register_transform(self, name: str, fn: Callable[[str], str]) -> None:
        """Register a custom transformation.

        Args:
            name: Unique name for the transform.
            fn: Callable that accepts a string and returns a transformed string.
        """
        self._transforms[name] = fn
        logger.debug("Registered custom transform: %s", name)

    def apply(self, text: str, transform_name: str) -> TransformResult:
        """Apply a single named transformation to text.

        Args:
            text: Input text.
            transform_name: Name of the transform to apply.

        Returns:
            A TransformResult with the output.

        Raises:
            ValueError: If the transform name is not registered.
        """
        if transform_name not in self._transforms:
            raise ValueError(f"Unknown transform: {transform_name}. Available: {list(self._transforms.keys())}")

        transformed = self._transforms[transform_name](text)
        return TransformResult(
            original=text,
            transformed=transformed,
            transform_name=transform_name,
            changes_count=sum(1 for a, b in zip(text, transformed) if a != b) + abs(len(text) - len(transformed)),
        )

    def apply_pipeline(self, text: str, transforms: List[str]) -> TransformResult:
        """Apply a sequence of named transformations.

        Args:
            text: Input text.
            transforms: List of transform names to apply in order.

        Returns:
            A TransformResult from the final transformation.

        Raises:
            ValueError: If any transform name is not registered.
        """
        current = text
        total_changes = 0
        for name in transforms:
            result = self.apply(current, name)
            current = result.transformed
            total_changes += result.changes_count

        return TransformResult(
            original=text,
            transformed=current,
            transform_name=" -> ".join(transforms),
            changes_count=total_changes,
            metadata={"pipeline": transforms},
        )

    def apply_custom(self, text: str, fn: Callable[[str], str], name: str = "custom") -> TransformResult:
        """Apply a one-off custom transformation.

        Args:
            text: Input text.
            fn: Transformation function.
            name: Optional name for logging.

        Returns:
            A TransformResult with the output.
        """
        transformed = fn(text)
        return TransformResult(
            original=text,
            transformed=transformed,
            transform_name=name,
            changes_count=sum(1 for a, b in zip(text, transformed) if a != b) + abs(len(text) - len(transformed)),
        )

    def available_transforms(self) -> List[str]:
        """Return list of registered transform names."""
        return list(self._transforms.keys())

    def _register_builtin_transforms(self) -> None:
        """Register built-in text transformations."""
        self._transforms["strip"] = lambda t: t.strip()
        self._transforms["lowercase"] = lambda t: t.lower()
        self._transforms["uppercase"] = lambda t: t.upper()
        self._transforms["title_case"] = lambda t: t.title()
        self._transforms["capitalize"] = lambda t: t.capitalize()
        self._transforms["normalize_whitespace"] = lambda t: re.sub(r'\s+', ' ', t).strip()
        self._transforms["remove_urls"] = lambda t: re.sub(r'https?://\S+|www\.\S+', '', t)
        self._transforms["remove_emails"] = lambda t: re.sub(r'\S+@\S+\.\S+', '', t)
        self._transforms["remove_html_tags"] = lambda t: re.sub(r'<[^>]+>', '', t)
        self._transforms["remove_digits"] = lambda t: re.sub(r'\d+', '', t)
        self._transforms["remove_punctuation"] = self._remove_punctuation
        self._transforms["normalize_unicode"] = lambda t: unicodedata.normalize('NFKC', t)
        self._transforms["remove_emojis"] = self._remove_emojis
        self._transforms["collapse_lines"] = lambda t: re.sub(r'\n{3,}', '\n\n', t)
        self._transforms["trim_lines"] = lambda t: '\n'.join(line.strip() for line in t.split('\n'))
        self._transforms["snake_case"] = self._to_snake_case
        self._transforms["camel_case"] = self._to_camel_case
        self._transforms["kebab_case"] = self._to_kebab_case
        self._transforms["deduplicate_lines"] = lambda t: '\n'.join(dict.fromkeys(t.split('\n')))
        self._transforms["remove_blank_lines"] = lambda t: '\n'.join(line for line in t.split('\n') if line.strip())

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation from text."""
        import string
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def _remove_emojis(text: str) -> str:
        """Remove emoji characters from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub('', text)

    @staticmethod
    def _to_snake_case(text: str) -> str:
        """Convert text to snake_case."""
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
        text = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', text)
        text = text.replace('-', '_').replace(' ', '_')
        return text.lower()

    @staticmethod
    def _to_camel_case(text: str) -> str:
        """Convert text to camelCase."""
        parts = re.split(r'[_\-\s]+', text)
        if not parts:
            return text
        return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])

    @staticmethod
    def _to_kebab_case(text: str) -> str:
        """Convert text to kebab-case."""
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1-\2', text)
        text = re.sub(r'([a-z\d])([A-Z])', r'\1-\2', text)
        text = text.replace('_', '-').replace(' ', '-')
        return text.lower()
