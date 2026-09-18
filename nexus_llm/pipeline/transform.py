"""Nexus-LLM Data Transforms.

Provides the DataTransform and TransformPipeline classes for
applying structured data transformations in processing pipelines.
"""

import copy
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataTransform:
    """A named, reversible data transformation.

    Each DataTransform wraps a function that transforms data and
    optionally an inverse function for reversal.

    Attributes:
        name: Transform name.
        description: Human-readable description.

    Example::

        upper = DataTransform("uppercase", lambda d: d.upper(), inverse=lambda d: d.lower())
        result = upper.apply("hello")  # "HELLO"
        original = upper.reverse(result)  # "hello"
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Any], Any],
        inverse: Optional[Callable[[Any], Any]] = None,
        description: str = "",
    ) -> None:
        self._name = name
        self._fn = fn
        self._inverse = inverse
        self._description = description
        logger.debug("DataTransform created: %s", name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_reversible(self) -> bool:
        """Whether this transform can be reversed."""
        return self._inverse is not None

    def apply(self, data: Any) -> Any:
        """Apply the forward transformation.

        Args:
            data: Input data.

        Returns:
            Transformed data.
        """
        return self._fn(data)

    def reverse(self, data: Any) -> Any:
        """Apply the inverse transformation.

        Args:
            data: Transformed data.

        Returns:
            Original data.

        Raises:
            ValueError: If the transform is not reversible.
        """
        if self._inverse is None:
            raise ValueError(f"Transform '{self._name}' is not reversible")
        return self._inverse(data)


class TransformPipeline:
    """Sequential pipeline of data transforms.

    Applies a chain of DataTransform objects to data, supporting
    both forward and reverse execution.

    Example::

        pipeline = TransformPipeline()
        pipeline.add(DataTransform("strip", lambda d: d.strip()))
        pipeline.add(DataTransform("upper", lambda d: d.upper()))
        result = pipeline.apply("  hello  ")  # "HELLO"
    """

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._transforms: List[DataTransform] = []
        logger.debug("TransformPipeline created: %s", name or "unnamed")

    @property
    def name(self) -> str:
        return self._name

    @property
    def transform_count(self) -> int:
        return len(self._transforms)

    def add(self, transform: DataTransform) -> None:
        """Add a transform to the pipeline.

        Args:
            transform: The DataTransform to add.
        """
        self._transforms.append(transform)
        logger.debug("Added transform '%s' to pipeline", transform.name)

    def remove(self, name: str) -> bool:
        """Remove a transform by name.

        Args:
            name: Transform name.

        Returns:
            True if found and removed.
        """
        for i, t in enumerate(self._transforms):
            if t.name == name:
                self._transforms.pop(i)
                return True
        return False

    def apply(self, data: Any) -> Any:
        """Apply all transforms in order.

        Args:
            data: Input data.

        Returns:
            Transformed data.
        """
        current = copy.deepcopy(data) if isinstance(data, (dict, list)) else data
        for transform in self._transforms:
            current = transform.apply(current)
        return current

    def reverse(self, data: Any) -> Any:
        """Apply inverse transforms in reverse order.

        Args:
            data: Transformed data.

        Returns:
            Reversed data.

        Raises:
            ValueError: If any transform is not reversible.
        """
        current = copy.deepcopy(data) if isinstance(data, (dict, list)) else data
        for transform in reversed(self._transforms):
            current = transform.reverse(current)
        return current

    def list_transforms(self) -> List[str]:
        """Return ordered list of transform names."""
        return [t.name for t in self._transforms]

    def is_reversible(self) -> bool:
        """Check if all transforms in the pipeline are reversible."""
        return all(t.is_reversible for t in self._transforms)
