"""Nexus-LLM Data Processor Module.

Provides chainable data transformations with both lazy and eager execution
modes. Supports filtering, mapping, column operations, shuffling, sampling,
batching, streaming, deduplication, and sorting.

The ``DataProcessor`` class builds a pipeline of transformation steps that
can be executed either eagerly (immediate) or lazily (deferred until
``collect()`` is called), enabling efficient data processing for large
datasets.

Standalone helper functions are also provided for quick one-off
transformations.

Example::

    from nexus_llm.data.processor import DataProcessor

    proc = DataProcessor(mode="lazy")
    result = (
        proc.filter(lambda row: len(row["text"]) > 10)
        .map(lambda row: {"text": row["text"].lower(), **row})
        .rename("old_field", "new_field")
        .select(["text", "label"])
        .shuffle(seed=42)
        .collect(data)
    )
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass for a single processing step
# ---------------------------------------------------------------------------

@dataclass
class ProcessingStep:
    """A named, composable transformation step.

    Attributes:
        name: Human-readable step name (used for logging and introspection).
        fn: Callable that transforms a list of row-dicts and returns a new list.
        description: Optional longer description of what the step does.
    """

    name: str
    fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
    description: str = ""


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def filter_empty(data: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    """Remove rows where *text_key* is blank or whitespace-only."""
    return [d for d in data if d.get(text_key, "").strip()]


def filter_by_length(
    data: List[Dict[str, Any]],
    min_length: int = 1,
    max_length: int = 10000,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Keep rows whose *text_key* length falls within [min_length, max_length]."""
    return [d for d in data if min_length <= len(d.get(text_key, "")) <= max_length]


def normalize_text(data: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    """Collapse runs of whitespace into single spaces and strip."""
    import re
    result: List[Dict[str, Any]] = []
    for d in data:
        new_d = dict(d)
        new_d[text_key] = re.sub(r"\s+", " ", new_d.get(text_key, "")).strip()
        result.append(new_d)
    return result


def lowercase_text(data: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
    """Lowercase the *text_key* field in every row."""
    result: List[Dict[str, Any]] = []
    for d in data:
        new_d = dict(d)
        new_d[text_key] = new_d.get(text_key, "").lower()
        result.append(new_d)
    return result


def deduplicate(data: List[Dict[str, Any]], key: str = "text") -> List[Dict[str, Any]]:
    """Remove rows with duplicate values for *key*, keeping the first occurrence."""
    seen: set = set()
    result: List[Dict[str, Any]] = []
    for d in data:
        val = d.get(key, "")
        if val not in seen:
            seen.add(val)
            result.append(d)
    return result


def sample_data(data: List[Dict[str, Any]], n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Return *n* randomly sampled rows (without replacement)."""
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def rename_key(
    data: List[Dict[str, Any]], old_key: str, new_key: str
) -> List[Dict[str, Any]]:
    """Rename *old_key* to *new_key* in every row that contains it."""
    result: List[Dict[str, Any]] = []
    for d in data:
        new_d = dict(d)
        if old_key in new_d:
            new_d[new_key] = new_d.pop(old_key)
        result.append(new_d)
    return result


# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------

class DataProcessor:
    """Chainable data transformation pipeline with lazy/eager execution.

    Steps are added via fluent methods (``filter``, ``map``, ``rename``,
    etc.) and executed either eagerly (``mode="eager"``) after each call
    or lazily (``mode="lazy"``) when :meth:`collect` is invoked.

    Args:
        mode: ``"eager"`` applies each step immediately;
              ``"lazy"`` defers execution until :meth:`collect`.

    Example::

        proc = DataProcessor(mode="lazy")
        result = proc.filter(lambda r: r["score"] > 0.5) \\
                     .map(lambda r: {**r, "text": r["text"].lower()}) \\
                     .collect(data)
    """

    def __init__(self, mode: str = "eager") -> None:
        if mode not in ("eager", "lazy"):
            raise ValueError(f"mode must be 'eager' or 'lazy', got '{mode}'")
        self._mode = mode
        self._steps: List[ProcessingStep] = []
        self._data: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_step(self, step: ProcessingStep) -> None:
        """Apply a step immediately (eager mode)."""
        if self._data is not None:
            self._data = step.fn(self._data)
        self._steps.append(step)

    def _add_step(self, name: str, fn: Callable, description: str = "") -> "DataProcessor":
        """Register a step; apply immediately in eager mode."""
        step = ProcessingStep(name=name, fn=fn, description=description)
        if self._mode == "eager" and self._data is not None:
            self._data = step.fn(self._data)
        self._steps.append(step)
        return self

    # ------------------------------------------------------------------
    # Fluent transformation API
    # ------------------------------------------------------------------

    def filter(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        description: str = "",
    ) -> "DataProcessor":
        """Keep only rows for which *predicate* returns True."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [row for row in data if predicate(row)]

        return self._add_step("filter", _fn, description or "filter(predicate)")

    def map(
        self,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
        description: str = "",
    ) -> "DataProcessor":
        """Apply *transform* to every row."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [transform(row) for row in data]

        return self._add_step("map", _fn, description or "map(transform)")

    def rename(self, old_key: str, new_key: str) -> "DataProcessor":
        """Rename a column from *old_key* to *new_key*."""
        desc = f"rename({old_key!r} -> {new_key!r})"
        return self._add_step("rename", lambda data: rename_key(data, old_key, new_key), desc)

    def select(self, columns: List[str]) -> "DataProcessor":
        """Keep only the specified *columns* in each row."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [{k: v for k, v in row.items() if k in columns} for row in data]

        return self._add_step("select", _fn, f"select({columns})")

    def drop(self, columns: List[str]) -> "DataProcessor":
        """Drop the specified *columns* from each row."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [{k: v for k, v in row.items() if k not in columns} for row in data]

        return self._add_step("drop", _fn, f"drop({columns})")

    def shuffle(self, seed: int = 42) -> "DataProcessor":
        """Shuffle the row order using the given random *seed*."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            items = list(data)
            random.seed(seed)
            random.shuffle(items)
            return items

        return self._add_step("shuffle", _fn, f"shuffle(seed={seed})")

    def sample(self, n: int, seed: int = 42) -> "DataProcessor":
        """Randomly sample *n* rows without replacement."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return sample_data(data, n=n, seed=seed)

        return self._add_step("sample", _fn, f"sample(n={n}, seed={seed})")

    def batch(self, batch_size: int) -> "DataProcessor":
        """Group rows into batches of *batch_size*.

        Returns a list of lists (each inner list is a batch).
        """

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            # Flatten-friendly: we store batches as special rows
            batches: List[Dict[str, Any]] = []
            for i in range(0, len(data), batch_size):
                batches.append({"_batch": data[i : i + batch_size], "_batch_index": i // batch_size})
            return batches

        return self._add_step("batch", _fn, f"batch(batch_size={batch_size})")

    def unique(self, key: str = "text") -> "DataProcessor":
        """Remove rows with duplicate values for *key*, keeping the first."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return deduplicate(data, key=key)

        return self._add_step("unique", _fn, f"unique(key={key!r})")

    def sort(self, key: str, reverse: bool = False) -> "DataProcessor":
        """Sort rows by the value of *key*."""

        def _fn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return sorted(data, key=lambda row: row.get(key, ""), reverse=reverse)

        return self._add_step("sort", _fn, f"sort(key={key!r}, reverse={reverse})")

    # ------------------------------------------------------------------
    # Step management (backwards-compatible)
    # ------------------------------------------------------------------

    def add_step(self, name: str, fn: Callable, description: str = "") -> None:
        """Add a processing step (non-fluent, backwards-compatible API)."""
        step = ProcessingStep(name=name, fn=fn, description=description)
        if self._mode == "eager" and self._data is not None:
            self._data = step.fn(self._data)
        self._steps.append(step)

    def remove_step(self, name: str) -> None:
        """Remove all steps matching *name*."""
        self._steps = [s for s in self._steps if s.name != name]

    def clear(self) -> None:
        """Remove all processing steps and reset internal data."""
        self._steps.clear()
        self._data = None

    def list_steps(self) -> List[str]:
        """Return the names of all registered steps."""
        return [s.name for s in self._steps]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute all steps on *data* and return the result.

        In eager mode this is a no-op if data was already set via
        :meth:`feed`; in lazy mode this runs the full pipeline.
        """
        result = list(data)
        for step in self._steps:
            result = step.fn(result)
        return result

    def process_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all steps on a single row."""
        result = dict(item)
        for step in self._steps:
            wrapped = step.fn([result])
            if not wrapped:
                return {}  # filtered out
            result = wrapped[0]
        return result

    def feed(self, data: List[Dict[str, Any]]) -> "DataProcessor":
        """Provide input data for eager-mode processing."""
        self._data = list(data)
        if self._mode == "eager":
            for step in self._steps:
                self._data = step.fn(self._data)
        return self

    def collect(self, data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Run the pipeline and return results.

        Args:
            data: Optional input data. If None, uses data previously
                  set via :meth:`feed`.

        Returns:
            Transformed list of row dictionaries.
        """
        if data is not None:
            self._data = list(data)
            for step in self._steps:
                self._data = step.fn(self._data)
        elif self._mode == "lazy" and self._data is not None:
            for step in self._steps:
                self._data = step.fn(self._data)
        return self._data if self._data is not None else []

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        data: Sequence[Dict[str, Any]],
        batch_size: int = 1,
    ) -> Iterator[Dict[str, Any]]:
        """Apply all steps incrementally and yield rows.

        This is a memory-friendly way to process large datasets: rows
        are processed through the pipeline in chunks and yielded one at
        a time (or *batch_size* at a time).

        Args:
            data: Input sequence of row dicts.
            batch_size: Number of rows to yield per iteration.

        Yields:
            Transformed row dictionaries.
        """
        buffer: List[Dict[str, Any]] = []
        chunk_size = max(1000, batch_size * 10)

        for i in range(0, len(data), chunk_size):
            chunk = list(data[i : i + chunk_size])
            for step in self._steps:
                chunk = step.fn(chunk)
            for row in chunk:
                if batch_size <= 1:
                    yield row
                else:
                    buffer.append(row)
                    if len(buffer) >= batch_size:
                        yield buffer  # type: ignore[misc]
                        buffer = []

        if buffer:
            yield buffer  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Return the execution mode ('eager' or 'lazy')."""
        return self._mode

    @property
    def num_steps(self) -> int:
        """Return the number of registered steps."""
        return len(self._steps)

    def describe(self) -> str:
        """Return a human-readable description of the pipeline."""
        lines = [f"DataProcessor(mode={self._mode!r}, steps={len(self._steps)})"]
        for i, step in enumerate(self._steps, 1):
            desc = f" - {step.description}" if step.description else ""
            lines.append(f"  {i}. {step.name}{desc}")
        return "\n".join(lines)
