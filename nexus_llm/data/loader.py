"""Nexus-LLM Data Loader Module.

Loads datasets from JSONL, CSV, JSON, Parquet, and HuggingFace datasets
with optional streaming support and automatic format detection.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset.

    Attributes:
        name: Dataset name or path.
        format: Source format (jsonl, csv, json, parquet, hf).
        num_rows: Total number of rows.
        num_columns: Number of columns/fields.
        columns: List of column names.
        size_mb: Approximate file size in MB.
    """

    name: str = ""
    format: str = ""
    num_rows: int = 0
    num_columns: int = 0
    columns: List[str] = field(default_factory=list)
    size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "columns": self.columns,
            "size_mb": round(self.size_mb, 2),
        }


class DataLoader:
    """Loads datasets from various file formats and HuggingFace Hub.

    Supported formats:
      - JSONL (.jsonl, .jsonlines)
      - CSV (.csv, .tsv)
      - JSON (.json)
      - Parquet (.parquet, .pq)
      - HuggingFace datasets (via datasets library)

    Streaming mode is supported for all formats, yielding one row at a
    time to minimize memory usage.

    Example::

        loader = DataLoader()
        data = loader.load("train.jsonl")
        for row in loader.stream("large_dataset.jsonl"):
            process(row)
    """

    FORMAT_EXTENSIONS = {
        ".jsonl": "jsonl",
        ".jsonlines": "jsonl",
        ".csv": "csv",
        ".tsv": "csv",
        ".json": "json",
        ".parquet": "parquet",
        ".pq": "parquet",
    }

    def load(
        self,
        source: str,
        format: Optional[str] = None,
        split: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Load a dataset into memory as a list of dicts.

        Args:
            source: File path, directory, or HuggingFace dataset ID.
            format: Force format override (jsonl, csv, json, parquet, hf).
            split: HuggingFace dataset split (train, test, validation).
            columns: Subset of columns to load.
            limit: Maximum number of rows to load.
            **kwargs: Format-specific options.

        Returns:
            List of row dictionaries.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            ValueError: If format cannot be detected.
        """
        fmt = format or self._detect_format(source)
        rows: List[Dict[str, Any]] = []

        if fmt == "jsonl":
            rows = self._load_jsonl(source, **kwargs)
        elif fmt == "csv":
            rows = self._load_csv(source, **kwargs)
        elif fmt == "json":
            rows = self._load_json(source, **kwargs)
        elif fmt == "parquet":
            rows = self._load_parquet(source, **kwargs)
        elif fmt == "hf":
            rows = self._load_hf(source, split=split, **kwargs)
        else:
            raise ValueError(f"Cannot detect format for '{source}'. Specify format= parameter.")

        if columns:
            rows = [{k: v for k, v in row.items() if k in columns} for row in rows]

        if limit is not None:
            rows = rows[:limit]

        return rows

    def stream(
        self,
        source: str,
        format: Optional[str] = None,
        split: Optional[str] = None,
        columns: Optional[List[str]] = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Stream dataset rows one at a time (memory-efficient).

        Args:
            source: File path or HuggingFace dataset ID.
            format: Force format override.
            split: HuggingFace dataset split.
            columns: Subset of columns.
            batch_size: Number of rows per yield (1 = row by row).
            **kwargs: Format-specific options.

        Yields:
            Row dictionaries.
        """
        fmt = format or self._detect_format(source)

        if fmt == "jsonl":
            iterator = self._stream_jsonl(source, **kwargs)
        elif fmt == "csv":
            iterator = self._stream_csv(source, **kwargs)
        elif fmt == "json":
            # JSON must be loaded fully first
            rows = self._load_json(source, **kwargs)
            iterator = iter(rows)
        elif fmt == "parquet":
            rows = self._load_parquet(source, **kwargs)
            iterator = iter(rows)
        elif fmt == "hf":
            iterator = self._stream_hf(source, split=split, **kwargs)
        else:
            raise ValueError(f"Cannot detect format for '{source}'.")

        batch: List[Dict[str, Any]] = []
        for row in iterator:
            if columns:
                row = {k: v for k, v in row.items() if k in columns}
            if batch_size <= 1:
                yield row
            else:
                batch.append(row)
                if len(batch) >= batch_size:
                    yield batch  # type: ignore[misc]
                    batch = []

    def info(self, source: str, format: Optional[str] = None) -> DatasetInfo:
        """Get metadata about a dataset without fully loading it.

        Args:
            source: File path or HuggingFace dataset ID.
            format: Force format override.

        Returns:
            DatasetInfo with row count, columns, and size.
        """
        fmt = format or self._detect_format(source)
        path = Path(source)
        size_mb = 0.0
        if path.exists():
            size_mb = path.stat().st_size / 1e6

        # Quick row count for JSONL
        if fmt == "jsonl" and path.exists():
            num_rows = sum(1 for _ in open(source))
            with open(source) as f:
                first_line = f.readline()
                sample = json.loads(first_line) if first_line.strip() else {}
            return DatasetInfo(
                name=source,
                format=fmt,
                num_rows=num_rows,
                num_columns=len(sample),
                columns=list(sample.keys()),
                size_mb=size_mb,
            )

        # Fall back to loading
        rows = self.load(source, format=fmt, limit=10)
        return DatasetInfo(
            name=source,
            format=fmt,
            num_rows=len(rows),
            columns=list(rows[0].keys()) if rows else [],
            num_columns=len(rows[0]) if rows else 0,
            size_mb=size_mb,
        )

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str, **kwargs: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _stream_jsonl(path: str, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _load_csv(path: str, delimiter: str = "", **kwargs: Any) -> List[Dict[str, Any]]:
        sep = delimiter or ("\t" if path.endswith(".tsv") else ",")
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                rows.append(dict(row))
        return rows

    @staticmethod
    def _stream_csv(path: str, delimiter: str = "", **kwargs: Any) -> Iterator[Dict[str, Any]]:
        sep = delimiter or ("\t" if path.endswith(".tsv") else ",")
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                yield dict(row)

    @staticmethod
    def _load_json(path: str, **kwargs: Any) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Try common keys
            for key in ("data", "records", "items", "rows", "examples"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return []

    @staticmethod
    def _load_parquet(path: str, **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        except ImportError:
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                return table.to_pylist()
            except ImportError:
                raise RuntimeError(
                    "pandas or pyarrow required for Parquet loading."
                )

    @staticmethod
    def _load_hf(
        dataset_id: str, split: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split=split or "train")
            return [dict(row) for row in ds]
        except ImportError:
            raise RuntimeError("datasets library required for HuggingFace loading.")

    @staticmethod
    def _stream_hf(
        dataset_id: str, split: Optional[str] = None, **kwargs: Any
    ) -> Iterator[Dict[str, Any]]:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split=split or "train", streaming=True)
            for row in ds:
                yield dict(row)
        except ImportError:
            raise RuntimeError("datasets library required for HuggingFace streaming.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_format(self, source: str) -> str:
        """Auto-detect format from file extension or source pattern."""
        path = Path(source)
        ext = path.suffix.lower()
        if ext in self.FORMAT_EXTENSIONS:
            return self.FORMAT_EXTENSIONS[ext]

        # HuggingFace pattern: org/name or org/name (no extension)
        if "/" in source and not path.exists():
            return "hf"

        # Directory with dataset files
        if path.is_dir():
            for f in path.iterdir():
                if f.suffix.lower() in self.FORMAT_EXTENSIONS:
                    return self.FORMAT_EXTENSIONS[f.suffix.lower()]

        return ""
