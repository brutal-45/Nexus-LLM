"""I/O utilities: file read/write, JSON, YAML, JSONL, CSV, async I/O, path management."""

import os
import json
import csv
import asyncio
import aiofiles
import logging
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        The path that was ensured.
    """
    os.makedirs(path, exist_ok=True)
    return path


def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read a text file.

    Args:
        path: File path.
        encoding: File encoding.

    Returns:
        File contents as string.
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """Write a text file.

    Args:
        path: File path.
        content: Content to write.
        encoding: File encoding.

    Returns:
        Path of the written file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return path


def read_json(path: str, encoding: str = "utf-8") -> Any:
    """Read a JSON file.

    Args:
        path: File path.
        encoding: File encoding.

    Returns:
        Parsed JSON data.
    """
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def write_json(
    path: str,
    data: Any,
    indent: int = 2,
    encoding: str = "utf-8",
    default: Optional[callable] = None,
) -> str:
    """Write data to a JSON file.

    Args:
        path: File path.
        data: Data to serialize.
        indent: JSON indentation.
        encoding: File encoding.
        default: Default serializer for non-serializable objects.

    Returns:
        Path of the written file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, default=default or str)
    return path


def read_yaml(path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read a YAML file.

    Args:
        path: File path.
        encoding: File encoding.

    Returns:
        Parsed YAML data.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    with open(path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def write_yaml(
    path: str,
    data: Any,
    encoding: str = "utf-8",
    default_flow_style: bool = False,
) -> str:
    """Write data to a YAML file.

    Args:
        path: File path.
        data: Data to serialize.
        encoding: File encoding.
        default_flow_style: YAML flow style setting.

    Returns:
        Path of the written file.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, allow_unicode=True)
    return path


def read_jsonl(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Read a JSONL file (one JSON object per line).

    Args:
        path: File path.
        encoding: File encoding.

    Returns:
        List of parsed JSON objects.
    """
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num} in {path}: {e}")
    return data


def write_jsonl(
    path: str,
    data: List[Dict[str, Any]],
    encoding: str = "utf-8",
) -> str:
    """Write data to a JSONL file.

    Args:
        path: File path.
        data: List of dictionaries to write.
        encoding: File encoding.

    Returns:
        Path of the written file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")
    return path


def read_csv(
    path: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> List[Dict[str, str]]:
    """Read a CSV file.

    Args:
        path: File path.
        encoding: File encoding.
        delimiter: CSV delimiter.

    Returns:
        List of row dictionaries.
    """
    data = []
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(dict(row))
    return data


def write_csv(
    path: str,
    data: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> str:
    """Write data to a CSV file.

    Args:
        path: File path.
        data: List of row dictionaries.
        fieldnames: Column names. Auto-detected from first row if None.
        encoding: File encoding.
        delimiter: CSV delimiter.

    Returns:
        Path of the written file.
    """
    if not data:
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = fieldnames or list(data[0].keys())

    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return path


async def async_read_file(path: str, encoding: str = "utf-8") -> str:
    """Asynchronously read a text file."""
    async with aiofiles.open(path, "r", encoding=encoding) as f:
        return await f.read()


async def async_write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """Asynchronously write a text file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    async with aiofiles.open(path, "w", encoding=encoding) as f:
        await f.write(content)
    return path


async def async_read_json(path: str, encoding: str = "utf-8") -> Any:
    """Asynchronously read a JSON file."""
    content = await async_read_file(path, encoding)
    return json.loads(content)


async def async_write_json(path: str, data: Any, indent: int = 2, encoding: str = "utf-8") -> str:
    """Asynchronously write a JSON file."""
    content = json.dumps(data, indent=indent, default=str)
    return await async_write_file(path, content, encoding)


async def async_read_jsonl(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Asynchronously read a JSONL file."""
    content = await async_read_file(path, encoding)
    data = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data


async def async_write_jsonl(path: str, data: List[Dict[str, Any]], encoding: str = "utf-8") -> str:
    """Asynchronously write a JSONL file."""
    lines = [json.dumps(item, default=str) for item in data]
    content = "\n".join(lines) + "\n"
    return await async_write_file(path, content, encoding)


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(path)


def file_exists(path: str) -> bool:
    """Check if a file exists."""
    return os.path.isfile(path)


def dir_exists(path: str) -> bool:
    """Check if a directory exists."""
    return os.path.isdir(path)


def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[str]:
    """List files in a directory.

    Args:
        directory: Directory path.
        pattern: Glob pattern for filtering.
        recursive: Whether to search recursively.

    Returns:
        List of file paths.
    """
    import glob
    if recursive:
        return glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    return glob.glob(os.path.join(directory, pattern))


def copy_file(src: str, dst: str) -> str:
    """Copy a file from src to dst."""
    import shutil
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def move_file(src: str, dst: str) -> str:
    """Move a file from src to dst."""
    import shutil
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.move(src, dst)
    return dst
