"""Tests for async I/O."""
import pytest
import asyncio
import tempfile
from pathlib import Path


def test_async_file_read():
    async def read_file():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("async content")
            name = f.name
        with open(name) as f:
            return f.read()
    result = asyncio.get_event_loop().run_until_complete(read_file())
    assert result == "async content"

def test_async_multiple_reads():
    async def read_multiple():
        results = []
        for i in range(3):
            results.append(f"item_{i}")
        return results
    result = asyncio.get_event_loop().run_until_complete(read_multiple())
    assert len(result) == 3

def test_async_file_write():
    async def write_file(tmp_dir):
        p = tmp_dir / "async.txt"
        p.write_text("async write")
        return p.read_text()
    with tempfile.TemporaryDirectory() as d:
        result = asyncio.get_event_loop().run_until_complete(write_file(Path(d)))
    assert result == "async write"
