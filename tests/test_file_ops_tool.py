"""Tests for nexus_llm.tools.file_ops module."""

import os
import tempfile
import pytest
from nexus_llm.tools.file_ops import FileOpsTool


class TestFileOpsTool:
    """Tests for the FileOpsTool class."""

    def test_init(self):
        tool = FileOpsTool()
        assert tool.name == "file_ops"

    def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        tool = FileOpsTool()
        result = tool.run(operation="read", path=str(test_file))
        assert result.success is True
        assert "hello world" in result.output

    def test_write_file(self, tmp_path):
        test_file = tmp_path / "output.txt"
        tool = FileOpsTool()
        result = tool.run(operation="write", path=str(test_file), content="test content")
        assert result.success is True
        assert test_file.read_text() == "test content"

    def test_list_directory(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        tool = FileOpsTool()
        result = tool.run(operation="list", path=str(tmp_path))
        assert result.success is True

    def test_file_exists(self, tmp_path):
        test_file = tmp_path / "exists.txt"
        test_file.write_text("yes")
        tool = FileOpsTool()
        result = tool.run(operation="exists", path=str(test_file))
        assert result.success is True

    def test_delete_file(self, tmp_path):
        test_file = tmp_path / "delete_me.txt"
        test_file.write_text("bye")
        tool = FileOpsTool()
        result = tool.run(operation="delete", path=str(test_file))
        assert result.success is True
        assert not test_file.exists()

    def test_unknown_operation(self, tmp_path):
        tool = FileOpsTool()
        result = tool.run(operation="unknown", path="/tmp")
        assert result.success is False
