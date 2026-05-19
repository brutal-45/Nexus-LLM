"""Test I/O utilities for Nexus-LLM."""
import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# --- I/O utility implementations to test ---

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: str, data: dict, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    keepchars = (" ", "-", "_", ".")
    return "".join(c if c.isalnum() or c in keepchars else "_" for c in name).strip()


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def copy_file(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(src, "rb") as sf:
        with open(dst, "wb") as df:
            df.write(sf.read())


class TestReadTextFile:
    def test_read_text(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("hello world")
        assert read_text_file(str(f)) == "hello world"

    def test_read_unicode(self, tmp_dir):
        f = tmp_dir / "unicode.txt"
        f.write_text("こんにちは世界", encoding="utf-8")
        assert read_text_file(str(f)) == "こんにちは世界"

    def test_read_nonexistent_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            read_text_file(str(tmp_dir / "nonexistent.txt"))

    def test_read_multiline(self, tmp_dir):
        f = tmp_dir / "multi.txt"
        content = "line1\nline2\nline3\n"
        f.write_text(content)
        assert read_text_file(str(f)) == content


class TestWriteTextFile:
    def test_write_text(self, tmp_dir):
        f = tmp_dir / "output.txt"
        write_text_file(str(f), "test content")
        assert f.read_text() == "test content"

    def test_write_creates_parent_dirs(self, tmp_dir):
        f = tmp_dir / "sub" / "dir" / "output.txt"
        write_text_file(str(f), "nested content")
        assert f.read_text() == "nested content"

    def test_write_overwrites(self, tmp_dir):
        f = tmp_dir / "output.txt"
        write_text_file(str(f), "first")
        write_text_file(str(f), "second")
        assert f.read_text() == "second"

    def test_write_empty_string(self, tmp_dir):
        f = tmp_dir / "empty.txt"
        write_text_file(str(f), "")
        assert f.read_text() == ""


class TestJsonIO:
    def test_read_json(self, tmp_dir):
        f = tmp_dir / "data.json"
        f.write_text('{"key": "value", "num": 42}')
        result = read_json_file(str(f))
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_write_json(self, tmp_dir):
        f = tmp_dir / "out.json"
        data = {"name": "test", "items": [1, 2, 3]}
        write_json_file(str(f), data)
        loaded = json.loads(f.read_text())
        assert loaded == data

    def test_write_json_unicode(self, tmp_dir):
        f = tmp_dir / "unicode.json"
        data = {"greeting": "こんにちは"}
        write_json_file(str(f), data)
        result = read_json_file(str(f))
        assert result["greeting"] == "こんにちは"

    def test_write_json_with_indent(self, tmp_dir):
        f = tmp_dir / "pretty.json"
        write_json_file(str(f), {"a": 1}, indent=4)
        content = f.read_text()
        assert "    " in content

    def test_read_invalid_json(self, tmp_dir):
        f = tmp_dir / "bad.json"
        f.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            read_json_file(str(f))


class TestFileUtilities:
    def test_file_exists_true(self, tmp_dir):
        f = tmp_dir / "exists.txt"
        f.write_text("content")
        assert file_exists(str(f)) is True

    def test_file_exists_false(self, tmp_dir):
        assert file_exists(str(tmp_dir / "nope.txt")) is False

    def test_ensure_directory(self, tmp_dir):
        new_dir = tmp_dir / "new" / "nested" / "dir"
        ensure_directory(str(new_dir))
        assert new_dir.is_dir()

    def test_ensure_directory_existing(self, tmp_dir):
        ensure_directory(str(tmp_dir))
        assert tmp_dir.is_dir()

    def test_safe_filename_normal(self):
        assert safe_filename("hello.txt") == "hello.txt"

    def test_safe_filename_special_chars(self):
        result = safe_filename("file with/special\\chars?.txt")
        assert "/" not in result
        assert "\\" not in result
        assert "?" not in result

    def test_safe_filename_empty(self):
        assert safe_filename("") == ""

    def test_count_lines(self, tmp_dir):
        f = tmp_dir / "lines.txt"
        f.write_text("line1\nline2\nline3\n")
        assert count_lines(str(f)) == 3

    def test_count_lines_empty(self, tmp_dir):
        f = tmp_dir / "empty.txt"
        f.write_text("")
        assert count_lines(str(f)) == 0

    def test_copy_file(self, tmp_dir):
        src = tmp_dir / "src.txt"
        dst = tmp_dir / "dst.txt"
        src.write_text("copy me")
        copy_file(str(src), str(dst))
        assert dst.read_text() == "copy me"
