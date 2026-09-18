"""Test format conversion utilities for Nexus-LLM."""
import json
import csv
import io
import pytest
from typing import Any, List, Dict, Optional


def dict_to_json(data: dict, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False)


def json_to_dict(text: str) -> dict:
    return json.loads(text)


def dict_to_yaml(data: dict, indent: int = 2) -> str:
    """Simple YAML-like serializer (not full YAML spec)."""
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"{' ' * indent}{k}: {_yaml_value(v)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"{' ' * indent}- {_yaml_value(item)}")
        else:
            lines.append(f"{key}: {_yaml_value(value)}")
    return "\n".join(lines)


def _yaml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if " " in value or ":" in value:
            return f'"{value}"'
        return value
    if value is None:
        return "null"
    return str(value)


def dict_to_csv_rows(data: List[dict]) -> str:
    if not data:
        return ""
    fieldnames = list(data[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def csv_rows_to_dict(text: str) -> List[dict]:
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]


def flatten_dict(data: dict, prefix: str = "", sep: str = ".") -> dict:
    items = {}
    for key, value in data.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value
    return items


def unflatten_dict(data: dict, sep: str = ".") -> dict:
    result = {}
    for key, value in data.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def bytes_to_human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def human_to_bytes(text: str) -> int:
    text = text.strip().upper()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for unit, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if text.endswith(unit):
            number = text[: -len(unit)].strip()
            return int(float(number) * multiplier)
    return int(text)


class TestDictToJson:
    def test_simple(self):
        result = dict_to_json({"key": "value"})
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_unicode(self):
        result = dict_to_json({"greeting": "こんにちは"})
        assert "こんにちは" in result

    def test_nested(self):
        data = {"outer": {"inner": 42}}
        result = dict_to_json(data)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == 42


class TestJsonToDict:
    def test_simple(self):
        result = json_to_dict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            json_to_dict("not json")


class TestDictToYaml:
    def test_simple(self):
        result = dict_to_yaml({"name": "test", "count": 5})
        assert "name: test" in result
        assert "count: 5" in result

    def test_nested(self):
        result = dict_to_yaml({"model": {"name": "gpt2", "size": 2}})
        assert "model:" in result
        assert "name: gpt2" in result

    def test_list(self):
        result = dict_to_yaml({"items": [1, 2, 3]})
        assert "- 1" in result

    def test_bool(self):
        result = dict_to_yaml({"enabled": True, "disabled": False})
        assert "true" in result
        assert "false" in result


class TestDictToCsvRows:
    def test_simple(self):
        data = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
        result = dict_to_csv_rows(data)
        assert "name" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_empty(self):
        assert dict_to_csv_rows([]) == ""


class TestCsvRowsToDict:
    def test_simple(self):
        csv_text = "name,age\nAlice,30\nBob,25\n"
        result = csv_rows_to_dict(csv_text)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"

    def test_roundtrip(self):
        data = [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]
        csv_text = dict_to_csv_rows(data)
        parsed = csv_rows_to_dict(csv_text)
        assert parsed == data


class TestFlattenDict:
    def test_simple(self):
        data = {"a": {"b": {"c": 1}}}
        result = flatten_dict(data)
        assert result == {"a.b.c": 1}

    def test_mixed(self):
        data = {"a": 1, "b": {"c": 2, "d": 3}}
        result = flatten_dict(data)
        assert result == {"a": 1, "b.c": 2, "b.d": 3}

    def test_custom_separator(self):
        data = {"a": {"b": 1}}
        result = flatten_dict(data, sep="/")
        assert result == {"a/b": 1}

    def test_empty_dict(self):
        assert flatten_dict({}) == {}


class TestUnflattenDict:
    def test_simple(self):
        data = {"a.b.c": 1}
        result = unflatten_dict(data)
        assert result == {"a": {"b": {"c": 1}}}

    def test_mixed(self):
        data = {"a": 1, "b.c": 2}
        result = unflatten_dict(data)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_roundtrip(self):
        original = {"a": {"b": 1}, "c": 2, "d": {"e": {"f": 3}}}
        flat = flatten_dict(original)
        unflat = unflatten_dict(flat)
        assert unflat == original


class TestBytesToHuman:
    def test_bytes(self):
        assert "B" in bytes_to_human(100)

    def test_kilobytes(self):
        assert "KB" in bytes_to_human(1024)

    def test_megabytes(self):
        assert "MB" in bytes_to_human(1024 * 1024)

    def test_gigabytes(self):
        assert "GB" in bytes_to_human(1024**3)

    def test_zero(self):
        assert bytes_to_human(0) == "0.0 B"


class TestHumanToBytes:
    def test_bytes(self):
        assert human_to_bytes("100 B") == 100

    def test_kilobytes(self):
        assert human_to_bytes("2 KB") == 2048

    def test_megabytes(self):
        assert human_to_bytes("1 MB") == 1024**2

    def test_gigabytes(self):
        assert human_to_bytes("1 GB") == 1024**3

    def test_roundtrip(self):
        for size in [100, 1024, 1024**2, 1024**3]:
            human = bytes_to_human(size)
            back = human_to_bytes(human)
            assert abs(back - size) < size * 0.1  # within 10%
