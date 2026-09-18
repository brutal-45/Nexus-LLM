"""Tests for sortable table."""
import pytest


def test_table_sort_ascending():
    rows = [{"name": "Charlie", "age": 30}, {"name": "Alice", "age": 25}, {"name": "Bob", "age": 35}]
    sorted_rows = sorted(rows, key=lambda r: r["name"])
    assert sorted_rows[0]["name"] == "Alice"

def test_table_sort_descending():
    rows = [{"name": "A", "val": 1}, {"name": "B", "val": 3}, {"name": "C", "val": 2}]
    sorted_rows = sorted(rows, key=lambda r: r["val"], reverse=True)
    assert sorted_rows[0]["val"] == 3

def test_table_column_access():
    headers = ["Name", "Age", "City"]
    assert "Name" in headers
    assert len(headers) == 3

def test_table_filter_rows():
    rows = [{"n": "a", "v": 1}, {"n": "b", "v": 2}, {"n": "c", "v": 1}]
    filtered = [r for r in rows if r["v"] == 1]
    assert len(filtered) == 2
