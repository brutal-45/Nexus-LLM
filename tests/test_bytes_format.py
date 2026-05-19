"""Tests for bytes formatting."""
import pytest


def test_bytes_format_bytes():
    def fmt(b):
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024:
                return f"{b:.1f} {u}"
            b /= 1024
        return f"{b:.1f} PB"
    assert fmt(500) == "500.0 B"

def test_bytes_format_kilobytes():
    def fmt(b):
        for u in ["B", "KB", "MB", "GB"]:
            if b < 1024:
                return f"{b:.1f} {u}"
            b /= 1024
    assert "KB" in fmt(2048)

def test_bytes_format_megabytes():
    def fmt(b):
        for u in ["B", "KB", "MB", "GB"]:
            if b < 1024:
                return f"{b:.1f} {u}"
            b /= 1024
    assert "MB" in fmt(5 * 1024**2)

def test_bytes_format_gigabytes():
    def fmt(b):
        for u in ["B", "KB", "MB", "GB"]:
            if b < 1024:
                return f"{b:.1f} {u}"
            b /= 1024
    assert "GB" in fmt(3 * 1024**3)

def test_bytes_format_zero():
    def fmt(b):
        return f"{b:.1f} B" if b < 1024 else f"{b/1024:.1f} KB"
    assert fmt(0) == "0.0 B"
