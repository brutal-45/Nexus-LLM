"""Tests for UI widgets."""
import pytest


class ProgressBar:
    """Simple progress bar widget."""
    def __init__(self, total=100, width=40):
        self.total = total
        self.width = width
        self.current = 0

    def update(self, value):
        self.current = min(value, self.total)

    @property
    def percent(self):
        return (self.current / self.total) * 100 if self.total > 0 else 0

    @property
    def filled(self):
        return int(self.width * self.current / self.total)


class StatusBar:
    """Simple status bar widget."""
    def __init__(self):
        self.items = {}

    def set(self, key, value):
        self.items[key] = value

    def get(self, key):
        return self.items.get(key)

    def render(self):
        parts = [f"{k}: {v}" for k, v in self.items.items()]
        return " | ".join(parts)


def test_progress_bar_creation():
    """Test creating a progress bar."""
    pb = ProgressBar(total=100)
    assert pb.total == 100
    assert pb.current == 0


def test_progress_bar_update():
    """Test updating progress bar."""
    pb = ProgressBar(total=100)
    pb.update(50)
    assert pb.percent == 50.0


def test_progress_bar_filled():
    """Test filled width calculation."""
    pb = ProgressBar(total=100, width=40)
    pb.update(50)
    assert pb.filled == 20


def test_progress_bar_complete():
    """Test progress bar at 100%."""
    pb = ProgressBar(total=100)
    pb.update(100)
    assert pb.percent == 100.0


def test_status_bar_set_get():
    """Test status bar set and get."""
    sb = StatusBar()
    sb.set("model", "nexus-7b")
    assert sb.get("model") == "nexus-7b"


def test_status_bar_render():
    """Test status bar rendering."""
    sb = StatusBar()
    sb.set("model", "nexus-7b")
    sb.set("tokens", "100")
    rendered = sb.render()
    assert "nexus-7b" in rendered
    assert "100" in rendered
