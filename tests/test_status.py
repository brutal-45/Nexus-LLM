"""Tests for status bar."""
import pytest


class StatusBarModel:
    """Status bar model for testing."""
    def __init__(self):
        self.sections = []

    def add_section(self, name, content="", width=20):
        self.sections.append({"name": name, "content": content, "width": width})

    def update(self, name, content):
        for s in self.sections:
            if s["name"] == name:
                s["content"] = content
                return True
        return False

    def render(self):
        parts = []
        for s in self.sections:
            text = f" {s['name']}: {s['content']} "
            parts.append(text[:s["width"]])
        return "|".join(parts)


@pytest.fixture
def status_bar():
    sb = StatusBarModel()
    sb.add_section("Model", "nexus-7b", width=20)
    sb.add_section("Tokens", "0", width=15)
    sb.add_section("Status", "Ready", width=15)
    return sb


def test_status_bar_sections(status_bar):
    """Test status bar has sections."""
    assert len(status_bar.sections) == 3


def test_status_bar_update(status_bar):
    """Test updating status bar content."""
    status_bar.update("Tokens", "150")
    assert status_bar.sections[1]["content"] == "150"


def test_status_bar_render(status_bar):
    """Test status bar rendering."""
    rendered = status_bar.render()
    assert "nexus-7b" in rendered
    assert "Ready" in rendered


def test_status_bar_update_nonexistent(status_bar):
    """Test updating a nonexistent section."""
    assert status_bar.update("NonExistent", "value") is False
