"""Tests for layout manager."""
import pytest


class LayoutManager:
    """Simple layout manager for terminal UI."""
    def __init__(self, width=120, height=40):
        self.width = width
        self.height = height
        self.regions = {}

    def add_region(self, name, x, y, w, h):
        self.regions[name] = {"x": x, "y": y, "w": w, "h": h}

    def get_region(self, name):
        return self.regions.get(name)

    def fits(self, name):
        r = self.regions.get(name)
        if r is None:
            return False
        return (r["x"] + r["w"] <= self.width and
                r["y"] + r["h"] <= self.height)


@pytest.fixture
def layout():
    return LayoutManager(width=120, height=40)


def test_layout_creation(layout):
    """Test creating a layout manager."""
    assert layout.width == 120
    assert layout.height == 40


def test_layout_add_region(layout):
    """Test adding a region."""
    layout.add_region("header", 0, 0, 120, 3)
    assert layout.get_region("header") is not None


def test_layout_region_fits(layout):
    """Test that a region fits within layout."""
    layout.add_region("main", 0, 3, 120, 37)
    assert layout.fits("main")


def test_layout_region_does_not_fit(layout):
    """Test that an oversized region does not fit."""
    layout.add_region("oversized", 0, 0, 200, 50)
    assert not layout.fits("oversized")


def test_layout_multiple_regions(layout):
    """Test layout with multiple regions."""
    layout.add_region("header", 0, 0, 120, 3)
    layout.add_region("body", 0, 3, 120, 34)
    layout.add_region("footer", 0, 37, 120, 3)
    assert len(layout.regions) == 3
    assert all(layout.fits(name) for name in layout.regions)
