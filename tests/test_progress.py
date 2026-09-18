"""Tests for progress indicators."""
import pytest
import time


class ProgressIndicator:
    """Simple progress indicator for testing."""
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def update(self, n=1):
        self.current = min(self.current + n, self.total)

    @property
    def percent(self):
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100

    @property
    def elapsed(self):
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def is_complete(self):
        return self.current >= self.total


@pytest.fixture
def progress():
    return ProgressIndicator(total=100, desc="Training")


def test_progress_start(progress):
    """Test starting progress indicator."""
    progress.start()
    assert progress.start_time is not None


def test_progress_update(progress):
    """Test updating progress."""
    progress.update(10)
    assert progress.current == 10
    assert progress.percent == 10.0


def test_progress_complete(progress):
    """Test completing progress."""
    progress.update(100)
    assert progress.is_complete
    assert progress.percent == 100.0


def test_progress_overflow(progress):
    """Test that progress doesn't exceed total."""
    progress.update(200)
    assert progress.current == 100


def test_progress_elapsed_time(progress):
    """Test elapsed time tracking."""
    progress.start()
    time.sleep(0.01)
    assert progress.elapsed > 0
