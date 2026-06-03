"""Tests for spinners."""
import pytest


class Spinner:
    """Simple spinner animation for testing."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message="Loading"):
        self.message = message
        self.frame_idx = 0
        self.active = False

    def start(self):
        self.active = True
        self.frame_idx = 0

    def stop(self):
        self.active = False

    def next_frame(self):
        frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
        self.frame_idx += 1
        return frame

    @property
    def current_frame(self):
        return self.FRAMES[self.frame_idx % len(self.FRAMES)]


@pytest.fixture
def spinner():
    return Spinner(message="Processing")


def test_spinner_creation(spinner):
    """Test creating a spinner."""
    assert spinner.message == "Processing"
    assert spinner.active is False


def test_spinner_start_stop(spinner):
    """Test starting and stopping spinner."""
    spinner.start()
    assert spinner.active is True
    spinner.stop()
    assert spinner.active is False


def test_spinner_frames(spinner):
    """Test spinner frame cycling."""
    frames = [spinner.next_frame() for _ in range(20)]
    assert len(frames) == 20
    # Frames should cycle
    assert frames[0] == frames[10]


def test_spinner_current_frame(spinner):
    """Test current frame access."""
    assert spinner.current_frame in Spinner.FRAMES
