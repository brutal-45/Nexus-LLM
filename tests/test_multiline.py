"""Tests for multiline input."""
import pytest


class MultilineInput:
    """Simple multiline input handler for testing."""
    def __init__(self):
        self.lines = []
        self.continuation_char = "\\"

    def add_line(self, line):
        self.lines.append(line)

    def is_complete(self):
        if not self.lines:
            return True
        return not self.lines[-1].endswith(self.continuation_char)

    def get_full_input(self):
        # Strip continuation characters
        cleaned = []
        for line in self.lines:
            if line.endswith(self.continuation_char):
                cleaned.append(line[:-1])
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def reset(self):
        self.lines = []


@pytest.fixture
def multiline():
    return MultilineInput()


def test_multiline_single_line(multiline):
    """Test single line input."""
    multiline.add_line("print('hello')")
    assert multiline.is_complete()
    assert multiline.get_full_input() == "print('hello')"


def test_multiline_continuation(multiline):
    """Test multiline with continuation character."""
    multiline.add_line("for i in range(10):\\")
    assert not multiline.is_complete()
    multiline.add_line("    print(i)")
    assert multiline.is_complete()


def test_multiline_get_full_input(multiline):
    """Test getting full multiline input."""
    multiline.add_line("line 1\\")
    multiline.add_line("line 2\\")
    multiline.add_line("line 3")
    result = multiline.get_full_input()
    assert "line 1" in result
    assert "line 3" in result


def test_multiline_reset(multiline):
    """Test resetting multiline input."""
    multiline.add_line("test")
    multiline.reset()
    assert len(multiline.lines) == 0
