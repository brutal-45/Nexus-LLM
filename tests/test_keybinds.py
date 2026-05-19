"""Tests for key bindings."""
import pytest


class KeyBinding:
    """A key binding mapping."""
    def __init__(self, key, action, description=""):
        self.key = key
        self.action = action
        self.description = description


class KeyBindingManager:
    """Manager for key bindings."""
    def __init__(self):
        self.bindings = {}

    def bind(self, key, action, description=""):
        self.bindings[key] = KeyBinding(key, action, description)

    def unbind(self, key):
        self.bindings.pop(key, None)

    def get_action(self, key):
        binding = self.bindings.get(key)
        return binding.action if binding else None


@pytest.fixture
def keybind_mgr():
    mgr = KeyBindingManager()
    mgr.bind("ctrl+c", "cancel", "Cancel current operation")
    mgr.bind("ctrl+d", "quit", "Exit the application")
    mgr.bind("enter", "submit", "Submit input")
    mgr.bind("tab", "autocomplete", "Autocomplete")
    return mgr


def test_keybind_bind(keybind_mgr):
    """Test binding a key."""
    keybind_mgr.bind("ctrl+z", "undo", "Undo last action")
    assert keybind_mgr.get_action("ctrl+z") == "undo"


def test_keybind_get_action(keybind_mgr):
    """Test getting action for a key."""
    assert keybind_mgr.get_action("ctrl+c") == "cancel"
    assert keybind_mgr.get_action("enter") == "submit"


def test_keybind_unbind(keybind_mgr):
    """Test unbinding a key."""
    keybind_mgr.unbind("ctrl+c")
    assert keybind_mgr.get_action("ctrl+c") is None


def test_keybind_unknown_key(keybind_mgr):
    """Test getting action for unbound key."""
    assert keybind_mgr.get_action("f12") is None
