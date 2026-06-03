"""Tests for training callbacks."""
import pytest


class Callback:
    """Base callback for testing."""
    def __init__(self):
        self.events = []

    def on_train_begin(self, **kwargs):
        self.events.append("train_begin")

    def on_train_end(self, **kwargs):
        self.events.append("train_end")

    def on_step_begin(self, step, **kwargs):
        self.events.append(f"step_begin_{step}")

    def on_step_end(self, step, **kwargs):
        self.events.append(f"step_end_{step}")


class CallbackManager:
    """Manages a list of callbacks."""
    def __init__(self):
        self.callbacks = []

    def add(self, callback):
        self.callbacks.append(callback)

    def fire(self, event, **kwargs):
        for cb in self.callbacks:
            handler = getattr(cb, f"on_{event}", None)
            if handler:
                handler(**kwargs)


@pytest.fixture
def callback():
    return Callback()


def test_callback_on_train_begin(callback):
    """Test on_train_begin callback."""
    callback.on_train_begin()
    assert "train_begin" in callback.events


def test_callback_on_train_end(callback):
    """Test on_train_end callback."""
    callback.on_train_end()
    assert "train_end" in callback.events


def test_callback_on_step(callback):
    """Test on_step callbacks."""
    callback.on_step_begin(step=1)
    callback.on_step_end(step=1)
    assert "step_begin_1" in callback.events
    assert "step_end_1" in callback.events


def test_callback_manager():
    """Test callback manager fires events."""
    mgr = CallbackManager()
    cb1 = Callback()
    cb2 = Callback()
    mgr.add(cb1)
    mgr.add(cb2)
    mgr.fire("train_begin")
    assert "train_begin" in cb1.events
    assert "train_begin" in cb2.events
