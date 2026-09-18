"""Test hook system for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional
from enum import Enum


class HookPriority(Enum):
    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100


@dataclass
class Hook:
    name: str
    callback: Callable
    priority: int = HookPriority.NORMAL.value
    description: str = ""

    def __lt__(self, other):
        return self.priority < other.priority


class HookSystem:
    def __init__(self):
        self._hooks: Dict[str, List[Hook]] = {}

    def register(self, event: str, callback: Callable, priority: int = HookPriority.NORMAL.value, name: str = ""):
        hook = Hook(name=name or callback.__name__, callback=callback, priority=priority)
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(hook)
        self._hooks[event].sort(key=lambda h: h.priority)

    def unregister(self, event: str, name: str = None, callback: Callable = None):
        if event not in self._hooks:
            return
        if callback:
            self._hooks[event] = [h for h in self._hooks[event] if h.callback != callback]
        elif name:
            self._hooks[event] = [h for h in self._hooks[event] if h.name != name]

    def trigger(self, event: str, *args, **kwargs) -> List[Any]:
        if event not in self._hooks:
            return []
        results = []
        for hook in self._hooks[event]:
            try:
                result = hook.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(e)
        return results

    def trigger_until_handled(self, event: str, *args, **kwargs) -> Optional[Any]:
        if event not in self._hooks:
            return None
        for hook in self._hooks[event]:
            try:
                result = hook.callback(*args, **kwargs)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def get_hooks(self, event: str) -> List[Hook]:
        return list(self._hooks.get(event, []))

    def has_hooks(self, event: str) -> bool:
        return event in self._hooks and len(self._hooks[event]) > 0

    def clear(self, event: str = None):
        if event:
            self._hooks.pop(event, None)
        else:
            self._hooks.clear()


class TestHook:
    def test_creation(self):
        hook = Hook(name="test", callback=lambda: None, priority=50)
        assert hook.name == "test"

    def test_comparison(self):
        h1 = Hook(name="low", callback=lambda: None, priority=25)
        h2 = Hook(name="high", callback=lambda: None, priority=75)
        assert h1 < h2


class TestHookSystem:
    def test_register_and_trigger(self):
        hooks = HookSystem()
        hooks.register("on_load", lambda: "loaded")
        results = hooks.trigger("on_load")
        assert len(results) == 1
        assert results[0] == "loaded"

    def test_multiple_hooks(self):
        hooks = HookSystem()
        hooks.register("on_save", lambda: "first")
        hooks.register("on_save", lambda: "second")
        results = hooks.trigger("on_save")
        assert len(results) == 2

    def test_priority_ordering(self):
        order = []
        hooks = HookSystem()
        hooks.register("event", lambda: order.append("low"), priority=HookPriority.LOW.value, name="low")
        hooks.register("event", lambda: order.append("high"), priority=HookPriority.HIGH.value, name="high")
        hooks.register("event", lambda: order.append("normal"), priority=HookPriority.NORMAL.value, name="normal")
        hooks.trigger("event")
        assert order == ["low", "normal", "high"]

    def test_unregister_by_name(self):
        hooks = HookSystem()
        hooks.register("event", lambda: "a", name="hook_a")
        hooks.register("event", lambda: "b", name="hook_b")
        hooks.unregister("event", name="hook_a")
        assert len(hooks.get_hooks("event")) == 1

    def test_unregister_by_callback(self):
        cb = lambda: "callback"
        hooks = HookSystem()
        hooks.register("event", cb)
        hooks.unregister("event", callback=cb)
        assert len(hooks.get_hooks("event")) == 0

    def test_trigger_no_hooks(self):
        hooks = HookSystem()
        results = hooks.trigger("nonexistent")
        assert results == []

    def test_trigger_with_args(self):
        hooks = HookSystem()
        hooks.register("add", lambda a, b: a + b)
        results = hooks.trigger("add", 3, 4)
        assert results == [7]

    def test_trigger_with_kwargs(self):
        hooks = HookSystem()
        hooks.register("greet", lambda name="world": f"hello {name}")
        results = hooks.trigger("greet", name="alice")
        assert results == ["hello alice"]

    def test_trigger_until_handled(self):
        hooks = HookSystem()
        hooks.register("event", lambda: None)
        hooks.register("event", lambda: "handled")
        hooks.register("event", lambda: "too late")
        result = hooks.trigger_until_handled("event")
        assert result == "handled"

    def test_trigger_until_handled_none(self):
        hooks = HookSystem()
        result = hooks.trigger_until_handled("nonexistent")
        assert result is None

    def test_get_hooks(self):
        hooks = HookSystem()
        hooks.register("event", lambda: "a", name="h1")
        hooks.register("event", lambda: "b", name="h2")
        hook_list = hooks.get_hooks("event")
        assert len(hook_list) == 2

    def test_has_hooks(self):
        hooks = HookSystem()
        assert hooks.has_hooks("event") is False
        hooks.register("event", lambda: None)
        assert hooks.has_hooks("event") is True

    def test_clear_event(self):
        hooks = HookSystem()
        hooks.register("event1", lambda: None)
        hooks.register("event2", lambda: None)
        hooks.clear("event1")
        assert hooks.has_hooks("event1") is False
        assert hooks.has_hooks("event2") is True

    def test_clear_all(self):
        hooks = HookSystem()
        hooks.register("event1", lambda: None)
        hooks.register("event2", lambda: None)
        hooks.clear()
        assert hooks.has_hooks("event1") is False
        assert hooks.has_hooks("event2") is False

    def test_exception_in_hook(self):
        hooks = HookSystem()
        hooks.register("event", lambda: "ok")
        hooks.register("event", lambda: 1 / 0)
        hooks.register("event", lambda: "also ok")
        results = hooks.trigger("event")
        assert len(results) == 3
        assert isinstance(results[1], ZeroDivisionError)
