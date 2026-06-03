"""Nexus-LLM State Management Module.

Provides the StateManager class for managing application state with
support for nested scopes, observers, and persistence. State changes
are tracked and can trigger callbacks through the observer pattern.
"""

import copy
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class StateObserver:
    """Observer that watches for state changes.

    Attributes:
        name: Name of the observer.
        callback: Function to call when state changes.
        keys: Set of state keys to observe (empty = all keys).
    """

    def __init__(
        self,
        name: str,
        callback: Callable[[str, Any, Any], None],
        keys: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the observer.

        Args:
            name: Observer name.
            callback: Called with (key, old_value, new_value).
            keys: Specific keys to observe. None means all keys.
        """
        self.name = name
        self.callback = callback
        self.keys = keys

    def matches(self, key: str) -> bool:
        """Check if this observer should be notified about a key change.

        Args:
            key: The state key that changed.

        Returns:
            True if the observer should be notified.
        """
        if self.keys is None:
            return True
        return key in self.keys


class StateManager:
    """Manages application state with observer support and persistence.

    Provides a centralized state store with support for nested scopes,
    change observers, and saving/loading state to disk.

    Example:
        >>> manager = StateManager()
        >>> manager.set("model.name", "gpt2-medium")
        >>> manager.get("model.name")
        'gpt2-medium'
        >>> manager.observe("logger", on_state_change, keys={"model.name"})
        >>> manager.set("model.name", "mistral-7b")  # triggers observer
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        """Initialize the state manager.

        Args:
            persist_path: Optional file path for persisting state.
        """
        self._state: Dict[str, Any] = {}
        self._observers: List[StateObserver] = []
        self._history: List[Dict[str, Any]] = []
        self._persist_path = persist_path
        self._lock = threading.RLock()
        self._max_history = 1000

        # Initialize with defaults
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Set default state values."""
        self._state = {
            "app": {
                "initialized": False,
                "shutting_down": False,
            },
            "model": {
                "current": None,
                "loaded_models": [],
                "device": "auto",
            },
            "chat": {
                "active": False,
                "session_id": None,
            },
            "server": {
                "running": False,
                "host": "0.0.0.0",
                "port": 8000,
            },
            "training": {
                "active": False,
                "stage": None,
                "epoch": 0,
                "step": 0,
                "loss": 0.0,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value by dot-separated key.

        Args:
            key: Dot-separated state key (e.g., "model.current").
            default: Default value if key not found.

        Returns:
            The state value, or default if not found.
        """
        with self._lock:
            keys = key.split(".")
            value = self._state
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return copy.deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        """Set a state value by dot-separated key.

        Triggers any observers watching this key.

        Args:
            key: Dot-separated state key.
            value: The value to set.
        """
        with self._lock:
            old_value = self.get(key)
            keys = key.split(".")
            current = self._state

            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

            # Record history
            self._history.append({
                "timestamp": datetime.now().isoformat(),
                "key": key,
                "old_value": old_value,
                "new_value": value,
            })
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Notify observers
            self._notify_observers(key, old_value, value)

    def delete(self, key: str) -> bool:
        """Delete a state value.

        Args:
            key: Dot-separated state key to delete.

        Returns:
            True if the key was found and deleted.
        """
        with self._lock:
            old_value = self.get(key)
            if old_value is None:
                return False

            keys = key.split(".")
            current = self._state

            for k in keys[:-1]:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return False

            if isinstance(current, dict) and keys[-1] in current:
                del current[keys[-1]]
                self._notify_observers(key, old_value, None)
                return True

            return False

    def has(self, key: str) -> bool:
        """Check if a state key exists.

        Args:
            key: Dot-separated state key.

        Returns:
            True if the key exists.
        """
        return self.get(key) is not None

    def get_all(self) -> Dict[str, Any]:
        """Get a copy of the entire state.

        Returns:
            Deep copy of the state dictionary.
        """
        with self._lock:
            return copy.deepcopy(self._state)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values at once.

        Args:
            updates: Dictionary of key-value pairs to update.
        """
        for key, value in updates.items():
            self.set(key, value)

    def observe(
        self,
        name: str,
        callback: Callable[[str, Any, Any], None],
        keys: Optional[Set[str]] = None,
    ) -> StateObserver:
        """Register an observer for state changes.

        Args:
            name: Observer name (for identification).
            callback: Function called with (key, old_value, new_value).
            keys: Specific keys to observe. None means all keys.

        Returns:
            The registered StateObserver.
        """
        observer = StateObserver(name=name, callback=callback, keys=keys)
        with self._lock:
            self._observers.append(observer)
        return observer

    def unobserve(self, observer: StateObserver) -> None:
        """Remove an observer.

        Args:
            observer: The observer to remove.
        """
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def unobserve_by_name(self, name: str) -> int:
        """Remove all observers with a given name.

        Args:
            name: Name of observers to remove.

        Returns:
            Number of observers removed.
        """
        with self._lock:
            original_count = len(self._observers)
            self._observers = [o for o in self._observers if o.name != name]
            return original_count - len(self._observers)

    def _notify_observers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify relevant observers of a state change.

        Args:
            key: The state key that changed.
            old_value: The previous value.
            new_value: The new value.
        """
        for observer in self._observers:
            if observer.matches(key):
                try:
                    observer.callback(key, old_value, new_value)
                except Exception as exc:
                    logger.error(
                        "Error in state observer %s for key %s: %s",
                        observer.name,
                        key,
                        exc,
                    )

    def get_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get state change history.

        Args:
            key: Filter by key. None returns all history.
            limit: Maximum number of entries to return.

        Returns:
            List of change history entries.
        """
        with self._lock:
            history = self._history[:]
        if key:
            history = [h for h in history if h["key"] == key]
        return history[-limit:]

    def save(self, path: Optional[str] = None) -> None:
        """Save current state to a file.

        Args:
            path: File path. Uses persist_path if not specified.
        """
        save_path = path or self._persist_path
        if save_path is None:
            logger.warning("No persist path configured for state saving.")
            return

        try:
            file_path = Path(save_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, default=str)
            logger.info("State saved to: %s", save_path)
        except Exception as exc:
            logger.error("Failed to save state to %s: %s", save_path, exc)

    def load(self, path: Optional[str] = None) -> None:
        """Load state from a file.

        Args:
            path: File path. Uses persist_path if not specified.
        """
        load_path = path or self._persist_path
        if load_path is None:
            logger.warning("No persist path configured for state loading.")
            return

        try:
            file_path = Path(load_path)
            if not file_path.exists():
                logger.warning("State file not found: %s", load_path)
                return

            with open(file_path, "r", encoding="utf-8") as f:
                loaded_state = json.load(f)

            with self._lock:
                self._state.update(loaded_state)

            logger.info("State loaded from: %s", load_path)
        except Exception as exc:
            logger.error("Failed to load state from %s: %s", load_path, exc)

    def reset(self) -> None:
        """Reset state to defaults."""
        with self._lock:
            self._state.clear()
            self._init_defaults()
            self._history.clear()
        logger.info("State reset to defaults.")

    @property
    def observer_count(self) -> int:
        """Get the number of registered observers."""
        return len(self._observers)

    @property
    def history_count(self) -> int:
        """Get the number of history entries."""
        return len(self._history)

    def __repr__(self) -> str:
        return f"StateManager(observers={self.observer_count}, history={self.history_count})"
