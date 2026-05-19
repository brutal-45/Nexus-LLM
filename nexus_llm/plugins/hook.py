"""Hook system for plugin event handling.

Provides a priority-based hook system supporting synchronous and
asynchronous hooks, hook registration, triggering, and management.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class HookPriority(IntEnum):
    """Priority levels for hooks. Lower numbers run first."""

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100
    MONITOR = 200  # Monitor hooks run last and should not modify data


@dataclass
class Hook:
    """A registered hook callback."""

    name: str
    callback: Callable
    hook_point: str
    priority: HookPriority = HookPriority.NORMAL
    is_async: bool = False
    description: str = ""
    owner: str = ""  # Plugin name that registered this hook

    def __lt__(self, other: "Hook") -> bool:
        """Compare by priority for sorting."""
        return self.priority < other.priority

    def __repr__(self) -> str:
        return f"Hook(name={self.name}, point={self.hook_point}, priority={self.priority.name})"


class HookManager:
    """Manages hook registration and triggering.

    Provides a centralized system for registering, triggering, and
    managing hooks. Supports both synchronous and asynchronous
    callbacks with priority-based execution ordering.
    """

    def __init__(self):
        self._hooks: Dict[str, List[Hook]] = {}
        self._all_hooks: Dict[str, Hook] = {}  # name -> Hook
        self._disabled: Set[str] = set()

    def register(
        self,
        hook_point: str,
        callback: Callable,
        name: Optional[str] = None,
        priority: HookPriority = HookPriority.NORMAL,
        description: str = "",
        owner: str = "",
    ) -> Hook:
        """Register a hook callback at a hook point.

        Args:
            hook_point: The hook point name to attach to.
            callback: The callback function to execute.
            name: Optional unique name for the hook.
            priority: Execution priority (lower = earlier).
            description: Human-readable description.
            owner: Plugin name that owns this hook.

        Returns:
            The registered Hook object.
        """
        import uuid

        hook_name = name or f"hook_{uuid.uuid4().hex[:8]}"

        # Check for duplicate names
        if hook_name in self._all_hooks:
            logger.warning("Hook '%s' already registered. Replacing.", hook_name)
            self.unregister(hook_name)

        is_async = asyncio.iscoroutinefunction(callback)

        hook = Hook(
            name=hook_name,
            callback=callback,
            hook_point=hook_point,
            priority=priority,
            is_async=is_async,
            description=description,
            owner=owner,
        )

        if hook_point not in self._hooks:
            self._hooks[hook_point] = []

        self._hooks[hook_point].append(hook)
        self._hooks[hook_point].sort()  # Sort by priority
        self._all_hooks[hook_name] = hook

        logger.debug(
            "Registered hook '%s' at point '%s' with priority %s.",
            hook_name, hook_point, priority.name,
        )
        return hook

    def unregister(self, name: str) -> bool:
        """Unregister a hook by name.

        Args:
            name: The hook name to unregister.

        Returns:
            True if the hook was found and removed.
        """
        if name not in self._all_hooks:
            return False

        hook = self._all_hooks.pop(name)
        if hook.hook_point in self._hooks:
            self._hooks[hook.hook_point] = [
                h for h in self._hooks[hook.hook_point] if h.name != name
            ]

        logger.debug("Unregistered hook '%s'.", name)
        return True

    def unregister_by_owner(self, owner: str) -> int:
        """Unregister all hooks owned by a specific plugin.

        Args:
            owner: The plugin name.

        Returns:
            Number of hooks unregistered.
        """
        hooks_to_remove = [
            name for name, hook in self._all_hooks.items()
            if hook.owner == owner
        ]
        for name in hooks_to_remove:
            self.unregister(name)
        return len(hooks_to_remove)

    def trigger(self, hook_point: str, *args, **kwargs) -> Any:
        """Trigger all hooks at a hook point synchronously.

        Hooks are called in priority order. The return value of each
        hook is passed as the first argument to the next hook
        (chaining pattern). The first hook receives the original args.

        Args:
            hook_point: The hook point to trigger.
            *args: Positional arguments to pass to hooks.
            **kwargs: Keyword arguments to pass to hooks.

        Returns:
            The result after all hooks have processed.
        """
        hooks = self._hooks.get(hook_point, [])
        active_hooks = [h for h in hooks if h.name not in self._disabled]

        if not active_hooks:
            return kwargs.get("result", args[0] if args else None)

        result = kwargs.pop("result", args[0] if args else None)

        for hook in active_hooks:
            if hook.is_async:
                logger.warning(
                    "Skipping async hook '%s' in sync trigger. Use trigger_async().",
                    hook.name,
                )
                continue

            try:
                hook_result = hook.callback(result, *args, **kwargs)
                if hook_result is not None:
                    result = hook_result
            except Exception as e:
                logger.error(
                    "Error in hook '%s' at point '%s': %s",
                    hook.name, hook_point, e,
                )

        return result

    async def trigger_async(self, hook_point: str, *args, **kwargs) -> Any:
        """Trigger all hooks at a hook point, supporting async callbacks.

        Args:
            hook_point: The hook point to trigger.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The result after all hooks have processed.
        """
        hooks = self._hooks.get(hook_point, [])
        active_hooks = [h for h in hooks if h.name not in self._disabled]

        if not active_hooks:
            return kwargs.get("result", args[0] if args else None)

        result = kwargs.pop("result", args[0] if args else None)

        for hook in active_hooks:
            try:
                if hook.is_async:
                    hook_result = await hook.callback(result, *args, **kwargs)
                else:
                    hook_result = hook.callback(result, *args, **kwargs)
                if hook_result is not None:
                    result = hook_result
            except Exception as e:
                logger.error(
                    "Error in async hook '%s' at point '%s': %s",
                    hook.name, hook_point, e,
                )

        return result

    def enable(self, name: str) -> bool:
        """Enable a disabled hook."""
        if name in self._disabled:
            self._disabled.discard(name)
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a hook without removing it."""
        if name in self._all_hooks:
            self._disabled.add(name)
            return True
        return False

    def get_hooks(self, hook_point: Optional[str] = None) -> List[Hook]:
        """Get hooks, optionally filtered by hook point."""
        if hook_point:
            return list(self._hooks.get(hook_point, []))
        return list(self._all_hooks.values())

    def get_hook_points(self) -> List[str]:
        """Get all registered hook point names."""
        return list(self._hooks.keys())

    def clear(self) -> None:
        """Clear all hooks."""
        self._hooks.clear()
        self._all_hooks.clear()
        self._disabled.clear()

    def count(self, hook_point: Optional[str] = None) -> int:
        """Count registered hooks."""
        if hook_point:
            return len(self._hooks.get(hook_point, []))
        return len(self._all_hooks)
