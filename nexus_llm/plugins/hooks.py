"""Hook system for Nexus-LLM plugin infrastructure.

Provides a priority-based callback registry that underpins the extensible
lifecycle of the application.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical hook names used throughout the application
BUILTIN_HOOKS = frozenset(
    {
        "pre_generate",
        "post_generate",
        "pre_chat",
        "post_chat",
        "on_load_model",
        "on_unload_model",
    }
)


class HookError(Exception):
    """Raised when a hook callback fails."""


class HookSystem:
    """Priority-based hook/callback registry.

    Callbacks are registered against named hooks with an integer priority
    (lower runs first). When a hook is triggered, all registered callbacks
    are invoked in priority order and their return values are collected.

    Example::

        hooks = HookSystem()
        hooks.register("pre_generate", my_callback, priority=10)
        results = hooks.trigger("pre_generate", prompt="Hello")
    """

    def __init__(self) -> None:
        # hook_name -> list of (priority, callback)
        self._hooks: Dict[str, List[Tuple[int, Callable[..., Any]]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        hook_name: str,
        callback: Callable[..., Any],
        priority: int = 50,
    ) -> None:
        """Register *callback* for *hook_name* with the given *priority*.

        Lower priority values execute first.  The default (50) places the
        callback in the middle of the range.

        Args:
            hook_name: The name of the hook to attach to.
            callback: A callable that will be invoked when the hook triggers.
            priority: Execution order — lower runs first.

        Raises:
            ValueError: If *callback* is not callable.
        """
        if not callable(callback):
            raise ValueError(f"Callback for hook {hook_name!r} must be callable")

        entry = (priority, callback)
        self._hooks[hook_name].append(entry)
        # Keep sorted by priority (stable — insertion order preserved for
        # equal priorities).
        self._hooks[hook_name].sort(key=lambda pair: pair[0])
        logger.debug(
            "Registered callback %s for hook %r with priority %d",
            getattr(callback, "__name__", repr(callback)),
            hook_name,
            priority,
        )

    def unregister(self, hook_name: str, callback: Callable[..., Any]) -> bool:
        """Remove a previously registered callback.

        Returns:
            True if the callback was found and removed, False otherwise.
        """
        entries = self._hooks.get(hook_name, [])
        for i, (_pri, cb) in enumerate(entries):
            if cb is callback:
                entries.pop(i)
                logger.debug(
                    "Unregistered callback %s from hook %r",
                    getattr(callback, "__name__", repr(callback)),
                    hook_name,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Triggering
    # ------------------------------------------------------------------

    def trigger(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """Trigger *hook_name*, invoking all registered callbacks in priority order.

        Args:
            hook_name: The hook to fire.
            *args: Positional arguments forwarded to every callback.
            **kwargs: Keyword arguments forwarded to every callback.

        Returns:
            A list of return values from each callback, in execution order.
        """
        results: List[Any] = []
        for priority, callback in self._hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Hook %r callback %s raised %s: %s",
                    hook_name,
                    getattr(callback, "__name__", repr(callback)),
                    type(exc).__name__,
                    exc,
                )
                raise HookError(
                    f"Callback {getattr(callback, '__name__', repr(callback))} "
                    f"for hook {hook_name!r} raised {type(exc).__name__}: {exc}"
                ) from exc
        return results

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_hooks(self) -> Dict[str, List[Tuple[int, str]]]:
        """Return a snapshot of all registered hooks and their callbacks.

        The callback representations are ``callback.__name__`` when
        available, otherwise ``repr(callback)``.
        """
        snapshot: Dict[str, List[Tuple[int, str]]] = {}
        for name, entries in self._hooks.items():
            snapshot[name] = [
                (pri, getattr(cb, "__name__", repr(cb))) for pri, cb in entries
            ]
        return snapshot

    def has_hook(self, hook_name: str) -> bool:
        """Return True if at least one callback is registered for *hook_name*."""
        return bool(self._hooks.get(hook_name))

    def clear(self, hook_name: Optional[str] = None) -> None:
        """Remove all callbacks.

        Args:
            hook_name: If given, only clear that hook. Otherwise clear all.
        """
        if hook_name is not None:
            self._hooks.pop(hook_name, None)
        else:
            self._hooks.clear()
