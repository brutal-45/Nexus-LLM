"""Event system for Nexus-LLM plugin infrastructure.

Provides a simple publish/subscribe mechanism decoupled from the hook
system.  While hooks are synchronous and ordered, events are fire-and-
forget notifications.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Canonical event names
BUILTIN_EVENTS = frozenset(
    {
        "model_loaded",
        "model_unloaded",
        "generation_complete",
        "chat_message",
        "error",
    }
)


class EventSystem:
    """Lightweight publish/subscribe event bus.

    Subscribers register handlers for named events.  When an event is
    published, all registered handlers are invoked in registration order.
    Exceptions in handlers are caught and logged so they never prevent
    other handlers from running.

    Example::

        bus = EventSystem()
        bus.subscribe("model_loaded", on_model_loaded)
        bus.publish("model_loaded", model_name="gpt2")
    """

    def __init__(self) -> None:
        # event_name -> ordered list of handlers
        self._subscribers: Dict[str, List[Callable[..., Any]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------

    def subscribe(self, event: str, handler: Callable[..., Any]) -> None:
        """Register *handler* for *event*.

        Args:
            event: The event name.
            handler: A callable that accepts keyword arguments published
                     with the event.

        Raises:
            ValueError: If *handler* is not callable.
        """
        if not callable(handler):
            raise ValueError(f"Handler for event {event!r} must be callable")
        if handler not in self._subscribers[event]:
            self._subscribers[event].append(handler)
            logger.debug(
                "Subscribed %s to event %r",
                getattr(handler, "__name__", repr(handler)),
                event,
            )

    def unsubscribe(self, event: str, handler: Callable[..., Any]) -> bool:
        """Remove *handler* from *event*.

        Returns:
            True if the handler was found and removed, False otherwise.
        """
        handlers = self._subscribers.get(event, [])
        try:
            handlers.remove(handler)
            logger.debug(
                "Unsubscribed %s from event %r",
                getattr(handler, "__name__", repr(handler)),
                event,
            )
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, event: str, **data: Any) -> None:
        """Publish *event* with arbitrary keyword *data*.

        Each subscribed handler is called with ``**data``.  Exceptions are
        caught per-handler so that one failing handler does not block
        others.
        """
        handlers = list(self._subscribers.get(event, []))
        if not handlers:
            logger.debug("No subscribers for event %r", event)
            return
        logger.debug(
            "Publishing event %r to %d handler(s)", event, len(handlers)
        )
        for handler in handlers:
            try:
                handler(**data)
            except Exception:
                logger.exception(
                    "Handler %s for event %r raised an exception",
                    getattr(handler, "__name__", repr(handler)),
                    event,
                )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def has_subscribers(self, event: str) -> bool:
        """Return True if at least one handler is registered for *event*."""
        return bool(self._subscribers.get(event))

    def list_events(self) -> Set[str]:
        """Return the set of event names that have at least one subscriber."""
        return {name for name, h in self._subscribers.items() if h}

    def clear(self, event: Optional[str] = None) -> None:
        """Remove all subscribers.

        Args:
            event: If given, only clear that event. Otherwise clear all.
        """
        if event is not None:
            self._subscribers.pop(event, None)
        else:
            self._subscribers.clear()
