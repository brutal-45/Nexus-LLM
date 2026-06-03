"""Nexus-LLM Event System Module.

Provides an event bus and event classes for decoupled inter-component
communication. Supports synchronous and asynchronous event handlers,
event filtering, and event history.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union

from nexus_llm.constants import EVENT_HANDLER_TIMEOUT, EVENT_MAX_HISTORY

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for event handlers."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15


class EventPhase(Enum):
    """Phases of event processing."""

    PRE = "pre"
    MAIN = "main"
    POST = "post"


@dataclass
class Event:
    """Base event class for the event system.

    Attributes:
        event_type: The type/category of the event.
        data: Event payload data.
        source: The component that emitted the event.
        timestamp: When the event was created.
        event_id: Unique identifier for the event.
        priority: Event priority level.
        metadata: Additional event metadata.
        cancelled: Whether the event has been cancelled.
    """

    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    def cancel(self) -> None:
        """Cancel the event, preventing further handler execution."""
        self.cancelled = True

    def is_cancelled(self) -> bool:
        """Check if the event has been cancelled."""
        return self.cancelled

    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "metadata": self.metadata,
            "cancelled": self.cancelled,
        }


# Specific event types
@dataclass
class ModelEvent(Event):
    """Event related to model operations."""

    model_name: str = ""

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = "model"
        self.data["model_name"] = self.model_name


@dataclass
class ModelLoadedEvent(ModelEvent):
    """Event emitted when a model is loaded."""

    def __post_init__(self) -> None:
        self.event_type = "model.loaded"


@dataclass
class ModelUnloadedEvent(ModelEvent):
    """Event emitted when a model is unloaded."""

    def __post_init__(self) -> None:
        self.event_type = "model.unloaded"


@dataclass
class InferenceEvent(Event):
    """Event related to inference operations."""

    model_name: str = ""
    prompt_length: int = 0
    output_length: int = 0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = "inference"
        self.data.update({
            "model_name": self.model_name,
            "prompt_length": self.prompt_length,
            "output_length": self.output_length,
            "latency_ms": self.latency_ms,
        })


@dataclass
class InferenceStartEvent(InferenceEvent):
    """Event emitted when inference starts."""

    def __post_init__(self) -> None:
        self.event_type = "inference.start"


@dataclass
class InferenceCompleteEvent(InferenceEvent):
    """Event emitted when inference completes."""

    def __post_init__(self) -> None:
        self.event_type = "inference.complete"


@dataclass
class TrainingEvent(Event):
    """Event related to training operations."""

    step: int = 0
    epoch: int = 0
    loss: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = "training"
        self.data.update({
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
        })


@dataclass
class TrainingStartEvent(TrainingEvent):
    """Event emitted when training starts."""

    def __post_init__(self) -> None:
        self.event_type = "training.start"


@dataclass
class TrainingStepEvent(TrainingEvent):
    """Event emitted after each training step."""

    def __post_init__(self) -> None:
        self.event_type = "training.step"


@dataclass
class TrainingCompleteEvent(TrainingEvent):
    """Event emitted when training completes."""

    def __post_init__(self) -> None:
        self.event_type = "training.complete"


@dataclass
class ErrorEvent(Event):
    """Event emitted when an error occurs."""

    error_type: str = ""
    error_message: str = ""
    traceback_str: str = ""

    def __post_init__(self) -> None:
        self.event_type = "error"
        self.data.update({
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback_str,
        })


EventHandler = Union[Callable[[Event], None], Callable[[Event], Coroutine[Any, Any, None]]]


class EventHandlerRegistration:
    """Registration information for an event handler.

    Attributes:
        handler: The handler function.
        event_type: The event type to handle (None for all events).
        priority: Handler priority (higher runs first).
        filter_fn: Optional filter function to further refine handled events.
        once: Whether to remove the handler after one invocation.
    """

    def __init__(
        self,
        handler: EventHandler,
        event_type: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        once: bool = False,
    ) -> None:
        self.handler = handler
        self.event_type = event_type
        self.priority = priority
        self.filter_fn = filter_fn
        self.once = once
        self.registration_id = str(uuid.uuid4())
        self.call_count = 0

    def matches(self, event: Event) -> bool:
        """Check if this handler should process the given event.

        Args:
            event: The event to check.

        Returns:
            True if the handler should process the event.
        """
        if self.event_type is not None and event.event_type != self.event_type:
            return False
        if self.filter_fn is not None and not self.filter_fn(event):
            return False
        return True

    def __lt__(self, other: "EventHandlerRegistration") -> bool:
        return self.priority.value > other.priority.value

    def __repr__(self) -> str:
        return (
            f"EventHandlerRegistration(id={self.registration_id[:8]}, "
            f"type={self.event_type}, priority={self.priority.name})"
        )


class EventBus:
    """Central event bus for publishing and subscribing to events.

    Provides a publish/subscribe pattern for decoupled communication
    between components. Supports both synchronous and asynchronous
    handlers, event filtering, and event history.

    Example:
        >>> bus = EventBus()
        >>> bus.subscribe("model.loaded", on_model_loaded)
        >>> bus.publish(Event(event_type="model.loaded", data={"name": "gpt2"}))
    """

    def __init__(self, max_history: int = EVENT_MAX_HISTORY) -> None:
        """Initialize the event bus.

        Args:
            max_history: Maximum number of events to keep in history.
        """
        self._handlers: List[EventHandlerRegistration] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._lock = threading.RLock()
        self._handler_count = 0

    def subscribe(
        self,
        event_type: Optional[str] = None,
        handler: Optional[EventHandler] = None,
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        once: bool = False,
    ) -> EventHandlerRegistration:
        """Subscribe to events.

        Can be used as a decorator or called directly.

        Args:
            event_type: The event type to subscribe to (None for all).
            handler: The handler function (if not using as decorator).
            priority: Handler priority.
            filter_fn: Optional additional filter function.
            once: Whether to remove after one invocation.

        Returns:
            The handler registration.
        """
        if handler is not None:
            registration = EventHandlerRegistration(
                handler=handler,
                event_type=event_type,
                priority=priority,
                filter_fn=filter_fn,
                once=once,
            )
            self._add_handler(registration)
            return registration

        # Used as decorator
        def decorator(func: EventHandler) -> EventHandler:
            registration = EventHandlerRegistration(
                handler=func,
                event_type=event_type,
                priority=priority,
                filter_fn=filter_fn,
                once=once,
            )
            self._add_handler(registration)
            return func

        return decorator  # type: ignore

    def _add_handler(self, registration: EventHandlerRegistration) -> None:
        """Add a handler registration.

        Args:
            registration: The handler registration to add.
        """
        with self._lock:
            self._handlers.append(registration)
            self._handlers.sort()
            self._handler_count += 1

    def unsubscribe(self, registration: EventHandlerRegistration) -> None:
        """Remove a handler registration.

        Args:
            registration: The registration to remove.
        """
        with self._lock:
            if registration in self._handlers:
                self._handlers.remove(registration)
                self._handler_count -= 1

    def unsubscribe_all(self, event_type: Optional[str] = None) -> int:
        """Remove all handlers, optionally filtered by event type.

        Args:
            event_type: If specified, only remove handlers for this event type.

        Returns:
            Number of handlers removed.
        """
        with self._lock:
            if event_type is None:
                count = len(self._handlers)
                self._handlers.clear()
            else:
                original = len(self._handlers)
                self._handlers = [h for h in self._handlers if h.event_type != event_type]
                count = original - len(self._handlers)
            self._handler_count = len(self._handlers)
            return count

    def publish(self, event: Event) -> None:
        """Publish an event to all matching handlers.

        Args:
            event: The event to publish.
        """
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            handlers_to_call = [h for h in self._handlers if h.matches(event)]

        for registration in handlers_to_call:
            if event.is_cancelled():
                logger.debug("Event %s cancelled, skipping remaining handlers", event.event_id)
                break

            try:
                if asyncio.iscoroutinefunction(registration.handler):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(registration.handler(event))
                    except RuntimeError:
                        asyncio.run(registration.handler(event))
                else:
                    registration.handler(event)

                registration.call_count += 1

                if registration.once:
                    self.unsubscribe(registration)

            except Exception as exc:
                logger.error(
                    "Error in event handler %s for event %s: %s",
                    registration.registration_id[:8],
                    event.event_type,
                    exc,
                )

    def publish_async(self, event: Event) -> asyncio.Task:
        """Publish an event asynchronously.

        Args:
            event: The event to publish.

        Returns:
            An asyncio Task that completes when all handlers are done.
        """
        async def _publish() -> None:
            with self._lock:
                self._history.append(event)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

                handlers_to_call = [h for h in self._handlers if h.matches(event)]

            tasks = []
            for registration in handlers_to_call:
                if event.is_cancelled():
                    break
                try:
                    if asyncio.iscoroutinefunction(registration.handler):
                        tasks.append(registration.handler(event))
                    else:
                        registration.handler(event)
                        registration.call_count += 1
                        if registration.once:
                            self.unsubscribe(registration)
                except Exception as exc:
                    logger.error("Error in async handler: %s", exc)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error("Async handler error: %s", result)
                    else:
                        if i < len(handlers_to_call):
                            handlers_to_call[i].call_count += 1
                            if handlers_to_call[i].once:
                                self.unsubscribe(handlers_to_call[i])

        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(_publish())
        except RuntimeError:
            return asyncio.ensure_future(_publish())

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        source: Optional[str] = None,
    ) -> List[Event]:
        """Get event history.

        Args:
            event_type: Filter by event type.
            limit: Maximum number of events to return.
            source: Filter by source.

        Returns:
            List of matching events.
        """
        with self._lock:
            events = self._history[:]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        return events[-limit:]

    @property
    def handler_count(self) -> int:
        """Get the number of registered handlers."""
        return self._handler_count

    @property
    def event_count(self) -> int:
        """Get the total number of events published."""
        return len(self._history)

    def __repr__(self) -> str:
        return f"EventBus(handlers={self.handler_count}, events={self.event_count})"


# Global event bus instance
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance.

    Returns:
        The global EventBus singleton.
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
