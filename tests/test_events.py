"""Tests for event system (EventBus, event handlers, async events)."""
import asyncio

import pytest

from nexus_llm.events import (
    Event,
    EventBus,
    EventPriority,
    EventHandlerRegistration,
    ModelLoadedEvent,
    ModelUnloadedEvent,
    InferenceStartEvent,
    InferenceCompleteEvent,
    TrainingStartEvent,
    TrainingStepEvent,
    TrainingCompleteEvent,
    ErrorEvent,
    get_event_bus,
)


class TestEvent:
    """Test Event dataclass."""

    def test_create_event(self):
        event = Event(event_type="test", data={"key": "value"})
        assert event.event_type == "test"
        assert event.data["key"] == "value"
        assert event.cancelled is False

    def test_cancel_event(self):
        event = Event(event_type="test")
        event.cancel()
        assert event.is_cancelled() is True

    def test_to_dict(self):
        event = Event(event_type="test", data={"x": 1}, source="unit")
        d = event.to_dict()
        assert d["event_type"] == "test"
        assert d["data"]["x"] == 1
        assert d["source"] == "unit"
        assert "event_id" in d
        assert "timestamp" in d

    def test_event_has_unique_id(self):
        e1 = Event(event_type="a")
        e2 = Event(event_type="a")
        assert e1.event_id != e2.event_id


class TestSpecificEventTypes:
    """Test specific event type classes."""

    def test_model_loaded_event(self):
        event = ModelLoadedEvent(event_type="model.loaded", model_name="gpt2")
        assert event.event_type == "model.loaded"
        assert event.model_name == "gpt2"

    def test_model_unloaded_event(self):
        event = ModelUnloadedEvent(event_type="model.unloaded", model_name="gpt2")
        assert event.event_type == "model.unloaded"
        assert event.model_name == "gpt2"

    def test_inference_start_event(self):
        event = InferenceStartEvent(event_type="inference.start", model_name="gpt2", prompt_length=10)
        assert event.event_type == "inference.start"
        assert event.model_name == "gpt2"
        assert event.prompt_length == 10

    def test_inference_complete_event(self):
        event = InferenceCompleteEvent(event_type="inference.complete", model_name="gpt2", latency_ms=100.0)
        assert event.event_type == "inference.complete"
        assert event.latency_ms == 100.0

    def test_training_start_event(self):
        event = TrainingStartEvent(event_type="training.start", epoch=1)
        assert event.event_type == "training.start"
        assert event.epoch == 1

    def test_training_step_event(self):
        event = TrainingStepEvent(event_type="training.step", step=100, loss=0.5)
        assert event.event_type == "training.step"
        assert event.step == 100
        assert event.loss == 0.5

    def test_training_complete_event(self):
        event = TrainingCompleteEvent(event_type="training.complete")
        assert event.event_type == "training.complete"

    def test_error_event(self):
        event = ErrorEvent(event_type="error", error_type="RuntimeError", error_message="boom")
        assert event.event_type == "error"
        assert event.error_type == "RuntimeError"
        assert event.error_message == "boom"


class TestEventHandlerRegistration:
    """Test EventHandlerRegistration."""

    def test_matches_by_event_type(self):
        def handler(event):
            pass
        reg = EventHandlerRegistration(handler=handler, event_type="model.loaded")
        e1 = Event(event_type="model.loaded")
        e2 = Event(event_type="model.unloaded")
        assert reg.matches(e1) is True
        assert reg.matches(e2) is False

    def test_matches_all_when_no_type(self):
        def handler(event):
            pass
        reg = EventHandlerRegistration(handler=handler)
        assert reg.matches(Event(event_type="any")) is True

    def test_matches_with_filter(self):
        def handler(event):
            pass
        reg = EventHandlerRegistration(
            handler=handler,
            event_type="model.loaded",
            filter_fn=lambda e: e.data.get("model_name") == "gpt2",
        )
        e1 = Event(event_type="model.loaded", data={"model_name": "gpt2"})
        e2 = Event(event_type="model.loaded", data={"model_name": "llama"})
        assert reg.matches(e1) is True
        assert reg.matches(e2) is False

    def test_priority_ordering(self):
        def handler(event):
            pass
        low = EventHandlerRegistration(handler=handler, priority=EventPriority.LOW)
        high = EventHandlerRegistration(handler=handler, priority=EventPriority.HIGH)
        assert high < low  # Higher priority sorts first


class TestEventBus:
    """Test EventBus."""

    @pytest.fixture
    def bus(self):
        return EventBus()

    def test_subscribe_and_publish(self, bus):
        received = []
        bus.subscribe(event_type="test", handler=lambda e: received.append(e))
        bus.publish(Event(event_type="test"))
        assert len(received) == 1

    def test_subscribe_with_decorator(self, bus):
        received = []
        @bus.subscribe(event_type="test")
        def handler(event):
            received.append(event)
        bus.publish(Event(event_type="test"))
        assert len(received) == 1

    def test_unsubscribe(self, bus):
        received = []
        reg = bus.subscribe(event_type="test", handler=lambda e: received.append(e))
        bus.unsubscribe(reg)
        bus.publish(Event(event_type="test"))
        assert len(received) == 0

    def test_unsubscribe_all(self, bus):
        bus.subscribe(event_type="a", handler=lambda e: None)
        bus.subscribe(event_type="b", handler=lambda e: None)
        count = bus.unsubscribe_all()
        assert count == 2
        assert bus.handler_count == 0

    def test_unsubscribe_all_by_type(self, bus):
        bus.subscribe(event_type="a", handler=lambda e: None)
        bus.subscribe(event_type="b", handler=lambda e: None)
        bus.subscribe(event_type="a", handler=lambda e: None)
        count = bus.unsubscribe_all(event_type="a")
        assert count == 2

    def test_event_cancellation(self, bus):
        results = []

        def cancelling_handler(event):
            results.append("first")
            event.cancel()

        def second_handler(event):
            results.append("second")

        bus.subscribe(event_type="test", handler=cancelling_handler, priority=EventPriority.HIGH)
        bus.subscribe(event_type="test", handler=second_handler, priority=EventPriority.LOW)
        bus.publish(Event(event_type="test"))
        assert results == ["first"]

    def test_once_handler(self, bus):
        count = [0]
        bus.subscribe(event_type="test", handler=lambda e: count.__setitem__(0, count[0] + 1), once=True)
        bus.publish(Event(event_type="test"))
        bus.publish(Event(event_type="test"))
        assert count[0] == 1

    def test_handler_count(self, bus):
        assert bus.handler_count == 0
        bus.subscribe(event_type="test", handler=lambda e: None)
        assert bus.handler_count == 1

    def test_event_count(self, bus):
        assert bus.event_count == 0
        bus.publish(Event(event_type="test"))
        assert bus.event_count == 1

    def test_get_history(self, bus):
        bus.publish(Event(event_type="a"))
        bus.publish(Event(event_type="b"))
        bus.publish(Event(event_type="a"))
        history = bus.get_history(event_type="a")
        assert len(history) == 2

    def test_get_history_with_source(self, bus):
        bus.publish(Event(event_type="test", source="src1"))
        bus.publish(Event(event_type="test", source="src2"))
        history = bus.get_history(source="src1")
        assert len(history) == 1

    def test_handler_error_does_not_crash(self, bus):
        def bad_handler(event):
            raise RuntimeError("boom")

        bus.subscribe(event_type="test", handler=bad_handler)
        bus.publish(Event(event_type="test"))  # Should not raise

    def test_repr(self, bus):
        r = repr(bus)
        assert "EventBus" in r


class TestGetEventBus:
    """Test global event bus."""

    def test_returns_event_bus(self):
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_singleton(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
