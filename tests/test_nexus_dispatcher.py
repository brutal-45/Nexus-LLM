"""Tests for nexus_llm.nexus.dispatcher module."""

import pytest
from unittest.mock import MagicMock
from nexus_llm.nexus.dispatcher import NexusDispatcher


class TestNexusDispatcher:
    """Tests for the NexusDispatcher class."""

    def test_init_default(self):
        dispatcher = NexusDispatcher()
        assert dispatcher is not None

    def test_register_handler(self):
        dispatcher = NexusDispatcher()
        handler = MagicMock(return_value={"result": "ok"})
        dispatcher.register_handler("test_action", handler)
        assert dispatcher.has_handler("test_action")

    def test_dispatch(self):
        dispatcher = NexusDispatcher()
        handler = MagicMock(return_value={"result": "ok"})
        dispatcher.register_handler("test_action", handler)
        result = dispatcher.dispatch("test_action", {"param": "value"})
        assert result == {"result": "ok"}
        handler.assert_called_once()

    def test_dispatch_unknown_action(self):
        dispatcher = NexusDispatcher()
        with pytest.raises(KeyError):
            dispatcher.dispatch("unknown", {})

    def test_unregister_handler(self):
        dispatcher = NexusDispatcher()
        handler = MagicMock()
        dispatcher.register_handler("test_action", handler)
        dispatcher.unregister_handler("test_action")
        assert not dispatcher.has_handler("test_action")

    def test_list_handlers(self):
        dispatcher = NexusDispatcher()
        dispatcher.register_handler("a", MagicMock())
        dispatcher.register_handler("b", MagicMock())
        handlers = dispatcher.list_handlers()
        assert "a" in handlers
        assert "b" in handlers

    def test_dispatch_with_error(self):
        dispatcher = NexusDispatcher()
        handler = MagicMock(side_effect=RuntimeError("fail"))
        dispatcher.register_handler("fail_action", handler)
        with pytest.raises(RuntimeError):
            dispatcher.dispatch("fail_action", {})
