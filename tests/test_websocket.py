"""Test WebSocket handlers for Nexus-LLM."""
import pytest
import json
import time
from typing import Dict, Any, Optional, List, Callable
from queue import Queue


class WebSocketError(Exception):
    pass


class MockWebSocket:
    def __init__(self):
        self._sent_messages: List[str] = []
        self._received: Queue = Queue()
        self._closed = False

    def send(self, message: str):
        if self._closed:
            raise WebSocketError("Connection closed")
        self._sent_messages.append(message)

    def receive(self) -> Optional[str]:
        if self._received.empty():
            return None
        return self._received.get()

    def inject_message(self, message: str):
        self._received.put(message)

    def close(self):
        self._closed = True

    @property
    def is_closed(self):
        return self._closed

    @property
    def sent_messages(self):
        return list(self._sent_messages)


class WebSocketHandler:
    def __init__(self):
        self._connections: Dict[str, MockWebSocket] = {}
        self._message_handlers: Dict[str, Callable] = {}

    def register_handler(self, message_type: str, handler: Callable):
        self._message_handlers[message_type] = handler

    def on_connect(self, client_id: str, ws: MockWebSocket):
        self._connections[client_id] = ws
        ws.send(json.dumps({"type": "connected", "client_id": client_id}))

    def on_disconnect(self, client_id: str):
        if client_id in self._connections:
            self._connections[client_id].close()
            del self._connections[client_id]

    def on_message(self, client_id: str, raw_message: str):
        if client_id not in self._connections:
            raise WebSocketError(f"Unknown client: {client_id}")
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            self._connections[client_id].send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            return

        msg_type = message.get("type", "")
        handler = self._message_handlers.get(msg_type)
        if handler:
            response = handler(message, client_id)
            if response:
                self._connections[client_id].send(json.dumps(response))
        else:
            self._connections[client_id].send(json.dumps({"type": "error", "message": f"Unknown type: {msg_type}"}))

    def broadcast(self, message: str):
        for ws in self._connections.values():
            ws.send(message)

    def get_connection_count(self) -> int:
        return len(self._connections)


# Default handlers
def handle_generate(message: dict, client_id: str) -> dict:
    prompt = message.get("prompt", "")
    if not prompt:
        return {"type": "error", "message": "prompt is required"}
    return {"type": "generate_response", "text": f"Generated: {prompt}"}


def handle_ping(message: dict, client_id: str) -> dict:
    return {"type": "pong"}


class TestMockWebSocket:
    def test_send(self):
        ws = MockWebSocket()
        ws.send("hello")
        assert ws.sent_messages == ["hello"]

    def test_receive(self):
        ws = MockWebSocket()
        ws.inject_message("test")
        assert ws.receive() == "test"

    def test_receive_empty(self):
        ws = MockWebSocket()
        assert ws.receive() is None

    def test_close(self):
        ws = MockWebSocket()
        ws.close()
        assert ws.is_closed is True

    def test_send_after_close(self):
        ws = MockWebSocket()
        ws.close()
        with pytest.raises(WebSocketError, match="closed"):
            ws.send("test")

    def test_multiple_sends(self):
        ws = MockWebSocket()
        ws.send("msg1")
        ws.send("msg2")
        assert len(ws.sent_messages) == 2


class TestWebSocketHandler:
    def test_connect(self):
        handler = WebSocketHandler()
        ws = MockWebSocket()
        handler.on_connect("client1", ws)
        assert handler.get_connection_count() == 1
        assert len(ws.sent_messages) == 1

    def test_disconnect(self):
        handler = WebSocketHandler()
        ws = MockWebSocket()
        handler.on_connect("client1", ws)
        handler.on_disconnect("client1")
        assert handler.get_connection_count() == 0
        assert ws.is_closed

    def test_disconnect_unknown(self):
        handler = WebSocketHandler()
        handler.on_disconnect("unknown")

    def test_message_handling(self):
        handler = WebSocketHandler()
        handler.register_handler("generate", handle_generate)
        ws = MockWebSocket()
        handler.on_connect("client1", ws)
        handler.on_message("client1", json.dumps({"type": "generate", "prompt": "hello"}))
        responses = [json.loads(m) for m in ws.sent_messages]
        gen_response = [r for r in responses if r.get("type") == "generate_response"]
        assert len(gen_response) >= 1

    def test_ping_pong(self):
        handler = WebSocketHandler()
        handler.register_handler("ping", handle_ping)
        ws = MockWebSocket()
        handler.on_connect("client1", ws)
        handler.on_message("client1", json.dumps({"type": "ping"}))
        responses = [json.loads(m) for m in ws.sent_messages]
        pong = [r for r in responses if r.get("type") == "pong"]
        assert len(pong) == 1

    def test_unknown_message_type(self):
        handler = WebSocketHandler()
        ws = MockWebSocket()
        handler.on_connect("client1", ws)
        handler.on_message("client1", json.dumps({"type": "unknown"}))
        responses = [json.loads(m) for m in ws.sent_messages]
        errors = [r for r in responses if r.get("type") == "error"]
        assert len(errors) >= 1

    def test_invalid_json(self):
        handler = WebSocketHandler()
        ws = MockWebSocket()
        handler.on_connect("client1", ws)
        handler.on_message("client1", "not json")
        responses = [json.loads(m) for m in ws.sent_messages]
        errors = [r for r in responses if r.get("type") == "error"]
        assert len(errors) >= 1

    def test_unknown_client(self):
        handler = WebSocketHandler()
        with pytest.raises(WebSocketError, match="Unknown client"):
            handler.on_message("unknown", json.dumps({"type": "ping"}))

    def test_broadcast(self):
        handler = WebSocketHandler()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        handler.on_connect("c1", ws1)
        handler.on_connect("c2", ws2)
        handler.broadcast(json.dumps({"type": "announcement"}))
        assert any("announcement" in m for m in ws1.sent_messages)
        assert any("announcement" in m for m in ws2.sent_messages)
