"""Nexus-LLM WebSocket Client.

Provides the WebSocketClient for real-time streaming communication
with the Nexus-LLM server via WebSocket protocol.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import websocket
try:
    import websocket
    _HAS_WEBSOCKET = True
except ImportError:
    _HAS_WEBSOCKET = False
    logger.warning("websocket-client not installed; WebSocketClient will use fallback")


@dataclass
class WSClientConfig:
    """Configuration for the WebSocket client.

    Attributes:
        url: WebSocket server URL.
        api_key: API key for authentication.
        reconnect: Whether to auto-reconnect on disconnect.
        reconnect_interval: Seconds between reconnection attempts.
        max_reconnects: Maximum reconnection attempts.
        ping_interval: WebSocket ping interval in seconds.
    """

    url: str = "ws://localhost:8000/ws"
    api_key: Optional[str] = None
    reconnect: bool = True
    reconnect_interval: float = 5.0
    max_reconnects: int = 10
    ping_interval: float = 30.0


@dataclass
class WSMessage:
    """A WebSocket message.

    Attributes:
        type: Message type.
        data: Message payload.
        timestamp: When the message was received/sent.
    """

    type: str = ""
    data: Any = None
    timestamp: float = field(default_factory=time.time)


class WebSocketClient:
    """WebSocket client for real-time communication with Nexus-LLM.

    Supports streaming chat, real-time events, and bidirectional
    message passing.

    Example::

        client = WebSocketClient(WSClientConfig(url="ws://localhost:8000/ws"))
        client.on_message = lambda msg: print(msg)
        client.connect()
        client.send_chat("Hello!")
        client.close()
    """

    def __init__(self, config: Optional[WSClientConfig] = None) -> None:
        self._config = config or WSClientConfig()
        self._ws: Any = None
        self._connected = False
        self._reconnect_count = 0
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._receive_thread: Optional[threading.Thread] = None
        logger.debug("WebSocketClient initialized: %s", self._config.url)

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently connected."""
        return self._connected

    def connect(self) -> bool:
        """Connect to the WebSocket server.

        Returns:
            True if connection was successful.
        """
        if not _HAS_WEBSOCKET:
            logger.error("websocket-client library not installed")
            return False

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            self._ws = websocket.WebSocketApp(
                self._config.url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message_raw,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            self._receive_thread = threading.Thread(
                target=self._ws.run_forever,
                kwargs={"ping_interval": self._config.ping_interval},
                daemon=True,
            )
            self._receive_thread.start()
            return True
        except Exception as exc:
            logger.error("WebSocket connection failed: %s", exc)
            return False

    def close(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws:
            self._ws.close()
            self._ws = None
        logger.info("WebSocket client closed")

    def send(self, message: WSMessage) -> bool:
        """Send a message through the WebSocket.

        Args:
            message: The WSMessage to send.

        Returns:
            True if the message was sent successfully.
        """
        if not self._connected or not self._ws:
            logger.warning("Cannot send: not connected")
            return False

        try:
            payload = json.dumps({"type": message.type, "data": message.data})
            self._ws.send(payload)
            return True
        except Exception as exc:
            logger.error("Failed to send message: %s", exc)
            return False

    def send_chat(self, content: str, model: str = "", **kwargs: Any) -> bool:
        """Send a chat message through the WebSocket.

        Args:
            content: Message content.
            model: Model name.
            **kwargs: Additional parameters.

        Returns:
            True if the message was sent.
        """
        return self.send(WSMessage(
            type="chat",
            data={"content": content, "model": model, **kwargs},
        ))

    def on(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type.

        Args:
            event_type: Message type to handle.
            handler: Callable to invoke with the message data.
        """
        self._message_handlers.setdefault(event_type, []).append(handler)

    def off(self, event_type: str, handler: Optional[Callable] = None) -> None:
        """Remove a handler for a message type.

        Args:
            event_type: Message type.
            handler: Specific handler to remove, or None for all.
        """
        if handler is None:
            self._message_handlers.pop(event_type, None)
        elif event_type in self._message_handlers:
            self._message_handlers[event_type] = [
                h for h in self._message_handlers[event_type] if h is not handler
            ]

    def _on_open(self, ws: Any) -> None:
        """Handle WebSocket connection open."""
        self._connected = True
        self._reconnect_count = 0
        logger.info("WebSocket connected to %s", self._config.url)
        self._dispatch("connected", None)

    def _on_message_raw(self, ws: Any, message: str) -> None:
        """Handle a raw WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            msg_data = data.get("data", data)
            self._dispatch(msg_type, msg_data)
        except json.JSONDecodeError:
            self._dispatch("raw", message)

    def _on_error(self, ws: Any, error: Any) -> None:
        """Handle WebSocket error."""
        logger.error("WebSocket error: %s", error)
        self._dispatch("error", str(error))

    def _on_close(self, ws: Any, close_status: Any, close_msg: Any) -> None:
        """Handle WebSocket close."""
        self._connected = False
        logger.info("WebSocket closed: %s %s", close_status, close_msg)
        self._dispatch("disconnected", {"status": close_status, "message": close_msg})

        # Auto-reconnect
        if self._config.reconnect and self._reconnect_count < self._config.max_reconnects:
            self._reconnect_count += 1
            time.sleep(self._config.reconnect_interval)
            logger.info("Reconnecting (attempt %d/%d)...", self._reconnect_count, self._config.max_reconnects)
            self.connect()

    def _dispatch(self, event_type: str, data: Any) -> None:
        """Dispatch an event to registered handlers."""
        for handler in self._message_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as exc:
                logger.error("Handler error for '%s': %s", event_type, exc)
