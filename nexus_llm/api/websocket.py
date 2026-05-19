"""WebSocket handlers: streaming generation, real-time chat, progress updates."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from nexus_llm.models.base import GenerationConfig

logger = logging.getLogger("nexus_llm.api.websocket")


class ConnectionManager:
    """Manages active WebSocket connections.

    Handles connection lifecycle, message broadcasting,
    and connection grouping by channels.
    """

    def __init__(self) -> None:
        self._connections: Dict[str, WebSocket] = {}
        self._channels: Dict[str, Set[str]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket instance.
            client_id: Optional client ID. Auto-generated if not provided.

        Returns:
            The assigned client ID.
        """
        await websocket.accept()
        cid = client_id or str(uuid.uuid4())
        self._connections[cid] = websocket
        self._metadata[cid] = {
            "connected_at": time.time(),
            "channels": set(),
        }
        logger.info("WebSocket client connected: %s", cid)
        return cid

    def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection.

        Args:
            client_id: The client ID to disconnect.
        """
        self._connections.pop(client_id, None)
        metadata = self._metadata.pop(client_id, None)
        if metadata:
            for channel in metadata.get("channels", set()):
                if channel in self._channels:
                    self._channels[channel].discard(client_id)
        logger.info("WebSocket client disconnected: %s", client_id)

    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send a JSON message to a specific client.

        Args:
            client_id: Target client ID.
            message: Message payload as a dictionary.

        Returns:
            True if message was sent successfully.
        """
        ws = self._connections.get(client_id)
        if ws is None:
            return False
        try:
            await ws.send_json(message)
            return True
        except Exception as e:
            logger.error("Failed to send message to %s: %s", client_id, e)
            self.disconnect(client_id)
            return False

    async def broadcast(self, message: Dict[str, Any], channel: Optional[str] = None) -> int:
        """Broadcast a message to all or channel-specific connections.

        Args:
            message: Message payload.
            channel: Optional channel to broadcast to.

        Returns:
            Number of clients that received the message.
        """
        if channel and channel in self._channels:
            client_ids = list(self._channels[channel])
        else:
            client_ids = list(self._connections.keys())

        sent_count = 0
        for cid in client_ids:
            if await self.send_message(cid, message):
                sent_count += 1
        return sent_count

    def subscribe(self, client_id: str, channel: str) -> None:
        """Subscribe a client to a channel.

        Args:
            client_id: Client ID.
            channel: Channel name.
        """
        if channel not in self._channels:
            self._channels[channel] = set()
        self._channels[channel].add(client_id)
        if client_id in self._metadata:
            self._metadata[client_id]["channels"].add(channel)

    def unsubscribe(self, client_id: str, channel: str) -> None:
        """Unsubscribe a client from a channel."""
        if channel in self._channels:
            self._channels[channel].discard(client_id)
        if client_id in self._metadata:
            self._metadata[client_id]["channels"].discard(channel)

    def get_active_count(self) -> int:
        """Return the number of active connections."""
        return len(self._connections)

    def get_channel_members(self, channel: str) -> Set[str]:
        """Return client IDs subscribed to a channel."""
        return self._channels.get(channel, set())

    def list_channels(self) -> List[str]:
        """Return all active channel names."""
        return [ch for ch, members in self._channels.items() if members]


# Global connection manager
_manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global WebSocket connection manager."""
    return _manager


class WebSocketMessageHandler:
    """Handles incoming WebSocket messages for generation and chat.

    Routes messages to appropriate handlers based on the
    message type field.
    """

    def __init__(self) -> None:
        self._model_manager: Optional[Any] = None
        self.manager = get_connection_manager()

    def set_model_manager(self, manager: Any) -> None:
        """Set the model manager for generation requests.

        Args:
            manager: Model manager instance with get_model() method.
        """
        self._model_manager = manager

    async def handle_message(self, client_id: str, data: Dict[str, Any]) -> None:
        """Route and handle an incoming WebSocket message.

        Args:
            client_id: The client that sent the message.
            data: The parsed JSON message.
        """
        msg_type = data.get("type", "")

        handlers = {
            "generate": self._handle_generate,
            "chat": self._handle_chat,
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "ping": self._handle_ping,
            "cancel": self._handle_cancel,
        }

        handler = handlers.get(msg_type)
        if handler:
            try:
                await handler(client_id, data)
            except Exception as e:
                await self.manager.send_message(client_id, {
                    "type": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request_id": data.get("request_id"),
                })
        else:
            await self.manager.send_message(client_id, {
                "type": "error",
                "error": f"Unknown message type: {msg_type}",
                "valid_types": list(handlers.keys()),
            })

    async def _handle_generate(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle a streaming generation request.

        Args:
            client_id: Client requesting generation.
            data: Message data with prompt and config.
        """
        if self._model_manager is None:
            await self.manager.send_message(client_id, {
                "type": "error",
                "error": "Model manager not configured",
            })
            return

        model_name = data.get("model", "default")
        prompt = data.get("prompt", "")
        request_id = data.get("request_id", str(uuid.uuid4()))

        if not prompt:
            await self.manager.send_message(client_id, {
                "type": "error",
                "error": "Prompt is required",
                "request_id": request_id,
            })
            return

        try:
            model = self._model_manager.get_model(model_name)
        except Exception as e:
            await self.manager.send_message(client_id, {
                "type": "error",
                "error": f"Model not available: {e}",
                "request_id": request_id,
            })
            return

        config = GenerationConfig(
            max_new_tokens=data.get("max_new_tokens", 512),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            do_sample=data.get("do_sample", True),
        )

        await self.manager.send_message(client_id, {
            "type": "generate_start",
            "request_id": request_id,
            "model": model_name,
        })

        start_time = time.time()
        full_text = ""
        token_count = 0

        try:
            for chunk in model.stream(prompt, config=config):
                full_text += chunk
                token_count += 1
                await self.manager.send_message(client_id, {
                    "type": "generate_chunk",
                    "request_id": request_id,
                    "chunk": chunk,
                })
                await asyncio.sleep(0)
        except Exception as e:
            await self.manager.send_message(client_id, {
                "type": "error",
                "error": f"Generation failed: {e}",
                "request_id": request_id,
            })
            return

        elapsed = time.time() - start_time
        await self.manager.send_message(client_id, {
            "type": "generate_complete",
            "request_id": request_id,
            "model": model_name,
            "total_tokens": token_count,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(token_count / elapsed, 2) if elapsed > 0 else 0,
        })

    async def _handle_chat(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle a real-time chat message.

        Args:
            client_id: Client sending the chat message.
            data: Message data with messages list and config.
        """
        if self._model_manager is None:
            await self.manager.send_message(client_id, {
                "type": "error", "error": "Model manager not configured",
            })
            return

        model_name = data.get("model", "default")
        messages = data.get("messages", [])
        request_id = data.get("request_id", str(uuid.uuid4()))

        if not messages:
            await self.manager.send_message(client_id, {
                "type": "error", "error": "Messages are required",
                "request_id": request_id,
            })
            return

        try:
            model = self._model_manager.get_model(model_name)
        except Exception as e:
            await self.manager.send_message(client_id, {
                "type": "error", "error": f"Model not available: {e}",
                "request_id": request_id,
            })
            return

        config = GenerationConfig(
            max_new_tokens=data.get("max_new_tokens", 1024),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
        )

        await self.manager.send_message(client_id, {
            "type": "chat_start",
            "request_id": request_id,
        })

        full_response = ""
        try:
            for chunk in model.stream(
                messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1]),
                config=config,
            ):
                full_response += chunk
                await self.manager.send_message(client_id, {
                    "type": "chat_chunk",
                    "request_id": request_id,
                    "chunk": chunk,
                })
                await asyncio.sleep(0)
        except Exception as e:
            await self.manager.send_message(client_id, {
                "type": "error",
                "error": f"Chat generation failed: {e}",
                "request_id": request_id,
            })
            return

        await self.manager.send_message(client_id, {
            "type": "chat_complete",
            "request_id": request_id,
            "message": {"role": "assistant", "content": full_response},
        })

    async def _handle_subscribe(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle channel subscription request."""
        channel = data.get("channel", "general")
        self.manager.subscribe(client_id, channel)
        await self.manager.send_message(client_id, {
            "type": "subscribed",
            "channel": channel,
        })

    async def _handle_unsubscribe(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle channel unsubscription request."""
        channel = data.get("channel", "")
        self.manager.unsubscribe(client_id, channel)
        await self.manager.send_message(client_id, {
            "type": "unsubscribed",
            "channel": channel,
        })

    async def _handle_ping(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle ping message with pong response."""
        await self.manager.send_message(client_id, {
            "type": "pong",
            "timestamp": time.time(),
        })

    async def _handle_cancel(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle generation cancellation request."""
        request_id = data.get("request_id")
        await self.manager.send_message(client_id, {
            "type": "cancelled",
            "request_id": request_id,
        })


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint handler.

    Handles the full lifecycle of a WebSocket connection including
    message parsing, routing, and cleanup on disconnect.

    Args:
        websocket: The WebSocket connection instance.
    """
    manager = get_connection_manager()
    handler = WebSocketMessageHandler()

    client_id = await manager.connect(websocket)

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await manager.send_message(client_id, {
                    "type": "error",
                    "error": "Invalid JSON message",
                })
                continue

            await handler.handle_message(client_id, data)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error for client %s: %s", client_id, e)
        manager.disconnect(client_id)
