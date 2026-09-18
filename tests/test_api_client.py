"""Tests for the API client module.

All HTTP calls are mocked — no live server required.
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from nexus_llm.api.client import NexusClient, ConnectionState
from nexus_llm.core.exceptions import ServerError


# ---------------------------------------------------------------------------
# NexusClient initialization
# ---------------------------------------------------------------------------

class TestNexusClientInit:
    """Tests for NexusClient initialization."""

    def test_default_init(self):
        client = NexusClient()
        assert client.base_url == "http://localhost:8000"
        assert client.state == ConnectionState.DISCONNECTED
        assert client.is_connected is False

    def test_custom_init(self):
        client = NexusClient(
            base_url="http://myserver:9999",
            api_key="secret-key",
            timeout=60.0,
            max_retries=5,
            retry_delay=2.0,
        )
        assert client.base_url == "http://myserver:9999"
        assert client.state == ConnectionState.DISCONNECTED

    def test_base_url_trailing_slash_stripped(self):
        client = NexusClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_connection_state_enum(self):
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# NexusClient — connect / disconnect (mocked requests)
# ---------------------------------------------------------------------------

class TestNexusClientConnect:
    """Tests for connect and disconnect methods."""

    @patch("requests.Session")
    def test_connect_success(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        # Mock the health-check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "model_loaded": False}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        client = NexusClient()
        client.connect()

        assert client.is_connected is True
        assert client.state == ConnectionState.CONNECTED

    @patch("requests.Session")
    def test_connect_failure(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        client = NexusClient()
        # Make the health check fail
        mock_session.request.side_effect = ConnectionError("Connection refused")
        client._session = mock_session

        with pytest.raises(ServerError, match="Cannot connect"):
            client.connect()

        assert client.state == ConnectionState.FAILED

    def test_disconnect(self):
        client = NexusClient()
        mock_session = MagicMock()
        client._session = mock_session
        client._state = ConnectionState.CONNECTED

        client.disconnect()
        assert client.state == ConnectionState.DISCONNECTED
        assert client.is_connected is False
        mock_session.close.assert_called_once()

    def test_disconnect_without_session(self):
        client = NexusClient()
        client._session = None
        client.disconnect()  # Should not raise
        assert client.state == ConnectionState.DISCONNECTED


# ---------------------------------------------------------------------------
# NexusClient — chat (mocked)
# ---------------------------------------------------------------------------

class TestNexusClientChat:
    """Tests for the chat method."""

    def test_chat_success(self):
        client = NexusClient()
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Hello! How can I help?",
            "generated_tokens": 6,
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        result = client.chat("Hello")
        assert result["text"] == "Hello! How can I help?"

    def test_chat_with_history(self):
        client = NexusClient()
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Follow-up response"}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        history = [{"role": "user", "content": "Previous message"}]
        result = client.chat("Follow up", history=history)
        assert result["text"] == "Follow-up response"

    def test_chat_with_model_override(self):
        client = NexusClient()
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Response from phi-2"}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        result = client.chat("Hello", model="phi-2")
        # Verify model was included in the request payload
        call_args = mock_session.request.call_args
        json_payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert json_payload["model"] == "phi-2"


# ---------------------------------------------------------------------------
# NexusClient — generate (mocked)
# ---------------------------------------------------------------------------

class TestNexusClientGenerate:
    """Tests for the generate method."""

    def test_generate_success(self):
        client = NexusClient()
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Once upon a time...",
            "generated_tokens": 10,
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        result = client.generate("Once upon a time")
        assert result["text"] == "Once upon a time..."

    def test_generate_with_params(self):
        client = NexusClient()
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Generated"}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        result = client.generate("Test", max_length=100, temperature=0.5, top_p=0.8, top_k=30)
        call_args = mock_session.request.call_args
        json_payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert json_payload["max_length"] == 100
        assert json_payload["temperature"] == 0.5
        assert json_payload["top_p"] == 0.8
        assert json_payload["top_k"] == 30


# ---------------------------------------------------------------------------
# NexusClient — health check (mocked)
# ---------------------------------------------------------------------------

class TestNexusClientHealth:
    """Tests for the health method."""

    def test_health_success(self):
        client = NexusClient()
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "model_loaded": True,
            "model_id": "gpt2-medium",
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        result = client.health()
        assert result["status"] == "healthy"
        assert result["model_loaded"] is True


# ---------------------------------------------------------------------------
# NexusClient — _request internals (mocked)
# ---------------------------------------------------------------------------

class TestNexusClientRequest:
    """Tests for the internal _request method."""

    def test_request_not_connected_raises(self):
        client = NexusClient()
        client._session = None
        with pytest.raises(ServerError, match="Not connected"):
            client._request("GET", "/health")

    def test_request_retry_on_failure(self):
        client = NexusClient(max_retries=2, retry_delay=0.01)
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status.side_effect = ConnectionError("timeout")

        mock_response_ok = MagicMock()
        mock_response_ok.json.return_value = {"status": "ok"}
        mock_response_ok.raise_for_status = MagicMock()

        mock_session.request.side_effect = [mock_response_fail, mock_response_ok]

        # Need to patch time.sleep to avoid delays
        with patch("time.sleep"):
            result = client._request("GET", "/health")

        assert result["status"] == "ok"
        assert mock_session.request.call_count == 2

    def test_request_all_retries_fail(self):
        client = NexusClient(max_retries=2, retry_delay=0.01)
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = ConnectionError("timeout")
        mock_session.request.return_value = mock_response

        with patch("time.sleep"):
            with pytest.raises(ServerError, match="failed"):
                client._request("GET", "/health")

    def test_request_no_retry(self):
        client = NexusClient(max_retries=3, retry_delay=0.01)
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = ConnectionError("fail")
        mock_session.request.return_value = mock_response

        with pytest.raises(ServerError):
            client._request("GET", "/health", retry=False)

        # Should only try once
        assert mock_session.request.call_count == 1


# ---------------------------------------------------------------------------
# NexusClient — headers
# ---------------------------------------------------------------------------

class TestNexusClientHeaders:
    """Tests for header building."""

    def test_headers_without_api_key(self):
        client = NexusClient()
        headers = client._build_headers()
        assert "Content-Type" in headers
        assert "Authorization" not in headers

    def test_headers_with_api_key(self):
        client = NexusClient(api_key="my-secret-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# NexusClient — context manager
# ---------------------------------------------------------------------------

class TestNexusClientContextManager:
    """Tests for context manager usage."""

    @patch("nexus_llm.api.client.NexusClient.connect")
    @patch("nexus_llm.api.client.NexusClient.disconnect")
    def test_context_manager(self, mock_disconnect, mock_connect):
        with NexusClient() as client:
            mock_connect.assert_called_once()
        mock_disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# NexusClient — model management (mocked)
# ---------------------------------------------------------------------------

class TestNexusClientModelManagement:
    """Tests for model management methods."""

    def _make_connected_client(self):
        client = NexusClient()
        client._state = ConnectionState.CONNECTED
        mock_session = MagicMock()
        client._session = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_session.request.return_value = mock_response

        return client, mock_session

    def test_load_model(self):
        client, mock_session = self._make_connected_client()
        result = client.load_model("gpt2-medium", device="cpu", precision="fp32")
        call_args = mock_session.request.call_args
        json_payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert json_payload["model_id"] == "gpt2-medium"
        assert json_payload["device"] == "cpu"
        assert json_payload["precision"] == "fp32"

    def test_unload_model(self):
        client, mock_session = self._make_connected_client()
        result = client.unload_model()
        call_args = mock_session.request.call_args
        assert "unload" in call_args.kwargs.get("url", "") or "unload" in str(call_args)

    def test_get_model_info(self):
        client, _ = self._make_connected_client()
        result = client.get_model_info()
        # Returns whatever the server responded
        assert result["status"] == "ok"

    def test_list_models(self):
        client, mock_session = self._make_connected_client()
        mock_session.request.return_value.json.return_value = {
            "models": [{"id": "gpt2-medium"}, {"id": "phi-2"}]
        }
        result = client.list_models()
        assert len(result) == 2
