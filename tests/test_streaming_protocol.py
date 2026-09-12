"""Tests for the streaming protocol: chunks, SSE framing and chunk factories."""

from __future__ import annotations

import json

import pytest

from nexus_llm.protocols.streaming_protocol import (
    StreamChunk,
    StreamConfig,
    StreamEventType,
    StreamingProtocol,
)


class EchoProtocol(StreamingProtocol):
    """Minimal concrete implementation used by these tests."""

    def stream(self, request):
        yield StreamChunk(data="hello", index=0)
        yield StreamChunk(data="world", index=1)


class TestStreamChunk:
    def test_defaults(self):
        chunk = StreamChunk()
        assert chunk.event is StreamEventType.TOKEN
        assert chunk.data == ""
        assert chunk.index == 0
        assert chunk.id

    def test_to_dict_round_trips_fields(self):
        chunk = StreamChunk(data="tok", index=3, model="gpt2", finish_reason="stop")
        payload = chunk.to_dict()
        assert payload["data"] == "tok"
        assert payload["index"] == 3
        assert payload["model"] == "gpt2"
        assert payload["finish_reason"] == "stop"
        assert payload["event"] == StreamEventType.TOKEN.value

    def test_ids_are_unique_per_chunk(self):
        assert StreamChunk().id != StreamChunk().id

    def test_to_sse_is_valid_sse_framing(self):
        raw = StreamChunk(data="hi").to_sse()
        assert raw.startswith("data: ")
        assert raw.endswith("\n\n")
        assert json.loads(raw[len("data: ") :].strip())["data"] == "hi"


class TestStreamConfig:
    def test_defaults_are_sane(self):
        config = StreamConfig()
        assert config.buffer_size > 0
        assert config.stream_id

    def test_include_usage_flag(self):
        assert StreamConfig(include_usage=True).include_usage is True


class TestChunkFactories:
    @pytest.fixture
    def protocol(self):
        return EchoProtocol()

    def test_start_chunk(self, protocol):
        chunk = protocol.create_start_chunk(model="gpt2")
        assert chunk.event is StreamEventType.START
        assert chunk.model == "gpt2"

    def test_token_chunk_carries_text_and_index(self, protocol):
        chunk = protocol.create_token_chunk("tok", index=4, model="gpt2")
        assert chunk.data == "tok"
        assert chunk.index == 4
        assert chunk.event is StreamEventType.TOKEN

    def test_end_chunk_can_carry_usage(self, protocol):
        chunk = protocol.create_end_chunk(usage={"completion_tokens": 2})
        assert chunk.event is StreamEventType.END
        assert chunk.usage == {"completion_tokens": 2}

    def test_error_chunk_reports_the_message(self, protocol):
        chunk = protocol.create_error_chunk("boom", model="gpt2")
        assert chunk.event is StreamEventType.ERROR
        assert "boom" in chunk.data


class TestStreaming:
    def test_stream_yields_chunks(self):
        protocol = EchoProtocol()
        chunks = list(protocol.stream("prompt"))
        assert [c.data for c in chunks] == ["hello", "world"]

    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            StreamingProtocol()
