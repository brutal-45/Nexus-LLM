"""Tests for nexus_llm.protocols.streaming_protocol module."""

import pytest
from nexus_llm.protocols.streaming_protocol import StreamingChunk, StreamingRequest, StreamingProtocol


class SampleStreamingProtocol(StreamingProtocol):
    def stream(self, request):
        chunks = [StreamingChunk(content="Hello"), StreamingChunk(content=" World")]
        return chunks


class TestStreamingChunk:
    def test_creation(self):
        chunk = StreamingChunk(content="hello", token_count=1)
        assert chunk.content == "hello"
        assert chunk.token_count == 1

    def test_to_dict(self):
        chunk = StreamingChunk(content="test")
        d = chunk.to_dict()
        assert "content" in d


class TestStreamingRequest:
    def test_default(self):
        req = StreamingRequest()
        assert req.stream is True

    def test_with_prompt(self):
        req = StreamingRequest(prompt="Hello", model="gpt-4")
        assert req.prompt == "Hello"


class TestStreamingProtocol:
    def test_stream(self):
        proto = SampleStreamingProtocol()
        req = StreamingRequest(prompt="Hello", model="gpt-4")
        chunks = proto.stream(req)
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
