"""Tests for nexus_llm.protocols.embedding_protocol module."""

import pytest
from nexus_llm.protocols.embedding_protocol import EmbeddingRequest, EmbeddingResponse, EmbeddingProtocol


class SampleEmbeddingProtocol(EmbeddingProtocol):
    def embed(self, request):
        vectors = [[0.1] * 128 for _ in request.input_texts]
        return EmbeddingResponse(model=request.model, embeddings=vectors)


class TestEmbeddingRequest:
    def test_default(self):
        req = EmbeddingRequest()
        assert req.input_texts == []

    def test_with_texts(self):
        req = EmbeddingRequest(input_texts=["hello", "world"], model="text-embedding")
        assert len(req.input_texts) == 2


class TestEmbeddingResponse:
    def test_default(self):
        resp = EmbeddingResponse()
        assert resp.embeddings == []

    def test_with_embeddings(self):
        resp = EmbeddingResponse(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        assert len(resp.embeddings) == 2


class TestEmbeddingProtocol:
    def test_validate_valid(self):
        proto = SampleEmbeddingProtocol()
        req = EmbeddingRequest(input_texts=["hello"], model="text-embedding")
        errors = proto.validate_request(req)
        assert errors == []

    def test_validate_empty_texts(self):
        proto = SampleEmbeddingProtocol()
        req = EmbeddingRequest(model="text-embedding")
        errors = proto.validate_request(req)
        assert len(errors) > 0

    def test_embed(self):
        proto = SampleEmbeddingProtocol()
        req = EmbeddingRequest(input_texts=["hello"], model="text-embedding")
        resp = proto.embed(req)
        assert len(resp.embeddings) == 1
        assert len(resp.embeddings[0]) == 128
