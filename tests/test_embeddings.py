"""Test embedding models for Nexus-LLM."""
import math
import random
import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EmbeddingConfig:
    model_name: str = "nexus-embed-base"
    dimension: int = 128
    max_length: int = 512
    normalize: bool = True
    batch_size: int = 32


class EmbeddingModel:
    def __init__(self, config: EmbeddingConfig = None):
        self._config = config or EmbeddingConfig()
        self._loaded = False
        random.seed(42)

    @property
    def config(self):
        return self._config

    @property
    def dimension(self):
        return self._config.dimension

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def embed(self, text: str) -> List[float]:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        if not text:
            return [0.0] * self._config.dimension
        random.seed(hash(text))
        embedding = [random.gauss(0, 1) for _ in range(self._config.dimension)]
        if self._config.normalize:
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        if len(emb1) != len(emb2):
            raise ValueError("Embedding dimensions must match")
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def most_similar(self, query_embedding: List[float], corpus_embeddings: Dict[str, List[float]], top_k: int = 5) -> List[tuple]:
        scored = []
        for doc_id, embedding in corpus_embeddings.items():
            sim = self.similarity(query_embedding, embedding)
            scored.append((doc_id, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class TestEmbeddingConfig:
    def test_defaults(self):
        config = EmbeddingConfig()
        assert config.model_name == "nexus-embed-base"
        assert config.dimension == 128
        assert config.normalize is True

    def test_custom(self):
        config = EmbeddingConfig(dimension=768, max_length=1024)
        assert config.dimension == 768


class TestEmbeddingModel:
    def test_load_unload(self):
        model = EmbeddingModel()
        model.load()
        assert model.is_loaded is True
        model.unload()
        assert model.is_loaded is False

    def test_embed_not_loaded(self):
        model = EmbeddingModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.embed("test")

    def test_embed_returns_correct_dimension(self):
        model = EmbeddingModel(EmbeddingConfig(dimension=64))
        model.load()
        emb = model.embed("hello world")
        assert len(emb) == 64

    def test_embed_normalized(self):
        model = EmbeddingModel(EmbeddingConfig(dimension=64, normalize=True))
        model.load()
        emb = model.embed("hello")
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 0.01

    def test_embed_deterministic(self):
        model = EmbeddingModel()
        model.load()
        emb1 = model.embed("test text")
        emb2 = model.embed("test text")
        assert emb1 == emb2

    def test_embed_empty(self):
        model = EmbeddingModel()
        model.load()
        emb = model.embed("")
        assert all(x == 0.0 for x in emb)

    def test_embed_batch(self):
        model = EmbeddingModel()
        model.load()
        embeddings = model.embed_batch(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == model.dimension

    def test_similarity_identical(self):
        model = EmbeddingModel()
        model.load()
        emb = model.embed("test")
        sim = model.similarity(emb, emb)
        assert abs(sim - 1.0) < 0.01

    def test_similarity_dimension_mismatch(self):
        model = EmbeddingModel()
        model.load()
        with pytest.raises(ValueError):
            model.similarity([1.0], [1.0, 2.0])

    def test_most_similar(self):
        model = EmbeddingModel(EmbeddingConfig(dimension=64))
        model.load()
        query = model.embed("machine learning")
        corpus = {
            "doc1": model.embed("ML algorithms"),
            "doc2": model.embed("cooking recipes"),
            "doc3": model.embed("neural networks"),
        }
        results = model.most_similar(query, corpus, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[1], float) for r in results)

    def test_dimension_property(self):
        model = EmbeddingModel(EmbeddingConfig(dimension=256))
        assert model.dimension == 256
