"""Embedding engine for Nexus-LLM.

Generates vector embeddings from text, computes cosine similarity,
and supports both mock (deterministic) and real model backends.
"""

import hashlib
import logging
import math
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# Default embedding dimensionality
DEFAULT_DIMENSION = 128


class EmbeddingEngine:
    """Generate embeddings and compute similarity.

    By default the engine uses a deterministic mock embedding that
    hashes the input text and returns a normalised float vector.  A
    real model backend can be plugged in via *embed_fn*.

    Example::

        engine = EmbeddingEngine()
        vec = engine.embed("Hello, world!")
        batch = engine.embed_batch(["Hello", "World"])
        sim = engine.similarity(vec, batch[0])
    """

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        """Initialise the engine.

        Args:
            dimension: Embedding vector dimensionality.
            embed_fn: Optional callable that maps a string to a list of
                      floats.  When ``None`` the mock embedding is used.
        """
        self.dimension = dimension
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text string.

        Args:
            text: Input text.

        Returns:
            A list of floats of length *dimension*.
        """
        if self._embed_fn is not None:
            result = self._embed_fn(text)
            if len(result) != self.dimension:
                logger.warning(
                    "Embedding dimension mismatch: expected %d, got %d",
                    self.dimension,
                    len(result),
                )
            return result

        return self._mock_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        # If the backend provides a batch function, prefer it
        batch_fn = getattr(self._embed_fn, "batch", None) if self._embed_fn else None
        if batch_fn is not None and callable(batch_fn):
            return batch_fn(texts)

        return [self.embed(text) for text in texts]

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def similarity(emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity in [-1, 1].

        Raises:
            ValueError: If vectors have different lengths or are zero-length.
        """
        if len(emb1) != len(emb2):
            raise ValueError(
                f"Vector length mismatch: {len(emb1)} vs {len(emb2)}"
            )
        if not emb1:
            raise ValueError("Cannot compute similarity of empty vectors")

        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot / (norm1 * norm2)

    # ------------------------------------------------------------------
    # Mock embedding
    # ------------------------------------------------------------------

    def _mock_embed(self, text: str) -> List[float]:
        """Produce a deterministic mock embedding from text.

        Uses SHA-256 hashing to generate reproducible float values
        in [0, 1], then normalises the vector to unit length.
        """
        # Generate enough hash bytes for the dimension
        raw = b""
        for i in range(max(1, (self.dimension * 4 + 31) // 32)):
            h = hashlib.sha256(f"{text}|chunk{i}".encode("utf-8")).digest()
            raw += h

        # Convert bytes to floats
        values: List[float] = []
        for i in range(self.dimension):
            # Use 4 bytes per float
            offset = i * 4
            int_val = int.from_bytes(raw[offset : offset + 4], "big")
            values.append(int_val / (2**32 - 1))  # Normalise to [0, 1]

        # Centre around 0
        mean = sum(values) / len(values)
        values = [v - mean for v in values]

        # Normalise to unit length
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]

        return values
