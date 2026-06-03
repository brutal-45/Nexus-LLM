"""Embedding models for vectorizing text.

Supports sentence-transformers and HuggingFace transformer models
with local embedding generation, batching, and normalization.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, model_name: str, dimension: int, normalize: bool = True):
        self.model_name = model_name
        self.dimension = dimension
        self.normalize = normalize
        self._cache: dict = {}

    @abstractmethod
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings. Subclasses must implement."""
        ...

    def embed(self, texts: str | List[str]) -> np.ndarray:
        """Embed one or more texts into vectors.

        Args:
            texts: A single text string or list of text strings.

        Returns:
            Numpy array of shape (n_texts, dimension).
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache for already-embedded texts
        uncached_indices = []
        uncached_texts = []
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        # Encode uncached texts in batches
        if uncached_texts:
            new_embeddings = self._encode_batch(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                cache_key = self._cache_key(texts[idx])
                self._cache[cache_key] = new_embeddings[j]

        # Assemble results from cache
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            result[i] = self._cache[cache_key]

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns:
            Numpy array of shape (dimension,).
        """
        result = self.embed([query])
        return result[0]

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _normalize_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embedding vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self.dimension})"


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Embedding model using the sentence-transformers library.

    Supports all models available on HuggingFace Hub via
    the sentence-transformers package.
    """

    # Known model dimensions for popular models
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "multi-qa-mpnet-base-dot-v1": 768,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: Optional[int] = None,
        normalize: bool = True,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        dimension = dimension or self.MODEL_DIMENSIONS.get(model_name, 384)
        super().__init__(model_name=model_name, dimension=dimension, normalize=normalize)
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazily load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                kwargs = {}
                if self.device:
                    kwargs["device"] = self.device
                self._model = SentenceTransformer(self.model_name, **kwargs)
                # Update dimension from actual model output
                test_emb = self._model.encode(["test"], show_progress_bar=False)
                self.dimension = test_emb.shape[1]
                logger.info(
                    "Loaded sentence-transformers model '%s' with dimension %d",
                    self.model_name,
                    self.dimension,
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence-transformers model."""
        self._load_model()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._model.encode(
                batch, show_progress_bar=False, convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        if self.normalize:
            embeddings = self._normalize_vectors(embeddings)

        return embeddings


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Embedding model using HuggingFace transformers directly.

    Uses the transformers library to load and run models
    for generating embeddings, with mean pooling over tokens.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: Optional[int] = None,
        normalize: bool = True,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        dimension = dimension or 384
        super().__init__(model_name=model_name, dimension=dimension, normalize=normalize)
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazily load the HuggingFace model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)

                if self.device:
                    device_id = self.device if self.device != "cuda" else "cuda:0"
                    self._model = self._model.to(device_id)
                elif torch.cuda.is_available():
                    self._model = self._model.to("cuda:0")
                    self.device = "cuda:0"

                self._model.eval()
                # Update dimension
                self.dimension = self._model.config.hidden_size
                logger.info(
                    "Loaded HuggingFace model '%s' with dimension %d",
                    self.model_name,
                    self.dimension,
                )
            except ImportError:
                raise ImportError(
                    "transformers and torch are required. "
                    "Install with: pip install transformers torch"
                )

    def _mean_pooling(self, model_output, attention_mask) -> np.ndarray:
        """Apply mean pooling to model output using attention mask."""
        import torch

        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return pooled.cpu().numpy().astype(np.float32)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts using HuggingFace transformers model."""
        self._load_model()
        import torch

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            device = next(self._model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)

            batch_embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        if self.normalize:
            embeddings = self._normalize_vectors(embeddings)

        return embeddings
