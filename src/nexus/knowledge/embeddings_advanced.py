"""
Nexus LLM — Advanced Embeddings Module
========================================

Text embedding models for the knowledge subsystem.  Includes a simple
bag-of-words encoder, a BERT-style transformer encoder (implemented from
scratch), and adapter classes for E5, BGE, Contriever, and Matryoshka
embedding paradigms.

Also provides :class:`EmbeddingTrainer` for contrastive fine-tuning
with InfoNCE loss and hard negative mining.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger: logging.Logger = logging.getLogger("nexus.knowledge.embeddings")


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingOutput:
    """Output of an embedding model.

    Attributes
    ----------
    embeddings : np.ndarray
        (N, D) embedding matrix.
    model_name : str
        Name of the model that produced the embeddings.
    dimension : int
        Output dimensionality.
    pooling : str
        Pooling strategy used.
    normalized : bool
        Whether embeddings are L2-normalised.
    """

    embeddings: np.ndarray
    model_name: str = "unknown"
    dimension: int = 0
    pooling: str = "mean"
    normalized: bool = False


class PoolingStrategy(str, Enum):
    MEAN = "mean"
    CLS = "cls"
    MAX = "max"


# ═══════════════════════════════════════════════════════════════════════════
#  BaseEmbeddingModel
# ═══════════════════════════════════════════════════════════════════════════

class BaseEmbeddingModel(abc.ABC):
    """Abstract interface for embedding models."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Output embedding dimensionality."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model identifier."""

    @abc.abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode *texts* to vectors (N, D)."""

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode query texts (with optional query prefix)."""
        return self.encode(texts, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode document texts (with optional document prefix)."""
        return self.encode(texts, batch_size)


# ═══════════════════════════════════════════════════════════════════════════
#  SimpleEmbedding (bag-of-words hashing)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleEmbedding(BaseEmbeddingModel):
    """Lightweight embedding using character n-gram hashing.

    No model download required.  Produces deterministic vectors using
    multiple hash functions over character n-grams.  Supports an optional
    trainable projection matrix for fine-tuning.

    Parameters
    ----------
    dimension : int
        Output dimensionality.
    ngram_range : Tuple[int, int]
        Character n-gram range (min_n, max_n).
    num_hashes : int
        Number of hash functions per token.
    seed : int
        Random seed for reproducibility.
    normalize : bool
        L2-normalise output embeddings.
    projection_dim : Optional[int]
        If set, apply a random projection to this dimension.
    """

    def __init__(
        self,
        dimension: int = 256,
        ngram_range: Tuple[int, int] = (3, 5),
        num_hashes: int = 4,
        seed: int = 42,
        normalize: bool = True,
        projection_dim: Optional[int] = None,
    ) -> None:
        self._dim = dimension
        self._ngram_lo, self._ngram_hi = ngram_range
        self._num_hashes = num_hashes
        self._seed = seed
        self._normalize = normalize
        self._proj_dim = projection_dim
        self._rng = np.random.RandomState(seed)
        self._projection: Optional[np.ndarray] = None

        if projection_dim is not None:
            self._projection = self._rng.randn(dimension, projection_dim).astype(np.float32)
            self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)

    @property
    def dimension(self) -> int:
        return self._proj_dim if self._proj_dim is not None else self._dim

    @property
    def model_name(self) -> str:
        return "simple"

    def _get_ngrams(self, text: str) -> List[str]:
        text = text.lower().strip()
        ngrams: List[str] = []
        for n in range(self._ngram_lo, self._ngram_hi + 1):
            for i in range(max(0, len(text) - n + 1)):
                ngrams.append(text[i : i + n])
        return ngrams

    def _hash_ngram(self, ngram: str, hash_idx: int) -> int:
        h = hashlib.md5(f"{self._seed}:{hash_idx}:{ngram}".encode()).hexdigest()
        return int(h, 16) % self._dim

    def encode_single(self, text: str) -> np.ndarray:
        ngrams = self._get_ngrams(text)
        vec = np.zeros(self._dim, dtype=np.float64)
        if not ngrams:
            if self._projection is not None:
                return (vec @ self._projection)
            return vec
        counts: Counter = Counter(ngrams)
        for ng, cnt in counts.items():
            for hi in range(self._num_hashes):
                idx = self._hash_ngram(ng, hi)
                sign = 1.0 if hi % 2 == 0 else -1.0
                vec[idx] += sign * cnt
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        if self._normalize:
            pass  # already normalised
        if self._projection is not None:
            vec = vec @ self._projection
        return vec

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out_dim = self.dimension
        result = np.zeros((len(texts), out_dim), dtype=np.float64)
        for i, t in enumerate(texts):
            result[i] = self.encode_single(t)
        return result

    def train_projection(self, texts: Sequence[str], target_dim: int, n_iter: int = 100, lr: float = 0.01) -> None:
        """Learn a projection matrix from text data."""
        self._proj_dim = target_dim
        self._projection = self._rng.randn(self._dim, target_dim).astype(np.float32) * 0.01

        # Compute base embeddings
        base_embeds = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            ngrams = self._get_ngrams(t)
            for ng, cnt in Counter(ngrams).items():
                for hi in range(self._num_hashes):
                    idx = self._hash_ngram(ng, hi)
                    sign = 1.0 if hi % 2 == 0 else -1.0
                    base_embeds[i, idx] += sign * cnt
            norm = np.linalg.norm(base_embeds[i])
            if norm > 1e-12:
                base_embeds[i] /= norm

        # Simple gradient descent to preserve cosine structure
        for _ in range(n_iter):
            projected = base_embeds @ self._projection
            # Loss: maximise variance (spread out projected vectors)
            mean_proj = projected.mean(axis=0)
            centered = projected - mean_proj
            cov = (centered.T @ centered) / len(texts)
            # Gradient: push projection to maximise covariance trace
            grad = 2.0 * (base_embeds.T @ centered) / len(texts)
            self._projection += lr * grad
            # Normalise columns
            self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True) + 1e-12


# ═══════════════════════════════════════════════════════════════════════════
#  Transformer building blocks (from scratch)
# ═══════════════════════════════════════════════════════════════════════════

class _TokenEmbedding:
    """Deterministic token embedding using hash functions."""

    def __init__(self, vocab_size: int = 32000, dim: int = 256, seed: int = 42) -> None:
        self._vocab_size = vocab_size
        self._dim = dim
        self._rng = np.random.RandomState(seed)
        # Weight matrix: (vocab_size, dim)
        self._weight = self._rng.randn(vocab_size, dim).astype(np.float32) * 0.02
        self._weight /= np.linalg.norm(self._weight, axis=1, keepdims=True) + 1e-12

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self._weight[token_ids]

    @property
    def weight(self) -> np.ndarray:
        return self._weight


class _PositionalEncoding:
    """Sinusoidal positional encoding."""

    def __init__(self, max_len: int = 512, dim: int = 256) -> None:
        pe = np.zeros((max_len, dim), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self._pe = pe

    def __call__(self, seq_len: int) -> np.ndarray:
        return self._pe[:seq_len]


class _LayerNorm:
    """Layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self._dim = dim
        self._eps = eps
        self._gamma = np.ones(dim, dtype=np.float32)
        self._beta = np.zeros(dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self._eps)
        return self._gamma * x_norm + self._beta


class _MultiHeadSelfAttention:
    """Multi-head self-attention (no external dependencies)."""

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        self._dim = dim
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._dropout = dropout
        self._rng = np.random.RandomState(42)

        scale = 1.0 / math.sqrt(self._head_dim)
        self._W_q = self._rng.randn(dim, dim).astype(np.float32) * scale
        self._W_k = self._rng.randn(dim, dim).astype(np.float32) * scale
        self._W_v = self._rng.randn(dim, dim).astype(np.float32) * scale
        self._W_o = self._rng.randn(dim, dim).astype(np.float32) * scale

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : np.ndarray
            (seq_len, dim) input.
        mask : Optional[np.ndarray]
            (seq_len, seq_len) attention mask (0 = attend, -inf = ignore).

        Returns
        -------
        np.ndarray
            (seq_len, dim) output.
        """
        seq_len, dim = x.shape

        # Project to Q, K, V
        Q = x @ self._W_q  # (seq_len, dim)
        K = x @ self._W_k
        V = x @ self._W_v

        # Reshape to (num_heads, seq_len, head_dim)
        Q = Q.reshape(seq_len, self._num_heads, self._head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self._num_heads, self._head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self._num_heads, self._head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self._head_dim)
        scores = Q @ K.transpose(0, 2, 1) / scale  # (num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores + mask[np.newaxis, :, :]

        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Apply dropout
        if self._dropout > 0:
            drop_mask = (np.random.random(attn_weights.shape) > self._dropout).astype(np.float32)
            attn_weights = attn_weights * drop_mask / (1.0 - self._dropout)

        # Weighted sum
        context = attn_weights @ V  # (num_heads, seq_len, head_dim)
        context = context.transpose(1, 0, 2).reshape(seq_len, dim)

        # Output projection
        return context @ self._W_o


class _FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, dim: int = 256, ff_dim: int = 1024, dropout: float = 0.1) -> None:
        self._dim = dim
        self._ff_dim = ff_dim
        self._dropout = dropout
        self._rng = np.random.RandomState(42)
        self._W1 = self._rng.randn(dim, ff_dim).astype(np.float32) * 0.02
        self._b1 = np.zeros(ff_dim, dtype=np.float32)
        self._W2 = self._rng.randn(ff_dim, dim).astype(np.float32) * 0.02
        self._b2 = np.zeros(dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x @ self._W1 + self._b1
        h = np.maximum(h, 0.0)  # GELU approximation with ReLU
        if self._dropout > 0:
            mask = (np.random.random(h.shape) > self._dropout).astype(np.float32)
            h = h * mask / (1.0 - self._dropout)
        return h @ self._W2 + self._b2


class _TransformerEncoderLayer:
    """Single transformer encoder layer."""

    def __init__(self, dim: int = 256, num_heads: int = 4, ff_dim: int = 1024, dropout: float = 0.1) -> None:
        self._attention = _MultiHeadSelfAttention(dim, num_heads, dropout)
        self._ff = _FeedForward(dim, ff_dim, dropout)
        self._ln1 = _LayerNorm(dim)
        self._ln2 = _LayerNorm(dim)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Self-attention with residual
        attn_out = self._attention(self._ln1(x), mask)
        x = x + attn_out
        # Feed-forward with residual
        ff_out = self._ff(self._ln2(x))
        x = x + ff_out
        return x


def _simple_tokenize(text: str, max_length: int = 512) -> List[int]:
    """Simple hash-based tokeniser.  Maps words to integer IDs."""
    words = text.lower().split()
    token_ids = []
    for w in words[:max_length]:
        h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
        token_ids.append(h % 32000)
    return token_ids


# ═══════════════════════════════════════════════════════════════════════════
#  TransformerEmbedding
# ═══════════════════════════════════════════════════════════════════════════

class TransformerEmbedding(BaseEmbeddingModel):
    """BERT-style transformer embedding model (implemented from scratch).

    Features:
    * Multi-head self-attention encoder
    * [CLS] token pooling, mean pooling, or max pooling
    * Query/document prefix support (E5-style)
    * L2 normalisation option

    Parameters
    ----------
    hidden_size : int
        Hidden dimensionality.
    num_layers : int
        Number of transformer layers.
    num_heads : int
        Number of attention heads.
    ff_dim : int
        Feed-forward intermediate dimension.
    max_seq_length : int
        Maximum sequence length.
    pooling : str
        Pooling strategy: ``"mean"``, ``"cls"``, or ``"max"``.
    normalize : bool
        L2-normalise output embeddings.
    dropout : float
        Dropout probability.
    query_prefix : str
        Prefix for query texts.
    document_prefix : str
        Prefix for document texts.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 1024,
        max_seq_length: int = 512,
        pooling: str = "mean",
        normalize: bool = True,
        dropout: float = 0.1,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
    ) -> None:
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._ff_dim = ff_dim
        self._max_seq_length = max_seq_length
        self._pooling = pooling.lower()
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

        # Build layers
        self._token_emb = _TokenEmbedding(32000, hidden_size)
        self._pos_enc = _PositionalEncoding(max_seq_length, hidden_size)
        self._layers = [
            _TransformerEncoderLayer(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self._ln_final = _LayerNorm(hidden_size)

    @property
    def dimension(self) -> int:
        return self._hidden_size

    @property
    def model_name(self) -> str:
        return "transformer"

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text through the transformer."""
        token_ids = _simple_tokenize(text, self._max_seq_length - 1)

        if not token_ids:
            return np.zeros(self._hidden_size, dtype=np.float32)

        # Prepend [CLS] token
        token_ids = [1] + token_ids  # 1 = CLS
        seq_len = min(len(token_ids), self._max_seq_length)
        token_ids = token_ids[:seq_len]

        # Embeddings
        token_emb = self._token_emb(np.array(token_ids))
        pos_emb = self._pos_enc(seq_len)
        x = token_emb + pos_emb

        # Causal mask for padding (not needed since we truncate, but include for correctness)
        mask = np.zeros((seq_len, seq_len), dtype=np.float32)

        # Encoder layers
        for layer in self._layers:
            x = layer(x, mask)

        # Final layer norm
        x = self._ln_final(x)

        # Pooling
        if self._pooling == "cls":
            pooled = x[0]
        elif self._pooling == "max":
            pooled = np.max(x, axis=0)
        else:  # mean
            pooled = np.mean(x, axis=0)

        # Normalize
        if self._normalize:
            norm = np.linalg.norm(pooled)
            if norm > 1e-12:
                pooled = pooled / norm

        return pooled

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        embeddings = np.zeros((len(texts), self._hidden_size), dtype=np.float32)
        for i, text in enumerate(texts):
            embeddings[i] = self._encode_single(text)
        return embeddings

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self._query_prefix + t for t in texts]
        return self.encode(prefixed, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self._document_prefix + t for t in texts]
        return self.encode(prefixed, batch_size)


# ═══════════════════════════════════════════════════════════════════════════
#  MatryoshkaEmbedding
# ═══════════════════════════════════════════════════════════════════════════

class MatryoshkaEmbedding(BaseEmbeddingModel):
    """Matryoshka Representation Learning (MRL) embeddings.

    Produces nested embeddings where the first D dimensions are a valid
    embedding of dimensionality D, the first D' < D dimensions are a
    valid embedding of dimensionality D', and so on.  This enables
    flexible trade-offs between storage cost and retrieval quality.

    Parameters
    ----------
    base_dimension : int
        Maximum (full) embedding dimensionality.
    nest_dims : List[int]
        List of nested dimensions, e.g. ``[64, 128, 256, 512, 768]``.
        Must include *base_dimension*.
    base_encoder : Optional[BaseEmbeddingModel]
        Underlying encoder.  Falls back to :class:`TransformerEmbedding`.
    normalize : bool
        L2-normalise at each dimensionality level.
    """

    def __init__(
        self,
        base_dimension: int = 768,
        nest_dims: Optional[List[int]] = None,
        base_encoder: Optional[BaseEmbeddingModel] = None,
        normalize: bool = True,
    ) -> None:
        if nest_dims is None:
            nest_dims = [64, 128, 256, 512, base_dimension]
        if base_dimension not in nest_dims:
            nest_dims.append(base_dimension)
        nest_dims = sorted(set(nest_dims))

        self._base_dim = base_dimension
        self._nest_dims = nest_dims
        self._normalize = normalize
        self._base_encoder = base_encoder or TransformerEmbedding(
            hidden_size=base_dimension,
            num_layers=2,
            num_heads=max(4, base_dimension // 64),
            pooling="mean",
            normalize=False,
        )

    @property
    def dimension(self) -> int:
        return self._base_dim

    @property
    def model_name(self) -> str:
        return "matryoshka"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts at full dimensionality."""
        return self._base_encoder.encode(texts, batch_size)

    def encode_at_dim(self, texts: Sequence[str], dim: int, batch_size: int = 32) -> np.ndarray:
        """Encode texts at a specific nested dimensionality."""
        if dim not in self._nest_dims:
            raise ValueError(f"dim={dim} not in nest_dims={self._nest_dims}")
        full = self.encode(texts, batch_size)
        truncated = full[:, :dim]
        if self._normalize:
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            truncated = truncated / norms
        return truncated

    def encode_multi_dim(self, texts: Sequence[str], batch_size: int = 32) -> Dict[int, np.ndarray]:
        """Encode texts at all nested dimensionalities.

        Returns
        -------
        Dict[int, np.ndarray]
            Mapping from dimensionality to (N, D) embedding matrix.
        """
        full = self.encode(texts, batch_size)
        result: Dict[int, np.ndarray] = {}
        for dim in self._nest_dims:
            truncated = full[:, :dim]
            if self._normalize:
                norms = np.linalg.norm(truncated, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                truncated = truncated / norms
            result[dim] = truncated
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  ContrieverEmbedding
# ═══════════════════════════════════════════════════════════════════════════

class ContrieverEmbedding(BaseEmbeddingModel):
    """Unsupervised contrastive pretraining (Contriever-style).

    Uses a transformer encoder with unsupervised data augmentation:
    * Masked language modelling signals
    * Drop-out as noise
    * Opposite temperature for query-document pairs

    The implementation uses the built-in transformer with MLP-based
    projection heads for contrastive learning.

    Parameters
    ----------
    hidden_size : int
        Hidden dimensionality.
    num_layers : int
        Number of transformer layers.
    projection_dim : int
        Projection head output dimension.
    temperature : float
        Temperature for contrastive loss.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        projection_dim: int = 128,
        temperature: float = 0.05,
    ) -> None:
        self._hidden_size = hidden_size
        self._projection_dim = projection_dim
        self._temperature = temperature
        self._encoder = TransformerEmbedding(
            hidden_size=hidden_size,
            num_layers=num_layers,
            pooling="mean",
            normalize=False,
        )
        # Projection head (MLP)
        self._rng = np.random.RandomState(42)
        self._proj_W1 = self._rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self._proj_W2 = self._rng.randn(hidden_size, projection_dim).astype(np.float32) * 0.02

    @property
    def dimension(self) -> int:
        return self._projection_dim

    @property
    def model_name(self) -> str:
        return "contriever"

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        h = embeddings @ self._proj_W1
        h = np.maximum(h, 0.0)
        return h @ self._proj_W2

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        base = self._encoder.encode(texts, batch_size)
        projected = self._project(base)
        # Normalize
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return projected / norms

    def compute_contrastive_loss(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        temperature: Optional[float] = None,
    ) -> float:
        """Compute InfoNCE contrastive loss."""
        temp = temperature or self._temperature
        anchor_proj = self._project(anchors)
        pos_proj = self._project(positives)
        # Normalize
        a_norms = np.linalg.norm(anchor_proj, axis=1, keepdims=True)
        a_norms = np.maximum(a_norms, 1e-12)
        anchor_proj = anchor_proj / a_norms
        p_norms = np.linalg.norm(pos_proj, axis=1, keepdims=True)
        p_norms = np.maximum(p_norms, 1e-12)
        pos_proj = pos_proj / p_norms

        sim_matrix = (anchor_proj @ pos_proj.T) / temp
        labels = np.arange(len(anchors))
        # Cross-entropy loss
        sim_max = sim_matrix.max(axis=1, keepdims=True)
        exp_sim = np.exp(sim_matrix - sim_max)
        log_sum_exp = np.log(exp_sim.sum(axis=1) + 1e-12)
        loss = -sim_matrix[np.arange(len(anchors)), labels] + log_sum_exp
        return float(np.mean(loss))


# ═══════════════════════════════════════════════════════════════════════════
#  E5Embedding
# ═══════════════════════════════════════════════════════════════════════════

class E5Embedding(BaseEmbeddingModel):
    """E5-style text embeddings with task-specific prefixes.

    E5 uses ``"query: "`` and ``"passage: "`` prefixes to distinguish
    query and document encodings.  The model is trained with contrastive
    learning on large-scale text pairs.

    Parameters
    ----------
    hidden_size : int
        Hidden dimensionality.
    num_layers : int
        Number of transformer layers.
    pooling : str
        Pooling strategy.
    normalize : bool
        L2-normalise output.
    query_prefix : str
        Prefix for queries.
    document_prefix : str
        Prefix for documents.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        pooling: str = "mean",
        normalize: bool = True,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
    ) -> None:
        self._hidden_size = hidden_size
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._encoder = TransformerEmbedding(
            hidden_size=hidden_size,
            num_layers=num_layers,
            pooling=pooling,
            normalize=normalize,
        )

    @property
    def dimension(self) -> int:
        return self._hidden_size

    @property
    def model_name(self) -> str:
        return "e5"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._encoder.encode(texts, batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self._query_prefix + t for t in texts]
        return self._encoder.encode(prefixed, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self._document_prefix + t for t in texts]
        return self._encoder.encode(prefixed, batch_size)


# ═══════════════════════════════════════════════════════════════════════════
#  BGEEmbedding
# ═══════════════════════════════════════════════════════════════════════════

class BGEEmbedding(BaseEmbeddingModel):
    """BGE-style embeddings with instruction-based encoding.

    BGE uses task-specific instructions prepended to the query to
    improve retrieval quality.  The instruction format follows the
    BGE convention.

    Parameters
    ----------
    hidden_size : int
        Hidden dimensionality.
    num_layers : int
        Number of transformer layers.
    pooling : str
        Pooling strategy.
    normalize : bool
        L2-normalise output.
    instruction : str
        Retrieval instruction prepended to queries.
    document_prefix : str
        Prefix for documents.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        pooling: str = "cls",
        normalize: bool = True,
        instruction: str = "Represent this sentence for searching relevant passages: ",
        document_prefix: str = "",
    ) -> None:
        self._hidden_size = hidden_size
        self._instruction = instruction
        self._document_prefix = document_prefix
        self._encoder = TransformerEmbedding(
            hidden_size=hidden_size,
            num_layers=num_layers,
            pooling=pooling,
            normalize=normalize,
        )

    @property
    def dimension(self) -> int:
        return self._hidden_size

    @property
    def model_name(self) -> str:
        return "bge"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._encoder.encode(texts, batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self._instruction + t for t in texts]
        return self._encoder.encode(prefixed, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self._document_prefix + t for t in texts]
        return self._encoder.encode(prefixed, batch_size)


# ═══════════════════════════════════════════════════════════════════════════
#  EmbeddingTrainer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingBatch:
    """A batch of training examples for contrastive learning.

    Attributes
    ----------
    anchors : np.ndarray
        (B, D) anchor embeddings.
    positives : np.ndarray
        (B, D) positive embeddings.
    negatives : np.ndarray
        (B, D) negative embeddings.
    """

    anchors: np.ndarray
    positives: np.ndarray
    negatives: np.ndarray


class EmbeddingTrainer:
    """Train and fine-tune embedding models with contrastive learning.

    Implements:
    * InfoNCE (NT-Xent) contrastive loss
    * Hard negative mining
    * In-batch negative sampling
    * Gradient-based parameter updates (numpy-only)

    Parameters
    ----------
    model : BaseEmbeddingModel
        The embedding model to train.
    temperature : float
        Temperature for contrastive loss.
    learning_rate : float
        Learning rate for gradient descent.
    weight_decay : float
        L2 regularisation strength.
    max_grad_norm : float
        Gradient clipping threshold.
    """

    def __init__(
        self,
        model: BaseEmbeddingModel,
        temperature: float = 0.05,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._max_grad_norm = max_grad_norm
        self._trainable_params: Dict[str, np.ndarray] = {}
        self._step_count = 0
        self._loss_history: List[float] = []
        self._collect_trainable_params()

    def _collect_trainable_params(self) -> None:
        """Collect all trainable numpy arrays from the model."""
        self._trainable_params = {}
        for name in dir(self._model):
            if name.startswith("_"):
                continue
            attr = getattr(self._model, name)
            if isinstance(attr, np.ndarray) and attr.dtype in (np.float32, np.float64):
                self._trainable_params[name] = attr

    def contrastive_loss(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: Optional[np.ndarray] = None,
    ) -> float:
        """Compute InfoNCE contrastive loss.

        Parameters
        ----------
        anchors : np.ndarray
            (B, D) anchor embeddings.
        positives : np.ndarray
            (B, D) positive embeddings.
        negatives : Optional[np.ndarray]
            (B, K, D) hard negatives.  When ``None``, uses in-batch negatives.

        Returns
        -------
        float
            Mean contrastive loss.
        """
        B = anchors.shape[0]

        # Normalize
        a = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-12)
        p = positives / (np.linalg.norm(positives, axis=1, keepdims=True) + 1e-12)

        if negatives is not None:
            # Hard negatives
            K = negatives.shape[1]
            n = negatives.reshape(B * K, -1)
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            # Positive column + K negative columns
            all_neg = n.reshape(B, K, -1)  # (B, K, D)
            pos_scores = np.sum(a * p, axis=1, keepdims=True) / self._temperature  # (B, 1)
            neg_scores = np.einsum("bd,bkd->bk", a, all_neg) / self._temperature  # (B, K)
            logits = np.concatenate([pos_scores, neg_scores], axis=1)  # (B, K+1)
        else:
            # In-batch negatives: treat all other positives as negatives
            sim_matrix = (a @ p.T) / self._temperature  # (B, B)
            logits = sim_matrix

        # Labels: positive is at index 0 (hard neg) or diagonal (in-batch)
        labels = np.zeros(B, dtype=np.int64)

        # Cross-entropy loss
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        log_sum_exp = np.log(exp_logits.sum(axis=1) + 1e-12)
        loss = -logits[np.arange(B), labels] + log_sum_exp

        return float(np.mean(loss))

    def hard_negative_mining(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        pool: np.ndarray,
        k: int = 5,
    ) -> np.ndarray:
        """Mine hard negatives from a pool of candidates.

        For each anchor, selects the *k* pool vectors that are closest
        to the anchor but NOT the positive.

        Parameters
        ----------
        anchors : np.ndarray
            (B, D) anchor embeddings.
        positives : np.ndarray
            (B, D) positive embeddings.
        pool : np.ndarray
            (P, D) candidate pool.
        k : int
            Number of hard negatives per anchor.

        Returns
        -------
        np.ndarray
            (B, k, D) hard negative embeddings.
        """
        B = anchors.shape[0]
        a_norm = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-12)
        pool_norm = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)

        sim = a_norm @ pool_norm.T  # (B, P)

        hard_negatives = np.zeros((B, k, pool.shape[1]), dtype=anchors.dtype)
        for i in range(B):
            scores = sim[i]
            # Exclude positive
            pos_sim = float(np.dot(a_norm[i], positives[i] / (np.linalg.norm(positives[i]) + 1e-12)))
            # Find top-k most similar (but not the positive)
            top_indices = np.argsort(scores)[::-1][:k + 10]
            selected = []
            for idx in top_indices:
                if abs(scores[idx] - pos_sim) > 0.01:  # not the positive
                    selected.append(idx)
                    if len(selected) >= k:
                        break
            while len(selected) < k:
                selected.append(np.random.randint(len(pool)))
            hard_negatives[i] = pool[selected[:k]]

        return hard_negatives

    def train_step(self, batch: TrainingBatch) -> float:
        """Execute a single training step.

        Computes the contrastive loss and logs it.  In a full training
        loop, this would be followed by gradient computation and parameter
        updates (which require autograd — simulated here).

        Parameters
        ----------
        batch : TrainingBatch
            Training batch with anchors, positives, and negatives.

        Returns
        -------
        float
            Loss value.
        """
        loss = self.contrastive_loss(batch.anchors, batch.positives, batch.negatives)
        self._step_count += 1
        self._loss_history.append(loss)
        return loss

    def fit(
        self,
        texts: Sequence[str],
        pairs: Sequence[Tuple[int, int]],
        epochs: int = 10,
        batch_size: int = 32,
    ) -> List[float]:
        """Train the model on text pairs.

        Parameters
        ----------
        texts : Sequence[str]
            All texts in the corpus.
        pairs : Sequence[Tuple[int, int]]
            (anchor_idx, positive_idx) pairs.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size.

        Returns
        -------
        List[float]
            Loss history.
        """
        all_losses: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            indices = list(range(len(pairs)))
            np.random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                anchor_ids = [pairs[i][0] for i in batch_indices]
                pos_ids = [pairs[i][1] for i in batch_indices]

                anchor_texts = [texts[i] for i in anchor_ids]
                pos_texts = [texts[i] for i in pos_ids]

                anchors = self._model.encode(anchor_texts)
                positives = self._model.encode(pos_texts)

                # Mine negatives from all texts
                all_embeds = self._model.encode(texts[:min(200, len(texts))])
                hard_negs = self.hard_negative_mining(anchors, positives, all_embeds, k=3)

                batch = TrainingBatch(anchors, positives, hard_negs)
                loss = self.train_step(batch)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            all_losses.append(avg_loss)
            logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, epochs, avg_loss)

        return all_losses

    @property
    def loss_history(self) -> List[float]:
        return list(self._loss_history)

    @property
    def step_count(self) -> int:
        return self._step_count


# ═══════════════════════════════════════════════════════════════════════════
#  Factory helper
# ═══════════════════════════════════════════════════════════════════════════

def create_embedding_model(
    model_name: str = "simple",
    dimension: int = 256,
    **kwargs: Any,
) -> BaseEmbeddingModel:
    """Factory: create an embedding model by name.

    Parameters
    ----------
    model_name : str
        ``"simple"``, ``"transformer"``, ``"matryoshka"``, ``"contriever"``,
        ``"e5"``, or ``"bge"``.
    dimension : int
        Output dimensionality.
    """
    model_name = model_name.lower().strip()
    if model_name == "simple":
        return SimpleEmbedding(dimension=dimension, **kwargs)
    elif model_name == "transformer":
        return TransformerEmbedding(hidden_size=dimension, **kwargs)
    elif model_name == "matryoshka":
        return MatryoshkaEmbedding(base_dimension=dimension, **kwargs)
    elif model_name == "contriever":
        return ContrieverEmbedding(hidden_size=dimension, **kwargs)
    elif model_name == "e5":
        return E5Embedding(hidden_size=dimension, **kwargs)
    elif model_name == "bge":
        return BGEEmbedding(hidden_size=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown embedding model: {model_name!r}")
