"""
Nexus Long-Term Memory
======================
Persistent vector-indexed long-term memory system with cosine similarity retrieval,
time-based decay, memory consolidation, and disk persistence.

This module implements the core long-term memory components:

- **MemoryConfig**: Configuration dataclass for all memory parameters.
- **MemoryEntry**: Individual memory record with embedding, metadata, and access tracking.
- **MemoryEncoder**: Neural text encoder that maps text to embedding vectors via
  a simple token-embedding + mean-pooling + linear-projection pipeline.
- **LongTermMemoryStore**: The main memory store supporting store, retrieve, forget,
  update, delete, search, import/export, and statistics.
- **MemoryConsolidator**: Moves important short-term memories to long-term storage,
  clusters related memories, and performs extractive summarization.
- **MemoryDecay**: Applies exponential time-based decay to memory importance scores,
  with optional pruning of weak memories.

Design Principles
-----------------
1. **Semantic retrieval**: All memories are stored with vector embeddings enabling
   cosine similarity search rather than keyword matching.
2. **Decay & forgetting**: Memories naturally decay over time unless reinforced
   by access, mimicking human forgetting curves.
3. **Persistence**: Memories can be saved to and loaded from disk as JSON files.
4. **Capacity management**: When capacity is reached, the least important memories
   are automatically pruned.
5. **Consolidation**: Short-term working memories can be consolidated into long-term
   storage through clustering and summarization.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json
import hashlib
import time
import os
import math
import collections
import copy
import random


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryConfig:
    """Configuration for the long-term memory system.

    All parameters control the behavior of LongTermMemoryStore, MemoryConsolidator,
    and MemoryDecay. Sensible defaults are provided for each parameter.

    Attributes:
        capacity: Maximum number of memories the store can hold. When exceeded,
            the least important memories are automatically pruned.
        embedding_dim: Dimensionality of embedding vectors produced by MemoryEncoder.
            Higher dimensions capture more semantic detail but use more memory.
        persistence_path: Filesystem directory path for saving/loading memory state.
            If None, memory exists only in RAM and is lost on process exit.
        decay_rate: Controls the speed of exponential importance decay.
            Formula: factor = exp(-decay_rate * hours_since_access).
            Higher values cause faster forgetting. Use 0.0 to disable decay.
        similarity_threshold: Minimum cosine similarity score for retrieval results.
            Results below this threshold are filtered out. Range: [0.0, 1.0].
        index_type: Type of vector index to use. Options: 'flat' (brute-force
            cosine similarity), 'hnsw' (hierarchical navigable small world,
            faster for large datasets), 'ivf' (inverted file index).
        auto_save: Whether to automatically persist to disk after modifications.
        auto_save_interval: Minimum seconds between automatic saves.
        encoder_vocab_size: Vocabulary size for the built-in MemoryEncoder.
        encoder_hidden_dim: Hidden dimension of the MemoryEncoder projection layer.
        max_retrieve_k: Maximum value allowed for top_k in retrieval operations.
        enable_access_tracking: Whether to track access count and timestamps.
    """
    capacity: int = 10000
    embedding_dim: int = 256
    persistence_path: Optional[str] = None
    decay_rate: float = 0.01
    similarity_threshold: float = 0.7
    index_type: str = "flat"
    auto_save: bool = False
    auto_save_interval: float = 60.0
    encoder_vocab_size: int = 32000
    encoder_hidden_dim: int = 512
    max_retrieve_k: int = 100
    enable_access_tracking: bool = True

    def validate(self) -> List[str]:
        """Validate configuration values and return list of warnings/errors.

        Returns:
            List of validation messages. Empty list means all values are valid.
        """
        issues = []
        if self.capacity <= 0:
            issues.append("capacity must be positive")
        if self.embedding_dim <= 0:
            issues.append("embedding_dim must be positive")
        if self.decay_rate < 0:
            issues.append("decay_rate must be non-negative")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            issues.append("similarity_threshold must be in [0.0, 1.0]")
        if self.index_type not in ("flat", "hnsw", "ivf"):
            issues.append(f"index_type must be 'flat', 'hnsw', or 'ivf', got '{self.index_type}'")
        if self.encoder_vocab_size <= 0:
            issues.append("encoder_vocab_size must be positive")
        if self.encoder_hidden_dim <= 0:
            issues.append("encoder_hidden_dim must be positive")
        return issues

    def to_dict(self) -> dict:
        """Serialize configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryConfig":
        """Deserialize configuration from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Entry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """A single memory record stored in the long-term memory system.

    Each memory entry contains the original content, its vector embedding
    for similarity search, optional metadata, and access statistics that
    inform decay and importance calculations.

    Attributes:
        id: Unique identifier for this memory entry. Generated as SHA-256
            hash of content + creation timestamp by default.
        content: The text content of this memory.
        embedding: PyTorch tensor of shape (embedding_dim,) containing the
            normalized embedding vector for this memory's content.
        metadata: Arbitrary key-value metadata attached to this memory.
            Useful for filtering, categorization, and provenance tracking.
        importance: Float in [0.0, 1.0] representing the importance or
            priority of this memory. Higher values resist decay and pruning.
        created_at: Unix timestamp (seconds since epoch) when this memory
            was first created.
        accessed_at: Unix timestamp of the most recent access to this memory.
        access_count: Number of times this memory has been retrieved or
            accessed. Higher counts resist decay.
    """
    id: str = ""
    content: str = ""
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0

    @staticmethod
    def generate_id(content: str, timestamp: Optional[float] = None) -> str:
        """Generate a deterministic unique ID from content and timestamp.

        Args:
            content: The text content of the memory.
            timestamp: Unix timestamp. If None, uses current time.

        Returns:
            SHA-256 hash string (first 16 hex characters).
        """
        if timestamp is None:
            timestamp = time.time()
        raw = f"{content}:{timestamp}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize memory entry to a JSON-compatible dictionary.

        The embedding tensor is converted to a list of floats.

        Returns:
            Dictionary representation of this memory entry.
        """
        embedding_list = None
        if self.embedding is not None:
            if isinstance(self.embedding, torch.Tensor):
                embedding_list = self.embedding.detach().cpu().tolist()
            else:
                embedding_list = list(self.embedding)
        return {
            "id": self.id,
            "content": self.content,
            "embedding": embedding_list,
            "metadata": dict(self.metadata),
            "importance": self.importance,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Deserialize a memory entry from a dictionary.

        The embedding list is converted back to a PyTorch tensor.

        Args:
            data: Dictionary with memory entry fields.

        Returns:
            Reconstructed MemoryEntry instance.
        """
        embedding = None
        if data.get("embedding") is not None:
            embedding_list = data["embedding"]
            if isinstance(embedding_list, list):
                embedding = torch.tensor(embedding_list, dtype=torch.float32)
            else:
                embedding = torch.tensor([embedding_list], dtype=torch.float32).squeeze()
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            embedding=embedding,
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", 0.0),
            accessed_at=data.get("accessed_at", 0.0),
            access_count=data.get("access_count", 0),
        )

    def touch(self) -> None:
        """Update access timestamp and increment access count.

        Called whenever this memory is retrieved or accessed.
        """
        self.accessed_at = time.time()
        self.access_count += 1

    def effective_importance(self, decay_rate: float = 0.0) -> float:
        """Calculate the effective importance after applying time decay.

        This combines the base importance with access frequency and
        recency to produce a single score used for ranking and pruning.

        Args:
            decay_rate: Decay rate to apply. If 0.0, returns base importance.

        Returns:
            Effective importance score, clamped to [0.0, 1.0].
        """
        if decay_rate <= 0.0:
            return self.importance

        now = time.time()
        hours_since_access = (now - self.accessed_at) / 3600.0

        # Exponential time decay
        time_factor = math.exp(-decay_rate * hours_since_access)

        # Access frequency bonus (logarithmic to prevent explosion)
        access_bonus = min(math.log1p(self.access_count) / 5.0, 0.3)

        effective = (self.importance * time_factor) + access_bonus
        return max(0.0, min(1.0, effective))

    def age_hours(self) -> float:
        """Get the age of this memory in hours since creation.

        Returns:
            Age in hours as a float.
        """
        return (time.time() - self.created_at) / 3600.0

    def __repr__(self) -> str:
        return (
            f"MemoryEntry(id={self.id!r}, content={self.content[:50]!r}..., "
            f"importance={self.importance:.3f}, access_count={self.access_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTokenizer:
    """A simple character-level and word-level hybrid tokenizer for the MemoryEncoder.

    This tokenizer provides basic tokenization without requiring external tokenizer
    libraries. It splits text into words and characters, maps them to integer IDs,
    and supports special tokens.

    The tokenizer is deterministic and reproducible — the same text always produces
    the same token sequence.

    Attributes:
        vocab_size: Maximum size of the vocabulary.
        special_tokens: Dictionary mapping special token names to IDs.
        word_to_id: Mapping from words to token IDs.
        id_to_word: Reverse mapping from IDs to words.
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    CLS_TOKEN = "[CLS]"

    def __init__(self, vocab_size: int = 32000):
        """Initialize the tokenizer.

        Args:
            vocab_size: Maximum vocabulary size. Words beyond this capacity
                will be mapped to UNK_TOKEN.
        """
        self.vocab_size = vocab_size

        # Special tokens occupy the first 5 IDs
        self.special_tokens = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
            self.CLS_TOKEN: 4,
        }

        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self._next_id = len(self.special_tokens)

        # Initialize with common English words and subwords for better coverage
        self._initialize_base_vocab()

    def _initialize_base_vocab(self) -> None:
        """Seed the vocabulary with common tokens."""
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "must", "can", "could", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "because", "but",
            "and", "or", "if", "while", "although", "though", "since", "until",
            "unless", "that", "which", "who", "whom", "what", "this", "that",
            "these", "those", "i", "me", "my", "myself", "we", "our", "ours",
            "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
            "it", "its", "they", "them", "their", "theirs", "what", "which",
            "who", "whom", "whose", "where", "when", "why", "how", "not", "no",
            "yes", "up", "down", "out", "off", "over", "under", "again", "now",
            "then", "here", "there", "always", "never", "often", "sometimes",
            "usually", "already", "still", "yet", "about", "also", "only", "even",
            "very", "quite", "rather", "really", "much", "many", "well", "too",
            "good", "bad", "new", "old", "big", "small", "high", "low", "long",
            "short", "great", "first", "last", "next", "best", "important",
            "different", "large", "right", "early", "young", "possible", "major",
            "local", "sure", "real", "whole", "special", "current", "hard",
            "human", "social", "political", "public", "economic", "national",
            "international", "military", "financial", "environmental", "medical",
            "legal", "technical", "scientific", "cultural", "historical",
        ]
        for word in common_words:
            self._add_word(word.lower())

        # Add common punctuation and symbols as separate tokens
        punctuation = [
            ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{",
            "}", "-", "/", "\\", "+", "=", "*", "%", "$", "#", "@", "&", "|",
            "^", "~", "<", ">", "_",
        ]
        for punct in punctuation:
            self._add_word(punct)

        # Add digits as tokens
        for digit in range(10):
            self._add_word(str(digit))

        # Add common prefixes and suffixes
        affixes = [
            "un", "re", "pre", "dis", "over", "mis", "sub", "inter", "non",
            "ing", "tion", "ment", "ness", "ous", "ful", "less", "ive", "able",
            "ible", "al", "ial", "ly", "ed", "er", "est", "ize", "ise", "ify",
            "ence", "ance", "ity", "dom", "ship", "hood", "ward", "wise",
        ]
        for affix in affixes:
            self._add_word(affix)

    def _add_word(self, word: str) -> None:
        """Add a word to the vocabulary if there is room.

        Args:
            word: The word to add.
        """
        if word not in self.word_to_id and self._next_id < self.vocab_size:
            self.word_to_id[word] = self._next_id
            self.id_to_word[self._next_id] = word
            self._next_id += 1

    def tokenize(self, text: str) -> List[int]:
        """Convert text to a list of token IDs.

        The tokenization strategy is:
        1. Add BOS and CLS tokens at the start.
        2. Split text into lowercase words (alphanumeric sequences).
        3. Map each word to its ID (UNK if not in vocab).
        4. Truncate to a reasonable max length (512 tokens).
        5. Add EOS token at the end.

        Args:
            text: Input text string to tokenize.

        Returns:
            List of integer token IDs.
        """
        if not text or not text.strip():
            return [self.special_tokens[self.BOS_TOKEN],
                    self.special_tokens[self.CLS_TOKEN],
                    self.special_tokens[self.EOS_TOKEN]]

        tokens = [self.special_tokens[self.BOS_TOKEN], self.special_tokens[self.CLS_TOKEN]]

        # Split into words: keep alphanumeric sequences, punctuation as separate tokens
        words = []
        current_word = []
        for char in text.lower():
            if char.isalnum() or char == "'":
                current_word.append(char)
            else:
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
                if char.strip():
                    words.append(char)
        if current_word:
            words.append("".join(current_word))

        # Also try splitting long words into subwords
        for word in words:
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                # Try splitting into subwords (greedy longest match)
                remaining = word
                while remaining:
                    matched = False
                    for length in range(min(len(remaining), 10), 1, -1):
                        subword = remaining[:length]
                        if subword in self.word_to_id:
                            tokens.append(self.word_to_id[subword])
                            remaining = remaining[length:]
                            matched = True
                            break
                    if not matched:
                        # Map individual characters
                        for char in remaining[:3]:  # limit character tokens
                            if char in self.word_to_id:
                                tokens.append(self.word_to_id[char])
                            else:
                                tokens.append(self.special_tokens[self.UNK_TOKEN])
                        remaining = remaining[3:] if len(remaining) > 3 else ""

        tokens.append(self.special_tokens[self.EOS_TOKEN])

        # Truncate to max length
        max_len = 512
        if len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + [self.special_tokens[self.EOS_TOKEN]]

        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text.

        Special tokens are skipped. Unknown tokens are replaced with spaces.

        Args:
            token_ids: List of token IDs to convert.

        Returns:
            Reconstructed text string.
        """
        words = []
        for tid in token_ids:
            if tid in self.id_to_word:
                token = self.id_to_word[tid]
                if token not in self.special_tokens:
                    # Add space before alphabetic tokens (after alphabetic tokens)
                    if words and words[-1][-1:].isalpha() and token[0:1].isalpha():
                        words.append(" ")
                    words.append(token)
        return "".join(words)

    @property
    def actual_vocab_size(self) -> int:
        """Return the current number of tokens in the vocabulary."""
        return self._next_id

    def vocab(self) -> Dict[str, int]:
        """Return a copy of the word-to-id mapping."""
        return dict(self.word_to_id)


class MemoryEncoder(nn.Module):
    """Neural text encoder that maps text to normalized embedding vectors.

    This is a lightweight encoder designed for the memory system. It uses a
    simple architecture:
    1. Tokenize text into integer IDs.
    2. Look up token embeddings (nn.Embedding).
    3. Apply mean pooling over all token positions.
    4. Project through a linear layer with ReLU activation.
    5. L2-normalize the output vector.

    The encoder is trained end-to-end and produces embeddings suitable for
    cosine similarity search.

    Architecture:
        Input (text) → Tokenizer → Embedding(V, D) → MeanPool → Linear(D, H)
        → ReLU → Linear(H, E) → L2Norm → Output (E-dim vector)

    where V = vocab_size, D = embed_dim, H = hidden_dim, E = embedding_dim.

    Args:
        vocab_size: Size of the embedding vocabulary.
        embed_dim: Dimension of the input embedding lookup table.
        hidden_dim: Dimension of the hidden projection layer.
        output_dim: Dimension of the output embedding vector.
        dropout: Dropout probability applied after pooling.
        max_seq_len: Maximum sequence length (tokens are truncated to this).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)

        # Embedding layer: maps token IDs to dense vectors
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # Projection layers with non-linearity
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)

    def forward(
        self,
        text: Union[str, List[str], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text into normalized embedding vectors.

        Args:
            text: Input text. Can be:
                - A single string → returns tensor of shape (output_dim,)
                - A list of strings → returns tensor of shape (batch_size, output_dim)
                - A tensor of token IDs → used directly
            attention_mask: Optional mask tensor of shape (batch_size, seq_len)
                where 1 indicates valid tokens and 0 indicates padding.
                If None, all tokens are considered valid.

        Returns:
            L2-normalized embedding tensor.
        """
        # Handle different input types
        if isinstance(text, str):
            text = [text]

        if isinstance(text, list):
            # Tokenize all texts
            all_token_ids = []
            for t in text:
                token_ids = self.tokenizer.tokenize(t)
                all_token_ids.append(token_ids)

            # Pad to same length within batch
            max_len = min(max(len(ids) for ids in all_token_ids), self.max_seq_len)
            padded = []
            masks = []
            for ids in all_token_ids:
                truncated = ids[:max_len]
                pad_len = max_len - len(truncated)
                padded.append(truncated + [0] * pad_len)
                masks.append([1] * len(truncated) + [0] * pad_len)

            token_tensor = torch.tensor(padded, dtype=torch.long, device=self._get_device())
            attention_mask = torch.tensor(masks, dtype=torch.float, device=self._get_device())

        elif isinstance(text, torch.Tensor):
            token_tensor = text.to(self._get_device())
        else:
            raise TypeError(f"Unsupported input type: {type(text)}")

        # Embedding lookup: (batch_size, seq_len) → (batch_size, seq_len, embed_dim)
        embedded = self.token_embedding(token_tensor)

        # Apply attention mask for mean pooling
        if attention_mask is not None:
            # Expand mask to match embedding dimension
            mask_expanded = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
            embedded = embedded * mask_expanded

            # Sum and divide by actual token counts
            sum_embeddings = embedded.sum(dim=1)  # (batch, embed_dim)
            token_counts = attention_mask.sum(dim=1, keepdim=True)
            token_counts = token_counts.clamp(min=1)  # Avoid division by zero
            pooled = sum_embeddings / token_counts  # (batch, embed_dim)
        else:
            # Simple mean pooling
            pooled = embedded.mean(dim=1)  # (batch, embed_dim)

        # Project through hidden layers
        projected = self.projection(pooled)  # (batch, output_dim)

        # Layer normalization
        normalized = self.layer_norm(projected)

        # L2 normalize for cosine similarity
        norms = normalized.norm(p=2, dim=-1, keepdim=True)
        norms = norms.clamp(min=1e-8)  # Avoid division by zero
        output = normalized / norms  # (batch, output_dim)

        # Squeeze if single input
        if output.shape[0] == 1 and isinstance(text, list) and len(text) == 1:
            return output.squeeze(0)

        return output

    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Convenience method to encode text without requiring gradients.

        Equivalent to calling forward() with torch.no_grad().

        Args:
            text: Input text string or list of strings.

        Returns:
            Normalized embedding tensor.
        """
        with torch.no_grad():
            return self.forward(text)

    def _get_device(self) -> torch.device:
        """Get the device of the model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_config(self) -> dict:
        """Return encoder configuration as a dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "max_seq_len": self.max_seq_len,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Cosine Similarity Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two tensors.

    Supports:
    - (dim,) × (dim,) → scalar
    - (dim,) × (N, dim) → (N,)
    - (M, dim) × (dim,) → (M,)
    - (M, dim) × (N, dim) → (M, N)

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Cosine similarity score(s) as a tensor.
    """
    if a.dim() == 1 and b.dim() == 1:
        return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)
    elif a.dim() == 1 and b.dim() == 2:
        return torch.matmul(b, a) / (b.norm(dim=1) * a.norm() + 1e-8)
    elif a.dim() == 2 and b.dim() == 1:
        return torch.matmul(a, b) / (a.norm(dim=1) * b.norm() + 1e-8)
    elif a.dim() == 2 and b.dim() == 2:
        a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
        return torch.matmul(a_norm, b_norm.t())
    else:
        raise ValueError(f"Unsupported tensor shapes: {a.shape}, {b.shape}")


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two numpy arrays.

    Args:
        a: First vector as numpy array.
        b: Second vector as numpy array.

    Returns:
        Cosine similarity score as a float.
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════════════════
# Long-Term Memory Store
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemoryStore:
    """Persistent vector-indexed long-term memory store.

    This is the core storage component of the long-term memory system. It manages
    a collection of MemoryEntry objects, supports efficient retrieval via cosine
    similarity search, and provides persistence to disk.

    Key Features:
    - Vector similarity search with configurable threshold
    - Automatic importance decay over time
    - Capacity management with automatic pruning
    - JSON-based import/export
    - Comprehensive statistics

    The store uses a flat index by default (brute-force cosine similarity over all
    entries). For very large collections (>100K entries), consider using an external
    vector database.

    Args:
        config: MemoryConfig instance controlling behavior.
        encoder: Optional pre-initialized MemoryEncoder. If None, a default
            encoder is created based on config parameters.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        encoder: Optional[MemoryEncoder] = None,
    ):
        """Initialize the long-term memory store.

        Args:
            config: Memory configuration. Uses defaults if None.
            encoder: Optional pre-initialized encoder. Created from config if None.
        """
        self.config = config or MemoryConfig()

        # Validate configuration
        issues = self.config.validate()
        if issues:
            raise ValueError(f"Invalid MemoryConfig: {'; '.join(issues)}")

        # Initialize or use provided encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = MemoryEncoder(
                vocab_size=self.config.encoder_vocab_size,
                embed_dim=128,
                hidden_dim=self.config.encoder_hidden_dim,
                output_dim=self.config.embedding_dim,
            )
            self.encoder.eval()

        # Core storage: ordered dictionary mapping memory_id → MemoryEntry
        self._memories: Dict[str, MemoryEntry] = collections.OrderedDict()

        # Counter for generating unique IDs
        self._store_count: int = 0

        # Timestamp of last modification
        self._last_modified: float = time.time()

        # Timestamp of last auto-save
        self._last_save_time: float = 0.0

        # Tracking statistics
        self._total_stored: int = 0
        self._total_retrieved: int = 0
        self._total_deleted: int = 0
        self._total_updated: int = 0

        # Load from disk if persistence is configured
        if self.config.persistence_path:
            os.makedirs(self.config.persistence_path, exist_ok=True)
            self._load_if_exists()

    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        memory_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store a new memory entry.

        Creates an embedding for the content, constructs a MemoryEntry, and adds
        it to the store. If the store is at capacity, the least important memory
        is automatically pruned.

        Args:
            content: The text content to remember.
            metadata: Optional metadata dictionary for categorization and filtering.
            importance: Importance score in [0.0, 1.0]. Higher values resist decay.
            memory_id: Optional explicit ID. If None, one is auto-generated.

        Returns:
            The newly created MemoryEntry.

        Raises:
            ValueError: If content is empty.
        """
        if not content or not content.strip():
            raise ValueError("Cannot store empty content")

        content = content.strip()
        importance = max(0.0, min(1.0, importance))
        metadata = metadata or {}

        # Check capacity and prune if needed
        if len(self._memories) >= self.config.capacity:
            self._prune_to_capacity(self.config.capacity - 1)

        # Generate embedding
        embedding = self.encoder.encode(content)

        # Generate unique ID
        now = time.time()
        if memory_id is None:
            memory_id = MemoryEntry.generate_id(content, now)

        # Create entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=dict(metadata),
            importance=importance,
            created_at=now,
            accessed_at=now,
            access_count=0,
        )

        self._memories[memory_id] = entry
        self._store_count += 1
        self._total_stored += 1
        self._last_modified = now

        # Auto-save if configured
        self._maybe_auto_save()

        return entry

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[Callable[[MemoryEntry], bool]] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve memories by semantic similarity to a query.

        Encodes the query into an embedding vector and computes cosine similarity
        against all stored memories. Results are returned sorted by descending
        similarity score.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return. Clamped to config.max_retrieve_k.
            filter_fn: Optional function that takes a MemoryEntry and returns True
                if the entry should be included in results.

        Returns:
            List of (MemoryEntry, similarity_score) tuples sorted by descending
            similarity. The similarity score is in [0.0, 1.0].
        """
        if not query or not query.strip():
            return []

        if not self._memories:
            return []

        top_k = min(top_k, self.config.max_retrieve_k, len(self._memories))

        # Encode query
        query_embedding = self.encoder.encode(query.strip())

        # Compute similarities
        results: List[Tuple[MemoryEntry, float]] = []
        for entry in self._memories.values():
            # Apply filter if provided
            if filter_fn is not None and not filter_fn(entry):
                continue

            if entry.embedding is None:
                continue

            sim = cosine_similarity(query_embedding, entry.embedding)
            score = float(sim.item()) if sim.dim() > 0 else float(sim)
            results.append((entry, score))

        # Sort by descending similarity
        results.sort(key=lambda x: x[1], reverse=True)

        # Take top_k
        results = results[:top_k]

        # Touch accessed entries to update their access stats
        if self.config.enable_access_tracking:
            for entry, score in results:
                entry.touch()

        self._total_retrieved += len(results)
        self._last_modified = time.time()

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search memories with similarity threshold filtering.

        This is similar to retrieve() but additionally filters out results
        whose similarity score falls below the threshold.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score. If None, uses config.similarity_threshold.

        Returns:
            List of (MemoryEntry, similarity_score) tuples sorted by descending
            similarity, filtered by the threshold.
        """
        if threshold is None:
            threshold = self.config.similarity_threshold

        results = self.retrieve(query, top_k=self.config.max_retrieve_k)

        # Filter by threshold
        filtered = [(entry, score) for entry, score in results if score >= threshold]

        return filtered[:top_k]

    def forget(
        self,
        criteria: str = "least_important",
        count: int = 1,
        min_age_hours: float = 0.0,
        max_importance: float = 1.0,
    ) -> List[str]:
        """Remove memories based on specified criteria.

        This is the intentional forgetting mechanism. Memories can be removed
        based on different strategies to manage capacity and relevance.

        Args:
            criteria: Strategy for selecting memories to forget. Options:
                - 'least_important': Remove memories with lowest effective importance.
                - 'oldest': Remove the oldest memories first.
                - 'least_accessed': Remove memories with fewest accesses.
                - 'by_age': Remove memories older than min_age_hours.
                - 'by_importance': Remove memories below max_importance.
                - 'random': Remove randomly selected memories.
            count: Maximum number of memories to remove.
            min_age_hours: For 'by_age' criteria, minimum age in hours.
            max_importance: For 'by_importance' criteria, maximum importance threshold.

        Returns:
            List of IDs of removed memories.
        """
        if not self._memories or count <= 0:
            return []

        count = min(count, len(self._memories))
        removed_ids: List[str] = []
        now = time.time()

        if criteria == "least_important":
            # Sort by effective importance (ascending)
            entries = list(self._memories.values())
            entries.sort(key=lambda e: e.effective_importance(self.config.decay_rate))
            for entry in entries[:count]:
                del self._memories[entry.id]
                removed_ids.append(entry.id)

        elif criteria == "oldest":
            # Sort by creation time (ascending)
            entries = list(self._memories.values())
            entries.sort(key=lambda e: e.created_at)
            for entry in entries[:count]:
                del self._memories[entry.id]
                removed_ids.append(entry.id)

        elif criteria == "least_accessed":
            # Sort by access count (ascending)
            entries = list(self._memories.values())
            entries.sort(key=lambda e: e.access_count)
            for entry in entries[:count]:
                del self._memories[entry.id]
                removed_ids.append(entry.id)

        elif criteria == "by_age":
            # Remove memories older than min_age_hours
            threshold_time = now - (min_age_hours * 3600)
            to_remove = [
                entry for entry in self._memories.values()
                if entry.created_at < threshold_time
            ]
            to_remove.sort(key=lambda e: e.effective_importance(self.config.decay_rate))
            for entry in to_remove[:count]:
                del self._memories[entry.id]
                removed_ids.append(entry.id)

        elif criteria == "by_importance":
            # Remove memories below importance threshold
            to_remove = [
                entry for entry in self._memories.values()
                if entry.importance < max_importance
            ]
            to_remove.sort(key=lambda e: e.effective_importance(self.config.decay_rate))
            for entry in to_remove[:count]:
                del self._memories[entry.id]
                removed_ids.append(entry.id)

        elif criteria == "random":
            entries = list(self._memories.values())
            random.shuffle(entries)
            for entry in entries[:count]:
                del self._memories[entry.id]
                removed_ids.append(entry.id)

        else:
            raise ValueError(f"Unknown forget criteria: {criteria}")

        self._total_deleted += len(removed_ids)
        self._last_modified = time.time()
        self._maybe_auto_save()

        return removed_ids

    def update(self, memory_id: str, content: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None,
               importance: Optional[float] = None) -> Optional[MemoryEntry]:
        """Update an existing memory entry.

        If content is changed, the embedding is recomputed. Other fields are
        updated in-place.

        Args:
            memory_id: ID of the memory to update.
            content: New content. If None, content is not changed.
            metadata: New metadata. If None, metadata is not changed.
                If provided, it is merged with existing metadata.
            importance: New importance. If None, importance is not changed.

        Returns:
            Updated MemoryEntry, or None if memory_id not found.
        """
        if memory_id not in self._memories:
            return None

        entry = self._memories[memory_id]

        if content is not None and content.strip():
            entry.content = content.strip()
            # Recompute embedding since content changed
            entry.embedding = self.encoder.encode(entry.content)

        if metadata is not None:
            entry.metadata.update(metadata)

        if importance is not None:
            entry.importance = max(0.0, min(1.0, importance))

        entry.accessed_at = time.time()
        self._total_updated += 1
        self._last_modified = time.time()
        self._maybe_auto_save()

        return entry

    def delete(self, memory_id: str) -> bool:
        """Delete a single memory entry by ID.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if the memory was found and deleted, False otherwise.
        """
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._total_deleted += 1
            self._last_modified = time.time()
            self._maybe_auto_save()
            return True
        return False

    def count(self) -> int:
        """Return the number of stored memories.

        Returns:
            Integer count of memories in the store.
        """
        return len(self._memories)

    def get_all(self) -> List[MemoryEntry]:
        """Return a list of all stored memory entries.

        Returns:
            List of all MemoryEntry objects in the store.
        """
        return list(self._memories.values())

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory entry by ID.

        Args:
            memory_id: ID of the memory to retrieve.

        Returns:
            MemoryEntry if found, None otherwise.
        """
        entry = self._memories.get(memory_id)
        if entry is not None and self.config.enable_access_tracking:
            entry.touch()
        return entry

    def has(self, memory_id: str) -> bool:
        """Check if a memory with the given ID exists.

        Args:
            memory_id: ID to check.

        Returns:
            True if a memory with this ID exists.
        """
        return memory_id in self._memories

    def export_json(self, path: str) -> None:
        """Export all memories to a JSON file.

        Each memory entry is serialized to a JSON-compatible dictionary.
        Embeddings are stored as lists of floats.

        Args:
            path: Filesystem path for the output JSON file.

        Raises:
            IOError: If the file cannot be written.
        """
        data = {
            "config": self.config.to_dict(),
            "memories": [entry.to_dict() for entry in self._memories.values()],
            "statistics": {
                "total_stored": self._total_stored,
                "total_retrieved": self._total_retrieved,
                "total_deleted": self._total_deleted,
                "total_updated": self._total_updated,
                "exported_at": time.time(),
            },
        }

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def import_json(self, path: str, merge: bool = True) -> int:
        """Import memories from a JSON file.

        Args:
            path: Filesystem path to the JSON file to import.
            merge: If True, imported memories are merged with existing ones
                (existing memories with the same ID are overwritten).
                If False, existing memories are cleared first.

        Returns:
            Number of memories imported.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not merge:
            self._memories.clear()

        imported_count = 0
        memories_data = data.get("memories", [])
        for entry_data in memories_data:
            entry = MemoryEntry.from_dict(entry_data)
            self._memories[entry.id] = entry
            imported_count += 1

        self._last_modified = time.time()
        self._total_stored += imported_count
        return imported_count

    def clear(self) -> None:
        """Remove all memories from the store.

        This is a destructive operation — all stored memories are permanently
        removed. Statistics counters are preserved.
        """
        self._memories.clear()
        self._last_modified = time.time()
        self._maybe_auto_save()

    def stats(self) -> Dict[str, Any]:
        """Return comprehensive statistics about the memory store.

        Returns:
            Dictionary containing:
            - count: Number of stored memories
            - capacity: Maximum capacity
            - utilization: Fraction of capacity used
            - total_stored: Cumulative number of memories ever stored
            - total_retrieved: Cumulative retrieval operations
            - total_deleted: Cumulative deletion operations
            - total_updated: Cumulative update operations
            - avg_importance: Mean importance across all memories
            - avg_access_count: Mean access count
            - oldest_memory_age_hours: Age of the oldest memory
            - newest_memory_age_hours: Age of the newest memory
            - memory_id_distribution: Counter of metadata categories (if present)
            - last_modified: Timestamp of last modification
        """
        if not self._memories:
            return {
                "count": 0,
                "capacity": self.config.capacity,
                "utilization": 0.0,
                "total_stored": self._total_stored,
                "total_retrieved": self._total_retrieved,
                "total_deleted": self._total_deleted,
                "total_updated": self._total_updated,
                "avg_importance": 0.0,
                "avg_access_count": 0.0,
                "oldest_memory_age_hours": 0.0,
                "newest_memory_age_hours": 0.0,
                "last_modified": self._last_modified,
            }

        entries = list(self._memories.values())
        now = time.time()

        importances = [e.importance for e in entries]
        access_counts = [e.access_count for e in entries]
        ages = [(now - e.created_at) / 3600.0 for e in entries]

        # Collect category distribution from metadata
        category_counts: Dict[str, int] = collections.Counter()
        for entry in entries:
            if "category" in entry.metadata:
                category_counts[entry.metadata["category"]] += 1

        # Importance distribution
        importance_buckets = {"high": 0, "medium": 0, "low": 0}
        for imp in importances:
            if imp >= 0.7:
                importance_buckets["high"] += 1
            elif imp >= 0.3:
                importance_buckets["medium"] += 1
            else:
                importance_buckets["low"] += 1

        return {
            "count": len(entries),
            "capacity": self.config.capacity,
            "utilization": len(entries) / self.config.capacity,
            "total_stored": self._total_stored,
            "total_retrieved": self._total_retrieved,
            "total_deleted": self._total_deleted,
            "total_updated": self._total_updated,
            "avg_importance": sum(importances) / len(importances),
            "min_importance": min(importances),
            "max_importance": max(importances),
            "avg_access_count": sum(access_counts) / len(access_counts),
            "total_access_count": sum(access_counts),
            "oldest_memory_age_hours": max(ages),
            "newest_memory_age_hours": min(ages),
            "importance_distribution": importance_buckets,
            "category_distribution": dict(category_counts),
            "last_modified": self._last_modified,
            "persistence_enabled": self.config.persistence_path is not None,
        }

    def _prune_to_capacity(self, target_count: int) -> int:
        """Prune memories to stay within capacity.

        Removes the memories with the lowest effective importance until
        the store is at or below target_count.

        Args:
            target_count: Target number of memories to keep.

        Returns:
            Number of memories removed.
        """
        if len(self._memories) <= target_count:
            return 0

        entries = list(self._memories.values())
        entries.sort(key=lambda e: e.effective_importance(self.config.decay_rate))

        to_remove = len(self._memories) - target_count
        for i in range(to_remove):
            entry = entries[i]
            if entry.id in self._memories:
                del self._memories[entry.id]
                self._total_deleted += 1

        return to_remove

    def _load_if_exists(self) -> bool:
        """Attempt to load persisted memory from disk.

        Returns:
            True if memory was successfully loaded, False otherwise.
        """
        if not self.config.persistence_path:
            return False

        json_path = os.path.join(self.config.persistence_path, "long_term_memory.json")
        if os.path.exists(json_path):
            try:
                self.import_json(json_path, merge=False)
                return True
            except (json.JSONDecodeError, IOError, OSError):
                return False
        return False

    def _maybe_auto_save(self) -> None:
        """Save to disk if auto-save is enabled and enough time has passed."""
        if not self.config.auto_save or not self.config.persistence_path:
            return

        now = time.time()
        if now - self._last_save_time >= self.config.auto_save_interval:
            self.save()

    def save(self) -> Optional[str]:
        """Explicitly save all memories to the configured persistence path.

        Returns:
            Path to the saved file, or None if persistence is not configured.
        """
        if not self.config.persistence_path:
            return None

        json_path = os.path.join(self.config.persistence_path, "long_term_memory.json")
        self.export_json(json_path)
        self._last_save_time = time.time()
        return json_path

    def __len__(self) -> int:
        return len(self._memories)

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._memories

    def __iter__(self):
        return iter(self._memories.values())

    def __repr__(self) -> str:
        return (
            f"LongTermMemoryStore(count={len(self._memories)}, "
            f"capacity={self.config.capacity}, "
            f"embedding_dim={self.config.embedding_dim})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Consolidator
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryConsolidator:
    """Consolidates short-term memories into long-term storage.

    The consolidation process involves:
    1. Filtering memories that meet importance criteria
    2. Clustering related memories by embedding similarity
    3. Summarizing clusters into condensed representations
    4. Storing the consolidated memories in the long-term store

    This mimics the biological process of memory consolidation during sleep,
    where short-term memories are transformed into stable long-term memories.

    Args:
        memory_store: The LongTermMemoryStore to consolidate into.
        min_importance: Minimum importance score for consolidation eligibility.
        min_access_count: Minimum access count for consolidation eligibility.
        similarity_threshold: Threshold for clustering related memories.
        max_summary_length: Maximum character length of summarized memories.
    """

    def __init__(
        self,
        memory_store: Optional[LongTermMemoryStore] = None,
        min_importance: float = 0.3,
        min_access_count: int = 1,
        similarity_threshold: float = 0.8,
        max_summary_length: int = 500,
    ):
        self.memory_store = memory_store
        self.min_importance = min_importance
        self.min_access_count = min_access_count
        self.similarity_threshold = similarity_threshold
        self.max_summary_length = max_summary_length

        # Statistics
        self._total_consolidated: int = 0
        self._total_clusters_formed: int = 0

    def should_consolidate(
        self,
        memory: MemoryEntry,
        min_importance: Optional[float] = None,
        min_access_count: Optional[int] = None,
    ) -> bool:
        """Determine if a memory should be consolidated into long-term storage.

        A memory qualifies for consolidation if:
        - Its importance score meets or exceeds the minimum threshold
        - It has been accessed at least the minimum number of times
        - Its effective importance (after decay) is still meaningful

        Args:
            memory: The memory entry to evaluate.
            min_importance: Override minimum importance threshold.
            min_access_count: Override minimum access count.

        Returns:
            True if the memory should be consolidated.
        """
        threshold = min_importance if min_importance is not None else self.min_importance
        access_threshold = min_access_count if min_access_count is not None else self.min_access_count

        # Check base importance
        if memory.importance < threshold:
            return False

        # Check access count
        if memory.access_count < access_threshold:
            return False

        # Check effective importance (must still be above a minimum)
        effective = memory.effective_importance(decay_rate=0.005)
        if effective < 0.1:
            return False

        # Content must be meaningful
        if not memory.content or len(memory.content.strip()) < 3:
            return False

        return True

    def consolidate(
        self,
        memories: List[MemoryEntry],
        target_store: Optional[LongTermMemoryStore] = None,
    ) -> List[MemoryEntry]:
        """Consolidate a list of memories into the long-term store.

        The consolidation process:
        1. Filter memories that meet consolidation criteria
        2. Cluster related memories by embedding similarity
        3. For each cluster, create a summarized representation
        4. Store consolidated memories in the target store

        Args:
            memories: List of MemoryEntry objects to consolidate.
            target_store: Override target store. Uses self.memory_store if None.

        Returns:
            List of newly created consolidated MemoryEntry objects.
        """
        store = target_store or self.memory_store
        if store is None:
            raise ValueError("No memory store available for consolidation")

        # Step 1: Filter eligible memories
        eligible = [m for m in memories if self.should_consolidate(m)]

        if not eligible:
            return []

        # Step 2: Cluster related memories
        num_clusters = max(1, len(eligible) // 3)  # ~3 memories per cluster
        clusters = self.cluster_memories(eligible, num_clusters=num_clusters)
        self._total_clusters_formed += len(clusters)

        # Step 3: Summarize each cluster and store
        consolidated_entries: List[MemoryEntry] = []
        for cluster in clusters:
            if len(cluster) == 0:
                continue

            if len(cluster) == 1:
                # Single memory: store directly
                entry = cluster[0]
                if not store.has(entry.id):
                    stored = store.store(
                        content=entry.content,
                        metadata={
                            **entry.metadata,
                            "consolidated": True,
                            "cluster_size": 1,
                        },
                        importance=entry.importance,
                        memory_id=entry.id,
                    )
                    consolidated_entries.append(stored)
            else:
                # Multiple memories: summarize and store
                summary = self.summarize_memories(cluster, self.max_summary_length)

                # Compute average importance of the cluster
                avg_importance = sum(m.importance for m in cluster) / len(cluster)

                # Combine metadata
                combined_metadata = {"consolidated": True, "cluster_size": len(cluster)}
                all_tags = set()
                for m in cluster:
                    if "tags" in m.metadata:
                        if isinstance(m.metadata["tags"], list):
                            all_tags.update(m.metadata["tags"])
                        elif isinstance(m.metadata["tags"], str):
                            all_tags.add(m.metadata["tags"])
                    for key, value in m.metadata.items():
                        if key != "tags" and key not in combined_metadata:
                            combined_metadata[key] = value
                if all_tags:
                    combined_metadata["tags"] = list(all_tags)

                stored = store.store(
                    content=summary,
                    metadata=combined_metadata,
                    importance=min(1.0, avg_importance * 1.1),  # Boost for consolidation
                )
                consolidated_entries.append(stored)

        self._total_consolidated += len(consolidated_entries)
        return consolidated_entries

    def summarize_memories(
        self,
        memories: List[MemoryEntry],
        max_length: int = 500,
    ) -> str:
        """Create an extractive summary of related memories.

        This method uses a scoring approach to select the most representative
        sentences from the input memories, combining them into a coherent summary.

        The algorithm:
        1. Split all memories into sentences
        2. Compute sentence embeddings (using the first sentence of each memory as context)
        3. Score each sentence by:
           - Position score (earlier sentences score higher)
           - Importance score (from the parent memory)
           - Length score (medium-length sentences preferred)
           - Uniqueness score (penalize redundancy)
        4. Select top sentences until max_length is reached
        5. Join selected sentences into a summary

        Args:
            memories: List of related MemoryEntry objects.
            max_length: Maximum character length of the summary.

        Returns:
            A text summary combining the most important information.
        """
        if not memories:
            return ""

        # Collect all sentences with their source memory importance
        all_sentences: List[Tuple[str, float]] = []
        for memory in memories:
            if not memory.content:
                continue
            # Split content into sentences (simple split on .!?)
            sentences = self._split_into_sentences(memory.content)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    # Position bonus: earlier sentences get higher scores
                    position_score = 1.0 / (1.0 + i * 0.3)
                    # Length score: prefer medium-length sentences
                    word_count = len(sentence.split())
                    if word_count < 3:
                        length_score = 0.1
                    elif word_count < 20:
                        length_score = 0.8
                    elif word_count < 50:
                        length_score = 1.0
                    else:
                        length_score = 0.7
                    # Combined score
                    combined_score = (
                        memory.importance * position_score * length_score
                    )
                    all_sentences.append((sentence.strip(), combined_score))

        if not all_sentences:
            return ""

        # Sort by combined score (descending)
        all_sentences.sort(key=lambda x: x[1], reverse=True)

        # Extractive selection: greedily add sentences
        selected: List[str] = []
        selected_text = ""
        used_substrings: Set[str] = set()

        for sentence, score in all_sentences:
            # Check if we've already used a very similar sentence
            sentence_norm = sentence.lower()[:50]
            is_duplicate = False
            for used in used_substrings:
                if self._text_overlap(sentence_norm, used) > 0.7:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Check length constraint
            candidate = selected_text + (" " if selected_text else "") + sentence
            if len(candidate) > max_length:
                break

            selected.append(sentence)
            selected_text = candidate
            used_substrings.add(sentence_norm)

        if not selected:
            # Fallback: return first sentence from the most important memory
            memories_sorted = sorted(memories, key=lambda m: m.importance, reverse=True)
            for m in memories_sorted:
                if m.content:
                    sentences = self._split_into_sentences(m.content)
                    if sentences:
                        return sentences[0].strip()[:max_length]
            return ""

        return selected_text.strip()

    def cluster_memories(
        self,
        memories: List[MemoryEntry],
        num_clusters: Optional[int] = None,
    ) -> List[List[MemoryEntry]]:
        """Cluster memories by embedding similarity using k-means.

        This implements a simple k-means clustering algorithm on memory embeddings.
        Memories with similar embeddings (representing similar content) are grouped
        together.

        Algorithm:
        1. Extract embeddings from all memories
        2. Initialize cluster centroids using k-means++ initialization
        3. Iterate: assign memories to nearest centroid, update centroids
        4. Repeat until convergence or max iterations

        Args:
            memories: List of MemoryEntry objects to cluster.
            num_clusters: Number of clusters. If None, automatically determined
                based on the number of memories (sqrt(n) heuristic).

        Returns:
            List of clusters, where each cluster is a list of MemoryEntry objects.
            Clusters with no assigned memories are omitted.
        """
        if not memories:
            return []

        # Filter to memories that have embeddings
        valid_memories = [m for m in memories if m.embedding is not None]
        if not valid_memories:
            return [[m] for m in memories]

        # Determine number of clusters
        if num_clusters is None:
            num_clusters = max(1, int(math.sqrt(len(valid_memories))))

        num_clusters = min(num_clusters, len(valid_memories))

        if num_clusters == 1:
            return [valid_memories]

        if num_clusters == len(valid_memories):
            return [[m] for m in valid_memories]

        # Extract embeddings as tensor
        embeddings = torch.stack([m.embedding for m in valid_memories])

        # K-means++ initialization for better centroids
        centroids = self._kmeans_plus_plus_init(embeddings, num_clusters)

        # K-means iterations
        max_iterations = 100
        assignments = torch.zeros(len(valid_memories), dtype=torch.long)

        for iteration in range(max_iterations):
            # Assign each memory to nearest centroid
            # Compute distances: (N, D) × (K, D) → (N, K)
            centroid_norms = centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
            centroid_normalized = centroids / centroid_norms
            embedding_norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            embeddings_normalized = embeddings / embedding_norms

            # Cosine similarity: higher is closer
            similarities = torch.matmul(embeddings_normalized, centroid_normalized.t())
            new_assignments = similarities.argmax(dim=1)

            # Check convergence
            if torch.equal(new_assignments, assignments) and iteration > 0:
                break
            assignments = new_assignments

            # Update centroids
            for k in range(num_clusters):
                mask = assignments == k
                if mask.any():
                    centroids[k] = embeddings[mask].mean(dim=0)
                    # Re-normalize centroid
                    centroids[k] = centroids[k] / (centroids[k].norm() + 1e-8)

        # Build clusters from assignments
        clusters: Dict[int, List[MemoryEntry]] = collections.defaultdict(list)
        for i, assignment in enumerate(assignments.tolist()):
            clusters[assignment].append(valid_memories[i])

        # Sort clusters by size (largest first)
        result = sorted(clusters.values(), key=len, reverse=True)
        return result

    def _kmeans_plus_plus_init(
        self,
        embeddings: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """Initialize k-means centroids using the k-means++ algorithm.

        This selects initial centroids that are well-spread across the data,
        leading to better convergence and cluster quality.

        Args:
            embeddings: Tensor of shape (N, D) containing data points.
            num_clusters: Number of centroids to select.

        Returns:
            Tensor of shape (K, D) with initial centroids.
        """
        n = embeddings.shape[0]
        centroids = torch.zeros(num_clusters, embeddings.shape[1])

        # Select first centroid randomly
        first_idx = random.randint(0, n - 1)
        centroids[0] = embeddings[first_idx]

        # Select remaining centroids
        for k in range(1, num_clusters):
            # Compute squared distances to nearest centroid
            distances = torch.zeros(n)
            for i in range(n):
                min_dist = float("inf")
                for j in range(k):
                    dist = 1.0 - cosine_similarity(embeddings[i], centroids[j])
                    min_dist = min(min_dist, max(0.0, float(dist)))
                distances[i] = min_dist

            # Sample proportional to distance squared
            distances = distances.clamp(min=1e-8)
            probabilities = distances / distances.sum()

            # Weighted random selection
            cumsum = torch.cumsum(probabilities, dim=0)
            r = random.random()
            idx = (cumsum < r).sum().item()
            idx = min(idx, n - 1)
            centroids[k] = embeddings[idx]

        return centroids

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation-based splitting.

        Handles common abbreviations and edge cases.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Simple sentence splitting on .!?
        import re

        # Protect abbreviations
        abbreviations = {
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.",
            "etc.", "e.g.", "i.e.", "a.k.a.", "approx.", "dept.", "est.",
        }

        text_protected = text
        for abbr in abbreviations:
            text_protected = text_protected.replace(abbr, abbr.replace(".", "<<DOT>>"))

        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text_protected)

        # Restore abbreviations and clean up
        cleaned = []
        for sentence in sentences:
            sentence = sentence.replace("<<DOT>>", ".")
            sentence = sentence.strip()
            if sentence:
                cleaned.append(sentence)

        return cleaned if cleaned else [text.strip()]

    def _text_overlap(self, text1: str, text2: str) -> float:
        """Compute word-level overlap between two text strings.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Overlap ratio in [0.0, 1.0].
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def get_stats(self) -> Dict[str, Any]:
        """Return consolidation statistics.

        Returns:
            Dictionary with total_consolidated, total_clusters_formed counts.
        """
        return {
            "total_consolidated": self._total_consolidated,
            "total_clusters_formed": self._total_clusters_formed,
            "min_importance": self.min_importance,
            "min_access_count": self.min_access_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Decay
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryDecay:
    """Time-based importance decay for long-term memories.

    Implements an exponential decay model inspired by the Ebbinghaus forgetting
    curve. Memories that are not accessed frequently or recently gradually lose
    importance, making them candidates for pruning.

    The decay formula is:

        decay_factor = exp(-decay_rate * hours_since_access)

        effective_importance = base_importance * decay_factor + access_bonus

    where:
    - decay_rate controls the speed of forgetting
    - hours_since_access is time since last access
    - access_bonus = min(log(1 + access_count) / 5, 0.3) provides resistance
      to decay for frequently accessed memories

    Args:
        decay_rate: Controls decay speed. Higher values = faster forgetting.
            Typical values: 0.001 (slow) to 0.05 (fast).
        access_resistance: How strongly access count resists decay.
            Range: [0.0, 1.0].
        min_importance: Memories below this are candidates for pruning.
        max_age_hours: Memories older than this are forcefully decayed.
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        access_resistance: float = 0.3,
        min_importance: float = 0.05,
        max_age_hours: float = 8760.0,  # 1 year
    ):
        self.decay_rate = max(0.0, decay_rate)
        self.access_resistance = max(0.0, min(1.0, access_resistance))
        self.min_importance = min_importance
        self.max_age_hours = max_age_hours

    def compute_decay_factor(
        self,
        memory: MemoryEntry,
        current_time: Optional[float] = None,
    ) -> float:
        """Compute the decay factor for a single memory.

        The decay factor multiplies the base importance to produce the effective
        importance. It is computed using exponential decay from the time of last
        access, with a bonus for access frequency.

        Args:
            memory: The memory entry to compute decay for.
            current_time: Current time as Unix timestamp. If None, uses current time.

        Returns:
            Decay factor in [0.0, 1.0]. Multiply by base importance for effective importance.
        """
        if current_time is None:
            current_time = time.time()

        hours_since_access = (current_time - memory.accessed_at) / 3600.0

        if hours_since_access < 0:
            hours_since_access = 0

        # Exponential decay
        time_decay = math.exp(-self.decay_rate * hours_since_access)

        # Access frequency bonus (logarithmic scaling)
        if memory.access_count > 0:
            access_bonus = self.access_resistance * min(
                math.log1p(memory.access_count) / 5.0, 1.0
            )
        else:
            access_bonus = 0.0

        # Age penalty: additional decay for very old memories
        hours_since_creation = (current_time - memory.created_at) / 3600.0
        if hours_since_creation > self.max_age_hours:
            age_penalty = math.exp(
                -0.1 * (hours_since_creation - self.max_age_hours)
            )
        else:
            age_penalty = 1.0

        # Combine all factors
        factor = time_decay * age_penalty + access_bonus
        return max(0.0, min(1.0, factor))

    def apply_decay(
        self,
        memories: List[MemoryEntry],
        current_time: Optional[float] = None,
        in_place: bool = True,
    ) -> List[MemoryEntry]:
        """Apply time-based decay to a list of memories.

        For each memory, the effective importance is computed based on the decay
        formula, and the importance field is updated accordingly.

        Args:
            memories: List of MemoryEntry objects to decay.
            current_time: Current time as Unix timestamp. If None, uses current time.
            in_place: If True, modifies the memories directly. If False, returns
                new MemoryEntry objects with decayed importance.

        Returns:
            List of memories with updated importance values.
        """
        if current_time is None:
            current_time = time.time()

        result = []
        for memory in memories:
            decay_factor = self.compute_decay_factor(memory, current_time)
            effective_importance = memory.importance * decay_factor

            if in_place:
                memory.importance = max(0.0, effective_importance)
                result.append(memory)
            else:
                new_entry = copy.deepcopy(memory)
                new_entry.importance = max(0.0, effective_importance)
                result.append(new_entry)

        return result

    def prune_weak(
        self,
        memories: List[MemoryEntry],
        threshold: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> Tuple[List[MemoryEntry], List[MemoryEntry]]:
        """Remove memories with importance below a threshold.

        First applies decay to all memories, then separates them into retained
        and pruned groups based on the importance threshold.

        Args:
            memories: List of MemoryEntry objects to evaluate.
            threshold: Minimum importance to retain. If None, uses self.min_importance.
            current_time: Current time for decay computation.

        Returns:
            Tuple of (retained_memories, pruned_memories).
        """
        if threshold is None:
            threshold = self.min_importance

        if current_time is None:
            current_time = time.time()

        # Apply decay first
        decayed = self.apply_decay(memories, current_time, in_place=True)

        # Separate by threshold
        retained = []
        pruned = []
        for memory in decayed:
            if memory.importance >= threshold:
                retained.append(memory)
            else:
                pruned.append(memory)

        return retained, pruned

    def batch_decay(
        self,
        memory_dict: Dict[str, MemoryEntry],
        current_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute decay factors for a dictionary of memories.

        This is efficient for batch operations where you need decay factors
        but don't want to modify the memories directly.

        Args:
            memory_dict: Dictionary mapping memory IDs to MemoryEntry objects.
            current_time: Current time for decay computation.

        Returns:
            Dictionary mapping memory IDs to computed decay factors.
        """
        if current_time is None:
            current_time = time.time()

        factors = {}
        for memory_id, memory in memory_dict.items():
            factors[memory_id] = self.compute_decay_factor(memory, current_time)

        return factors

    def get_decay_curve(
        self,
        hours: int = 720,
        access_counts: Optional[List[int]] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Generate the decay curve for visualization and analysis.

        Returns importance values over time for different access counts.

        Args:
            hours: Number of hours to simulate.
            access_counts: List of access counts to generate curves for.
                Default: [0, 1, 3, 5, 10, 20, 50].

        Returns:
            Dictionary mapping access_count to list of (hours, importance) tuples.
        """
        if access_counts is None:
            access_counts = [0, 1, 3, 5, 10, 20, 50]

        base_importance = 0.8
        curves: Dict[str, List[Tuple[float, float]]] = {}

        for ac in access_counts:
            curve = []
            for h in range(0, hours + 1, max(1, hours // 100)):
                # Create a synthetic memory
                fake_memory = MemoryEntry(
                    content="test",
                    importance=base_importance,
                    created_at=time.time() - h * 3600 - 3600,  # created 1h before access
                    accessed_at=time.time() - h * 3600,
                    access_count=ac,
                )
                factor = self.compute_decay_factor(fake_memory)
                effective = base_importance * factor
                curve.append((float(h), effective))
            curves[str(ac)] = curve

        return curves

    def estimate_memory_lifetime(
        self,
        memory: MemoryEntry,
        threshold: float = 0.1,
    ) -> float:
        """Estimate how many hours until a memory's importance falls below threshold.

        Solves the decay equation for time.

        Args:
            memory: The memory entry to estimate lifetime for.
            threshold: Importance threshold at which the memory is considered "dead".

        Returns:
            Estimated hours until the memory falls below the threshold.
            Returns 0 if already below threshold, float('inf') if never will.
        """
        current_effective = memory.importance * self.compute_decay_factor(memory)
        if current_effective <= threshold:
            return 0.0

        # Account for access bonus
        if memory.access_count > 0:
            access_bonus = self.access_resistance * min(
                math.log1p(memory.access_count) / 5.0, 1.0
            )
        else:
            access_bonus = 0.0

        # effective = importance * exp(-rate * t) + access_bonus
        # We want: importance * exp(-rate * t) + access_bonus = threshold
        # exp(-rate * t) = (threshold - access_bonus) / importance
        numerator = threshold - access_bonus
        if numerator <= 0:
            return float("inf")  # Access bonus alone keeps it above threshold

        ratio = numerator / memory.importance
        if ratio <= 0:
            return float("inf")
        if ratio >= 1.0:
            return 0.0

        try:
            hours = -math.log(ratio) / self.decay_rate
            return max(0.0, hours)
        except (ValueError, ZeroDivisionError):
            return float("inf")

    def get_stats(self) -> Dict[str, Any]:
        """Return decay configuration statistics.

        Returns:
            Dictionary with current decay parameters.
        """
        return {
            "decay_rate": self.decay_rate,
            "access_resistance": self.access_resistance,
            "min_importance": self.min_importance,
            "max_age_hours": self.max_age_hours,
            "half_life_hours": (
                math.log(2) / self.decay_rate if self.decay_rate > 0
                else float("inf")
            ),
        }
