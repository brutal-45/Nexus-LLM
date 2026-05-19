"""FAISS-based vector store for embedding storage and retrieval.

Supports cosine similarity and L2 distance metrics with add,
search, and delete operations.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """A document stored in the vector store with its embedding and metadata."""

    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary (excluding embedding array)."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorDocument":
        """Deserialize from dictionary."""
        return cls(
            doc_id=data.get("doc_id", str(uuid.uuid4())),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
        )


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents to the store. Returns list of document IDs."""
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents. Returns list of (document, score) tuples."""
        ...

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents by ID. Returns number of documents deleted."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        ...


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store supporting cosine similarity and L2 distance.

    Stores document embeddings in a FAISS index for efficient
    similarity search. Supports persistence to disk and incremental
    addition/deletion of vectors.
    """

    def __init__(
        self,
        dimension: int = 384,
        metric: str = "cosine",
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """Initialize the FAISS vector store.

        Args:
            dimension: Dimension of the embedding vectors.
            metric: Distance metric - 'cosine' or 'l2'.
            index_type: Type of FAISS index - 'flat', 'ivf', or 'hnsw'.
            nlist: Number of clusters for IVF index.
            nprobe: Number of clusters to search in IVF index.
        """
        if metric not in ("cosine", "l2"):
            raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'l2'.")
        if index_type not in ("flat", "ivf", "hnsw"):
            raise ValueError(f"Unsupported index_type: {index_type}. Use 'flat', 'ivf', or 'hnsw'.")

        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe

        self._index = None
        self._documents: Dict[str, VectorDocument] = {}
        self._id_to_internal: Dict[str, int] = {}
        self._internal_to_id: Dict[int, str] = {}
        self._next_internal_id = 0

        self._init_index()

    def _init_index(self):
        """Initialize the FAISS index based on configuration."""
        import faiss

        if self.metric == "cosine":
            # For cosine similarity, we use IndexFlatIP with normalized vectors
            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                self._index.nprobe = self.nprobe
            else:  # hnsw
                self._index = faiss.IndexHNSWFlat(self.dimension, 32)
                self._index.hnsw.efSearch = 64
        else:  # l2
            if self.index_type == "flat":
                self._index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                self._index.nprobe = self.nprobe
            else:  # hnsw
                self._index = faiss.IndexHNSWFlat(self.dimension, 32)
                self._index.hnsw.efSearch = 64

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Prepare vectors for insertion (normalize if using cosine)."""
        vectors = vectors.astype(np.float32)
        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors)
        return vectors

    def add(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents with embeddings to the vector store.

        Args:
            documents: List of VectorDocument objects with embeddings.

        Returns:
            List of document IDs that were added.
        """
        if not documents:
            return []

        embeddings = []
        doc_ids = []

        for doc in documents:
            if doc.embedding is None:
                logger.warning("Document %s has no embedding, skipping.", doc.doc_id)
                continue
            embeddings.append(doc.embedding)
            doc_ids.append(doc.doc_id)
            self._documents[doc.doc_id] = doc

        if not embeddings:
            return []

        vectors = np.vstack(embeddings)
        vectors = self._prepare_vectors(vectors)

        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            train_size = max(len(vectors), self.nlist * 10)
            if len(vectors) < self.nlist:
                logger.warning(
                    "IVF index needs at least %d vectors for training. Got %d.",
                    self.nlist,
                    len(vectors),
                )
            self._index.train(vectors)

        # Assign internal IDs
        internal_ids = []
        for doc_id in doc_ids:
            int_id = self._next_internal_id
            self._next_internal_id += 1
            self._id_to_internal[doc_id] = int_id
            self._internal_to_id[int_id] = doc_id
            internal_ids.append(int_id)

        # Create ID selector for adding with specific IDs
        id_array = np.array(internal_ids, dtype=np.int64)

        # Add vectors to index
        if self.index_type == "ivf" and self._index.is_trained:
            self._index.add_with_ids(vectors, id_array)
        elif self.index_type != "ivf":
            # For flat and HNSW, we may need to use IndexIDMap
            import faiss

            if not isinstance(self._index, faiss.IndexIDMap):
                self._index = faiss.IndexIDMap(self._index)
            self._index.add_with_ids(vectors, id_array)
        else:
            # Fallback: just add without IDs
            self._index.add(vectors)

        logger.info("Added %d documents to vector store. Total: %d", len(doc_ids), self.count())
        return doc_ids

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents by query embedding.

        Args:
            query_embedding: Query embedding vector of shape (dimension,) or (1, dimension).
            top_k: Number of results to return.

        Returns:
            List of (VectorDocument, score) tuples sorted by relevance.
        """
        if self.count() == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = self._prepare_vectors(query_embedding)
        top_k = min(top_k, self.count())

        distances, indices = self._index.search(query_embedding, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0:
                continue
            doc_id = self._internal_to_id.get(int(idx))
            if doc_id and doc_id in self._documents:
                score = float(distances[0][i])
                # For cosine with IP, higher is better; for L2, lower is better
                if self.metric == "l2":
                    score = 1.0 / (1.0 + score)
                results.append((self._documents[doc_id], score))

        return results

    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents from the store.

        Note: FAISS does not natively support efficient deletion.
        This removes documents from the metadata map. For full
        removal, call rebuild_index() after deletions.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        deleted = 0
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                if doc_id in self._id_to_internal:
                    int_id = self._id_to_internal.pop(doc_id)
                    self._internal_to_id.pop(int_id, None)
                deleted += 1

        if deleted > 0:
            logger.info("Deleted %d documents from metadata. Call rebuild_index() for full cleanup.", deleted)

        return deleted

    def rebuild_index(self):
        """Rebuild the FAISS index from stored documents.

        Useful after deletions to reclaim space and maintain
        search performance.
        """
        docs_with_embeddings = [
            doc for doc in self._documents.values() if doc.embedding is not None
        ]

        if not docs_with_embeddings:
            self._init_index()
            self._id_to_internal.clear()
            self._internal_to_id.clear()
            self._next_internal_id = 0
            return

        # Reset index and ID mappings
        self._init_index()
        self._id_to_internal.clear()
        self._internal_to_id.clear()
        self._next_internal_id = 0

        # Re-add all documents
        self.add(docs_with_embeddings)
        logger.info("Rebuilt index with %d documents.", len(docs_with_embeddings))

    def count(self) -> int:
        """Return the number of documents in the store."""
        try:
            return self._index.ntotal if self._index else 0
        except Exception:
            return len(self._documents)

    def save(self, path: str | Path):
        """Save the vector store to disk.

        Args:
            path: Directory path to save index and metadata.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        import faiss

        index_path = path / "index.faiss"
        faiss.write_index(self._index, str(index_path))

        # Save metadata
        meta_path = path / "metadata.json"
        metadata = {
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self._documents.items()},
            "id_to_internal": {k: v for k, v in self._id_to_internal.items()},
            "next_internal_id": self._next_internal_id,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save embeddings as numpy
        if self._documents:
            embeddings = []
            for doc in self._documents.values():
                if doc.embedding is not None:
                    embeddings.append(doc.embedding)
            if embeddings:
                emb_array = np.vstack(embeddings)
                np.save(str(path / "embeddings.npy"), emb_array)

        logger.info("Saved vector store to %s with %d documents.", path, self.count())

    @classmethod
    def load(cls, path: str | Path) -> "FAISSVectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path containing saved index and metadata.

        Returns:
            Loaded FAISSVectorStore instance.
        """
        path = Path(path)

        # Load metadata
        meta_path = path / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls(
            dimension=metadata["dimension"],
            metric=metadata["metric"],
            index_type=metadata["index_type"],
            nlist=metadata.get("nlist", 100),
            nprobe=metadata.get("nprobe", 10),
        )

        # Load FAISS index
        import faiss

        index_path = path / "index.faiss"
        store._index = faiss.read_index(str(index_path))

        # Restore documents
        for doc_id, doc_data in metadata.get("documents", {}).items():
            store._documents[doc_id] = VectorDocument.from_dict(doc_data)

        # Restore ID mappings
        store._id_to_internal = metadata.get("id_to_internal", {})
        store._internal_to_id = {v: k for k, v in store._id_to_internal.items()}
        store._next_internal_id = metadata.get("next_internal_id", 0)

        # Load embeddings
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            emb_array = np.load(str(emb_path))
            doc_ids = list(store._documents.keys())
            for i, doc_id in enumerate(doc_ids):
                if i < len(emb_array):
                    store._documents[doc_id].embedding = emb_array[i]

        logger.info("Loaded vector store from %s with %d documents.", path, store.count())
        return store
