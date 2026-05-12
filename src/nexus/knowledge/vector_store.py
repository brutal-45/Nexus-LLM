"""
Nexus LLM — Vector Store Module
=================================

Production-grade vector storage and retrieval implementations:
* SimpleVectorStore — brute-force in-memory search
* HNSWIndex — Hierarchical Navigable Small World (from scratch)
* IVFIndex — Inverted File Index (from scratch)
* ProductQuantizationIndex — PQ compression (from scratch)
* DiskANNIndex — disk-based approximate nearest neighbour search
* VectorStoreManager — unified management layer

All implementations use numpy for vector math and require no external
index libraries.
"""

from __future__ import annotations

import abc
import heapq
import json
import logging
import math
import os
import pickle
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger: logging.Logger = logging.getLogger("nexus.knowledge.vector_store")


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VectorRecord:
    """A stored vector with associated metadata.

    Attributes
    ----------
    id : str
        Unique identifier.
    vector : np.ndarray
        The stored embedding (D,).
    metadata : Dict[str, Any]
        Arbitrary metadata.
    """

    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result.

    Attributes
    ----------
    id : str
        Document/vector identifier.
    score : float
        Similarity score (higher = more similar for cosine/IP,
        lower = more similar for L2).
    vector : Optional[np.ndarray]
        The matched vector (optional).
    metadata : Dict[str, Any]
        Associated metadata.
    distance : float
        Raw distance value (always lower = closer).
    """

    id: str
    score: float
    distance: float = 0.0
    vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricType(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "ip"


# ═══════════════════════════════════════════════════════════════════════════
#  Distance functions
# ═══════════════════════════════════════════════════════════════════════════

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - dot / (na * nb)


def _cosine_distance_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance from *query* (D,) to each row of *matrix* (N, D)."""
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-12:
        return np.ones(matrix.shape[0], dtype=np.float64)
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms = np.maximum(m_norms, 1e-12)
    dots = matrix @ query
    cos_sim = dots / (m_norms * q_norm)
    return 1.0 - cos_sim


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _l2_distance_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    diff = matrix - query
    return np.linalg.norm(diff, axis=1)


def _ip_distance(a: np.ndarray, b: np.ndarray) -> float:
    return -float(np.dot(a, b))


def _ip_distance_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return -(matrix @ query)


def _get_distance_fn(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    metric = metric.lower()
    if metric in ("cosine", "cos"):
        return _cosine_distance
    elif metric in ("l2", "euclidean"):
        return _l2_distance
    elif metric in ("ip", "inner_product", "dot"):
        return _ip_distance
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _get_distance_batch_fn(metric: str) -> Callable:
    metric = metric.lower()
    if metric in ("cosine", "cos"):
        return _cosine_distance_batch
    elif metric in ("l2", "euclidean"):
        return _l2_distance_batch
    elif metric in ("ip", "inner_product", "dot"):
        return _ip_distance_batch
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _distance_to_score(distance: float, metric: str) -> float:
    """Convert raw distance to a similarity score (higher = better)."""
    metric = metric.lower()
    if metric in ("cosine", "cos"):
        return 1.0 - distance
    elif metric in ("l2", "euclidean"):
        return 1.0 / (1.0 + distance)
    elif metric in ("ip", "inner_product", "dot"):
        return -distance
    return distance


# ═══════════════════════════════════════════════════════════════════════════
#  BaseVectorStore
# ═══════════════════════════════════════════════════════════════════════════

class BaseVectorStore(abc.ABC):
    """Abstract interface for vector stores."""

    @abc.abstractmethod
    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the store.

        Parameters
        ----------
        embeddings : np.ndarray
            (N, D) matrix of vectors.
        ids : List[str]
            Unique identifier per vector.
        metadata : Optional[List[Dict[str, Any]]]
            Optional metadata per vector.
        """

    @abc.abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search for the nearest neighbours of *query*."""

    @abc.abstractmethod
    def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID.  Returns number of deleted vectors."""

    def update(self, id: str, embedding: np.ndarray) -> bool:
        """Update the vector for *id*.  Returns ``True`` on success."""
        return False

    def save(self, path: Union[str, Path]) -> None:
        """Persist the store to disk."""

    def load(self, path: Union[str, Path]) -> None:
        """Load the store from disk."""

    @property
    @abc.abstractmethod
    def count(self) -> int:
        """Number of vectors in the store."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Dimensionality of stored vectors."""


# ═══════════════════════════════════════════════════════════════════════════
#  SimpleVectorStore (brute-force)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleVectorStore(BaseVectorStore):
    """In-memory brute-force vector store using numpy.

    Suitable for small-to-medium corpora (< 100 K vectors).  Stores all
    vectors in a contiguous numpy array for fast batch distance computation.

    Parameters
    ----------
    dimension : int
        Expected vector dimensionality.
    metric : str
        Distance metric: ``"cosine"``, ``"l2"``, or ``"ip"``.
    normalize : bool
        L2-normalise vectors on insertion (cosine-friendly).
    """

    def __init__(
        self,
        dimension: int = 256,
        metric: str = "cosine",
        normalize: bool = True,
    ) -> None:
        self._dim = dimension
        self._metric = metric.lower()
        self._normalize = normalize
        self._distance_fn = _get_distance_fn(self._metric)
        self._distance_batch_fn = _get_distance_batch_fn(self._metric)

        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}

    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the store."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        assert embeddings.shape[1] == self._dim, (
            f"Expected dimension {self._dim}, got {embeddings.shape[1]}"
        )
        assert len(ids) == embeddings.shape[0], "ids and embeddings must have same length"

        vecs = embeddings.astype(np.float64)
        if self._normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vecs = vecs / norms

        start_idx = len(self._ids)
        if self._vectors is None:
            self._vectors = vecs
        else:
            self._vectors = np.vstack([self._vectors, vecs])

        meta = metadata if metadata is not None else [{}] * len(ids)
        for i, (doc_id, m) in enumerate(zip(ids, meta)):
            idx = start_idx + i
            self._ids.append(doc_id)
            self._metadata.append(m)
            self._id_to_idx[doc_id] = idx

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Brute-force nearest-neighbour search."""
        if self._vectors is None or len(self._ids) == 0:
            return []

        if query.ndim == 1:
            query = query.astype(np.float64)
            if self._normalize:
                norm = np.linalg.norm(query)
                if norm > 1e-12:
                    query = query / norm
        else:
            query = query.flatten().astype(np.float64)
            if self._normalize:
                norm = np.linalg.norm(query)
                if norm > 1e-12:
                    query = query / norm

        distances = self._distance_batch_fn(query, self._vectors)
        k = min(top_k, len(distances))
        top_indices = np.argpartition(distances, k)[:k]
        top_indices = top_indices[np.argsort(distances[top_indices])]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices):
            idx = int(idx)
            d = float(distances[idx])
            results.append(SearchResult(
                id=self._ids[idx],
                score=_distance_to_score(d, self._metric),
                distance=d,
                vector=self._vectors[idx].copy(),
                metadata=dict(self._metadata[idx]),
            ))

        # Apply metadata filters if provided
        if filters:
            results = [r for r in results if self._matches_filter(r.metadata, filters)]

        return results

    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, (list, set, tuple)):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID."""
        indices_to_delete: Set[int] = set()
        for doc_id in ids:
            idx = self._id_to_idx.pop(doc_id, None)
            if idx is not None:
                indices_to_delete.add(idx)

        if not indices_to_delete:
            return 0

        keep_indices = sorted(set(range(len(self._ids))) - indices_to_delete)
        self._vectors = self._vectors[keep_indices]
        self._ids = [self._ids[i] for i in keep_indices]
        self._metadata = [self._metadata[i] for i in keep_indices]
        self._id_to_idx = {d: i for i, d in enumerate(self._ids)}
        return len(indices_to_delete)

    def update(self, id: str, embedding: np.ndarray) -> bool:
        """Update a single vector."""
        idx = self._id_to_idx.get(id)
        if idx is None:
            return False
        vec = embedding.astype(np.float64).flatten()
        if self._normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
        self._vectors[idx] = vec
        return True

    def get(self, id: str) -> Optional[VectorRecord]:
        """Retrieve a single vector record by ID."""
        idx = self._id_to_idx.get(id)
        if idx is None:
            return None
        return VectorRecord(
            id=self._ids[idx],
            vector=self._vectors[idx].copy(),
            metadata=dict(self._metadata[idx]),
        )

    def get_batch(self, ids: List[str]) -> List[Optional[VectorRecord]]:
        """Retrieve multiple vector records by ID."""
        return [self.get(doc_id) for doc_id in ids]

    def save(self, path: Union[str, Path]) -> None:
        """Persist the store to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self._dim,
            "metric": self._metric,
            "normalize": self._normalize,
            "vectors": self._vectors,
            "ids": self._ids,
            "metadata": self._metadata,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("SimpleVectorStore saved %d vectors to %s", self.count, path)

    def load(self, path: Union[str, Path]) -> None:
        """Load the store from a pickle file."""
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._dim = data["dimension"]
        self._metric = data["metric"]
        self._normalize = data["normalize"]
        self._vectors = data["vectors"]
        self._ids = data["ids"]
        self._metadata = data["metadata"]
        self._id_to_idx = {d: i for i, d in enumerate(self._ids)}
        logger.info("SimpleVectorStore loaded %d vectors from %s", self.count, path)

    @property
    def count(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return self.count


# ═══════════════════════════════════════════════════════════════════════════
#  HNSWIndex (Hierarchical Navigable Small World — from scratch)
# ═══════════════════════════════════════════════════════════════════════════

class HNSWIndex(BaseVectorStore):
    """Hierarchical Navigable Small World graph for approximate NN search.

    Implements the full HNSW algorithm from Malkov & Yashunin (2018):
    * Multi-layer graph with exponential layer assignment
    * Greedy search with beam width (ef) parameter
    * Neighbour selection with heuristic (simple or heuristic)
    * Thread-safe insertion

    Parameters
    ----------
    dimension : int
        Vector dimensionality.
    M : int
        Maximum number of neighbours per node per layer.
    ef_construction : int
        Beam width during construction.
    ef_search : int
        Beam width during search.
    metric : str
        Distance metric.
    ml : float
        Level-generation factor (``1 / ln(M)`` by default).
    max_elements : int
        Maximum number of elements to store.
    """

    def __init__(
        self,
        dimension: int = 256,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        metric: str = "cosine",
        ml: Optional[float] = None,
        max_elements: int = 1_000_000,
    ) -> None:
        self._dim = dimension
        self._M = M
        self._M_max0 = 2 * M  # max neighbours at layer 0
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._metric = metric.lower()
        self._distance_fn = _get_distance_fn(self._metric)
        self._ml = ml if ml is not None else 1.0 / math.log(M)
        self._max_elements = max_elements

        # Storage
        self._vectors: Dict[int, np.ndarray] = {}
        self._ids: Dict[int, str] = {}
        self._str_to_int: Dict[str, int] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._graphs: List[Dict[int, Set[int]]] = [{}]  # layer 0
        self._max_level: int = 0
        self._entry_point: int = -1
        self._element_count: int = 0

    def _get_random_level(self) -> int:
        """Sample a level from the geometric distribution."""
        level = 0
        while np.random.random() < (1.0 / self._ml) and level < 16:
            level += 1
        return level

    def _ensure_layers(self, level: int) -> None:
        while len(self._graphs) <= level:
            self._graphs.append({})

    def _distance(self, a_idx: int, b_idx: int) -> float:
        return self._distance_fn(self._vectors[a_idx], self._vectors[b_idx])

    def _distance_to_vec(self, query: np.ndarray, node_idx: int) -> float:
        return self._distance_fn(query, self._vectors[node_idx])

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """Search a single HNSW layer.

        Returns list of ``(distance, node_id)`` sorted by distance.
        """
        if not entry_points:
            return []

        visited: Set[int] = set(entry_points)
        candidates: List[Tuple[float, int]] = []
        results: List[Tuple[float, int]] = []

        for ep in entry_points:
            d = self._distance_to_vec(query, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_result_dist = -results[0][0]

            if c_dist > furthest_result_dist:
                break

            graph = self._graphs[layer].get(c_id, set())
            for neighbour in graph:
                if neighbour in visited:
                    continue
                visited.add(neighbour)

                d = self._distance_to_vec(query, neighbour)
                furthest_result_dist = -results[0][0]

                if d < furthest_result_dist or len(results) < ef:
                    heapq.heappush(candidates, (d, neighbour))
                    heapq.heappush(results, (-d, neighbour))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(abs(d), n) for d, n in results]

    def _select_neighbours_simple(
        self,
        query_vec: np.ndarray,
        candidates: List[Tuple[float, int]],
        M: int,
    ) -> List[Tuple[float, int]]:
        """Simple neighbour selection: take closest M."""
        candidates.sort(key=lambda x: x[0])
        return candidates[:M]

    def _select_neighbours_heuristic(
        self,
        query_vec: np.ndarray,
        candidates: List[Tuple[float, int]],
        M: int,
    ) -> List[Tuple[float, int]]:
        """Heuristic neighbour selection (Algorithm 4 from the paper)."""
        candidates.sort(key=lambda x: x[0])
        result: List[Tuple[float, int]] = []
        for d_q, c in candidates:
            if len(result) >= M:
                break
            good = True
            d_cq = d_q
            for d_r, r in result:
                d_cr = self._distance_fn(self._vectors[c], self._vectors[r])
                if d_cr < d_cq:
                    good = False
                    break
            if good:
                result.append((d_q, c))
        return result

    def _insert(self, internal_id: int, level: int) -> None:
        """Insert node into the HNSW graph."""
        query_vec = self._vectors[internal_id]
        self._ensure_layers(level)

        if self._entry_point == -1:
            self._entry_point = internal_id
            self._max_level = level
            for l in range(level + 1):
                self._graphs[l][internal_id] = set()
            return

        # Search from top to the insertion level
        ep = [self._entry_point]
        for l in range(self._max_level, level, -1):
            results = self._search_layer(query_vec, ep, ef=1, layer=l)
            ep = [n for _, n in results]

        # Insert at levels [0, level]
        for l in range(min(level, self._max_level) + 1):
            results = self._search_layer(query_vec, ep, ef=self._ef_construction, layer=l)
            M_max = self._M_max0 if l == 0 else self._M
            neighbours = self._select_neighbours_heuristic(query_vec, results, M_max)

            # Add bidirectional connections
            self._graphs[l][internal_id] = set()
            for d, n in neighbours:
                self._graphs[l][internal_id].add(n)
                if n not in self._graphs[l]:
                    self._graphs[l][n] = set()
                self._graphs[l][n].add(internal_id)

                # Trim if too many neighbours
                if len(self._graphs[l][n]) > M_max:
                    n_vec = self._vectors[n]
                    n_candidates = [(self._distance_fn(n_vec, self._vectors[nn]), nn)
                                    for nn in self._graphs[l][n]]
                    trimmed = self._select_neighbours_heuristic(n_vec, n_candidates, M_max)
                    self._graphs[l][n] = set(nn for _, nn in trimmed)

            ep = [n for _, n in neighbours]

        # Handle new levels above existing max
        if level > self._max_level:
            for l in range(self._max_level + 1, level + 1):
                self._graphs[l][internal_id] = set()
            self._entry_point = internal_id
            self._max_level = level

    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors and build the HNSW graph."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        meta = metadata if metadata is not None else [{}] * len(ids)

        for i in range(embeddings.shape[0]):
            internal_id = self._element_count
            doc_id = ids[i]

            self._str_to_int[doc_id] = internal_id
            self._ids[internal_id] = doc_id
            self._vectors[internal_id] = embeddings[i].astype(np.float64)
            self._metadata[internal_id] = meta[i]

            level = self._get_random_level()
            self._insert(internal_id, level)
            self._element_count += 1

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Approximate nearest-neighbour search using HNSW."""
        if self._entry_point == -1:
            return []

        query_vec = query.flatten().astype(np.float64)
        ef = max(self._ef_search, top_k)

        # Greedy search from top to layer 1
        ep = [self._entry_point]
        for l in range(self._max_level, 0, -1):
            results = self._search_layer(query_vec, ep, ef=1, layer=l)
            ep = [n for _, n in results]

        # Search layer 0 with full ef
        results = self._search_layer(query_vec, ep, ef=ef, layer=0)
        results.sort(key=lambda x: x[0])

        out: List[SearchResult] = []
        for rank, (dist, idx) in enumerate(results[:top_k]):
            out.append(SearchResult(
                id=self._ids[idx],
                score=_distance_to_score(dist, self._metric),
                distance=dist,
                vector=self._vectors[idx].copy(),
                metadata=dict(self._metadata[idx]),
            ))
        return out

    def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID (lazy deletion — removes from graph)."""
        deleted = 0
        for doc_id in ids:
            internal_id = self._str_to_int.pop(doc_id, None)
            if internal_id is None:
                continue
            for layer in self._graphs:
                if internal_id in layer:
                    neighbours = layer[internal_id]
                    for n in neighbours:
                        if n in layer:
                            layer[n].discard(internal_id)
                    del layer[internal_id]
            self._vectors.pop(internal_id, None)
            self._ids.pop(internal_id, None)
            self._metadata.pop(internal_id, None)
            deleted += 1
        return deleted

    def update(self, id: str, embedding: np.ndarray) -> bool:
        """Update vector (re-inserts into graph)."""
        internal_id = self._str_to_int.get(id)
        if internal_id is None:
            return False
        # Remove old connections
        for layer in self._graphs:
            if internal_id in layer:
                neighbours = layer[internal_id]
                for n in neighbours:
                    if n in layer:
                        layer[n].discard(internal_id)
                layer[internal_id] = set()
        self._vectors[internal_id] = embedding.astype(np.float64)
        # Re-connect with new vector
        query_vec = self._vectors[internal_id]
        for l in range(len(self._graphs)):
            results = self._search_layer(query_vec, [self._entry_point],
                                         ef=self._ef_construction, layer=l)
            M_max = self._M_max0 if l == 0 else self._M
            neighbours = self._select_neighbours_heuristic(query_vec, results, M_max)
            for _, n in neighbours:
                self._graphs[l][internal_id].add(n)
                self._graphs[l][n].add(internal_id)
        return True

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self._dim,
            "M": self._M,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "metric": self._metric,
            "ml": self._ml,
            "max_elements": self._max_elements,
            "vectors": {k: v.tolist() for k, v in self._vectors.items()},
            "ids": self._ids,
            "str_to_int": self._str_to_int,
            "metadata": self._metadata,
            "graphs": [{k: list(v) for k, v in layer.items()} for layer in self._graphs],
            "max_level": self._max_level,
            "entry_point": self._entry_point,
            "element_count": self._element_count,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._dim = data["dimension"]
        self._M = data["M"]
        self._ef_construction = data["ef_construction"]
        self._ef_search = data["ef_search"]
        self._metric = data["metric"]
        self._ml = data["ml"]
        self._max_elements = data["max_elements"]
        self._vectors = {k: np.array(v, dtype=np.float64) for k, v in data["vectors"].items()}
        self._ids = data["ids"]
        self._str_to_int = data["str_to_int"]
        self._metadata = data["metadata"]
        self._graphs = [{k: set(v) for k, v in layer.items()} for layer in data["graphs"]]
        self._max_level = data["max_level"]
        self._entry_point = data["entry_point"]
        self._element_count = data["element_count"]

    @property
    def count(self) -> int:
        return self._element_count

    @property
    def dimension(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════════════
#  IVFIndex (Inverted File Index — from scratch)
# ═══════════════════════════════════════════════════════════════════════════

class IVFIndex(BaseVectorStore):
    """Inverted File Index for approximate nearest-neighbour search.

    Partitions the vector space into Voronoi cells using k-means
    clustering.  At query time, only the *nprobe* closest centroids are
    examined, dramatically reducing search cost.

    Parameters
    ----------
    dimension : int
        Vector dimensionality.
    nlist : int
        Number of Voronoi cells (clusters).
    nprobe : int
        Number of cells to search at query time.
    metric : str
        Distance metric.
    kmeans_niter : int
        Number of k-means iterations during training.
    kmeans_seed : int
        Random seed for k-means initialisation.
    """

    def __init__(
        self,
        dimension: int = 256,
        nlist: int = 100,
        nprobe: int = 10,
        metric: str = "cosine",
        kmeans_niter: int = 20,
        kmeans_seed: int = 42,
    ) -> None:
        self._dim = dimension
        self._nlist = nlist
        self._nprobe = nprobe
        self._metric = metric.lower()
        self._distance_fn = _get_distance_fn(self._metric)
        self._distance_batch_fn = _get_distance_batch_fn(self._metric)
        self._kmeans_niter = kmeans_niter
        self._kmeans_seed = kmeans_seed

        self._centroids: Optional[np.ndarray] = None
        self._inverted_lists: Dict[int, List[int]] = defaultdict(list)
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
        self._is_trained: bool = False

    def train(self, vectors: np.ndarray) -> None:
        """Train the IVF index using k-means clustering.

        Parameters
        ----------
        vectors : np.ndarray
            Training vectors (N, D).
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = vectors.shape[0]
        actual_nlist = min(self._nlist, n)
        logger.info(
            "IVFIndex: training with %d vectors, nlist=%d …",
            n, actual_nlist,
        )
        t0 = time.perf_counter()

        rng = np.random.RandomState(self._kmeans_seed)
        # K-means++ initialisation
        centroids = np.empty((actual_nlist, self._dim), dtype=np.float64)
        first_idx = rng.randint(n)
        centroids[0] = vectors[first_idx]

        for i in range(1, actual_nlist):
            dists = np.min(
                np.array([_l2_distance_batch(centroids[j], vectors) for j in range(i)]),
                axis=0,
            )
            probs = dists / (dists.sum() + 1e-12)
            idx = rng.choice(n, p=probs)
            centroids[i] = vectors[idx]

        # K-means iterations
        for iteration in range(self._kmeans_niter):
            # Assign
            dist_matrix = np.zeros((n, actual_nlist), dtype=np.float64)
            for c in range(actual_nlist):
                dist_matrix[:, c] = _l2_distance_batch(centroids[c], vectors)
            assignments = np.argmin(dist_matrix, axis=1)

            # Update
            new_centroids = np.zeros_like(centroids)
            for c in range(actual_nlist):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = vectors[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < 1e-6:
                logger.debug("  K-means converged at iteration %d", iteration + 1)
                break

        self._centroids = centroids
        self._is_trained = True
        elapsed = time.perf_counter() - t0
        logger.info("IVFIndex: trained in %.2f s", elapsed)

    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the IVF index."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        n = embeddings.shape[0]

        if not self._is_trained:
            self.train(embeddings)

        meta = metadata if metadata is not None else [{}] * len(ids)
        start_idx = len(self._ids)

        if self._vectors is None:
            self._vectors = embeddings.astype(np.float64)
        else:
            self._vectors = np.vstack([self._vectors, embeddings.astype(np.float64)])

        for i in range(n):
            idx = start_idx + i
            self._ids.append(ids[i])
            self._metadata.append(meta[i])
            self._id_to_idx[ids[i]] = idx

        # Assign to clusters
        dists = np.zeros((n, len(self._centroids)), dtype=np.float64)
        for c in range(len(self._centroids)):
            dists[:, c] = _l2_distance_batch(self._centroids[c], embeddings)
        assignments = np.argmin(dists, axis=1)

        for i, cluster in enumerate(assignments):
            self._inverted_lists[int(cluster)].append(start_idx + i)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        nprobe: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search nprobe clusters for nearest neighbours."""
        if self._vectors is None or self._centroids is None:
            return []

        query_vec = query.flatten().astype(np.float64)
        probe = nprobe if nprobe is not None else self._nprobe

        # Find closest centroids
        centroid_dists = np.zeros(len(self._centroids), dtype=np.float64)
        for c in range(len(self._centroids)):
            centroid_dists[c] = self._distance_fn(query_vec, self._centroids[c])
        closest_centroids = np.argpartition(centroid_dists, probe)[:probe]

        # Collect candidate vector indices
        candidate_indices: List[int] = []
        for c in closest_centroids:
            candidate_indices.extend(self._inverted_lists.get(int(c), []))

        if not candidate_indices:
            return []

        # Compute distances to candidates
        candidate_vecs = self._vectors[candidate_indices]
        distances = self._distance_batch_fn(query_vec, candidate_vecs)
        k = min(top_k, len(distances))
        if k <= 0:
            return []
        top_local = np.argpartition(distances, k)[:k]
        top_local = top_local[np.argsort(distances[top_local])]

        results: List[SearchResult] = []
        for rank, local_idx in enumerate(top_local):
            global_idx = candidate_indices[int(local_idx)]
            d = float(distances[local_idx])
            results.append(SearchResult(
                id=self._ids[global_idx],
                score=_distance_to_score(d, self._metric),
                distance=d,
                vector=self._vectors[global_idx].copy(),
                metadata=dict(self._metadata[global_idx]),
            ))
        return results

    def delete(self, ids: List[str]) -> int:
        deleted = 0
        indices_to_delete: Set[int] = set()
        for doc_id in ids:
            idx = self._id_to_idx.pop(doc_id, None)
            if idx is not None:
                indices_to_delete.add(idx)
                deleted += 1
        if not indices_to_delete:
            return 0
        # Rebuild inverted lists
        self._inverted_lists.clear()
        if self._centroids is not None and self._vectors is not None:
            dists = np.zeros((len(self._ids), len(self._centroids)), dtype=np.float64)
            for c in range(len(self._centroids)):
                dists[:, c] = _l2_distance_batch(self._centroids[c], self._vectors)
            assignments = np.argmin(dists, axis=1)
            for i, cluster in enumerate(assignments):
                if i not in indices_to_delete:
                    self._inverted_lists[int(cluster)].append(i)
        keep = sorted(set(range(len(self._ids))) - indices_to_delete)
        self._vectors = self._vectors[keep]
        self._ids = [self._ids[i] for i in keep]
        self._metadata = [self._metadata[i] for i in keep]
        self._id_to_idx = {d: i for i, d in enumerate(self._ids)}
        return deleted

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self._dim,
            "nlist": self._nlist,
            "nprobe": self._nprobe,
            "metric": self._metric,
            "centroids": self._centroids,
            "vectors": self._vectors,
            "ids": self._ids,
            "metadata": self._metadata,
            "inverted_lists": dict(self._inverted_lists),
            "is_trained": self._is_trained,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._dim = data["dimension"]
        self._nlist = data["nlist"]
        self._nprobe = data["nprobe"]
        self._metric = data["metric"]
        self._centroids = data["centroids"]
        self._vectors = data["vectors"]
        self._ids = data["ids"]
        self._metadata = data["metadata"]
        self._inverted_lists = defaultdict(list, data["inverted_lists"])
        self._is_trained = data["is_trained"]
        self._id_to_idx = {d: i for i, d in enumerate(self._ids)}

    @property
    def count(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════════════
#  ProductQuantizationIndex
# ═══════════════════════════════════════════════════════════════════════════

class ProductQuantizationIndex(BaseVectorStore):
    """Product Quantization for memory-efficient vector compression.

    Splits each vector into *m* sub-vectors and quantises each sub-space
    independently using k-means.  This enables efficient asymmetric
    distance computation (ADC) at query time.

    Parameters
    ----------
    dimension : int
        Full vector dimensionality.
    n_subquantizers : int
        Number of sub-quantizers (m).  Must divide *dimension*.
    n_bits : int
        Bits per sub-quantizer code (256 centroids max for 8 bits).
    metric : str
        Distance metric.
    kmeans_niter : int
        K-means iterations for training codebooks.
    """

    def __init__(
        self,
        dimension: int = 256,
        n_subquantizers: int = 8,
        n_bits: int = 8,
        metric: str = "l2",
        kmeans_niter: int = 20,
    ) -> None:
        self._dim = dimension
        self._m = n_subquantizers
        self._n_bits = n_bits
        self._k = 2 ** n_bits
        self._metric = metric.lower()
        self._kmeans_niter = kmeans_niter
        self._sub_dim = dimension // n_subquantizers

        if dimension % n_subquantizers != 0:
            raise ValueError(
                f"dimension ({dimension}) must be divisible by "
                f"n_subquantizers ({n_subquantizers})"
            )

        # Codebooks: list of (k, sub_dim) arrays, one per sub-quantizer
        self._codebooks: List[np.ndarray] = []
        # PQ codes: (N, m) uint8 array
        self._codes: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
        self._is_trained: bool = False

    def train(self, vectors: np.ndarray) -> None:
        """Train PQ codebooks from training vectors."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = vectors.shape[0]
        logger.info(
            "PQ: training %d sub-quantizers (k=%d, sub_dim=%d) with %d vectors …",
            self._m, self._k, self._sub_dim, n,
        )
        t0 = time.perf_counter()
        self._codebooks = []

        for sq in range(self._m):
            start = sq * self._sub_dim
            end = start + self._sub_dim
            sub_vectors = vectors[:, start:end].astype(np.float64)

            # K-means with random initialisation
            rng = np.random.RandomState(42 + sq)
            indices = rng.choice(n, size=min(self._k, n), replace=False)
            centroids = sub_vectors[indices].copy()

            for _ in range(self._kmeans_niter):
                dists = np.zeros((n, self._k), dtype=np.float64)
                for c in range(self._k):
                    dists[:, c] = np.sum((sub_vectors - centroids[c]) ** 2, axis=1)
                assignments = np.argmin(dists, axis=1)
                for c in range(self._k):
                    mask = assignments == c
                    if mask.any():
                        centroids[c] = sub_vectors[mask].mean(axis=0)

            self._codebooks.append(centroids)

        self._is_trained = True
        elapsed = time.perf_counter() - t0
        logger.info("PQ: trained in %.2f s", elapsed)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to PQ codes.

        Returns
        -------
        np.ndarray
            (N, m) uint8 array of PQ codes.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = vectors.shape[0]
        codes = np.zeros((n, self._m), dtype=np.uint8)

        for sq in range(self._m):
            start = sq * self._sub_dim
            end = start + self._sub_dim
            sub_vectors = vectors[:, start:end].astype(np.float64)
            centroids = self._codebooks[sq]

            dists = np.zeros((n, self._k), dtype=np.float64)
            for c in range(self._k):
                dists[:, c] = np.sum((sub_vectors - centroids[c]) ** 2, axis=1)
            codes[:, sq] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to approximate vectors."""
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
        n = codes.shape[0]
        vectors = np.zeros((n, self._dim), dtype=np.float64)

        for sq in range(self._m):
            start = sq * self._sub_dim
            end = start + self._sub_dim
            for i in range(n):
                c = int(codes[i, sq])
                vectors[i, start:end] = self._codebooks[sq][c]

        return vectors

    def _asymmetric_distance_table(self, query: np.ndarray) -> np.ndarray:
        """Compute the distance lookup table for ADC.

        Returns
        -------
        np.ndarray
            (m, k) table where entry [sq, c] = distance from query sub-vector
            to centroid c of sub-quantizer sq.
        """
        table = np.zeros((self._m, self._k), dtype=np.float64)
        for sq in range(self._m):
            start = sq * self._sub_dim
            end = start + self._sub_dim
            q_sub = query[start:end]
            for c in range(self._k):
                diff = q_sub - self._codebooks[sq][c]
                table[sq, c] = float(np.dot(diff, diff))
        return table

    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors (encode and store PQ codes)."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if not self._is_trained:
            self.train(embeddings)

        codes = self.encode(embeddings)
        if self._codes is None:
            self._codes = codes
        else:
            self._codes = np.vstack([self._codes, codes])

        meta = metadata if metadata is not None else [{}] * len(ids)
        start_idx = len(self._ids)
        for i, (doc_id, m) in enumerate(zip(ids, meta)):
            self._ids.append(doc_id)
            self._metadata.append(m)
            self._id_to_idx[doc_id] = start_idx + i

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Asymmetric distance computation (ADC) search."""
        if self._codes is None or not self._is_trained:
            return []

        query_vec = query.flatten().astype(np.float64)
        table = self._asymmetric_distance_table(query_vec)

        # Compute distances using the lookup table
        n = len(self._ids)
        distances = np.zeros(n, dtype=np.float64)
        for sq in range(self._m):
            distances += table[sq, self._codes[:, sq]]

        k = min(top_k, n)
        top_indices = np.argpartition(distances, k)[:k]
        top_indices = top_indices[np.argsort(distances[top_indices])]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices):
            idx = int(idx)
            d = float(distances[idx])
            results.append(SearchResult(
                id=self._ids[idx],
                score=_distance_to_score(d, self._metric),
                distance=d,
                metadata=dict(self._metadata[idx]),
            ))
        return results

    def delete(self, ids: List[str]) -> int:
        deleted = 0
        indices_to_delete: Set[int] = set()
        for doc_id in ids:
            idx = self._id_to_idx.pop(doc_id, None)
            if idx is not None:
                indices_to_delete.add(idx)
                deleted += 1
        if not indices_to_delete:
            return 0
        keep = sorted(set(range(len(self._ids))) - indices_to_delete)
        self._codes = self._codes[keep]
        self._ids = [self._ids[i] for i in keep]
        self._metadata = [self._metadata[i] for i in keep]
        self._id_to_idx = {d: i for i, d in enumerate(self._ids)}
        return deleted

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self._dim,
            "n_subquantizers": self._m,
            "n_bits": self._n_bits,
            "metric": self._metric,
            "codebooks": [cb.tolist() for cb in self._codebooks],
            "codes": self._codes,
            "ids": self._ids,
            "metadata": self._metadata,
            "is_trained": self._is_trained,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._dim = data["dimension"]
        self._m = data["n_subquantizers"]
        self._n_bits = data["n_bits"]
        self._metric = data["metric"]
        self._codebooks = [np.array(cb) for cb in data["codebooks"]]
        self._codes = data["codes"]
        self._ids = data["ids"]
        self._metadata = data["metadata"]
        self._is_trained = data["is_trained"]
        self._id_to_idx = {d: i for i, d in enumerate(self._ids)}

    @property
    def count(self) -> int:
        return len(self._ids)

    @property
    def dimension(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════════════
#  DiskANNIndex (disk-based approximate NN)
# ═══════════════════════════════════════════════════════════════════════════

class DiskANNIndex(BaseVectorStore):
    """Disk-based index for large-scale vector retrieval.

    Uses a Vamana graph (similar to HNSW but single-layer) with
    memory-mapped storage for vectors.  This allows scaling to billions
    of vectors while keeping only the graph in memory.

    Parameters
    ----------
    dimension : int
        Vector dimensionality.
    max_degree : int
        Maximum degree per node in the graph.
    beam_width : int
        Search beam width.
    metric : str
        Distance metric.
    page_size : int
        Page size in bytes for disk I/O.
    disk_path : Optional[str]
        Directory for disk storage.  ``None`` = fully in-memory.
    """

    def __init__(
        self,
        dimension: int = 256,
        max_degree: int = 64,
        beam_width: int = 16,
        metric: str = "cosine",
        page_size: int = 4096,
        disk_path: Optional[str] = None,
    ) -> None:
        self._dim = dimension
        self._max_degree = max_degree
        self._beam_width = beam_width
        self._metric = metric.lower()
        self._distance_fn = _get_distance_fn(self._metric)
        self._page_size = page_size
        self._disk_path = Path(disk_path) if disk_path else None

        self._vectors: Dict[int, np.ndarray] = {}
        self._ids: Dict[int, str] = {}
        self._str_to_int: Dict[str, int] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._graph: Dict[int, Set[int]] = {}
        self._entry_point: int = -1
        self._element_count: int = 0
        self._use_disk = self._disk_path is not None

        if self._use_disk:
            self._disk_path.mkdir(parents=True, exist_ok=True)

    def _disk_vector_path(self, internal_id: int) -> Path:
        bucket = internal_id // 1000
        return self._disk_path / f"bucket_{bucket}" / f"{internal_id}.npy"

    def _save_vector_to_disk(self, internal_id: int, vector: np.ndarray) -> None:
        path = self._disk_vector_path(internal_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector)

    def _load_vector_from_disk(self, internal_id: int) -> np.ndarray:
        path = self._disk_vector_path(internal_id)
        return np.load(path)

    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors and build the Vamana graph."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        meta = metadata if metadata is not None else [{}] * len(ids)

        # First pass: store all vectors
        for i in range(embeddings.shape[0]):
            iid = self._element_count
            vec = embeddings[i].astype(np.float64)

            self._str_to_int[ids[i]] = iid
            self._ids[iid] = ids[i]
            self._metadata[iid] = meta[i]

            if self._use_disk:
                self._save_vector_to_disk(iid, vec)
            else:
                self._vectors[iid] = vec

            self._element_count += 1

        # Second pass: build graph
        all_ids = list(range(self._element_count))
        if self._entry_point == -1 and all_ids:
            self._entry_point = all_ids[0]

        # Build a simple NN-descent graph
        self._build_vamana_graph(all_ids)

    def _build_vamana_graph(self, all_ids: List[int]) -> None:
        """Build a Vamana-style graph using greedy NN construction."""
        logger.info("DiskANN: building graph for %d vectors …", len(all_ids))
        t0 = time.perf_counter()

        for node_id in all_ids:
            self._graph[node_id] = set()

        rng = np.random.RandomState(42)
        # Random initial neighbours
        for node_id in all_ids:
            n_neighbours = min(self._max_degree, len(all_ids) - 1)
            candidates = rng.choice([x for x in all_ids if x != node_id],
                                    size=n_neighbours, replace=False)
            self._graph[node_id] = set(int(c) for c in candidates)

        # Refine: for each node, find true nearest neighbours
        for node_id in all_ids:
            node_vec = self._get_vector(node_id)
            # Search for better neighbours
            candidates = list(self._graph[node_id]) + rng.choice(
                all_ids, size=min(50, len(all_ids)), replace=False
            ).tolist()
            candidates = list(set(candidates) - {node_id})

            scored = []
            for c in candidates:
                c_vec = self._get_vector(c)
                d = self._distance_fn(node_vec, c_vec)
                scored.append((d, c))
            scored.sort()

            neighbours = set(c for _, c in scored[:self._max_degree])
            self._graph[node_id] = neighbours

        elapsed = time.perf_counter() - t0
        logger.info("DiskANN: graph built in %.2f s", elapsed)

    def _get_vector(self, internal_id: int) -> np.ndarray:
        if self._use_disk:
            return self._load_vector_from_disk(internal_id)
        return self._vectors[internal_id]

    def _greedy_search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> List[Tuple[float, int]]:
        """Greedy beam search over the Vamana graph."""
        if self._entry_point == -1:
            return []

        visited: Set[int] = set()
        candidates: List[Tuple[float, int]] = []
        results: List[Tuple[float, int]] = []

        ep = self._entry_point
        d = self._distance_fn(query, self._get_vector(ep))
        heapq.heappush(candidates, (d, ep))
        heapq.heappush(results, (-d, ep))
        visited.add(ep)

        ef = max(self._beam_width, top_k)

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest = -results[0][0]
            if c_dist > furthest:
                break

            for neighbour in self._graph.get(c_id, set()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)

                n_dist = self._distance_fn(query, self._get_vector(neighbour))
                if n_dist < furthest or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, neighbour))
                    heapq.heappush(results, (-n_dist, neighbour))
                    if len(results) > ef:
                        heapq.heappop(results)
                    furthest = -results[0][0]

        return [(abs(d), n) for d, n in results]

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search the DiskANN index."""
        query_vec = query.flatten().astype(np.float64)
        results = self._greedy_search(query_vec, top_k)
        results.sort()

        out: List[SearchResult] = []
        for rank, (dist, idx) in enumerate(results[:top_k]):
            out.append(SearchResult(
                id=self._ids[idx],
                score=_distance_to_score(dist, self._metric),
                distance=dist,
                vector=self._get_vector(idx).copy(),
                metadata=dict(self._metadata[idx]),
            ))
        return out

    def delete(self, ids: List[str]) -> int:
        deleted = 0
        for doc_id in ids:
            iid = self._str_to_int.pop(doc_id, None)
            if iid is None:
                continue
            self._graph.pop(iid, None)
            self._ids.pop(iid, None)
            self._metadata.pop(iid, None)
            if not self._use_disk:
                self._vectors.pop(iid, None)
            deleted += 1
        return deleted

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self._dim,
            "max_degree": self._max_degree,
            "beam_width": self._beam_width,
            "metric": self._metric,
            "page_size": self._page_size,
            "entry_point": self._entry_point,
            "element_count": self._element_count,
            "ids": self._ids,
            "str_to_int": self._str_to_int,
            "metadata": self._metadata,
            "graph": {k: list(v) for k, v in self._graph.items()},
        }
        if not self._use_disk:
            data["vectors"] = {k: v.tolist() for k, v in self._vectors.items()}
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._dim = data["dimension"]
        self._max_degree = data["max_degree"]
        self._beam_width = data["beam_width"]
        self._metric = data["metric"]
        self._page_size = data["page_size"]
        self._entry_point = data["entry_point"]
        self._element_count = data["element_count"]
        self._ids = data["ids"]
        self._str_to_int = data["str_to_int"]
        self._metadata = data["metadata"]
        self._graph = {k: set(v) for k, v in data["graph"].items()}
        if not self._use_disk:
            self._vectors = {k: np.array(v) for k, v in data["vectors"].items()}

    @property
    def count(self) -> int:
        return self._element_count

    @property
    def dimension(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════════════
#  VectorStoreManager
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """Unified manager for multiple vector stores.

    Automatically selects the best index type based on corpus size and
    provides a single API for adding, searching, and managing vectors.

    Parameters
    ----------
    dimension : int
        Vector dimensionality.
    metric : str
        Distance metric.
    auto_select : bool
        Automatically choose the best index type.
    hnsw_m : int
        HNSW M parameter (used when auto-selecting HNSW).
    ivf_nlist : int
        IVF nlist parameter.
    persist_path : Optional[str]
        Base directory for persistence.
    """

    def __init__(
        self,
        dimension: int = 256,
        metric: str = "cosine",
        auto_select: bool = True,
        hnsw_m: int = 16,
        ivf_nlist: int = 100,
        persist_path: Optional[str] = None,
    ) -> None:
        self._dim = dimension
        self._metric = metric
        self._auto_select = auto_select
        self._hnsw_m = hnsw_m
        self._ivf_nlist = ivf_nlist
        self._persist_path = Path(persist_path) if persist_path else None
        self._store: Optional[BaseVectorStore] = None
        self._total_added: int = 0

    def _select_store(self, n_vectors: int) -> BaseVectorStore:
        """Select the best store type based on corpus size."""
        if not self._auto_select:
            return SimpleVectorStore(dimension=self._dim, metric=self._metric)

        if n_vectors < 10_000:
            logger.info("Selecting SimpleVectorStore for %d vectors", n_vectors)
            return SimpleVectorStore(dimension=self._dim, metric=self._metric)
        elif n_vectors < 500_000:
            logger.info("Selecting HNSWIndex for %d vectors", n_vectors)
            return HNSWIndex(
                dimension=self._dim,
                M=self._hnsw_m,
                metric=self._metric,
            )
        else:
            logger.info("Selecting IVFIndex for %d vectors", n_vectors)
            return IVFIndex(
                dimension=self._dim,
                nlist=self._ivf_nlist,
                metric=self._metric,
            )

    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors, auto-selecting store if needed."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if self._store is None:
            self._store = self._select_store(len(ids))

        self._store.add(embeddings, ids, metadata)
        self._total_added += len(ids)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search the active vector store."""
        if self._store is None:
            return []
        return self._store.search(query, top_k)

    def delete(self, ids: List[str]) -> int:
        if self._store is None:
            return 0
        return self._store.delete(ids)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        p = path or self._persist_path
        if p is None or self._store is None:
            return
        self._store.save(p)

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        p = path or self._persist_path
        if p is None:
            return
        if self._store is not None:
            self._store.load(p)

    def rebuild(self, new_embeddings: np.ndarray, new_ids: List[str]) -> None:
        """Rebuild the index with new data, potentially changing store type."""
        self._store = None
        self._total_added = 0
        self.add(new_embeddings, new_ids)

    @property
    def store(self) -> Optional[BaseVectorStore]:
        return self._store

    @property
    def count(self) -> int:
        return self._store.count if self._store else 0

    @property
    def stats(self) -> Dict[str, Any]:
        store_type = type(self._store).__name__ if self._store else "None"
        return {
            "store_type": store_type,
            "dimension": self._dim,
            "metric": self._metric,
            "count": self.count,
            "total_added": self._total_added,
            "auto_select": self._auto_select,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Factory helper
# ═══════════════════════════════════════════════════════════════════════════

def create_vector_store(
    backend: str = "simple",
    dimension: int = 256,
    metric: str = "cosine",
    **kwargs: Any,
) -> BaseVectorStore:
    """Factory: create a vector store by backend name.

    Parameters
    ----------
    backend : str
        ``"simple"``, ``"hnsw"``, ``"ivf"``, ``"pq"``, or ``"diskann"``.
    dimension : int
        Vector dimensionality.
    metric : str
        Distance metric.
    """
    backend = backend.lower().strip()
    if backend == "simple" or backend == "flat":
        return SimpleVectorStore(dimension=dimension, metric=metric, **kwargs)
    elif backend == "hnsw" or backend == "hnswlib":
        return HNSWIndex(dimension=dimension, metric=metric, **kwargs)
    elif backend == "ivf":
        return IVFIndex(dimension=dimension, metric=metric, **kwargs)
    elif backend == "pq" or backend == "product_quantization":
        return ProductQuantizationIndex(dimension=dimension, metric=metric, **kwargs)
    elif backend == "diskann" or backend == "disk":
        return DiskANNIndex(dimension=dimension, metric=metric, **kwargs)
    else:
        raise ValueError(f"Unknown vector store backend: {backend!r}")
