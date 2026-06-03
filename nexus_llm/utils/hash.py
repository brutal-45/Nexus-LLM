"""Hashing utilities: file hashing, content hashing, consistent hashing."""

import hashlib
import struct
import logging
from typing import Optional, List, Any, Union

logger = logging.getLogger(__name__)


def file_hash(
    filepath: str,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """Compute the hash of a file.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm name (md5, sha1, sha256, sha512).
        chunk_size: Size of chunks to read at a time.

    Returns:
        Hexadecimal hash string.
    """
    hash_func = hashlib.new(algorithm)

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_func.update(chunk)

    return hash_func.hexdigest()


def content_hash(
    content: Union[str, bytes],
    algorithm: str = "sha256",
    encoding: str = "utf-8",
) -> str:
    """Compute the hash of string or bytes content.

    Args:
        content: Content to hash.
        algorithm: Hash algorithm name.
        encoding: Encoding for string content.

    Returns:
        Hexadecimal hash string.
    """
    if isinstance(content, str):
        content = content.encode(encoding)

    hash_func = hashlib.new(algorithm)
    hash_func.update(content)
    return hash_func.hexdigest()


def consistent_hash(
    key: str,
    num_buckets: int = 16,
    algorithm: str = "md5",
) -> int:
    """Compute a consistent hash for distributing keys to buckets.

    Uses hash-based key distribution that is consistent regardless
    of the number of previous/future assignments.

    Args:
        key: Key string to hash.
        num_buckets: Number of buckets to distribute across.
        algorithm: Hash algorithm.

    Returns:
        Bucket index (0 to num_buckets-1).
    """
    hash_value = hashlib.new(algorithm, key.encode("utf-8")).digest()
    # Take first 4 bytes as unsigned integer
    int_value = struct.unpack("<I", hash_value[:4])[0]
    return int_value % num_buckets


class ConsistentHashRing:
    """Consistent hashing ring for distributing keys across nodes.

    Uses virtual nodes to ensure even distribution.
    """

    def __init__(
        self,
        nodes: Optional[List[str]] = None,
        num_virtual_nodes: int = 100,
        algorithm: str = "md5",
    ):
        """Initialize the consistent hash ring.

        Args:
            nodes: Initial list of node names.
            num_virtual_nodes: Number of virtual nodes per physical node.
            algorithm: Hash algorithm.
        """
        self.num_virtual_nodes = num_virtual_nodes
        self.algorithm = algorithm
        self._ring: List[tuple] = []  # (hash_value, node_name)
        self._nodes = set()

        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node: str):
        """Add a node to the ring.

        Args:
            node: Node name/identifier.
        """
        if node in self._nodes:
            return

        self._nodes.add(node)
        for i in range(self.num_virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self._ring.append((hash_value, node))

        self._ring.sort(key=lambda x: x[0])

    def remove_node(self, node: str):
        """Remove a node from the ring.

        Args:
            node: Node name/identifier.
        """
        if node not in self._nodes:
            return

        self._nodes.discard(node)
        self._ring = [(h, n) for h, n in self._ring if n != node]

    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a given key.

        Args:
            key: Key to look up.

        Returns:
            Node name, or None if the ring is empty.
        """
        if not self._ring:
            return None

        hash_value = self._hash(key)

        # Binary search for the first node with hash >= key_hash
        low, high = 0, len(self._ring)
        while low < high:
            mid = (low + high) // 2
            if self._ring[mid][0] < hash_value:
                low = mid + 1
            else:
                high = mid

        # Wrap around if necessary
        if low >= len(self._ring):
            low = 0

        return self._ring[low][1]

    def get_nodes(self, key: str, count: int = 3) -> List[str]:
        """Get multiple nodes for a key (for replication).

        Args:
            key: Key to look up.
            count: Number of distinct nodes to return.

        Returns:
            List of node names.
        """
        if not self._ring:
            return []

        hash_value = self._hash(key)
        nodes = []
        seen = set()

        low, high = 0, len(self._ring)
        while low < high:
            mid = (low + high) // 2
            if self._ring[mid][0] < hash_value:
                low = mid + 1
            else:
                high = mid

        idx = low if low < len(self._ring) else 0

        for i in range(len(self._ring)):
            node = self._ring[(idx + i) % len(self._ring)][1]
            if node not in seen:
                seen.add(node)
                nodes.append(node)
                if len(nodes) >= count:
                    break

        return nodes

    def _hash(self, key: str) -> int:
        """Compute hash for a key, returning an integer."""
        digest = hashlib.new(self.algorithm, key.encode("utf-8")).digest()
        return struct.unpack("<I", digest[:4])[0]

    @property
    def num_nodes(self) -> int:
        """Number of physical nodes in the ring."""
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"ConsistentHashRing(nodes={len(self._nodes)}, vnodes={self.num_virtual_nodes})"


def hash_dict(data: dict, algorithm: str = "sha256") -> str:
    """Compute a deterministic hash of a dictionary.

    Args:
        data: Dictionary to hash.
        algorithm: Hash algorithm.

    Returns:
        Hexadecimal hash string.
    """
    import json
    serialized = json.dumps(data, sort_keys=True, default=str)
    return content_hash(serialized, algorithm)


def hash_list(items: list, algorithm: str = "sha256") -> str:
    """Compute a deterministic hash of a list.

    Args:
        items: List to hash.
        algorithm: Hash algorithm.

    Returns:
        Hexadecimal hash string.
    """
    import json
    serialized = json.dumps(items, default=str)
    return content_hash(serialized, algorithm)
