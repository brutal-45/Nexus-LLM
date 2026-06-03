"""Crypto: hash computation (MD5, SHA256), file integrity verification."""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_md5(data: str | bytes, encoding: str = "utf-8") -> str:
    """Compute MD5 hash of a string or bytes.

    Args:
        data: Input data.
        encoding: Encoding to use for string input.

    Returns:
        Hexadecimal MD5 hash string.
    """
    if isinstance(data, str):
        data = data.encode(encoding)
    return hashlib.md5(data).hexdigest()


def compute_sha256(data: str | bytes, encoding: str = "utf-8") -> str:
    """Compute SHA-256 hash of a string or bytes.

    Args:
        data: Input data.
        encoding: Encoding to use for string input.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    if isinstance(data, str):
        data = data.encode(encoding)
    return hashlib.sha256(data).hexdigest()


def compute_sha1(data: str | bytes, encoding: str = "utf-8") -> str:
    """Compute SHA-1 hash of a string or bytes."""
    if isinstance(data, str):
        data = data.encode(encoding)
    return hashlib.sha1(data).hexdigest()


def compute_sha512(data: str | bytes, encoding: str = "utf-8") -> str:
    """Compute SHA-512 hash of a string or bytes."""
    if isinstance(data, str):
        data = data.encode(encoding)
    return hashlib.sha512(data).hexdigest()


def compute_file_hash(
    filepath: str,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """Compute hash of a file.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm (md5, sha1, sha256, sha512).
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


def verify_file_integrity(
    filepath: str,
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    """Verify file integrity by comparing its hash against an expected value.

    Args:
        filepath: Path to the file.
        expected_hash: Expected hash value (hexadecimal).
        algorithm: Hash algorithm to use.

    Returns:
        True if the file hash matches the expected hash.
    """
    actual_hash = compute_file_hash(filepath, algorithm)
    match = actual_hash.lower() == expected_hash.lower()

    if match:
        logger.debug(f"File integrity verified: {filepath}")
    else:
        logger.warning(
            f"File integrity check FAILED for {filepath}. "
            f"Expected: {expected_hash}, Got: {actual_hash}"
        )

    return match


def compute_hash_with_salt(data: str, salt: str, algorithm: str = "sha256") -> str:
    """Compute a salted hash for password-like data.

    Args:
        data: Input string.
        salt: Salt value.
        algorithm: Hash algorithm.

    Returns:
        Hexadecimal salted hash string.
    """
    salted_data = salt + data
    return hashlib.new(algorithm, salted_data.encode("utf-8")).hexdigest()


def compute_multi_hash(filepath: str, algorithms: Optional[list] = None) -> dict:
    """Compute multiple hash algorithms for a file simultaneously.

    Args:
        filepath: Path to the file.
        algorithms: List of algorithm names. Defaults to ['md5', 'sha256'].

    Returns:
        Dictionary mapping algorithm names to hash strings.
    """
    algorithms = algorithms or ["md5", "sha256"]
    hash_funcs = {alg: hashlib.new(alg) for alg in algorithms}

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            for hf in hash_funcs.values():
                hf.update(chunk)

    return {alg: hf.hexdigest() for alg, hf in hash_funcs.items()}
