"""Nexus-LLM Encryption Utilities.

Provides symmetric encryption and decryption utilities using the
cryptography library (or a fallback base64 encoding when unavailable).
"""

import base64
import hashlib
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import cryptography; fall back to basic encoding
try:
    from cryptography.fernet import Fernet
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False
    logger.warning("cryptography library not available; using base64 fallback (NOT secure)")


class EncryptionManager:
    """Manages symmetric encryption operations.

    Uses Fernet (AES-128-CBC) when the cryptography library is
    available, otherwise falls back to base64 encoding (insecure).

    Attributes:
        algorithm: The encryption algorithm in use.
    """

    def __init__(self, key: Optional[bytes] = None) -> None:
        if _HAS_CRYPTO:
            self._key = key or Fernet.generate_key()
            self._fernet = Fernet(self._key)
            self._algorithm = "Fernet/AES-128-CBC"
        else:
            self._key = key or b"nexus-llm-fallback-key"
            self._fernet = None
            self._algorithm = "base64-fallback"
        logger.debug("EncryptionManager initialized with algorithm: %s", self._algorithm)

    @property
    def algorithm(self) -> str:
        """The encryption algorithm in use."""
        return self._algorithm

    def encrypt(self, data: str) -> str:
        """Encrypt a string.

        Args:
            data: Plain text to encrypt.

        Returns:
            Base64-encoded ciphertext string.
        """
        if self._fernet:
            encrypted = self._fernet.encrypt(data.encode("utf-8"))
            return encrypted.decode("utf-8")
        # Fallback: base64 encode (NOT secure)
        encoded = base64.b64encode(data.encode("utf-8"))
        return encoded.decode("utf-8")

    def decrypt(self, token: str) -> str:
        """Decrypt a previously encrypted string.

        Args:
            token: Base64-encoded ciphertext.

        Returns:
            Decrypted plain text.
        """
        if self._fernet:
            decrypted = self._fernet.decrypt(token.encode("utf-8"))
            return decrypted.decode("utf-8")
        # Fallback: base64 decode
        decoded = base64.b64decode(token.encode("utf-8"))
        return decoded.decode("utf-8")

    def hash(self, data: str, algorithm: str = "sha256") -> str:
        """Compute a hash of the given data.

        Args:
            data: Input string.
            algorithm: Hash algorithm (sha256, sha512, md5).

        Returns:
            Hex-encoded hash string.
        """
    def hash(self, data: str, algorithm: str = "sha256") -> str:
        h = hashlib.new(algorithm)
        h.update(data.encode("utf-8"))
        return h.hexdigest()

    def generate_key(self) -> bytes:
        """Generate a new encryption key.

        Returns:
            A new Fernet key (or a deterministic key in fallback mode).
        """
        if _HAS_CRYPTO:
            return Fernet.generate_key()
        return os.urandom(32)

    def export_key(self) -> str:
        """Export the current key as a base64 string.

        Returns:
            Base64-encoded key.
        """
        return base64.b64encode(self._key).decode("utf-8")

    @classmethod
    def from_key_string(cls, key_string: str) -> "EncryptionManager":
        """Create an EncryptionManager from a base64-encoded key string.

        Args:
            key_string: Base64-encoded key.

        Returns:
            A new EncryptionManager instance.
        """
        key = base64.b64decode(key_string.encode("utf-8"))
        return cls(key=key)


def encrypt_data(data: str, key: Optional[bytes] = None) -> Dict[str, str]:
    """Convenience function to encrypt data.

    Args:
        data: Plain text to encrypt.
        key: Optional encryption key.

    Returns:
        Dictionary with 'ciphertext' and 'key' (base64-encoded).
    """
    manager = EncryptionManager(key=key)
    return {
        "ciphertext": manager.encrypt(data),
        "key": manager.export_key(),
        "algorithm": manager.algorithm,
    }


def decrypt_data(ciphertext: str, key_string: str) -> str:
    """Convenience function to decrypt data.

    Args:
        ciphertext: Encrypted text.
        key_string: Base64-encoded key.

    Returns:
        Decrypted plain text.
    """
    manager = EncryptionManager.from_key_string(key_string)
    return manager.decrypt(ciphertext)
