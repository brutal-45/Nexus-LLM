"""Nexus-LLM Secrets Management.

Provides secure storage and retrieval of sensitive configuration values
such as API keys, tokens, and passwords, with support for environment
variable overrides and encrypted file-based storage.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from nexus_llm.security.encryption import EncryptionManager

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manages secure storage and retrieval of secrets.

    Supports multiple backends for secret storage:
    - Environment variables (highest priority)
    - Encrypted file storage
    - In-memory secrets

    Example::

        sm = SecretsManager()
        sm.set("api_key", "sk-12345")
        key = sm.get("api_key")
    """

    def __init__(
        self,
        env_prefix: str = "NEXUS_LLM_",
        encrypted_file: Optional[str] = None,
        master_key: Optional[bytes] = None,
    ) -> None:
        """Initialize the SecretsManager.

        Args:
            env_prefix: Prefix for environment variable lookups.
            encrypted_file: Path to an encrypted secrets file.
            master_key: Master encryption key.
        """
        self._env_prefix = env_prefix
        self._encrypted_file = encrypted_file
        self._secrets: Dict[str, str] = {}
        self._encryption = EncryptionManager(key=master_key)
        logger.debug("SecretsManager initialized with env_prefix=%s", env_prefix)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value.

        Lookup order: environment variables -> in-memory -> default.

        Args:
            key: Secret key name.
            default: Default value if not found.

        Returns:
            The secret value, or default if not found.
        """
        # Check environment variables first
        env_key = f"{self._env_prefix}{key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value

        # Check in-memory secrets
        if key in self._secrets:
            return self._secrets[key]

        return default

    def set(self, key: str, value: str, persist: bool = False) -> None:
        """Set a secret value.

        Args:
            key: Secret key name.
            value: Secret value.
            persist: Whether to persist to encrypted file.
        """
        self._secrets[key] = value
        logger.debug("Secret set: %s", key)
        if persist and self._encrypted_file:
            self._save_to_file()

    def delete(self, key: str) -> bool:
        """Delete a secret.

        Args:
            key: Secret key to delete.

        Returns:
            True if the secret was found and deleted.
        """
        if key in self._secrets:
            del self._secrets[key]
            logger.debug("Secret deleted: %s", key)
            return True
        return False

    def has(self, key: str) -> bool:
        """Check if a secret exists.

        Args:
            key: Secret key name.

        Returns:
            True if the secret is available.
        """
        env_key = f"{self._env_prefix}{key.upper()}"
        return os.environ.get(env_key) is not None or key in self._secrets

    def list_keys(self) -> list:
        """List all available secret key names.

        Returns:
            List of secret key names.
        """
        keys = set(self._secrets.keys())
        # Add env var keys with prefix stripped
        for k, v in os.environ.items():
            if k.startswith(self._env_prefix):
                stripped = k[len(self._env_prefix):].lower()
                keys.add(stripped)
        return sorted(keys)

    def require(self, key: str) -> str:
        """Get a required secret, raising an error if missing.

        Args:
            key: Secret key name.

        Returns:
            The secret value.

        Raises:
            KeyError: If the secret is not found.
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required secret '{key}' not found")
        return value

    def _save_to_file(self) -> None:
        """Save secrets to the encrypted file."""
        if not self._encrypted_file:
            return
        try:
            import json
            data = json.dumps(self._secrets)
            encrypted = self._encryption.encrypt(data)
            path = Path(self._encrypted_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encrypted)
            logger.debug("Secrets saved to %s", self._encrypted_file)
        except Exception as exc:
            logger.error("Failed to save secrets: %s", exc)

    def _load_from_file(self) -> None:
        """Load secrets from the encrypted file."""
        if not self._encrypted_file:
            return
        path = Path(self._encrypted_file)
        if not path.exists():
            return
        try:
            import json
            encrypted = path.read_text()
            decrypted = self._encryption.decrypt(encrypted)
            loaded = json.loads(decrypted)
            self._secrets.update(loaded)
            logger.debug("Secrets loaded from %s", self._encrypted_file)
        except Exception as exc:
            logger.error("Failed to load secrets: %s", exc)

    def load(self) -> None:
        """Load secrets from the encrypted file."""
        self._load_from_file()

    def export_redacted(self) -> Dict[str, str]:
        """Export secrets with values redacted for logging.

        Returns:
            Dictionary with keys and redacted values.
        """
        return {k: "***REDACTED***" for k in self._secrets}
