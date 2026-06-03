"""Authentication: API key auth, token auth, rate limit per key."""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from nexus_llm.api.errors import AuthenticationError

logger = logging.getLogger("nexus_llm.api.auth")


@dataclass
class APIKey:
    """Represents an API key with associated metadata and permissions."""
    key_hash: str
    name: str
    created_at: float = 0.0
    last_used_at: float = 0.0
    is_active: bool = True
    rate_limit: int = 60
    rate_limit_period: int = 60
    allowed_models: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "is_active": self.is_active,
            "rate_limit": self.rate_limit,
            "rate_limit_period": self.rate_limit_period,
            "allowed_models": self.allowed_models,
        }


class KeyStore:
    """Storage and management for API keys.

    Supports in-memory storage with optional file persistence,
    key generation, validation, and lookup.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self._keys: Dict[str, APIKey] = {}
        self._persistence_path = persistence_path
        if persistence_path and os.path.isfile(persistence_path):
            self._load_from_file()

    def generate_key(
        self,
        name: str,
        rate_limit: int = 60,
        rate_limit_period: int = 60,
        allowed_models: Optional[List[str]] = None,
    ) -> str:
        """Generate a new API key.

        Args:
            name: Human-readable name for the key.
            rate_limit: Maximum requests per period.
            rate_limit_period: Rate limit period in seconds.
            allowed_models: List of allowed model names. None means all models.

        Returns:
            The generated API key string (shown only once).
        """
        raw_key = f"nxs-{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)

        api_key = APIKey(
            key_hash=key_hash,
            name=name,
            created_at=time.time(),
            rate_limit=rate_limit,
            rate_limit_period=rate_limit_period,
            allowed_models=allowed_models,
        )
        self._keys[key_hash] = api_key
        self._persist()

        logger.info("API key generated: %s", name)
        return raw_key

    def validate_key(self, raw_key: str) -> Tuple[bool, Optional[APIKey]]:
        """Validate an API key and return its metadata.

        Args:
            raw_key: The raw API key string.

        Returns:
            Tuple of (is_valid, api_key_or_none).
        """
        key_hash = self._hash_key(raw_key)
        api_key = self._keys.get(key_hash)

        if api_key is None:
            return False, None

        if not api_key.is_active:
            return False, None

        api_key.last_used_at = time.time()
        self._persist()
        return True, api_key

    def revoke_key(self, name: str) -> bool:
        """Revoke an API key by name.

        Args:
            name: Name of the key to revoke.

        Returns:
            True if the key was found and revoked.
        """
        for key_hash, api_key in self._keys.items():
            if api_key.name == name:
                api_key.is_active = False
                self._persist()
                logger.info("API key revoked: %s", name)
                return True
        return False

    def delete_key(self, name: str) -> bool:
        """Delete an API key by name.

        Args:
            name: Name of the key to delete.

        Returns:
            True if the key was found and deleted.
        """
        to_delete = None
        for key_hash, api_key in self._keys.items():
            if api_key.name == name:
                to_delete = key_hash
                break
        if to_delete:
            del self._keys[to_delete]
            self._persist()
            logger.info("API key deleted: %s", name)
            return True
        return False

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without revealing the actual key values)."""
        return [api_key.to_dict() for api_key in self._keys.values()]

    def get_key_by_name(self, name: str) -> Optional[APIKey]:
        """Find an API key by name."""
        for api_key in self._keys.values():
            if api_key.name == name:
                return api_key
        return None

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _persist(self) -> None:
        """Save key store to file if persistence is enabled."""
        if not self._persistence_path:
            return
        data = {kh: ak.to_dict() for kh, ak in self._keys.items()}
        data["__key_hashes"] = {
            kh: {"key_hash": ak.key_hash, "name": ak.name, "is_active": ak.is_active,
                 "created_at": ak.created_at, "last_used_at": ak.last_used_at,
                 "rate_limit": ak.rate_limit, "rate_limit_period": ak.rate_limit_period,
                 "allowed_models": ak.allowed_models}
            for kh, ak in self._keys.items()
        }
        try:
            os.makedirs(os.path.dirname(self._persistence_path), exist_ok=True)
            with open(self._persistence_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error("Failed to persist key store: %s", e)

    def _load_from_file(self) -> None:
        """Load key store from file."""
        try:
            with open(self._persistence_path, "r") as f:
                data = json.load(f)
            hashes = data.get("__key_hashes", data)
            for kh, ak_data in hashes.items():
                if kh.startswith("__"):
                    continue
                self._keys[kh] = APIKey(
                    key_hash=ak_data.get("key_hash", kh),
                    name=ak_data.get("name", "unknown"),
                    created_at=ak_data.get("created_at", 0),
                    last_used_at=ak_data.get("last_used_at", 0),
                    is_active=ak_data.get("is_active", True),
                    rate_limit=ak_data.get("rate_limit", 60),
                    rate_limit_period=ak_data.get("rate_limit_period", 60),
                    allowed_models=ak_data.get("allowed_models"),
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load key store: %s", e)


# Security schemes
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


class AuthManager:
    """Manages authentication for the Nexus-LLM API.

    Supports API key authentication and Bearer token authentication.
    """

    def __init__(
        self,
        key_store: Optional[KeyStore] = None,
        require_auth: bool = True,
        admin_key: Optional[str] = None,
    ):
        self.key_store = key_store or KeyStore()
        self.require_auth = require_auth
        self.admin_key = admin_key
        self._bearer_tokens: Dict[str, Dict[str, Any]] = {}

    def register_bearer_token(
        self,
        token: str,
        name: str = "default",
        rate_limit: int = 60,
        allowed_models: Optional[List[str]] = None,
    ) -> None:
        """Register a Bearer token for authentication.

        Args:
            token: The bearer token string.
            name: Name identifier for the token.
            rate_limit: Requests per minute.
            allowed_models: Optional model whitelist.
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._bearer_tokens[token_hash] = {
            "name": name,
            "rate_limit": rate_limit,
            "allowed_models": allowed_models,
            "created_at": time.time(),
        }
        logger.info("Bearer token registered: %s", name)

    async def authenticate_request(
        self,
        api_key: Optional[str] = Security(_api_key_header),
        bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    ) -> APIKey:
        """Authenticate an incoming request.

        Checks API key header first, then Bearer token.

        Args:
            api_key: API key from X-API-Key header.
            bearer: Bearer token from Authorization header.

        Returns:
            APIKey object for the authenticated user.

        Raises:
            AuthenticationError: If authentication fails.
        """
        if not self.require_auth:
            return APIKey(
                key_hash="anonymous",
                name="anonymous",
                rate_limit=1000,
                rate_limit_period=60,
            )

        if api_key:
            is_valid, key_obj = self.key_store.validate_key(api_key)
            if is_valid and key_obj:
                return key_obj
            raise AuthenticationError("Invalid API key")

        if bearer:
            token_hash = hashlib.sha256(bearer.credentials.encode()).hexdigest()
            token_data = self._bearer_tokens.get(token_hash)
            if token_data:
                return APIKey(
                    key_hash=token_hash,
                    name=token_data["name"],
                    rate_limit=token_data["rate_limit"],
                    allowed_models=token_data.get("allowed_models"),
                )
            raise AuthenticationError("Invalid bearer token")

        raise AuthenticationError("No authentication provided. Use X-API-Key header or Bearer token.")

    def check_model_access(self, api_key: APIKey, model_name: str) -> bool:
        """Check if an API key has access to a specific model.

        Args:
            api_key: The authenticated API key.
            model_name: The model being requested.

        Returns:
            True if access is allowed.
        """
        if api_key.allowed_models is None:
            return True
        return model_name in api_key.allowed_models

    def is_admin(self, api_key: APIKey) -> bool:
        """Check if an API key has admin privileges.

        Args:
            api_key: The API key to check.

        Returns:
            True if the key has admin access.
        """
        if self.admin_key is None:
            return False
        return api_key.key_hash == self.admin_key


# Global auth manager
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the global authentication manager singleton."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager(require_auth=False)
    return _auth_manager


def init_auth(
    key_store: Optional[KeyStore] = None,
    require_auth: bool = True,
    admin_key: Optional[str] = None,
) -> AuthManager:
    """Initialize the global authentication manager.

    Args:
        key_store: Optional key store instance.
        require_auth: Whether authentication is required.
        admin_key: Optional admin key hash.

    Returns:
        The initialized AuthManager.
    """
    global _auth_manager
    _auth_manager = AuthManager(
        key_store=key_store,
        require_auth=require_auth,
        admin_key=admin_key,
    )
    return _auth_manager
