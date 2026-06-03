"""
Token Management for Nexus-LLM

Provides JWT-based token creation, validation, and revocation.
Tokens are used for stateless authentication of API requests
and WebSocket connections.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TokenError(Exception):
    """Base exception for token errors."""


class TokenExpiredError(TokenError):
    """Raised when a token has expired."""

    def __init__(self, token_id: Optional[str] = None) -> None:
        self.token_id = token_id
        super().__init__(f"Token expired" + (f" (id={token_id})" if token_id else ""))


class TokenInvalidError(TokenError):
    """Raised when a token is invalid or malformed."""

    def __init__(self, reason: str = "invalid token") -> None:
        self.reason = reason
        super().__init__(f"Invalid token: {reason}")


class TokenRevokedError(TokenError):
    """Raised when a token has been revoked."""

    def __init__(self, token_id: Optional[str] = None) -> None:
        self.token_id = token_id
        super().__init__(f"Token revoked" + (f" (id={token_id})" if token_id else ""))


# ---------------------------------------------------------------------------
# Token data
# ---------------------------------------------------------------------------

@dataclass
class TokenInfo:
    """Stores metadata about a token for tracking purposes."""

    token_id: str
    subject: str
    token_type: str  # "access" or "refresh"
    created_at: float
    expires_at: float
    is_revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token Manager
# ---------------------------------------------------------------------------

class TokenManager:
    """Manages JWT-like tokens for Nexus-LLM authentication.

    Implements a simplified JWT format with HMAC-SHA256 signing.
    Supports token creation, validation, refresh, and revocation.

    Note: This is a self-contained implementation that does not depend
    on external JWT libraries. For production use with external services,
    consider using PyJWT for full JWT compliance.

    Example::

        tm = TokenManager(secret_key="my-secret")
        token = tm.create_token({"sub": "alice", "roles": ["admin"]})
        claims = tm.validate_token(token)
        tm.revoke_token(token)
    """

    # Token defaults
    DEFAULT_ACCESS_EXPIRY = 3600       # 1 hour
    DEFAULT_REFRESH_EXPIRY = 604800    # 7 days
    ALGORITHM = "HS256"
    TOKEN_PREFIX = "nxt_"  # Nexus token prefix

    def __init__(
        self,
        secret_key: Optional[str] = None,
        access_expiry: int = DEFAULT_ACCESS_EXPIRY,
        refresh_expiry: int = DEFAULT_REFRESH_EXPIRY,
        issuer: str = "nexus-llm",
    ) -> None:
        """Initialize the TokenManager.

        Args:
            secret_key: HMAC signing key. Auto-generated if not provided.
            access_expiry: Access token lifetime in seconds.
            refresh_expiry: Refresh token lifetime in seconds.
            issuer: Token issuer claim.
        """
        self._secret_key = secret_key or self._generate_secret()
        self._access_expiry = access_expiry
        self._refresh_expiry = refresh_expiry
        self._issuer = issuer
        self._revoked_ids: Set[str] = set()
        self._token_store: Dict[str, TokenInfo] = {}

    # ------------------------------------------------------------------
    # Token creation
    # ------------------------------------------------------------------

    def create_token(
        self,
        claims: Dict[str, Any],
        *,
        token_type: str = "access",
        expiry: Optional[int] = None,
        token_id: Optional[str] = None,
    ) -> str:
        """Create a signed token.

        Args:
            claims: Payload claims (e.g., {"sub": "alice", "roles": ["admin"]}).
            token_type: "access" or "refresh".
            expiry: Custom expiry in seconds. Defaults based on token_type.
            token_id: Custom token ID. Auto-generated if not provided.

        Returns:
            Encoded token string.
        """
        now = time.time()
        tid = token_id or str(uuid.uuid4())

        if expiry is None:
            expiry = (
                self._refresh_expiry
                if token_type == "refresh"
                else self._access_expiry
            )

        payload = {
            **claims,
            "iss": self._issuer,
            "iat": int(now),
            "exp": int(now + expiry),
            "jti": tid,
            "type": token_type,
        }

        # Encode and sign
        header = {"alg": self.ALGORITHM, "typ": "JWT"}
        header_b64 = self._base64url_encode(json.dumps(header, separators=(",", ":")))
        payload_b64 = self._base64url_encode(json.dumps(payload, separators=(",", ":")))
        signing_input = f"{header_b64}.{payload_b64}"
        signature = self._sign(signing_input)

        token = f"{self.TOKEN_PREFIX}{signing_input}.{signature}"

        # Track token
        self._token_store[tid] = TokenInfo(
            token_id=tid,
            subject=claims.get("sub", ""),
            token_type=token_type,
            created_at=now,
            expires_at=now + expiry,
        )

        return token

    def create_refresh_token(self, access_claims: Dict[str, Any]) -> str:
        """Create a refresh token based on access token claims.

        Args:
            access_claims: Claims from the original access token.

        Returns:
            Refresh token string.
        """
        refresh_claims = {
            "sub": access_claims.get("sub", ""),
            "roles": access_claims.get("roles", []),
        }
        return self.create_token(refresh_claims, token_type="refresh")

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate a token and return its claims.

        Args:
            token: The token string to validate.

        Returns:
            Dictionary of token claims.

        Raises:
            TokenInvalidError: If the token is malformed or signature invalid.
            TokenExpiredError: If the token has expired.
            TokenRevokedError: If the token has been revoked.
        """
        # Strip prefix
        if token.startswith(self.TOKEN_PREFIX):
            token = token[len(self.TOKEN_PREFIX):]

        parts = token.split(".")
        if len(parts) != 3:
            raise TokenInvalidError("Token must have 3 parts")

        header_b64, payload_b64, signature = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = self._sign(signing_input)
        if not hmac.compare_digest(signature, expected_sig):
            raise TokenInvalidError("Invalid signature")

        # Decode payload
        try:
            payload = json.loads(self._base64url_decode(payload_b64))
        except (json.JSONDecodeError, ValueError) as exc:
            raise TokenInvalidError(f"Invalid payload: {exc}")

        # Check expiration
        exp = payload.get("exp", 0)
        if time.time() > exp:
            raise TokenExpiredError(payload.get("jti"))

        # Check revocation
        jti = payload.get("jti")
        if jti and jti in self._revoked_ids:
            raise TokenRevokedError(jti)

        return payload

    def is_valid(self, token: str) -> bool:
        """Check if a token is valid without raising exceptions.

        Args:
            token: The token string.

        Returns:
            True if valid, False otherwise.
        """
        try:
            self.validate_token(token)
            return True
        except TokenError:
            return False

    def get_token_info(self, token_id: str) -> Optional[TokenInfo]:
        """Retrieve token metadata by ID.

        Args:
            token_id: The token's jti claim.

        Returns:
            TokenInfo if found, None otherwise.
        """
        return self._token_store.get(token_id)

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def refresh_token(self, refresh_token: str) -> str:
        """Exchange a refresh token for a new access token.

        Args:
            refresh_token: A valid refresh token.

        Returns:
            New access token string.

        Raises:
            TokenInvalidError: If the refresh token is not a refresh type.
            TokenError: If the refresh token is invalid.
        """
        claims = self.validate_token(refresh_token)

        if claims.get("type") != "refresh":
            raise TokenInvalidError("Token is not a refresh token")

        # Revoke the old refresh token
        old_jti = claims.get("jti")
        if old_jti:
            self._revoked_ids.add(old_jti)

        # Create new access token
        new_claims = {
            "sub": claims.get("sub", ""),
            "roles": claims.get("roles", []),
        }
        return self.create_token(new_claims, token_type="access")

    # ------------------------------------------------------------------
    # Token revocation
    # ------------------------------------------------------------------

    def revoke_token(self, token: str) -> None:
        """Revoke a token so it can no longer be used.

        Args:
            token: The token string to revoke.
        """
        try:
            claims = self.validate_token(token)
            jti = claims.get("jti")
            if jti:
                self._revoked_ids.add(jti)
                if jti in self._token_store:
                    self._token_store[jti].is_revoked = True
        except TokenExpiredError:
            # Still revoke even if expired
            pass
        except TokenInvalidError:
            raise

    def revoke_all_for_user(self, username: str) -> int:
        """Revoke all tokens for a specific user.

        Args:
            username: The user's username (sub claim).

        Returns:
            Number of tokens revoked.
        """
        count = 0
        for tid, info in self._token_store.items():
            if info.subject == username and not info.is_revoked:
                self._revoked_ids.add(tid)
                info.is_revoked = True
                count += 1
        return count

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove expired token records from the store.

        Returns:
            Number of expired tokens removed.
        """
        now = time.time()
        expired = [
            tid
            for tid, info in self._token_store.items()
            if info.expires_at < now
        ]
        for tid in expired:
            del self._token_store[tid]
            self._revoked_ids.discard(tid)
        return len(expired)

    def active_token_count(self) -> int:
        """Return the number of currently active (non-expired, non-revoked) tokens."""
        now = time.time()
        return sum(
            1
            for info in self._token_store.values()
            if info.expires_at > now and not info.is_revoked
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sign(self, data: str) -> str:
        """Create HMAC-SHA256 signature for data."""
        sig = hmac.new(
            self._secret_key.encode(),
            data.encode(),
            hashlib.sha256,
        ).digest()
        return self._base64url_encode(sig)

    @staticmethod
    def _base64url_encode(data: Any) -> str:
        """Base64url-encode data without padding."""
        if isinstance(data, str):
            data = data.encode()
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def _base64url_decode(data: str) -> bytes:
        """Base64url-decode data with padding restoration."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    @staticmethod
    def _generate_secret() -> str:
        """Generate a random secret key."""
        import secrets as _secrets
        return _secrets.token_hex(32)
