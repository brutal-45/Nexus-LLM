"""
Authentication Manager for Nexus-LLM

Manages user authentication, login/logout flows, and session tracking.
Integrates with TokenManager for JWT-based session tokens and
RoleManager for authorization checks.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from nexus_llm.auth.tokens import TokenManager, TokenExpiredError, TokenInvalidError
from nexus_llm.auth.permissions import Permission, PermissionDeniedError, Role, RoleManager


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AuthenticationError(Exception):
    """Base exception for authentication errors."""


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are invalid."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"Invalid credentials for user '{username}'")


class UserNotFoundError(AuthenticationError):
    """Raised when a user does not exist."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"User '{username}' not found")


class UserAlreadyExistsError(AuthenticationError):
    """Raised when attempting to create a user that already exists."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"User '{username}' already exists")


class AccountLockedError(AuthenticationError):
    """Raised when a user account is locked."""

    def __init__(self, username: str) -> None:
        self.username = username
        super().__init__(f"Account '{username}' is locked")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class User:
    """Represents a registered user."""

    username: str
    password_hash: str
    salt: str
    roles: List[str] = field(default_factory=lambda: ["user"])
    is_active: bool = True
    is_locked: bool = False
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize user to a dictionary (excluding sensitive fields)."""
        return {
            "username": self.username,
            "roles": self.roles,
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Auth Manager
# ---------------------------------------------------------------------------

class AuthManager:
    """Manages user authentication for Nexus-LLM.

    Handles user registration, login, logout, password management,
    and session tracking. Uses JWT tokens for stateless authentication.

    Example::

        auth = AuthManager()
        auth.register("alice", "secure_password", roles=["admin"])
        token = auth.login("alice", "secure_password")
        user = auth.authenticate(token)
        auth.logout(token)
    """

    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = 900  # 15 minutes in seconds

    def __init__(
        self,
        token_manager: Optional[TokenManager] = None,
        role_manager: Optional[RoleManager] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        """Initialize the AuthManager.

        Args:
            token_manager: Custom TokenManager instance. Created if not provided.
            role_manager: Custom RoleManager instance. Created if not provided.
            secret_key: Secret key for token signing. Auto-generated if not provided.
        """
        self._secret_key = secret_key or secrets.token_hex(32)
        self._token_manager = token_manager or TokenManager(secret_key=self._secret_key)
        self._role_manager = role_manager or RoleManager()
        self._users: Dict[str, User] = {}
        self._active_sessions: Dict[str, str] = {}  # token_id -> username

        # Initialize default roles
        self._init_default_roles()

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def register(
        self,
        username: str,
        password: str,
        *,
        roles: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Register a new user.

        Args:
            username: Unique username.
            password: Plaintext password (will be hashed).
            roles: Optional list of role names. Defaults to ['user'].
            metadata: Optional user metadata.

        Returns:
            The newly created User object.

        Raises:
            UserAlreadyExistsError: If the username is taken.
            ValueError: If username or password is empty.
        """
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")
        if not password or len(password) < 4:
            raise ValueError("Password must be at least 4 characters")

        if username in self._users:
            raise UserAlreadyExistsError(username)

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)

        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            roles=roles or ["user"],
            metadata=metadata or {},
        )

        self._users[username] = user
        return user

    def login(self, username: str, password: str) -> str:
        """Authenticate a user and return a JWT token.

        Args:
            username: The user's username.
            password: The user's plaintext password.

        Returns:
            JWT token string.

        Raises:
            UserNotFoundError: If the user does not exist.
            InvalidCredentialsError: If the password is wrong.
            AccountLockedError: If the account is locked.
        """
        if username not in self._users:
            raise UserNotFoundError(username)

        user = self._users[username]

        # Check lockout
        if user.is_locked:
            if user.last_login and (
                time.time() - user.last_login < self.LOCKOUT_DURATION
            ):
                raise AccountLockedError(username)
            else:
                # Unlock after lockout duration
                user.is_locked = False
                user.failed_login_attempts = 0

        if not user.is_active:
            raise AccountLockedError(username)

        # Verify password
        password_hash = self._hash_password(password, user.salt)
        if password_hash != user.password_hash:
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.is_locked = True
            raise InvalidCredentialsError(username)

        # Reset failed attempts on success
        user.failed_login_attempts = 0
        user.last_login = time.time()

        # Generate token with user claims
        claims = {
            "sub": username,
            "roles": user.roles,
            "type": "access",
        }
        token = self._token_manager.create_token(claims)

        # Track active session
        self._active_sessions[token] = username

        return token

    def logout(self, token: str) -> None:
        """Invalidate a user session.

        Args:
            token: The JWT token to invalidate.
        """
        self._active_sessions.pop(token, None)
        try:
            self._token_manager.revoke_token(token)
        except (TokenExpiredError, TokenInvalidError):
            pass  # Token already invalid

    def authenticate(self, token: str) -> User:
        """Validate a token and return the associated user.

        Args:
            token: JWT token string.

        Returns:
            The authenticated User object.

        Raises:
            TokenExpiredError: If the token has expired.
            TokenInvalidError: If the token is invalid.
            UserNotFoundError: If the token refers to a non-existent user.
        """
        claims = self._token_manager.validate_token(token)
        username = claims.get("sub")
        if username not in self._users:
            raise UserNotFoundError(username)
        return self._users[username]

    def change_password(
        self, username: str, old_password: str, new_password: str
    ) -> None:
        """Change a user's password.

        Args:
            username: The user's username.
            old_password: Current password for verification.
            new_password: New password to set.

        Raises:
            UserNotFoundError: If the user does not exist.
            InvalidCredentialsError: If old_password is wrong.
            ValueError: If new_password is too short.
        """
        if username not in self._users:
            raise UserNotFoundError(username)

        user = self._users[username]
        old_hash = self._hash_password(old_password, user.salt)
        if old_hash != user.password_hash:
            raise InvalidCredentialsError(username)

        if len(new_password) < 4:
            raise ValueError("Password must be at least 4 characters")

        new_salt = secrets.token_hex(16)
        user.salt = new_salt
        user.password_hash = self._hash_password(new_password, new_salt)

    def reset_password(self, username: str, new_password: str) -> None:
        """Reset a user's password without verifying the old one.

        Intended for admin use only.

        Args:
            username: The user's username.
            new_password: New password to set.

        Raises:
            UserNotFoundError: If the user does not exist.
        """
        if username not in self._users:
            raise UserNotFoundError(username)

        user = self._users[username]
        new_salt = secrets.token_hex(16)
        user.salt = new_salt
        user.password_hash = self._hash_password(new_password, new_salt)
        user.is_locked = False
        user.failed_login_attempts = 0

    def deactivate_user(self, username: str) -> None:
        """Deactivate a user account.

        Args:
            username: The user to deactivate.

        Raises:
            UserNotFoundError: If the user does not exist.
        """
        if username not in self._users:
            raise UserNotFoundError(username)
        self._users[username].is_active = False

    def activate_user(self, username: str) -> None:
        """Re-activate a deactivated user account.

        Args:
            username: The user to activate.

        Raises:
            UserNotFoundError: If the user does not exist.
        """
        if username not in self._users:
            raise UserNotFoundError(username)
        self._users[username].is_active = True
        self._users[username].is_locked = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_user(self, username: str) -> User:
        """Retrieve a user by username.

        Raises:
            UserNotFoundError: If the user does not exist.
        """
        if username not in self._users:
            raise UserNotFoundError(username)
        return self._users[username]

    def list_users(self) -> List[Dict[str, Any]]:
        """Return a list of all users (excluding sensitive data)."""
        return [user.to_dict() for user in self._users.values()]

    def get_user_permissions(self, username: str) -> Set[Permission]:
        """Get all permissions for a user based on their roles.

        Args:
            username: The user's username.

        Returns:
            Set of Permission objects.
        """
        if username not in self._users:
            raise UserNotFoundError(username)
        user = self._users[username]
        permissions: Set[Permission] = set()
        for role_name in user.roles:
            role = self._role_manager.get_role(role_name)
            if role:
                permissions.update(role.permissions)
        return permissions

    def check_permission(self, username: str, permission: Permission) -> bool:
        """Check if a user has a specific permission.

        Args:
            username: The user's username.
            permission: The permission to check.

        Returns:
            True if the user has the permission.
        """
        return permission in self.get_user_permissions(username)

    def require_permission(self, username: str, permission: Permission) -> None:
        """Require a user to have a specific permission.

        Raises:
            UserNotFoundError: If the user does not exist.
            PermissionDeniedError: If the user lacks the permission.
        """
        if not self.check_permission(username, permission):
            raise PermissionDeniedError(username, str(permission))

    # ------------------------------------------------------------------
    # API Key authentication
    # ------------------------------------------------------------------

    def create_api_key(self, username: str, name: str = "default") -> str:
        """Create an API key for a user.

        Args:
            username: The user's username.
            name: A label for the API key.

        Returns:
            The generated API key string.

        Raises:
            UserNotFoundError: If the user does not exist.
        """
        if username not in self._users:
            raise UserNotFoundError(username)

        api_key = f"nexus_{secrets.token_urlsafe(32)}"
        user = self._users[username]

        if "api_keys" not in user.metadata:
            user.metadata["api_keys"] = {}

        user.metadata["api_keys"][name] = {
            "key_hash": hashlib.sha256(api_key.encode()).hexdigest(),
            "created_at": time.time(),
        }

        return api_key

    def authenticate_api_key(self, api_key: str) -> User:
        """Authenticate using an API key.

        Args:
            api_key: The API key string.

        Returns:
            The associated User object.

        Raises:
            AuthenticationError: If the API key is invalid.
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        for user in self._users.values():
            api_keys = user.metadata.get("api_keys", {})
            for key_info in api_keys.values():
                if key_info.get("key_hash") == key_hash:
                    return user

        raise AuthenticationError("Invalid API key")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """Hash a password with a salt using SHA-256.

        Args:
            password: Plaintext password.
            salt: Hex-encoded salt.

        Returns:
            Hex-encoded password hash.
        """
        return hashlib.sha256(
            f"{salt}{password}{salt}".encode()
        ).hexdigest()

    def _init_default_roles(self) -> None:
        """Set up default roles and permissions."""
        # Admin role - full access
        admin_permissions = {p for p in Permission}
        self._role_manager.create_role(
            Role(name="admin", permissions=admin_permissions, description="Full access administrator")
        )

        # User role - basic access
        user_permissions = {
            Permission.CHAT_READ,
            Permission.CHAT_WRITE,
            Permission.MODEL_LIST,
            Permission.MODEL_INFER,
            Permission.PRESET_LIST,
            Permission.PRESET_USE,
        }
        self._role_manager.create_role(
            Role(name="user", permissions=user_permissions, description="Standard user")
        )

        # Viewer role - read-only
        viewer_permissions = {
            Permission.CHAT_READ,
            Permission.MODEL_LIST,
            Permission.PRESET_LIST,
        }
        self._role_manager.create_role(
            Role(name="viewer", permissions=viewer_permissions, description="Read-only access")
        )

        # Operator role - can manage models and training
        operator_permissions = {
            Permission.CHAT_READ,
            Permission.CHAT_WRITE,
            Permission.MODEL_LIST,
            Permission.MODEL_INFER,
            Permission.MODEL_LOAD,
            Permission.MODEL_UNLOAD,
            Permission.TRAIN_START,
            Permission.TRAIN_STOP,
            Permission.TRAIN_VIEW,
            Permission.PRESET_LIST,
            Permission.PRESET_USE,
        }
        self._role_manager.create_role(
            Role(name="operator", permissions=operator_permissions, description="Model and training operator")
        )
