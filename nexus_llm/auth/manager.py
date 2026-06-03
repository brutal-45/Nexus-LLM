"""Core authentication and authorization manager."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from nexus_llm.auth.api_keys import APIKeyManager
from nexus_llm.auth.permissions import Permission, Role, BUILTIN_ROLES


@dataclass
class User:
    """Represents an authenticated user.

    Attributes:
        username: Unique user identifier.
        role: The user's role.
        user_id: Internal unique ID.
        api_key: The API key assigned to this user (if any).
        active: Whether the user account is active.
    """

    username: str
    role: Role
    user_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    api_key: Optional[str] = None
    active: bool = True

    def has_permission(self, permission: Permission) -> bool:
        """Check whether the user has a specific permission."""
        return self.role.has_permission(permission)


class AuthManager:
    """Central authentication and authorization manager.

    Provides user lifecycle management (create / delete), authentication
    via API keys, and authorization checks against role-based
    permissions.

    Example::

        auth = AuthManager()
        user = auth.create_user("alice", BUILTIN_ROLES["admin"])
        assert auth.authorize(user, Permission.admin)
    """

    def __init__(self) -> None:
        self._users: Dict[str, User] = {}
        self._key_manager = APIKeyManager()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def key_manager(self) -> APIKeyManager:
        """Return the associated API key manager."""
        return self._key_manager

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, api_key: str) -> User:
        """Authenticate a request by API key.

        Args:
            api_key: The API key to validate.

        Returns:
            The authenticated :class:`User`.

        Raises:
            PermissionError: If the key is invalid or revoked.
        """
        valid, user_info = self._key_manager.validate_key(api_key)
        if not valid:
            raise PermissionError("Invalid or revoked API key.")

        username = user_info.get("username")
        if username is None:
            raise PermissionError("API key has no associated user.")

        with self._lock:
            user = self._users.get(username)
            if user is None or not user.active:
                raise PermissionError(
                    f"User '{username}' does not exist or is deactivated."
                )
            return user

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def authorize(self, user: User, permission: Permission) -> bool:
        """Check whether *user* holds *permission*.

        Args:
            user: The user to check.
            permission: The required permission.

        Returns:
            ``True`` if the user's role grants the permission.
        """
        return user.has_permission(permission)

    # ------------------------------------------------------------------
    # User lifecycle
    # ------------------------------------------------------------------

    def create_user(
        self,
        username: str,
        role: Optional[Role] = None,
        permissions: Optional[Set[Permission]] = None,
    ) -> User:
        """Create a new user and generate an API key.

        If *role* is not provided, the default ``"user"`` role is used.
        If *permissions* is provided, a custom role is created that
        extends the given base role.

        Args:
            username: Unique user name.
            role: The user's role (defaults to ``user`` role).
            permissions: Optional custom permission set.

        Returns:
            The newly created :class:`User`.

        Raises:
            ValueError: If the username is already taken.
        """
        with self._lock:
            if username in self._users:
                raise ValueError(f"User '{username}' already exists.")

            if role is None:
                role = BUILTIN_ROLES["user"]

            # Create a custom role if extra permissions are requested
            if permissions is not None:
                merged = role.permissions | permissions
                role = Role(
                    name=f"custom_{username}",
                    permissions=merged,
                )

            # Generate API key
            key = self._key_manager.generate_key(
                name=username,
                permissions=role.permissions,
            )

            user = User(username=username, role=role, api_key=key)
            self._users[username] = user

            # Register user info in key manager for later lookup
            self._key_manager._key_metadata[key]["username"] = username

            return user

    def delete_user(self, username: str) -> None:
        """Delete a user and revoke their API key.

        Args:
            username: The user to delete.

        Raises:
            KeyError: If the user does not exist.
        """
        with self._lock:
            user = self._users.pop(username, None)
            if user is None:
                raise KeyError(f"User '{username}' does not exist.")

            if user.api_key:
                self._key_manager.revoke_key(user.api_key)

    def get_user(self, username: str) -> Optional[User]:
        """Retrieve a user by username, or ``None``."""
        with self._lock:
            return self._users.get(username)

    def list_users(self) -> List[User]:
        """Return a list of all registered users."""
        with self._lock:
            return list(self._users.values())
