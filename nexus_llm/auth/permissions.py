"""
Permission System for Nexus-LLM

Implements role-based access control (RBAC) with fine-grained permissions.
Defines all available permissions, roles, and a RoleManager for managing
role-permission assignments.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PermissionError(Exception):
    """Base exception for permission errors."""


class PermissionDeniedError(PermissionError):
    """Raised when a user lacks a required permission."""

    def __init__(self, username: str, permission: str) -> None:
        self.username = username
        self.permission = permission
        super().__init__(
            f"User '{username}' lacks permission '{permission}'"
        )


class RoleNotFoundError(PermissionError):
    """Raised when a role does not exist."""

    def __init__(self, role_name: str) -> None:
        self.role_name = role_name
        super().__init__(f"Role '{role_name}' not found")


class RoleAlreadyExistsError(PermissionError):
    """Raised when a role already exists."""

    def __init__(self, role_name: str) -> None:
        self.role_name = role_name
        super().__init__(f"Role '{role_name}' already exists")


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

class Permission(enum.Enum):
    """All available permissions in Nexus-LLM.

    Permissions are organized by resource category:
    - CHAT_*: Chat and conversation operations
    - MODEL_*: Model management operations
    - TRAIN_*: Training operations
    - SERVER_*: Server administration
    - PRESET_*: Preset management
    - PLUGIN_*: Plugin management
    - USER_*: User management
    - RAG_*: RAG pipeline operations
    - AGENT_*: Agent operations
    - SYSTEM_*: System-level operations
    """

    # Chat permissions
    CHAT_READ = "chat:read"
    CHAT_WRITE = "chat:write"
    CHAT_DELETE = "chat:delete"
    CHAT_EXPORT = "chat:export"

    # Model permissions
    MODEL_LIST = "model:list"
    MODEL_VIEW = "model:view"
    MODEL_LOAD = "model:load"
    MODEL_UNLOAD = "model:unload"
    MODEL_INFER = "model:infer"
    MODEL_DOWNLOAD = "model:download"
    MODEL_DELETE = "model:delete"
    MODEL_QUANTIZE = "model:quantize"
    MODEL_CONVERT = "model:convert"

    # Training permissions
    TRAIN_VIEW = "train:view"
    TRAIN_START = "train:start"
    TRAIN_STOP = "train:stop"
    TRAIN_MANAGE = "train:manage"
    TRAIN_EXPORT = "train:export"

    # Server permissions
    SERVER_VIEW = "server:view"
    SERVER_CONFIGURE = "server:configure"
    SERVER_RESTART = "server:restart"
    SERVER_SHUTDOWN = "server:shutdown"
    SERVER_LOGS = "server:logs"

    # Preset permissions
    PRESET_LIST = "preset:list"
    PRESET_VIEW = "preset:view"
    PRESET_USE = "preset:use"
    PRESET_CREATE = "preset:create"
    PRESET_DELETE = "preset:delete"

    # Plugin permissions
    PLUGIN_LIST = "plugin:list"
    PLUGIN_VIEW = "plugin:view"
    PLUGIN_INSTALL = "plugin:install"
    PLUGIN_UNINSTALL = "plugin:uninstall"
    PLUGIN_CONFIGURE = "plugin:configure"

    # User management permissions
    USER_LIST = "user:list"
    USER_VIEW = "user:view"
    USER_CREATE = "user:create"
    USER_DELETE = "user:delete"
    USER_UPDATE = "user:update"
    USER_ROLE_ASSIGN = "user:role_assign"
    USER_ROLE_REVOKE = "user:role_revoke"

    # RAG permissions
    RAG_INDEX = "rag:index"
    RAG_SEARCH = "rag:search"
    RAG_MANAGE = "rag:manage"

    # Agent permissions
    AGENT_USE = "agent:use"
    AGENT_CONFIGURE = "agent:configure"

    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_METRICS = "system:metrics"
    SYSTEM_CONFIG = "system:config"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "Permission":
        """Parse a permission string like 'chat:read' into a Permission enum.

        Args:
            value: The permission string.

        Returns:
            The corresponding Permission enum member.

        Raises:
            ValueError: If the string does not match any permission.
        """
        for perm in cls:
            if perm.value == value:
                return perm
        raise ValueError(f"Unknown permission: {value}")


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Role:
    """Represents a named set of permissions.

    Roles are immutable; use the RoleManager to create or modify roles.
    """

    name: str
    permissions: FrozenSet[Permission] = field(default_factory=frozenset)
    description: str = ""
    is_builtin: bool = False

    def has_permission(self, permission: Permission) -> bool:
        """Check if this role includes a specific permission."""
        return permission in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the role to a dictionary."""
        return {
            "name": self.name,
            "permissions": [str(p) for p in sorted(self.permissions, key=str)],
            "description": self.description,
            "is_builtin": self.is_builtin,
        }


# ---------------------------------------------------------------------------
# Role Manager
# ---------------------------------------------------------------------------

class RoleManager:
    """Manages roles and their associated permissions.

    Provides CRUD operations for roles and supports checking whether
    a set of roles collectively grants a specific permission.

    Example::

        rm = RoleManager()
        rm.create_role(Role(
            name="editor",
            permissions={Permission.CHAT_READ, Permission.CHAT_WRITE},
            description="Can read and write chats",
        ))
        editor_role = rm.get_role("editor")
        has_perm = editor_role.has_permission(Permission.CHAT_READ)
    """

    def __init__(self) -> None:
        self._roles: Dict[str, Role] = {}

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_role(self, role: Role) -> Role:
        """Create a new role.

        Args:
            role: The Role object to register.

        Returns:
            The created Role.

        Raises:
            RoleAlreadyExistsError: If a role with the same name exists.
        """
        if role.name in self._roles:
            raise RoleAlreadyExistsError(role.name)
        self._roles[role.name] = role
        return role

    def get_role(self, name: str) -> Optional[Role]:
        """Retrieve a role by name.

        Args:
            name: The role name.

        Returns:
            The Role object, or None if not found.
        """
        return self._roles.get(name)

    def update_role(
        self,
        name: str,
        *,
        permissions: Optional[Set[Permission]] = None,
        description: Optional[str] = None,
    ) -> Role:
        """Update an existing role's permissions and/or description.

        Args:
            name: The role name.
            permissions: New set of permissions (if provided).
            description: New description (if provided).

        Returns:
            The updated Role.

        Raises:
            RoleNotFoundError: If the role does not exist.
            PermissionError: If the role is a builtin role.
        """
        if name not in self._roles:
            raise RoleNotFoundError(name)

        existing = self._roles[name]
        if existing.is_builtin:
            raise PermissionError(f"Cannot modify builtin role '{name}'")

        updated = Role(
            name=name,
            permissions=frozenset(permissions) if permissions is not None else existing.permissions,
            description=description if description is not None else existing.description,
            is_builtin=False,
        )
        self._roles[name] = updated
        return updated

    def delete_role(self, name: str) -> None:
        """Delete a custom role.

        Args:
            name: The role name.

        Raises:
            RoleNotFoundError: If the role does not exist.
            PermissionError: If the role is a builtin role.
        """
        if name not in self._roles:
            raise RoleNotFoundError(name)

        if self._roles[name].is_builtin:
            raise PermissionError(f"Cannot delete builtin role '{name}'")

        del self._roles[name]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_roles(self) -> List[Role]:
        """Return all registered roles."""
        return list(self._roles.values())

    def list_role_names(self) -> List[str]:
        """Return all role names."""
        return list(self._roles.keys())

    def check_permission(self, role_names: List[str], permission: Permission) -> bool:
        """Check if any of the given roles grant a specific permission.

        Args:
            role_names: List of role names to check.
            permission: The permission to check for.

        Returns:
            True if at least one role grants the permission.
        """
        for name in role_names:
            role = self._roles.get(name)
            if role and role.has_permission(permission):
                return True
        return False

    def get_all_permissions(self, role_names: List[str]) -> Set[Permission]:
        """Get the union of permissions from multiple roles.

        Args:
            role_names: List of role names.

        Returns:
            Combined set of permissions.
        """
        permissions: Set[Permission] = set()
        for name in role_names:
            role = self._roles.get(name)
            if role:
                permissions.update(role.permissions)
        return permissions

    def role_exists(self, name: str) -> bool:
        """Check if a role exists."""
        return name in self._roles

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all roles to a dictionary."""
        return {name: role.to_dict() for name, role in self._roles.items()}
