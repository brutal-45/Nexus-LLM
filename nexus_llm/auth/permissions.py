"""Permission enum and Role-based access control."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set


class Permission(enum.Enum):
    """Enumerated permissions for the Nexus-LLM system.

    Each permission represents a discrete capability that can be
    granted to a role.
    """

    read = "read"
    write = "write"
    admin = "admin"
    generate = "generate"
    chat = "chat"
    train = "train"
    manage_models = "manage_models"
    manage_users = "manage_users"

    def __str__(self) -> str:  # noqa: D105
        return self.value

    @classmethod
    def from_string(cls, name: str) -> Permission:
        """Look up a permission by name (case-insensitive).

        Args:
            name: Permission name string.

        Returns:
            The matching :class:`Permission`.

        Raises:
            ValueError: If no such permission exists.
        """
        try:
            return cls(name.lower())
        except ValueError:
            valid = ", ".join(p.value for p in cls)
            raise ValueError(
                f"Unknown permission '{name}'. Valid: {valid}"
            ) from None


@dataclass(frozen=True)
class Role:
    """A named set of permissions.

    Attributes:
        name: Human-readable role name.
        permissions: The set of permissions this role grants.
    """

    name: str
    permissions: FrozenSet[Permission] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Normalize to frozenset so the dataclass is hashable
        if isinstance(self.permissions, (set, list)):
            object.__setattr__(
                self, "permissions", frozenset(self.permissions)
            )

    def has_permission(self, permission: Permission) -> bool:
        """Check whether this role includes *permission*.

        The ``admin`` permission is special — if a role holds it, all
        other permissions are implicitly granted.

        Args:
            permission: The permission to check.

        Returns:
            ``True`` if the role grants the permission.
        """
        if Permission.admin in self.permissions:
            return True
        return permission in self.permissions

    def grant(self, *permissions: Permission) -> Role:
        """Return a new Role with additional permissions.

        Args:
            *permissions: Permissions to add.

        Returns:
            A new :class:`Role` instance.
        """
        merged = self.permissions | frozenset(permissions)
        return Role(name=self.name, permissions=merged)

    def revoke(self, *permissions: Permission) -> Role:
        """Return a new Role with permissions removed.

        Args:
            *permissions: Permissions to remove.

        Returns:
            A new :class:`Role` instance.
        """
        reduced = self.permissions - frozenset(permissions)
        return Role(name=self.name, permissions=reduced)

    def __contains__(self, permission: Permission) -> bool:  # type: ignore[override]
        return self.has_permission(permission)

    def __str__(self) -> str:  # noqa: D105
        perms = ", ".join(sorted(str(p) for p in self.permissions))
        return f"Role({self.name!r}, [{perms}])"


# ------------------------------------------------------------------
# Built-in roles
# ------------------------------------------------------------------

BUILTIN_ROLES: Dict[str, Role] = {
    "viewer": Role(
        name="viewer",
        permissions=frozenset({Permission.read}),
    ),
    "user": Role(
        name="user",
        permissions=frozenset({
            Permission.read,
            Permission.write,
            Permission.chat,
            Permission.generate,
        }),
    ),
    "power_user": Role(
        name="power_user",
        permissions=frozenset({
            Permission.read,
            Permission.write,
            Permission.chat,
            Permission.generate,
            Permission.train,
        }),
    ),
    "admin": Role(
        name="admin",
        permissions=frozenset({
            Permission.read,
            Permission.write,
            Permission.admin,
            Permission.generate,
            Permission.chat,
            Permission.train,
            Permission.manage_models,
            Permission.manage_users,
        }),
    ),
}


def get_role(name: str) -> Optional[Role]:
    """Look up a built-in role by name.

    Args:
        name: Role name (e.g. ``"admin"``).

    Returns:
        The :class:`Role`, or ``None`` if not found.
    """
    return BUILTIN_ROLES.get(name)
