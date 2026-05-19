"""
Nexus-LLM Authentication Module

Provides authentication and authorization capabilities including
user management, token handling, and role-based permissions.
"""

from nexus_llm.auth.manager import AuthManager, AuthenticationError
from nexus_llm.auth.tokens import TokenManager, TokenExpiredError, TokenInvalidError
from nexus_llm.auth.permissions import (
    Permission,
    Role,
    PermissionDeniedError,
    RoleManager,
)

__all__ = [
    "AuthManager",
    "AuthenticationError",
    "TokenManager",
    "TokenExpiredError",
    "TokenInvalidError",
    "Permission",
    "Role",
    "PermissionDeniedError",
    "RoleManager",
]
