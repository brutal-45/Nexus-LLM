"""Tests for the auth module.

Covers AuthManager, APIKeyManager, RateLimiter, Permission, and Role.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from nexus_llm.auth.manager import AuthManager, User
from nexus_llm.auth.api_keys import APIKeyManager
from nexus_llm.auth.rate_limiter import RateLimiter, RateLimitConfig
from nexus_llm.auth.permissions import Permission, Role, BUILTIN_ROLES, get_role


# ---------------------------------------------------------------------------
# Permission
# ---------------------------------------------------------------------------

class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self):
        assert Permission.read.value == "read"
        assert Permission.write.value == "write"
        assert Permission.admin.value == "admin"

    def test_from_string(self):
        perm = Permission.from_string("read")
        assert perm == Permission.read

    def test_from_string_case_insensitive(self):
        perm = Permission.from_string("READ")
        assert perm == Permission.read

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown permission"):
            Permission.from_string("nonexistent")

    def test_str(self):
        assert str(Permission.read) == "read"


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

class TestRole:
    """Tests for Role."""

    def test_create_role(self):
        role = Role(name="viewer", permissions=frozenset({Permission.read}))
        assert role.name == "viewer"
        assert Permission.read in role.permissions

    def test_has_permission(self):
        role = Role(name="editor", permissions=frozenset({Permission.read, Permission.write}))
        assert role.has_permission(Permission.read) is True
        assert role.has_permission(Permission.admin) is False

    def test_admin_grants_all(self):
        role = Role(name="admin", permissions=frozenset({Permission.admin}))
        assert role.has_permission(Permission.read) is True
        assert role.has_permission(Permission.write) is True

    def test_grant(self):
        role = Role(name="viewer", permissions=frozenset({Permission.read}))
        new_role = role.grant(Permission.write)
        assert new_role.has_permission(Permission.write) is True
        # Original unchanged
        assert role.has_permission(Permission.write) is False

    def test_revoke(self):
        role = Role(name="editor", permissions=frozenset({Permission.read, Permission.write}))
        new_role = role.revoke(Permission.write)
        assert new_role.has_permission(Permission.write) is False

    def test_contains(self):
        role = Role(name="test", permissions=frozenset({Permission.read}))
        assert Permission.read in role
        assert Permission.write not in role

    def test_builtin_roles(self):
        assert "admin" in BUILTIN_ROLES
        assert "user" in BUILTIN_ROLES
        assert BUILTIN_ROLES["admin"].has_permission(Permission.admin)

    def test_get_role(self):
        role = get_role("admin")
        assert role is not None
        assert role.name == "admin"

    def test_get_role_nonexistent(self):
        assert get_role("nonexistent") is None


# ---------------------------------------------------------------------------
# APIKeyManager
# ---------------------------------------------------------------------------

class TestAPIKeyManager:
    """Tests for APIKeyManager."""

    def test_generate_key(self):
        akm = APIKeyManager()
        key = akm.generate_key("test-user")
        assert key.startswith("nxl-")
        assert len(key) > 0

    def test_validate_valid_key(self):
        akm = APIKeyManager()
        key = akm.generate_key("test-user", {Permission.read})
        valid, info = akm.validate_key(key)
        assert valid is True
        assert info["name"] == "test-user"

    def test_validate_invalid_key(self):
        akm = APIKeyManager()
        valid, info = akm.validate_key("nxl-invalid-key")
        assert valid is False
        assert info == {}

    def test_revoke_key(self):
        akm = APIKeyManager()
        key = akm.generate_key("test-user")
        akm.revoke_key(key)
        valid, info = akm.validate_key(key)
        assert valid is False

    def test_list_keys(self):
        akm = APIKeyManager()
        akm.generate_key("user1")
        akm.generate_key("user2")
        keys = akm.list_keys()
        assert len(keys) == 2

    def test_key_with_permissions(self):
        akm = APIKeyManager()
        key = akm.generate_key("power_user", {Permission.read, Permission.write})
        valid, info = akm.validate_key(key)
        assert valid is True
        assert Permission.read in info["permissions"]


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allow_within_limit(self):
        config = RateLimitConfig(requests_per_minute=60, requests_per_hour=1000, requests_per_day=10000)
        rl = RateLimiter(config=config)
        allowed, remaining, reset_time = rl.check_rate("user1")
        assert allowed is True
        assert remaining >= 0

    def test_multiple_requests(self):
        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=100, requests_per_day=1000)
        rl = RateLimiter(config=config)
        for i in range(5):
            allowed, _, _ = rl.check_rate("user1")
            assert allowed is True

    def test_different_users_independent(self):
        config = RateLimitConfig(requests_per_minute=2, requests_per_hour=100, requests_per_day=1000)
        rl = RateLimiter(config=config)
        rl.check_rate("user1")
        rl.check_rate("user1")
        # user1 may be rate-limited, but user2 should still be allowed
        allowed, _, _ = rl.check_rate("user2")
        assert allowed is True

    def test_reset(self):
        config = RateLimitConfig(requests_per_minute=2, requests_per_hour=100, requests_per_day=1000)
        rl = RateLimiter(config=config)
        rl.check_rate("user1")
        rl.check_rate("user1")
        rl.reset("user1")
        allowed, _, _ = rl.check_rate("user1")
        assert allowed is True

    def test_get_remaining(self):
        config = RateLimitConfig(requests_per_minute=60, requests_per_hour=1000, requests_per_day=10000)
        rl = RateLimiter(config=config)
        rl.check_rate("user1")
        remaining = rl.get_remaining("user1")
        assert "minute" in remaining
        assert "hour" in remaining
        assert "day" in remaining


# ---------------------------------------------------------------------------
# AuthManager
# ---------------------------------------------------------------------------

class TestAuthManager:
    """Tests for AuthManager."""

    def test_init(self):
        am = AuthManager()
        assert am is not None
        assert am.key_manager is not None

    def test_create_user(self):
        am = AuthManager()
        user = am.create_user("alice", BUILTIN_ROLES["admin"])
        assert user.username == "alice"
        assert user.api_key is not None

    def test_create_user_default_role(self):
        am = AuthManager()
        user = am.create_user("bob")
        assert user.role.name == "user"

    def test_create_duplicate_user_raises(self):
        am = AuthManager()
        am.create_user("alice")
        with pytest.raises(ValueError, match="already exists"):
            am.create_user("alice")

    def test_authenticate_with_key(self):
        am = AuthManager()
        user = am.create_user("alice", BUILTIN_ROLES["admin"])
        authenticated = am.authenticate(user.api_key)
        assert authenticated.username == "alice"

    def test_authenticate_invalid_key(self):
        am = AuthManager()
        with pytest.raises(PermissionError):
            am.authenticate("nxl-invalid-key")

    def test_authorize(self):
        am = AuthManager()
        user = am.create_user("alice", BUILTIN_ROLES["admin"])
        assert am.authorize(user, Permission.read) is True
        assert am.authorize(user, Permission.admin) is True

    def test_authorize_insufficient_permission(self):
        am = AuthManager()
        user = am.create_user("viewer_user", BUILTIN_ROLES["viewer"])
        assert am.authorize(user, Permission.read) is True
        assert am.authorize(user, Permission.write) is False

    def test_delete_user(self):
        am = AuthManager()
        am.create_user("alice")
        am.delete_user("alice")
        assert am.get_user("alice") is None

    def test_delete_nonexistent_user_raises(self):
        am = AuthManager()
        with pytest.raises(KeyError):
            am.delete_user("nonexistent")

    def test_list_users(self):
        am = AuthManager()
        am.create_user("alice")
        am.create_user("bob")
        users = am.list_users()
        assert len(users) == 2
