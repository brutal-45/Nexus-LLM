"""Test authentication for Nexus-LLM."""
import hashlib
import hmac
import os
import time
import pytest
from typing import Dict, Optional, List
from dataclasses import dataclass


class AuthError(Exception):
    pass


@dataclass
class User:
    username: str
    password_hash: str
    roles: List[str]
    api_key: str = ""
    created_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def has_role(self, role: str) -> bool:
        return role in self.roles


def hash_password(password: str, salt: str = "") -> str:
    if not salt:
        salt = os.urandom(16).hex()
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    if ":" not in stored_hash:
        return False
    salt, _ = stored_hash.split(":", 1)
    new_hash = hash_password(password, salt)
    return hmac.compare_digest(new_hash, stored_hash)


class TokenManager:
    def __init__(self, secret: str = "nexus-secret"):
        self._secret = secret
        self._tokens: Dict[str, dict] = {}

    def create_token(self, username: str, expires_in: int = 3600) -> str:
        token = os.urandom(32).hex()
        self._tokens[token] = {
            "username": username,
            "expires_at": time.time() + expires_in,
        }
        return token

    def validate_token(self, token: str) -> Optional[dict]:
        if token not in self._tokens:
            return None
        info = self._tokens[token]
        if time.time() > info["expires_at"]:
            del self._tokens[token]
            return None
        return info

    def revoke_token(self, token: str):
        self._tokens.pop(token, None)


class AuthManager:
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._token_manager = TokenManager()
        self._api_keys: Dict[str, str] = {}

    def register_user(self, username: str, password: str, roles: List[str] = None) -> User:
        if username in self._users:
            raise AuthError(f"User '{username}' already exists")
        if not password or len(password) < 4:
            raise AuthError("Password must be at least 4 characters")
        password_hash = hash_password(password)
        api_key = f"nx_{os.urandom(16).hex()}"
        user = User(username=username, password_hash=password_hash, roles=roles or ["user"], api_key=api_key)
        self._users[username] = user
        self._api_keys[api_key] = username
        return user

    def authenticate(self, username: str, password: str) -> str:
        if username not in self._users:
            raise AuthError("Invalid credentials")
        user = self._users[username]
        if not verify_password(password, user.password_hash):
            raise AuthError("Invalid credentials")
        return self._token_manager.create_token(username)

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        if api_key not in self._api_keys:
            return None
        username = self._api_keys[api_key]
        return self._users.get(username)

    def validate_token(self, token: str) -> Optional[User]:
        info = self._token_manager.validate_token(token)
        if info is None:
            return None
        return self._users.get(info["username"])

    def revoke_token(self, token: str):
        self._token_manager.revoke_token(token)

    def get_user(self, username: str) -> Optional[User]:
        return self._users.get(username)


class TestPasswordHashing:
    def test_hash_password(self):
        hashed = hash_password("secret123")
        assert ":" in hashed

    def test_verify_correct_password(self):
        hashed = hash_password("secret123")
        assert verify_password("secret123", hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("secret123")
        assert verify_password("wrong", hashed) is False

    def test_different_hashes_for_same_password(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_invalid_hash_format(self):
        assert verify_password("test", "invalid") is False


class TestTokenManager:
    def test_create_and_validate(self):
        tm = TokenManager()
        token = tm.create_token("user1")
        info = tm.validate_token(token)
        assert info is not None
        assert info["username"] == "user1"

    def test_invalid_token(self):
        tm = TokenManager()
        assert tm.validate_token("invalid") is None

    def test_revoke_token(self):
        tm = TokenManager()
        token = tm.create_token("user1")
        tm.revoke_token(token)
        assert tm.validate_token(token) is None

    def test_expired_token(self):
        tm = TokenManager()
        token = tm.create_token("user1", expires_in=-1)
        assert tm.validate_token(token) is None


class TestAuthManager:
    def test_register_user(self):
        am = AuthManager()
        user = am.register_user("alice", "password123")
        assert user.username == "alice"
        assert user.has_role("user")

    def test_register_duplicate_user(self):
        am = AuthManager()
        am.register_user("alice", "password123")
        with pytest.raises(AuthError, match="already exists"):
            am.register_user("alice", "password456")

    def test_register_short_password(self):
        am = AuthManager()
        with pytest.raises(AuthError, match="4 characters"):
            am.register_user("alice", "ab")

    def test_authenticate_success(self):
        am = AuthManager()
        am.register_user("alice", "password123")
        token = am.authenticate("alice", "password123")
        assert token is not None

    def test_authenticate_wrong_password(self):
        am = AuthManager()
        am.register_user("alice", "password123")
        with pytest.raises(AuthError, match="Invalid"):
            am.authenticate("alice", "wrong")

    def test_authenticate_unknown_user(self):
        am = AuthManager()
        with pytest.raises(AuthError, match="Invalid"):
            am.authenticate("unknown", "password")

    def test_validate_token(self):
        am = AuthManager()
        am.register_user("alice", "password123")
        token = am.authenticate("alice", "password123")
        user = am.validate_token(token)
        assert user is not None
        assert user.username == "alice"

    def test_authenticate_api_key(self):
        am = AuthManager()
        user = am.register_user("alice", "password123")
        found = am.authenticate_api_key(user.api_key)
        assert found is not None
        assert found.username == "alice"

    def test_invalid_api_key(self):
        am = AuthManager()
        assert am.authenticate_api_key("invalid_key") is None

    def test_revoke_token(self):
        am = AuthManager()
        am.register_user("alice", "password123")
        token = am.authenticate("alice", "password123")
        am.revoke_token(token)
        assert am.validate_token(token) is None

    def test_user_roles(self):
        am = AuthManager()
        user = am.register_user("admin", "pass123", roles=["user", "admin"])
        assert user.has_role("admin") is True
        assert user.has_role("superuser") is False
