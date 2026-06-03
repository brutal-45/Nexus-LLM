"""Test CORS for Nexus-LLM."""
import pytest
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CORSConfig:
    allow_origins: List[str] = None
    allow_methods: List[str] = None
    allow_headers: List[str] = None
    expose_headers: List[str] = None
    allow_credentials: bool = False
    max_age: int = 600

    def __post_init__(self):
        if self.allow_origins is None:
            self.allow_origins = ["*"]
        if self.allow_methods is None:
            self.allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        if self.allow_headers is None:
            self.allow_headers = ["Content-Type", "Authorization"]
        if self.expose_headers is None:
            self.expose_headers = []


class CORSHandler:
    def __init__(self, config: CORSConfig = None):
        self._config = config or CORSConfig()

    @property
    def config(self):
        return self._config

    def is_origin_allowed(self, origin: str) -> bool:
        if "*" in self._config.allow_origins:
            return True
        return origin in self._config.allow_origins

    def is_method_allowed(self, method: str) -> bool:
        return method.upper() in [m.upper() for m in self._config.allow_methods]

    def is_header_allowed(self, header: str) -> bool:
        if "*" in self._config.allow_headers:
            return True
        return header.lower() in [h.lower() for h in self._config.allow_headers]

    def get_preflight_headers(self, origin: str, request_method: str, request_headers: List[str] = None) -> Dict[str, str]:
        headers = {}
        if not self.is_origin_allowed(origin):
            return headers
        headers["Access-Control-Allow-Origin"] = origin if "*" not in self._config.allow_origins else "*"
        if self.is_method_allowed(request_method):
            headers["Access-Control-Allow-Methods"] = ", ".join(self._config.allow_methods)
        if request_headers:
            allowed = [h for h in request_headers if self.is_header_allowed(h)]
            if allowed:
                headers["Access-Control-Allow-Headers"] = ", ".join(allowed)
        if self._config.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Max-Age"] = str(self._config.max_age)
        return headers

    def get_response_headers(self, origin: str) -> Dict[str, str]:
        headers = {}
        if self.is_origin_allowed(origin):
            headers["Access-Control-Allow-Origin"] = origin if "*" not in self._config.allow_origins else "*"
        if self._config.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        if self._config.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self._config.expose_headers)
        return headers


class TestCORSConfig:
    def test_defaults(self):
        config = CORSConfig()
        assert "*" in config.allow_origins
        assert "GET" in config.allow_methods
        assert "POST" in config.allow_methods
        assert config.allow_credentials is False
        assert config.max_age == 600

    def test_custom(self):
        config = CORSConfig(
            allow_origins=["https://example.com"],
            allow_methods=["GET", "POST"],
            allow_credentials=True,
        )
        assert config.allow_origins == ["https://example.com"]
        assert config.allow_credentials is True


class TestCORSHandler:
    def test_wildcard_origin(self):
        handler = CORSHandler()
        assert handler.is_origin_allowed("https://any-site.com") is True

    def test_specific_origin_allowed(self):
        handler = CORSHandler(CORSConfig(allow_origins=["https://example.com"]))
        assert handler.is_origin_allowed("https://example.com") is True

    def test_specific_origin_not_allowed(self):
        handler = CORSHandler(CORSConfig(allow_origins=["https://example.com"]))
        assert handler.is_origin_allowed("https://evil.com") is False

    def test_method_allowed(self):
        handler = CORSHandler()
        assert handler.is_method_allowed("GET") is True
        assert handler.is_method_allowed("POST") is True

    def test_method_not_allowed(self):
        handler = CORSHandler(CORSConfig(allow_methods=["GET"]))
        assert handler.is_method_allowed("DELETE") is False

    def test_header_allowed(self):
        handler = CORSHandler()
        assert handler.is_header_allowed("Content-Type") is True

    def test_header_not_allowed(self):
        handler = CORSHandler(CORSConfig(allow_headers=["Authorization"]))
        assert handler.is_header_allowed("X-Custom-Header") is False

    def test_preflight_headers(self):
        handler = CORSHandler()
        headers = handler.get_preflight_headers("https://example.com", "POST", ["Content-Type"])
        assert "Access-Control-Allow-Origin" in headers
        assert "Access-Control-Allow-Methods" in headers
        assert "Access-Control-Max-Age" in headers

    def test_preflight_disallowed_origin(self):
        handler = CORSHandler(CORSConfig(allow_origins=["https://allowed.com"]))
        headers = handler.get_preflight_headers("https://evil.com", "POST")
        assert "Access-Control-Allow-Origin" not in headers

    def test_preflight_with_credentials(self):
        handler = CORSHandler(CORSConfig(allow_credentials=True))
        headers = handler.get_preflight_headers("https://example.com", "POST")
        assert headers.get("Access-Control-Allow-Credentials") == "true"

    def test_response_headers(self):
        handler = CORSHandler()
        headers = handler.get_response_headers("https://example.com")
        assert "Access-Control-Allow-Origin" in headers

    def test_response_headers_disallowed(self):
        handler = CORSHandler(CORSConfig(allow_origins=["https://allowed.com"]))
        headers = handler.get_response_headers("https://evil.com")
        assert "Access-Control-Allow-Origin" not in headers

    def test_expose_headers(self):
        handler = CORSHandler(CORSConfig(expose_headers=["X-Custom-Header"]))
        headers = handler.get_response_headers("https://example.com")
        assert "X-Custom-Header" in headers.get("Access-Control-Expose-Headers", "")
