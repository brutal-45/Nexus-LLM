"""CORS configuration: origins, methods, headers, credentials."""

import logging
from typing import Any, Dict, List, Optional, Sequence

from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("nexus_llm.api.cors")


class CORSConfig:
    """Configuration for Cross-Origin Resource Sharing (CORS).

    Provides fine-grained control over allowed origins, methods,
    headers, and credential policies.
    """

    def __init__(
        self,
        allow_origins: Optional[List[str]] = None,
        allow_methods: Optional[List[str]] = None,
        allow_headers: Optional[List[str]] = None,
        expose_headers: Optional[List[str]] = None,
        allow_credentials: bool = False,
        max_age: int = 86400,
        allow_origin_regex: Optional[str] = None,
    ):
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or [
            "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD",
        ]
        self.allow_headers = allow_headers or [
            "Accept",
            "Accept-Encoding",
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "X-Process-Time-Ms",
        ]
        self.expose_headers = expose_headers or [
            "X-Request-ID",
            "X-Process-Time-Ms",
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.allow_origin_regex = allow_origin_regex

    def to_cors_middleware_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for FastAPI CORSMiddleware.

        Returns:
            Dictionary suitable for app.add_middleware(CORSMiddleware, **kwargs).
        """
        kwargs: Dict[str, Any] = {
            "allow_origins": self.allow_origins,
            "allow_methods": self.allow_methods,
            "allow_headers": self.allow_headers,
            "expose_headers": self.expose_headers,
            "allow_credentials": self.allow_credentials,
            "max_age": self.max_age,
        }
        if self.allow_origin_regex:
            kwargs["allow_origin_regex"] = self.allow_origin_regex
        return kwargs

    def validate(self) -> List[str]:
        """Validate the CORS configuration.

        Returns:
            List of warning messages for potential issues.
        """
        warnings: List[str] = []

        if "*" in self.allow_origins and self.allow_credentials:
            warnings.append(
                "Wildcard origin (*) with allow_credentials=True is not recommended "
                "and may be rejected by browsers. Specify explicit origins instead."
            )

        if "*" in self.allow_origins:
            warnings.append(
                "Wildcard origin (*) allows any domain to access the API. "
                "Consider restricting to specific domains in production."
            )

        if self.max_age < 0:
            warnings.append("max_age should be a positive integer.")

        if self.max_age > 604800:
            warnings.append(
                "max_age exceeds 7 days. Some browsers may not cache preflight "
                "responses for that long."
            )

        return warnings

    def add_origin(self, origin: str) -> None:
        """Add an allowed origin.

        Args:
            origin: Origin URL to allow (e.g., 'https://example.com').
        """
        if "*" in self.allow_origins:
            self.allow_origins.remove("*")
        if origin not in self.allow_origins:
            self.allow_origins.append(origin)

    def remove_origin(self, origin: str) -> None:
        """Remove an allowed origin.

        Args:
            origin: Origin URL to remove.
        """
        if origin in self.allow_origins:
            self.allow_origins.remove(origin)

    def add_header(self, header: str) -> None:
        """Add an allowed request header.

        Args:
            header: Header name to allow.
        """
        if header not in self.allow_headers:
            self.allow_headers.append(header)

    def add_expose_header(self, header: str) -> None:
        """Add a header to expose to the client.

        Args:
            header: Header name to expose.
        """
        if header not in self.expose_headers:
            self.expose_headers.append(header)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as a dictionary."""
        return {
            "allow_origins": self.allow_origins,
            "allow_methods": self.allow_methods,
            "allow_headers": self.allow_headers,
            "expose_headers": self.expose_headers,
            "allow_credentials": self.allow_credentials,
            "max_age": self.max_age,
            "allow_origin_regex": self.allow_origin_regex,
        }


class CORSConfigBuilder:
    """Builder pattern for constructing CORS configurations."""

    def __init__(self) -> None:
        self._origins: List[str] = []
        self._methods: List[str] = []
        self._headers: List[str] = []
        self._expose_headers: List[str] = []
        self._credentials: bool = False
        self._max_age: int = 86400
        self._origin_regex: Optional[str] = None

    def allow_all_origins(self) -> "CORSConfigBuilder":
        """Allow all origins (wildcard)."""
        self._origins = ["*"]
        return self

    def add_origin(self, origin: str) -> "CORSConfigBuilder":
        """Add a specific allowed origin."""
        if "*" in self._origins:
            self._origins.remove("*")
        self._origins.append(origin)
        return self

    def add_origins(self, origins: List[str]) -> "CORSConfigBuilder":
        """Add multiple allowed origins."""
        for origin in origins:
            self.add_origin(origin)
        return self

    def allow_all_methods(self) -> "CORSConfigBuilder":
        """Allow all standard HTTP methods."""
        self._methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"]
        return self

    def add_method(self, method: str) -> "CORSConfigBuilder":
        """Add an allowed HTTP method."""
        method = method.upper()
        if method not in self._methods:
            self._methods.append(method)
        return self

    def allow_all_headers(self) -> "CORSConfigBuilder":
        """Allow all headers (wildcard)."""
        self._headers = ["*"]
        return self

    def add_header(self, header: str) -> "CORSConfigBuilder":
        """Add an allowed request header."""
        if "*" in self._headers:
            self._headers.remove("*")
        self._headers.append(header)
        return self

    def expose_header(self, header: str) -> "CORSConfigBuilder":
        """Add a header to expose to clients."""
        self._expose_headers.append(header)
        return self

    def allow_credentials(self, enabled: bool = True) -> "CORSConfigBuilder":
        """Set whether credentials (cookies, auth) are allowed."""
        self._credentials = enabled
        return self

    def max_age(self, seconds: int) -> "CORSConfigBuilder":
        """Set the preflight cache duration in seconds."""
        self._max_age = seconds
        return self

    def origin_regex(self, pattern: str) -> "CORSConfigBuilder":
        """Set a regex pattern for matching origins."""
        self._origin_regex = pattern
        return self

    def development_preset(self) -> "CORSConfigBuilder":
        """Apply development-friendly CORS settings."""
        return (
            self.allow_all_origins()
            .allow_all_methods()
            .allow_all_headers()
            .max_age(600)
        )

    def production_preset(
        self,
        domains: Optional[List[str]] = None,
    ) -> "CORSConfigBuilder":
        """Apply production-hardened CORS settings.

        Args:
            domains: List of allowed domains.
        """
        origins = domains or []
        for d in origins:
            self.add_origin(d)
        self.add_method("GET")
        self.add_method("POST")
        self.add_method("OPTIONS")
        self.add_header("Content-Type")
        self.add_header("Authorization")
        self.add_header("X-API-Key")
        self.expose_header("X-Request-ID")
        self.allow_credentials(True)
        self.max_age(86400)
        return self

    def build(self) -> CORSConfig:
        """Build the CORSConfig instance.

        Returns:
            Configured CORSConfig.
        """
        return CORSConfig(
            allow_origins=self._origins or ["*"],
            allow_methods=self._methods or ["GET", "POST", "OPTIONS"],
            allow_headers=self._headers or ["Content-Type", "Authorization"],
            expose_headers=self._expose_headers or ["X-Request-ID"],
            allow_credentials=self._credentials,
            max_age=self._max_age,
            allow_origin_regex=self._origin_regex,
        )


def setup_cors(app: Any, config: Optional[CORSConfig] = None) -> None:
    """Configure and add CORS middleware to a FastAPI application.

    Args:
        app: FastAPI application instance.
        config: CORSConfig instance. Uses development defaults if None.
    """
    if config is None:
        config = CORSConfigBuilder().development_preset().build()

    warnings = config.validate()
    for warning in warnings:
        logger.warning("CORS configuration warning: %s", warning)

    kwargs = config.to_cors_middleware_kwargs()
    app.add_middleware(CORSMiddleware, **kwargs)

    logger.info(
        "CORS middleware configured: origins=%s, methods=%s, credentials=%s",
        config.allow_origins,
        config.allow_methods,
        config.allow_credentials,
    )


def create_development_cors() -> CORSConfig:
    """Create a permissive CORS config for development."""
    return CORSConfigBuilder().development_preset().build()


def create_production_cors(domains: Optional[List[str]] = None) -> CORSConfig:
    """Create a restrictive CORS config for production."""
    return CORSConfigBuilder().production_preset(domains).build()
