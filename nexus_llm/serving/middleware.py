"""Middleware classes for Nexus-LLM serving."""

import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Rate-limits requests per IP address.

    Uses a sliding-window counter to enforce a maximum number of
    requests per minute per client IP.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self._max_rpm = requests_per_minute
        self._windows: Dict[str, List[float]] = defaultdict(list)
        self._blocked: Set[str] = set()

    def is_allowed(self, client_ip: str) -> bool:
        """Check whether a request from the given IP is allowed.

        Args:
            client_ip: Client IP address.

        Returns:
            ``True`` if the request is within rate limits.
        """
        now = time.time()
        window_start = now - 60.0  # 1-minute sliding window

        # Prune old entries
        timestamps = self._windows[client_ip]
        self._windows[client_ip] = [
            ts for ts in timestamps if ts > window_start
        ]

        if len(self._windows[client_ip]) >= self._max_rpm:
            self._blocked.add(client_ip)
            return False

        self._windows[client_ip].append(now)
        self._blocked.discard(client_ip)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Return the number of remaining requests in the current window."""
        now = time.time()
        window_start = now - 60.0
        count = sum(
            1 for ts in self._windows.get(client_ip, [])
            if ts > window_start
        )
        return max(0, self._max_rpm - count)

    def get_blocked_ips(self) -> Set[str]:
        """Return the set of currently blocked IPs."""
        return set(self._blocked)

    def reset(self, client_ip: Optional[str] = None) -> None:
        """Reset rate-limit counters.

        Args:
            client_ip: If provided, reset only that IP. Otherwise reset all.
        """
        if client_ip:
            self._windows.pop(client_ip, None)
            self._blocked.discard(client_ip)
        else:
            self._windows.clear()
            self._blocked.clear()


class AuthMiddleware:
    """Validates API keys for authenticated access.

    Supports a static set of valid keys and optional per-key metadata.
    """

    def __init__(
        self,
        valid_keys: Optional[Set[str]] = None,
        header_name: str = "X-API-Key",
    ) -> None:
        self._valid_keys: Set[str] = valid_keys or set()
        self._header_name = header_name
        self._key_metadata: Dict[str, Dict[str, Any]] = {}

    def add_key(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a valid API key.

        Args:
            key: The API key string.
            metadata: Optional metadata (e.g. owner, rate limit).
        """
        self._valid_keys.add(key)
        if metadata:
            self._key_metadata[key] = metadata

    def remove_key(self, key: str) -> None:
        """Revoke an API key."""
        self._valid_keys.discard(key)
        self._key_metadata.pop(key, None)

    def validate(self, api_key: Optional[str]) -> bool:
        """Validate an API key.

        Args:
            api_key: The key to validate.

        Returns:
            ``True`` if the key is valid.
        """
        if not self._valid_keys:
            # No keys configured = auth disabled
            return True
        return api_key in self._valid_keys

    def get_key_metadata(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Return metadata associated with a key."""
        return self._key_metadata.get(api_key)

    @property
    def header_name(self) -> str:
        """The HTTP header name where the API key is expected."""
        return self._header_name


class LoggingMiddleware:
    """Logs request and response details.

    Captures method, path, status code, duration, and request IDs
    for every request passing through.
    """

    def __init__(self, log_bodies: bool = False, max_body_length: int = 1000) -> None:
        self._log_bodies = log_bodies
        self._max_body_length = max_body_length
        self._request_log: List[Dict[str, Any]] = []

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        request_id: Optional[str] = None,
        body: Optional[str] = None,
    ) -> str:
        """Log a completed request.

        Args:
            method: HTTP method.
            path: Request path.
            status_code: Response status code.
            duration: Request duration in seconds.
            request_id: Optional request ID (auto-generated if missing).
            body: Optional request body snippet.

        Returns:
            The request ID.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        entry: Dict[str, Any] = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            "timestamp": time.time(),
        }

        if self._log_bodies and body:
            entry["body"] = body[: self._max_body_length]

        self._request_log.append(entry)
        logger.info(
            "[%s] %s %s %d %.1fms",
            request_id[:8], method, path, status_code, duration * 1000,
        )
        return request_id

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent request log entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of log entry dicts (most recent last).
        """
        return list(self._request_log[-limit:])

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate logging statistics."""
        if not self._request_log:
            return {
                "total_requests": 0,
                "avg_duration_ms": 0.0,
                "status_codes": {},
            }

        durations = [e["duration_ms"] for e in self._request_log]
        codes: Dict[int, int] = defaultdict(int)
        for e in self._request_log:
            codes[e["status_code"]] += 1

        return {
            "total_requests": len(self._request_log),
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "status_codes": dict(codes),
        }

    def clear(self) -> None:
        """Clear the request log."""
        self._request_log.clear()


class CORSMiddleware:
    """Handles Cross-Origin Resource Sharing (CORS).

    Manages allowed origins, methods, and headers for browser-based
    API access.
    """

    def __init__(
        self,
        allow_origins: Optional[List[str]] = None,
        allow_methods: Optional[List[str]] = None,
        allow_headers: Optional[List[str]] = None,
        allow_credentials: bool = False,
        max_age: int = 600,
    ) -> None:
        self._allow_origins = set(allow_origins or ["*"])
        self._allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self._allow_headers = allow_headers or ["Content-Type", "Authorization", "X-API-Key"]
        self._allow_credentials = allow_credentials
        self._max_age = max_age

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is permitted.

        Args:
            origin: The ``Origin`` header value.

        Returns:
            ``True`` if the origin is allowed.
        """
        if "*" in self._allow_origins:
            return True
        return origin in self._allow_origins

    def get_cors_headers(self, origin: Optional[str] = None) -> Dict[str, str]:
        """Generate CORS response headers.

        Args:
            origin: The request ``Origin`` header value.

        Returns:
            Dict of CORS headers to include in the response.
        """
        headers: Dict[str, str] = {}

        if origin and self.is_origin_allowed(origin):
            if "*" in self._allow_origins:
                headers["Access-Control-Allow-Origin"] = "*"
            else:
                headers["Access-Control-Allow-Origin"] = origin

        headers["Access-Control-Allow-Methods"] = ", ".join(self._allow_methods)
        headers["Access-Control-Allow-Headers"] = ", ".join(self._allow_headers)
        headers["Access-Control-Max-Age"] = str(self._max_age)

        if self._allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"

        return headers

    def add_origin(self, origin: str) -> None:
        """Add an allowed origin."""
        self._allow_origins.discard("*")
        self._allow_origins.add(origin)

    def remove_origin(self, origin: str) -> None:
        """Remove an allowed origin."""
        self._allow_origins.discard(origin)
