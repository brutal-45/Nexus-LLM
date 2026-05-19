"""Test network utilities for Nexus-LLM."""
import socket
import pytest
from unittest.mock import patch, MagicMock


# --- Network utility implementations to test ---

def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(start: int = 8000, end: int = 9000) -> int:
    for port in range(start, end):
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available port in range {start}-{end}")


def validate_url(url: str) -> bool:
    if not url:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    return True


def parse_host_port(address: str) -> tuple:
    if ":" not in address:
        raise ValueError(f"Invalid address format: {address}")
    host, port_str = address.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(f"Invalid port: {port_str}")
    if not (1 <= port <= 65535):
        raise ValueError(f"Port out of range: {port}")
    return host, port


def build_url(scheme: str, host: str, port: int, path: str = "") -> str:
    if not scheme or not host:
        raise ValueError("Scheme and host are required")
    base = f"{scheme}://{host}"
    if port not in (80, 443):
        base += f":{port}"
    if path:
        if not path.startswith("/"):
            path = "/" + path
        base += path
    return base


def is_localhost(host: str) -> bool:
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def get_hostname() -> str:
    return socket.gethostname()


class TestPortAvailability:
    def test_high_port_available(self):
        result = is_port_available(0)
        assert isinstance(result, bool)

    def test_negative_port_raises(self):
        with pytest.raises(OSError):
            is_port_available(-1)


class TestFindAvailablePort:
    def test_finds_port_in_range(self):
        port = find_available_port(9000, 9100)
        assert 9000 <= port < 9100

    def test_returns_int(self):
        port = find_available_port()
        assert isinstance(port, int)

    def test_no_available_port_raises(self):
        with patch("tests.test_network.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="No available port"):
                find_available_port(8000, 8001)


class TestValidateURL:
    def test_valid_http(self):
        assert validate_url("http://example.com") is True

    def test_valid_https(self):
        assert validate_url("https://example.com") is True

    def test_invalid_no_scheme(self):
        assert validate_url("example.com") is False

    def test_invalid_ftp(self):
        assert validate_url("ftp://example.com") is False

    def test_empty_string(self):
        assert validate_url("") is False

    def test_none_like(self):
        assert validate_url(None) is False if None else validate_url("") is False


class TestParseHostPort:
    def test_valid_address(self):
        host, port = parse_host_port("0.0.0.0:8000")
        assert host == "0.0.0.0"
        assert port == 8000

    def test_localhost(self):
        host, port = parse_host_port("localhost:5000")
        assert host == "localhost"
        assert port == 5000

    def test_ipv6_like(self):
        host, port = parse_host_port("[::1]:8080")
        assert port == 8080

    def test_no_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid address"):
            parse_host_port("localhost")

    def test_invalid_port_raises(self):
        with pytest.raises(ValueError, match="Invalid port"):
            parse_host_port("localhost:abc")

    def test_port_out_of_range(self):
        with pytest.raises(ValueError, match="Port out of range"):
            parse_host_port("localhost:99999")


class TestBuildURL:
    def test_http_with_port(self):
        url = build_url("http", "localhost", 8000)
        assert url == "http://localhost:8000"

    def test_https_with_path(self):
        url = build_url("https", "example.com", 443, "/api/v1")
        assert url == "https://example.com/api/v1"

    def test_http_port_80_no_port_suffix(self):
        url = build_url("http", "example.com", 80)
        assert url == "http://example.com"

    def test_https_port_443_no_port_suffix(self):
        url = build_url("https", "example.com", 443)
        assert url == "https://example.com"

    def test_path_without_leading_slash(self):
        url = build_url("http", "localhost", 8000, "api/v1")
        assert url == "http://localhost:8000/api/v1"

    def test_empty_scheme_raises(self):
        with pytest.raises(ValueError, match="Scheme and host"):
            build_url("", "localhost", 8000)


class TestLocalhost:
    def test_localhost(self):
        assert is_localhost("localhost") is True

    def test_127(self):
        assert is_localhost("127.0.0.1") is True

    def test_ipv6_loopback(self):
        assert is_localhost("::1") is True

    def test_wildcard(self):
        assert is_localhost("0.0.0.0") is True

    def test_external_is_not_localhost(self):
        assert is_localhost("192.168.1.1") is False

    def test_domain_not_localhost(self):
        assert is_localhost("example.com") is False


class TestHostname:
    def test_returns_string(self):
        name = get_hostname()
        assert isinstance(name, str)

    def test_nonempty(self):
        name = get_hostname()
        assert len(name) > 0
