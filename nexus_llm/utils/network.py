"""Network utilities: download with progress, retry, HTTP client, proxy support."""

import os
import time
import logging
from typing import Optional, Dict, Any, Callable, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class HttpClient:
    """HTTP client with retry logic, proxy support, and session management."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        retry_status_codes: Optional[list] = None,
        proxies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.proxies = proxies
        self.headers = headers or {}
        self.verify_ssl = verify_ssl

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff,
            status_forcelist=retry_status_codes or [429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD"],
        )

        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if self.proxies:
            self.session.proxies.update(self.proxies)
        if self.headers:
            self.session.headers.update(self.headers)
        self.session.verify = self.verify_ssl

    def _build_url(self, path: str) -> str:
        """Build full URL from base URL and path."""
        if self.base_url:
            return f"{self.base_url}/{path.lstrip('/')}"
        return path

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> requests.Response:
        """Send a GET request."""
        url = self._build_url(path)
        response = self.session.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> requests.Response:
        """Send a POST request."""
        url = self._build_url(path)
        response = self.session.post(
            url,
            json=json,
            data=data,
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> requests.Response:
        """Send a PUT request."""
        url = self._build_url(path)
        response = self.session.put(
            url,
            json=json,
            data=data,
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def delete(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> requests.Response:
        """Send a DELETE request."""
        url = self._build_url(path)
        response = self.session.delete(
            url,
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def download(
        self,
        url: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 8192,
    ) -> str:
        """Download a file from a URL with progress tracking.

        Args:
            url: URL to download from.
            output_path: Local path to save the file.
            progress_callback: Optional callback(downloaded_bytes, total_bytes).
            chunk_size: Download chunk size in bytes.

        Returns:
            Path to the downloaded file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        response = self.session.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

        logger.info(f"Downloaded {url} -> {output_path} ({downloaded} bytes)")
        return output_path

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def download_file(
    url: str,
    output_path: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    chunk_size: int = 8192,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    proxies: Optional[Dict[str, str]] = None,
    verify_ssl: bool = True,
) -> str:
    """Download a file from a URL with retry logic and progress tracking.

    Args:
        url: URL to download.
        output_path: Local file path to save to.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.
        chunk_size: Download chunk size.
        progress_callback: Optional callback(downloaded_bytes, total_bytes).
        proxies: Optional proxy configuration.
        verify_ssl: Whether to verify SSL certificates.

    Returns:
        Path to the downloaded file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    last_error = None
    for attempt in range(max_retries):
        try:
            client = HttpClient(proxies=proxies, verify_ssl=verify_ssl)
            return client.download(
                url, output_path,
                progress_callback=progress_callback,
                chunk_size=chunk_size,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Download failed after {max_retries} attempts: {e}")

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {last_error}")


def is_url(path: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_filename_from_url(url: str) -> str:
    """Extract filename from a URL."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    return filename if filename else "downloaded_file"
