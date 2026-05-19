"""Test model downloader for Nexus-LLM."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional


class DownloadError(Exception):
    pass


@dataclass
class DownloadConfig:
    model_name: str = ""
    save_dir: str = "/tmp/nexus_models"
    chunk_size: int = 8192
    max_retries: int = 3
    verify_checksum: bool = True


class ModelDownloader:
    def __init__(self, config: DownloadConfig = None):
        self._config = config or DownloadConfig()
        self._progress = 0.0
        self._downloaded_bytes = 0

    @property
    def progress(self):
        return self._progress

    def get_model_path(self, model_name: str = None) -> str:
        name = model_name or self._config.model_name
        if not name:
            raise DownloadError("Model name is required")
        return os.path.join(self._config.save_dir, name)

    def validate_url(self, url: str) -> bool:
        if not url:
            return False
        return url.startswith(("http://", "https://"))

    def parse_model_id(self, model_id: str) -> tuple:
        if "/" not in model_id:
            raise DownloadError("Model ID must be in 'org/model' format")
        parts = model_id.split("/")
        if len(parts) != 2 or not all(parts):
            raise DownloadError("Invalid model ID format")
        return parts[0], parts[1]

    def compute_download_size(self, file_sizes: dict) -> int:
        return sum(file_sizes.values())

    def format_size(self, size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def simulate_download(self, total_size: int) -> float:
        self._downloaded_bytes = total_size
        self._progress = 1.0
        return self._progress


class TestDownloadConfig:
    def test_defaults(self):
        config = DownloadConfig()
        assert config.chunk_size == 8192
        assert config.max_retries == 3
        assert config.verify_checksum is True

    def test_custom(self):
        config = DownloadConfig(model_name="test-model", max_retries=5)
        assert config.model_name == "test-model"


class TestModelDownloader:
    def test_get_model_path(self):
        dl = ModelDownloader(DownloadConfig(model_name="gpt2"))
        path = dl.get_model_path()
        assert "gpt2" in path

    def test_get_model_path_no_name(self):
        dl = ModelDownloader()
        with pytest.raises(DownloadError, match="required"):
            dl.get_model_path()

    def test_validate_url_https(self):
        dl = ModelDownloader()
        assert dl.validate_url("https://example.com/model") is True

    def test_validate_url_http(self):
        dl = ModelDownloader()
        assert dl.validate_url("http://example.com/model") is True

    def test_validate_url_invalid(self):
        dl = ModelDownloader()
        assert dl.validate_url("ftp://example.com") is False
        assert dl.validate_url("") is False

    def test_parse_model_id(self):
        dl = ModelDownloader()
        org, name = dl.parse_model_id("org/model-name")
        assert org == "org"
        assert name == "model-name"

    def test_parse_model_id_no_slash(self):
        dl = ModelDownloader()
        with pytest.raises(DownloadError, match="org/model"):
            dl.parse_model_id("just-a-name")

    def test_parse_model_id_empty_parts(self):
        dl = ModelDownloader()
        with pytest.raises(DownloadError, match="Invalid"):
            dl.parse_model_id("/model")

    def test_compute_download_size(self):
        dl = ModelDownloader()
        sizes = {"model.bin": 1000, "config.json": 50, "tokenizer.json": 200}
        assert dl.compute_download_size(sizes) == 1250

    def test_format_size(self):
        dl = ModelDownloader()
        assert "B" in dl.format_size(100)
        assert "KB" in dl.format_size(2048)
        assert "MB" in dl.format_size(1024 * 1024)
        assert "GB" in dl.format_size(1024**3)

    def test_simulate_download(self):
        dl = ModelDownloader()
        progress = dl.simulate_download(1000000)
        assert progress == 1.0
        assert dl.progress == 1.0
