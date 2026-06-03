"""Image processing for Nexus-LLM multimodal support."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


class ImageProcessor:
    """Handles loading, transforming, and encoding images.

    Supports loading from local paths or URLs, resizing, and
    base64 encoding/decoding for API transport.
    """

    def __init__(self, default_format: str = "PNG") -> None:
        self._default_format = default_format

    # -- Loading --------------------------------------------------------------

    def load_image(self, path_or_url: str) -> Any:
        """Load an image from a local path or URL.

        Args:
            path_or_url: Filesystem path or HTTP(S) URL.

        Returns:
            PIL ``Image`` object if Pillow is available, otherwise raw bytes.

        Raises:
            FileNotFoundError: If the local file does not exist.
            ValueError: If the source is neither a valid path nor URL.
        """
        # Try URL first
        if path_or_url.startswith(("http://", "https://")):
            return self._load_from_url(path_or_url)

        # Local path
        file_path = Path(path_or_url)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {path_or_url}")

        try:
            from PIL import Image
            return Image.open(file_path).copy()
        except ImportError:
            return file_path.read_bytes()

    # -- Transforms -----------------------------------------------------------

    def resize(
        self,
        image: Any,
        size: Union[Tuple[int, int], int],
    ) -> Any:
        """Resize an image to the given dimensions.

        Args:
            image: PIL ``Image`` object.
            size: ``(width, height)`` tuple or a single int for the
                  shorter side (maintaining aspect ratio).

        Returns:
            Resized PIL ``Image``.
        """
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required for image resizing")

        if isinstance(size, int):
            w, h = image.size
            if w < h:
                new_w = size
                new_h = int(h * size / w)
            else:
                new_h = size
                new_w = int(w * size / h)
            return image.resize((new_w, new_h), Image.LANCZOS)

        return image.resize(size, Image.LANCZOS)

    # -- Encoding -------------------------------------------------------------

    def encode_base64(self, image: Any, format: Optional[str] = None) -> str:
        """Encode a PIL Image to a base64 string.

        Args:
            image: PIL ``Image`` object.
            format: Image format (PNG, JPEG, etc.). Defaults to the
                    processor's ``default_format``.

        Returns:
            Base64-encoded string of the image bytes.
        """
        fmt = format or self._default_format

        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")

        try:
            buffer = io.BytesIO()
            image.save(buffer, format=fmt)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except AttributeError:
            raise TypeError(
                "Expected a PIL Image or bytes for encode_base64"
            )

    def decode_base64(self, data: str) -> Any:
        """Decode a base64 string back to a PIL Image.

        Args:
            data: Base64-encoded image string.

        Returns:
            PIL ``Image`` object if Pillow is available, otherwise bytes.
        """
        raw = base64.b64decode(data)
        try:
            from PIL import Image
            return Image.open(io.BytesIO(raw)).copy()
        except ImportError:
            return raw

    # -- Info -----------------------------------------------------------------

    def get_image_info(self, image: Any) -> Dict[str, Any]:
        """Extract metadata from an image.

        Args:
            image: PIL ``Image`` object.

        Returns:
            Dict with keys: ``width``, ``height``, ``mode``, ``format``,
            ``size_bytes``.
        """
        try:
            info: Dict[str, Any] = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": getattr(image, "format", None),
            }

            # Estimate size in memory
            buffer = io.BytesIO()
            fmt = getattr(image, "format", None) or self._default_format
            image.save(buffer, format=fmt)
            info["size_bytes"] = buffer.tell()

            return info
        except AttributeError:
            if isinstance(image, bytes):
                return {
                    "width": None,
                    "height": None,
                    "mode": None,
                    "format": None,
                    "size_bytes": len(image),
                }
            raise TypeError("Expected a PIL Image or bytes")

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _load_from_url(url: str) -> Any:
        """Fetch an image from a URL."""
        try:
            import urllib.request
            data = urllib.request.urlopen(url, timeout=30).read()
            from PIL import Image
            return Image.open(io.BytesIO(data)).copy()
        except ImportError:
            import urllib.request
            return urllib.request.urlopen(url, timeout=30).read()
