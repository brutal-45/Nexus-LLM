"""Image Processor for Nexus-LLM.

Provides image loading, resizing, normalization, format conversion,
and metadata extraction capabilities using PIL/Pillow as the backend.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ImageFormat(str, Enum):
    """Supported image output formats."""
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    BMP = "BMP"
    GIF = "GIF"
    TIFF = "TIFF"


class ResizeMode(str, Enum):
    """Image resize strategy."""
    FIT = "fit"             # Keep aspect ratio, fit within dimensions
    CROP = "crop"           # Crop to exact dimensions
    STRETCH = "stretch"     # Stretch to exact dimensions
    PAD = "pad"             # Keep aspect ratio, pad to dimensions


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImageMetadata:
    """Metadata extracted from an image file."""
    width: int = 0
    height: int = 0
    format: Optional[str] = None
    mode: Optional[str] = None
    file_size: int = 0
    exif: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "mode": self.mode,
            "file_size": self.file_size,
            "exif": self.exif,
        }


@dataclass
class NormalizeConfig:
    """Configuration for image normalization."""
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    max_pixel_value: float = 255.0


# ---------------------------------------------------------------------------
# Image Processor
# ---------------------------------------------------------------------------

class ImageProcessor:
    """Image processing utilities for multimodal workflows.

    Provides resizing, normalization, format conversion, and metadata
    extraction backed by Pillow.

    Example::

        proc = ImageProcessor()
        img = proc.load("photo.jpg")
        resized = proc.resize(img, (224, 224), mode=ResizeMode.FIT)
        meta = proc.extract_metadata("photo.jpg")
        proc.save(resized, "photo_resized.png", format=ImageFormat.PNG)
    """

    def __init__(self, default_resize_mode: ResizeMode = ResizeMode.FIT) -> None:
        self._default_resize_mode = default_resize_mode

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self, source: Union[str, bytes, io.BytesIO]) -> "Image.Image":
        """Load an image from a file path, raw bytes, or BytesIO stream.

        Args:
            source: File path, raw image bytes, or BytesIO buffer.

        Returns:
            PIL Image object.

        Raises:
            ImportError: If Pillow is not installed.
            FileNotFoundError: If the file path does not exist.
            ValueError: If the source type is unsupported or data is corrupt.
        """
        if not HAS_PIL:
            raise ImportError("Pillow is required for image processing: pip install Pillow")

        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Image file not found: {source}")
            return Image.open(source).copy()
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).copy()
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            return Image.open(source).copy()
        else:
            raise ValueError(f"Unsupported image source type: {type(source).__name__}")

    def save(
        self,
        image: "Image.Image",
        destination: Union[str, io.BytesIO],
        format: ImageFormat = ImageFormat.PNG,
        quality: int = 95,
    ) -> Union[str, int]:
        """Save an image to a file or BytesIO stream.

        Args:
            image: PIL Image to save.
            destination: File path or BytesIO buffer.
            format: Output image format.
            quality: Compression quality (1-100) for lossy formats.

        Returns:
            File path string when saving to disk, or byte count when saving to BytesIO.
        """
        if not HAS_PIL:
            raise ImportError("Pillow is required for image processing")

        kwargs: Dict[str, Any] = {"format": format.value}
        if format in (ImageFormat.JPEG, ImageFormat.WEBP):
            kwargs["quality"] = quality

        if isinstance(destination, str):
            os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
            image.save(destination, **kwargs)
            return destination
        else:
            image.save(destination, **kwargs)
            return destination.tell()

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resize(
        self,
        image: "Image.Image",
        size: Tuple[int, int],
        mode: Optional[ResizeMode] = None,
        resample: Optional[int] = None,
        pad_color: Tuple[int, ...] = (0, 0, 0),
    ) -> "Image.Image":
        """Resize an image according to the specified strategy.

        Args:
            image: PIL Image to resize.
            size: Target (width, height).
            mode: Resize strategy. Defaults to the processor's default.
            resample: PIL resampling filter. Defaults to LANCZOS.
            pad_color: RGB color tuple used for padding in PAD mode.

        Returns:
            Resized PIL Image.
        """
        if not HAS_PIL:
            raise ImportError("Pillow is required for image processing")

        if resample is None:
            resample = Image.LANCZOS

        mode = mode or self._default_resize_mode
        target_w, target_h = size
        orig_w, orig_h = image.size

        if mode == ResizeMode.STRETCH:
            return image.resize((target_w, target_h), resample)

        if mode == ResizeMode.CROP:
            ratio = max(target_w / orig_w, target_h / orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
            resized = image.resize((new_w, new_h), resample)
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            return resized.crop((left, top, left + target_w, top + target_h))

        if mode == ResizeMode.PAD:
            ratio = min(target_w / orig_w, target_h / orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
            resized = image.resize((new_w, new_h), resample)
            canvas = Image.new(image.mode, (target_w, target_h), pad_color)
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            return canvas

        # FIT: keep aspect ratio, fit within dimensions
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        return image.resize((new_w, new_h), resample)

    # ------------------------------------------------------------------
    # Normalize / Convert
    # ------------------------------------------------------------------

    def normalize(
        self,
        image: "Image.Image",
        config: Optional[NormalizeConfig] = None,
    ) -> List[List[List[float]]]:
        """Normalize pixel values to the range expected by vision models.

        Converts the image to RGB, rescales pixel values to [0, 1], then
        applies channel-wise mean/std normalization.

        Args:
            image: PIL Image (any mode, will be converted to RGB).
            config: Normalization parameters. Uses ImageNet defaults if None.

        Returns:
            Nested list of shape [channels][height][width] with float values.
        """
        if not HAS_PIL:
            raise ImportError("Pillow is required for image processing")

        cfg = config or NormalizeConfig()
        rgb = image.convert("RGB")
        pixels = list(rgb.getdata())
        w, h = rgb.size
        channels = 3

        result = [[[0.0] * w for _ in range(h)] for _ in range(channels)]

        for y in range(h):
            for x in range(w):
                r, g, b = pixels[y * w + x]
                result[0][y][x] = (r / cfg.max_pixel_value - cfg.mean[0]) / cfg.std[0]
                result[1][y][x] = (g / cfg.max_pixel_value - cfg.mean[1]) / cfg.std[1]
                result[2][y][x] = (b / cfg.max_pixel_value - cfg.mean[2]) / cfg.std[2]

        return result

    def convert_format(
        self,
        image: "Image.Image",
        target_format: ImageFormat = ImageFormat.PNG,
        quality: int = 95,
    ) -> bytes:
        """Convert an image to a different format and return raw bytes.

        Args:
            image: PIL Image to convert.
            target_format: Desired output format.
            quality: Compression quality for lossy formats.

        Returns:
            Raw bytes of the converted image.
        """
        buf = io.BytesIO()
        self.save(image, buf, format=target_format, quality=quality)
        buf.seek(0)
        return buf.read()

    def to_base64(self, image: "Image.Image", format: ImageFormat = ImageFormat.PNG) -> str:
        """Encode an image as a base64 string.

        Args:
            image: PIL Image to encode.
            format: Output format for the encoded image.

        Returns:
            Base64-encoded string of the image data.
        """
        import base64
        raw = self.convert_format(image, format)
        return base64.b64encode(raw).decode("utf-8")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def extract_metadata(self, source: Union[str, "Image.Image"]) -> ImageMetadata:
        """Extract metadata from an image file or PIL Image.

        Args:
            source: File path or PIL Image object.

        Returns:
            ImageMetadata with dimensions, format, mode, size, and EXIF data.
        """
        if not HAS_PIL:
            raise ImportError("Pillow is required for image processing")

        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Image file not found: {source}")
            img = Image.open(source)
            file_size = os.path.getsize(source)
        else:
            img = source
            file_size = 0

        exif_data: Dict[str, Any] = {}
        raw_exif = img.getexif() if hasattr(img, "getexif") else {}
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = repr(value)
                exif_data[tag_name] = value

        return ImageMetadata(
            width=img.width,
            height=img.height,
            format=img.format if hasattr(img, "format") and img.format else None,
            mode=img.mode,
            file_size=file_size,
            exif=exif_data,
        )

    def thumbnail(
        self,
        image: "Image.Image",
        max_size: Tuple[int, int] = (256, 256),
    ) -> "Image.Image":
        """Create a thumbnail that preserves aspect ratio.

        Args:
            image: PIL Image to thumbnail.
            max_size: Maximum (width, height) of the thumbnail.

        Returns:
            A new PIL Image thumbnail.
        """
        copy = image.copy()
        copy.thumbnail(max_size, Image.LANCZOS)
        return copy
