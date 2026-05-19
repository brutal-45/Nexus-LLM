"""OCR Text Extraction for Nexus-LLM.

Provides optical character recognition (OCR) capabilities for
extracting text from images.  Supports multiple OCR backends
including Tesseract and EasyOCR, with automatic fallback.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OCRBackend(str, Enum):
    """Supported OCR backends."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box for a text region."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class OCRWord:
    """A single recognized word with position and confidence."""
    text: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class OCRLine:
    """A line of recognized text."""
    text: str
    words: List[OCRWord] = field(default_factory=list)
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict() if self.bbox else None,
        }


@dataclass
class OCRResult:
    """Complete OCR result for an image."""
    text: str = ""
    lines: List[OCRLine] = field(default_factory=list)
    words: List[OCRWord] = field(default_factory=list)
    confidence: float = 0.0
    backend: Optional[str] = None
    processing_time: float = 0.0
    language: str = "eng"

    @property
    def word_count(self) -> int:
        return len(self.words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "lines": [l.to_dict() for l in self.lines],
            "words": [w.to_dict() for w in self.words],
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
            "processing_time": round(self.processing_time, 4),
            "language": self.language,
            "word_count": self.word_count,
        }


# ---------------------------------------------------------------------------
# OCR Engine
# ---------------------------------------------------------------------------

class OCREngine:
    """Optical character recognition engine for Nexus-LLM.

    Supports multiple backends with automatic detection and fallback.
    Tesseract is used when available, with EasyOCR as an alternative.

    Example::

        ocr = OCREngine(backend=OCRBackend.AUTO)
        result = ocr.extract("scanned_document.png")
        print(result.text)

        # With detailed word-level results
        result = ocr.extract("receipt.jpg", detail_level="word")
        for word in result.words:
            print(f"{word.text} (conf: {word.confidence:.2f})")
    """

    def __init__(
        self,
        backend: OCRBackend = OCRBackend.AUTO,
        languages: Optional[List[str]] = None,
        tesseract_config: Optional[str] = None,
    ) -> None:
        """Initialize the OCR engine.

        Args:
            backend: OCR backend to use. AUTO will try Tesseract first, then EasyOCR.
            languages: List of language codes (e.g., ["eng", "fra"]).
            tesseract_config: Additional Tesseract configuration string.
        """
        self._backend = backend
        self._languages = languages or ["eng"]
        self._tesseract_config = tesseract_config or ""
        self._easyocr_reader = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        source: Union[str, "Image.Image", bytes],
        languages: Optional[List[str]] = None,
        detail_level: str = "line",
        preprocessing: bool = True,
    ) -> OCRResult:
        """Extract text from an image using OCR.

        Args:
            source: Image file path, PIL Image, or raw bytes.
            languages: Override language codes for this call.
            detail_level: 'line' for line-level, 'word' for word-level detail.
            preprocessing: Whether to apply preprocessing (grayscale, contrast).

        Returns:
            OCRResult with extracted text, confidence, and word/line details.
        """
        import time as _time
        start = _time.time()

        image = self._load_image(source)
        if preprocessing:
            image = self._preprocess(image)

        langs = languages or self._languages
        backend = self._resolve_backend()

        if backend == OCRBackend.TESSERACT:
            result = self._extract_tesseract(image, langs, detail_level)
        elif backend == OCRBackend.EASYOCR:
            result = self._extract_easyocr(image, langs, detail_level)
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")

        result.backend = backend.value
        result.processing_time = _time.time() - start
        result.language = ",".join(langs)
        return result

    def extract_pdf(
        self,
        path: str,
        languages: Optional[List[str]] = None,
        detail_level: str = "line",
    ) -> List[OCRResult]:
        """Extract text from each page of a scanned PDF.

        Args:
            path: Path to the PDF file.
            languages: Language codes for OCR.
            detail_level: Level of detail in results.

        Returns:
            List of OCRResult, one per page.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PDF OCR requires PyMuPDF: pip install pymupdf")

        doc = fitz.open(path)
        results = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            result = self.extract(img_bytes, languages=languages, detail_level=detail_level)
            results.append(result)
        doc.close()
        return results

    def is_available(self, backend: Optional[OCRBackend] = None) -> bool:
        """Check if an OCR backend is available.

        Args:
            backend: Backend to check. Uses configured backend if None.

        Returns:
            True if the backend is installed and functional.
        """
        b = backend or self._resolve_backend()
        if b == OCRBackend.TESSERACT:
            return self._tesseract_available()
        elif b == OCRBackend.EASYOCR:
            return self._easyocr_available()
        return False

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> OCRBackend:
        """Resolve AUTO to a concrete backend."""
        if self._backend != OCRBackend.AUTO:
            return self._backend
        if self._tesseract_available():
            return OCRBackend.TESSERACT
        if self._easyocr_available():
            return OCRBackend.EASYOCR
        raise ImportError(
            "No OCR backend available. Install Tesseract or EasyOCR: "
            "pip install pytesseract  or  pip install easyocr"
        )

    @staticmethod
    def _tesseract_available() -> bool:
        """Check if Tesseract is installed."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def _easyocr_available() -> bool:
        """Check if EasyOCR is installed."""
        try:
            import easyocr
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Tesseract extraction
    # ------------------------------------------------------------------

    def _extract_tesseract(
        self,
        image: "Image.Image",
        languages: List[str],
        detail_level: str,
    ) -> OCRResult:
        """Extract text using Tesseract."""
        import pytesseract

        lang = "+".join(languages)
        config = self._tesseract_config

        if detail_level == "word":
            data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            words: List[OCRWord] = []
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                conf = float(data["conf"][i])
                if conf < 0:
                    conf = 0.0
                bbox = BoundingBox(
                    x=int(data["left"][i]),
                    y=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                )
                words.append(OCRWord(text=txt, confidence=conf / 100.0, bbox=bbox))

            # Group words into lines by y-coordinate proximity
            lines = self._group_words_into_lines(words)
            full_text = "\n".join(line.text for line in lines)
            avg_conf = sum(w.confidence for w in words) / max(len(words), 1)

            return OCRResult(
                text=full_text, lines=lines, words=words,
                confidence=avg_conf, language=",".join(languages),
            )
        else:
            # Line-level extraction
            full_text = pytesseract.image_to_string(image, lang=lang, config=config)
            line_texts = [l for l in full_text.split("\n") if l.strip()]

            lines: List[OCRLine] = []
            all_words: List[OCRWord] = []

            # Also get word data for confidence
            try:
                data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
                word_idx = 0
                for line_text in line_texts:
                    line_words = []
                    while word_idx < len(data["text"]):
                        txt = data["text"][word_idx].strip()
                        if not txt:
                            word_idx += 1
                            continue
                        conf = max(0.0, float(data["conf"][word_idx]) / 100.0)
                        bbox = BoundingBox(
                            x=int(data["left"][word_idx]),
                            y=int(data["top"][word_idx]),
                            width=int(data["width"][word_idx]),
                            height=int(data["height"][word_idx]),
                        )
                        w = OCRWord(text=txt, confidence=conf, bbox=bbox)
                        line_words.append(w)
                        all_words.append(w)
                        word_idx += 1
                        # Simple heuristic: if next word is far below, new line
                        if word_idx < len(data["top"]):
                            if data["top"][word_idx] - bbox.y > bbox.height:
                                break
                    line_conf = sum(w.confidence for w in line_words) / max(len(line_words), 1)
                    lines.append(OCRLine(text=line_text, words=line_words, confidence=line_conf))
            except Exception:
                lines = [OCRLine(text=lt) for lt in line_texts]

            avg_conf = sum(w.confidence for w in all_words) / max(len(all_words), 1)
            return OCRResult(
                text=full_text.strip(), lines=lines, words=all_words,
                confidence=avg_conf, language=",".join(languages),
            )

    # ------------------------------------------------------------------
    # EasyOCR extraction
    # ------------------------------------------------------------------

    def _extract_easyocr(
        self,
        image: "Image.Image",
        languages: List[str],
        detail_level: str,
    ) -> OCRResult:
        """Extract text using EasyOCR."""
        import easyocr

        # Map language codes for EasyOCR
        lang_map = {"eng": "en", "fra": "fr", "deu": "de", "spa": "es", "chi_sim": "ch_sim", "jpn": "ja", "kor": "ko"}
        easy_langs = [lang_map.get(l, l) for l in languages]

        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(easy_langs, gpu=False)

        import numpy as np
        img_array = np.array(image)
        raw_results = self._easyocr_reader.readtext(img_array)

        words: List[OCRWord] = []
        for bbox_pts, text, conf in raw_results:
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            bbox = BoundingBox(
                x=int(min(xs)), y=int(min(ys)),
                width=int(max(xs) - min(xs)),
                height=int(max(ys) - min(ys)),
            )
            words.append(OCRWord(text=text, confidence=conf, bbox=bbox))

        lines = self._group_words_into_lines(words)
        full_text = "\n".join(line.text for line in lines)
        avg_conf = sum(w.confidence for w in words) / max(len(words), 1)

        return OCRResult(
            text=full_text, lines=lines, words=words,
            confidence=avg_conf, language=",".join(languages),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(source: Union[str, "Image.Image", bytes]) -> "Image.Image":
        """Load an image from various source types."""
        if not HAS_PIL:
            raise ImportError("Pillow is required for OCR: pip install Pillow")

        if isinstance(source, Image.Image):
            return source
        elif isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Image file not found: {source}")
            return Image.open(source).copy()
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).copy()
        else:
            raise ValueError(f"Unsupported image source: {type(source).__name__}")

    @staticmethod
    def _preprocess(image: "Image.Image") -> "Image.Image":
        """Apply preprocessing to improve OCR accuracy."""
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")

        # Increase contrast using histogram equalization (simplified)
        try:
            from PIL import ImageOps
            image = ImageOps.autocontrast(image)
        except Exception:
            pass

        # Resize if too small (OCR works better with larger images)
        w, h = image.size
        if w < 300 or h < 300:
            scale = max(300 / w, 300 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)

        return image

    @staticmethod
    def _group_words_into_lines(words: List[OCRWord], y_threshold: float = 0.5) -> List[OCRLine]:
        """Group words into lines based on vertical proximity.

        Args:
            words: List of OCRWord objects.
            y_threshold: Threshold as fraction of average word height.

        Returns:
            List of OCRLine objects.
        """
        if not words:
            return []

        # Sort by y, then x
        sorted_words = sorted(words, key=lambda w: (w.bbox.y, w.bbox.x))

        avg_height = sum(w.bbox.height for w in sorted_words) / len(sorted_words) if sorted_words else 20
        threshold = avg_height * y_threshold

        lines: List[OCRLine] = []
        current_words: List[OCRWord] = [sorted_words[0]]

        for word in sorted_words[1:]:
            last_y = current_words[-1].bbox.y
            if abs(word.bbox.y - last_y) <= threshold:
                current_words.append(word)
            else:
                # Finish current line
                line_text = " ".join(w.text for w in sorted(current_words, key=lambda w: w.bbox.x))
                line_conf = sum(w.confidence for w in current_words) / len(current_words)
                min_x = min(w.bbox.x for w in current_words)
                min_y = min(w.bbox.y for w in current_words)
                max_x2 = max(w.bbox.x2 for w in current_words)
                max_y2 = max(w.bbox.y2 for w in current_words)
                lines.append(OCRLine(
                    text=line_text,
                    words=current_words,
                    confidence=line_conf,
                    bbox=BoundingBox(x=min_x, y=min_y, width=max_x2 - min_x, height=max_y2 - min_y),
                ))
                current_words = [word]

        # Last line
        if current_words:
            line_text = " ".join(w.text for w in sorted(current_words, key=lambda w: w.bbox.x))
            line_conf = sum(w.confidence for w in current_words) / len(current_words)
            min_x = min(w.bbox.x for w in current_words)
            min_y = min(w.bbox.y for w in current_words)
            max_x2 = max(w.bbox.x2 for w in current_words)
            max_y2 = max(w.bbox.y2 for w in current_words)
            lines.append(OCRLine(
                text=line_text,
                words=current_words,
                confidence=line_conf,
                bbox=BoundingBox(x=min_x, y=min_y, width=max_x2 - min_x, height=max_y2 - min_y),
            ))

        return lines
