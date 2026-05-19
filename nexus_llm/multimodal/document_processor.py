"""Document Processor for Nexus-LLM.

Provides document loading, text extraction, file type detection,
and format-aware processing for PDF, DOCX, plain text, and other
document formats.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentContent:
    """Extracted content from a document."""
    text: str = ""
    pages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    format: Optional[DocumentFormat] = None
    source: Optional[str] = None
    page_count: int = 0

    @property
    def word_count(self) -> int:
        """Approximate word count of the extracted text."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Character count of the extracted text."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "pages": self.pages,
            "metadata": self.metadata,
            "format": self.format.value if self.format else None,
            "source": self.source,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }


@dataclass
class DocumentChunk:
    """A chunk of document content for RAG or further processing."""
    text: str
    index: int = 0
    page: Optional[int] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "page": self.page,
            "source": self.source,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

# Magic bytes for common formats
_MAGIC_BYTES: List[Tuple[bytes, DocumentFormat]] = [
    (b"%PDF", DocumentFormat.PDF),
    (b"PK\x03\x04", DocumentFormat.DOCX),  # ZIP-based (DOCX, XLSX, etc.)
]

_DOCX_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_EXTENSIONS: Dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".doc": DocumentFormat.UNKNOWN,
    ".txt": DocumentFormat.TXT,
    ".csv": DocumentFormat.CSV,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".json": DocumentFormat.JSON,
    ".xml": DocumentFormat.XML,
}


def detect_format(path: str) -> DocumentFormat:
    """Detect the document format from a file path.

    Uses both file extension and magic byte inspection for robust
    format detection.

    Args:
        path: Path to the document file.

    Returns:
        Detected DocumentFormat enum value.
    """
    if not os.path.isfile(path):
        return DocumentFormat.UNKNOWN

    # Try magic bytes first
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        for magic, fmt in _MAGIC_BYTES:
            if header.startswith(magic):
                if fmt == DocumentFormat.DOCX:
                    return _disambiguate_zip(path, header)
                return fmt
    except OSError:
        pass

    # Fall back to extension
    _, ext = os.path.splitext(path)
    return _EXTENSIONS.get(ext.lower(), DocumentFormat.UNKNOWN)


def _disambiguate_zip(path: str, header: bytes) -> DocumentFormat:
    """Disambiguate a ZIP-based format (DOCX vs other)."""
    try:
        import zipfile
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if "word/document.xml" in names:
                return DocumentFormat.DOCX
    except Exception:
        pass
    return DocumentFormat.UNKNOWN


# ---------------------------------------------------------------------------
# Document Processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """Document processing for multimodal workflows.

    Handles text extraction from PDF, DOCX, and various plain-text
    formats.  Includes chunking support for RAG pipelines.

    Example::

        proc = DocumentProcessor()
        content = proc.load("report.pdf")
        print(content.text)
        chunks = proc.chunk(content, chunk_size=500)
    """

    def __init__(self, chunk_overlap: int = 50, default_chunk_size: int = 1000) -> None:
        self._chunk_overlap = chunk_overlap
        self._default_chunk_size = default_chunk_size

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, path: str) -> DocumentContent:
        """Load and extract text from a document file.

        Args:
            path: Path to the document.

        Returns:
            DocumentContent with extracted text, pages, and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Document not found: {path}")

        fmt = detect_format(path)

        if fmt == DocumentFormat.PDF:
            return self._load_pdf(path)
        elif fmt == DocumentFormat.DOCX:
            return self._load_docx(path)
        elif fmt == DocumentFormat.TXT:
            return self._load_text(path, DocumentFormat.TXT)
        elif fmt == DocumentFormat.CSV:
            return self._load_text(path, DocumentFormat.CSV)
        elif fmt == DocumentFormat.HTML:
            return self._load_html(path)
        elif fmt == DocumentFormat.MARKDOWN:
            return self._load_text(path, DocumentFormat.MARKDOWN)
        elif fmt == DocumentFormat.JSON:
            return self._load_text(path, DocumentFormat.JSON)
        elif fmt == DocumentFormat.XML:
            return self._load_xml(path)
        else:
            # Attempt plain text as a fallback
            try:
                return self._load_text(path, DocumentFormat.UNKNOWN)
            except Exception:
                raise ValueError(f"Unsupported document format: {path}")

    def load_text(self, text: str, source: Optional[str] = None) -> DocumentContent:
        """Create a DocumentContent directly from a text string.

        Args:
            text: The document text.
            source: Optional source label.

        Returns:
            DocumentContent wrapping the text.
        """
        return DocumentContent(
            text=text,
            pages=[text],
            metadata={},
            format=DocumentFormat.TXT,
            source=source,
            page_count=1,
        )

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _load_pdf(self, path: str) -> DocumentContent:
        """Extract text from a PDF file."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            text = "\n\n".join(pages)
            meta = dict(doc.metadata) if doc.metadata else {}
            page_count = len(doc)
            doc.close()
            return DocumentContent(
                text=text, pages=pages, metadata=meta,
                format=DocumentFormat.PDF, source=path, page_count=page_count,
            )
        except ImportError:
            pass

        # Fallback: pdfminer
        try:
            from pdfminer.high_level import extract_text, extract_pages
            from pdfminer.pdfpage import PDFPage
            pages = []
            with open(path, "rb") as f:
                page_objs = list(PDFPage.get_pages(f))
                for _ in page_objs:
                    f.seek(0)
                    pages.append(extract_text(path, page_numbers=[len(pages)]))
            text = "\n\n".join(pages) if pages else extract_text(path)
            return DocumentContent(
                text=text, pages=pages or [text], metadata={},
                format=DocumentFormat.PDF, source=path,
                page_count=len(pages),
            )
        except ImportError:
            raise ImportError(
                "PDF extraction requires PyMuPDF or pdfminer: "
                "pip install pymupdf  or  pip install pdfminer.six"
            )

    # ------------------------------------------------------------------
    # DOCX
    # ------------------------------------------------------------------

    def _load_docx(self, path: str) -> DocumentContent:
        """Extract text from a DOCX file."""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            core_props = doc.core_properties
            meta: Dict[str, Any] = {}
            if core_props.title:
                meta["title"] = core_props.title
            if core_props.author:
                meta["author"] = core_props.author
            if core_props.created:
                meta["created"] = str(core_props.created)
            if core_props.modified:
                meta["modified"] = str(core_props.modified)
            return DocumentContent(
                text=text, pages=[text], metadata=meta,
                format=DocumentFormat.DOCX, source=path, page_count=1,
            )
        except ImportError:
            # Fallback: parse the ZIP XML directly
            return self._load_docx_xml(path)

    def _load_docx_xml(self, path: str) -> DocumentContent:
        """Fallback DOCX extraction by parsing the raw XML."""
        import zipfile
        if not HAS_XML:
            raise ImportError(
                "DOCX extraction requires python-docx or xml.etree.ElementTree"
            )
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("word/document.xml") as doc_xml:
                tree = ET.parse(doc_xml)

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for p in tree.findall(".//w:p", ns):
            texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
            if texts:
                paragraphs.append("".join(texts))

        text = "\n\n".join(paragraphs)
        return DocumentContent(
            text=text, pages=[text], metadata={},
            format=DocumentFormat.DOCX, source=path, page_count=1,
        )

    # ------------------------------------------------------------------
    # Plain text / CSV / Markdown / JSON
    # ------------------------------------------------------------------

    def _load_text(self, path: str, fmt: DocumentFormat) -> DocumentContent:
        """Load a plain-text variant file."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        if fmt == DocumentFormat.CSV:
            text = self._csv_to_text(text)

        return DocumentContent(
            text=text, pages=[text], metadata={},
            format=fmt, source=path, page_count=1,
        )

    @staticmethod
    def _csv_to_text(csv_text: str) -> str:
        """Convert CSV content to a readable text representation."""
        import csv as csv_mod
        reader = csv_mod.reader(io.StringIO(csv_text))
        rows = list(reader)
        if not rows:
            return ""
        headers = rows[0]
        lines = []
        for row in rows[1:]:
            parts = [f"{h}: {v}" for h, v in zip(headers, row) if h and v]
            if parts:
                lines.append("; ".join(parts))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _load_html(self, path: str) -> DocumentContent:
        """Extract text from an HTML file."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()

        text = self._strip_html(html)
        return DocumentContent(
            text=text, pages=[text], metadata={"original_format": "html"},
            format=DocumentFormat.HTML, source=path, page_count=1,
        )

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags and decode entities."""
        # Remove script and style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Replace block tags with newlines
        html = re.sub(r"<(br|p|div|li|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Remove remaining tags
        html = re.sub(r"<[^>]+>", "", html)
        # Decode common entities
        for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
            html = html.replace(entity, char)
        # Collapse whitespace
        html = re.sub(r"\n\s*\n", "\n\n", html)
        return html.strip()

    # ------------------------------------------------------------------
    # XML
    # ------------------------------------------------------------------

    def _load_xml(self, path: str) -> DocumentContent:
        """Extract text content from an XML file."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        if not HAS_XML:
            text = re.sub(r"<[^>]+>", "", raw)
        else:
            root = ET.fromstring(raw)
            texts = [elem.text for elem in root.iter() if elem.text and elem.text.strip()]
            text = "\n".join(texts)

        return DocumentContent(
            text=text, pages=[text], metadata={},
            format=DocumentFormat.XML, source=path, page_count=1,
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk(
        self,
        content: DocumentContent,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Split document content into overlapping chunks.

        Args:
            content: DocumentContent to chunk.
            chunk_size: Maximum characters per chunk.
            overlap: Overlap characters between consecutive chunks.

        Returns:
            List of DocumentChunk objects.
        """
        cs = chunk_size or self._default_chunk_size
        ov = overlap if overlap is not None else self._chunk_overlap
        text = content.text
        chunks: List[DocumentChunk] = []

        if len(text) <= cs:
            chunks.append(DocumentChunk(
                text=text, index=0, source=content.source,
                metadata=content.metadata,
            ))
            return chunks

        start = 0
        idx = 0
        while start < len(text):
            end = start + cs
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(".")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > cs // 2:
                    end = start + break_point + 1
                    chunk_text = text[start:end]

            chunks.append(DocumentChunk(
                text=chunk_text.strip(),
                index=idx,
                source=content.source,
                metadata=content.metadata,
            ))
            idx += 1
            start = end - ov

        return chunks
