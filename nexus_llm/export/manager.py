"""Export manager for Nexus-LLM.

Central facade that auto-detects the target format from the file
extension and delegates to the appropriate serialiser.
"""

import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Supported format identifiers
SUPPORTED_FORMATS = {"json", "csv", "markdown", "html", "pdf"}

# Extension → format mapping
_EXTENSION_MAP: Dict[str, str] = {
    ".json": "json",
    ".csv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


class ExportManager:
    """High-level export facade with auto-detection of output format.

    Example::

        em = ExportManager()
        em.export(data, "json", "results.json")
        em.export(data, format=None, path="report.md")   # auto-detect from extension
    """

    def export(
        self,
        data: Any,
        format: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export *data* in the given *format* and optionally write to *path*.

        If *format* is ``None`` it is inferred from *path*'s extension.
        If *path* is ``None`` the serialised string is returned without
        writing to disk.

        Args:
            data: The data to export (dict, list, or any serialisable object).
            format: One of ``"json"``, ``"csv"``, ``"markdown"``, ``"html"``,
                    ``"pdf"``.  ``None`` triggers auto-detection from *path*.
            path: Destination file path.  If provided the result is written
                  to disk and the path string is returned.

        Returns:
            The file path (if *path* was given) or the serialised string.

        Raises:
            ValueError: If the format cannot be determined or is unsupported.
        """
        # Resolve format
        if format is None and path is not None:
            ext = Path(path).suffix.lower()
            format = _EXTENSION_MAP.get(ext)
            if format is None:
                raise ValueError(
                    f"Cannot determine export format from extension {ext!r}. "
                    f"Supported extensions: {sorted(_EXTENSION_MAP.keys())}"
                )
        if format is None:
            raise ValueError(
                "Either *format* or *path* (for auto-detection) must be provided"
            )
        format = format.lower()
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {format!r}. "
                f"Supported: {sorted(SUPPORTED_FORMATS)}"
            )

        # Serialise
        result = self._serialise(data, format)

        # Write to disk if path given
        if path is not None:
            path = str(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(result)
            logger.info("Exported data to %s (format=%s)", path, format)
            return path

        return result

    # ------------------------------------------------------------------
    # Internal serialisers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise(data: Any, format: str) -> str:
        """Dispatch to the appropriate serialiser."""
        if format == "json":
            return ExportManager._to_json(data)
        if format == "csv":
            return ExportManager._to_csv(data)
        if format == "markdown":
            return ExportManager._to_markdown(data)
        if format == "html":
            return ExportManager._to_html(data)
        if format == "pdf":
            return ExportManager._to_pdf_mock(data)
        raise ValueError(f"Unhandled format: {format!r}")

    @staticmethod
    def _to_json(data: Any) -> str:
        return json.dumps(data, indent=2, default=str, ensure_ascii=False)

    @staticmethod
    def _to_csv(data: Any) -> str:
        """Convert a list of dicts (or a single dict) to CSV."""
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not data:
            return ""

        # Collect all keys
        if isinstance(data[0], dict):
            fieldnames = sorted({k for item in data for k in item.keys()})
        else:
            fieldnames = ["value"]

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in data:
            if isinstance(item, dict):
                writer.writerow(item)
            else:
                writer.writerow({"value": item})
        return output.getvalue()

    @staticmethod
    def _to_markdown(data: Any) -> str:
        """Convert data to a Markdown representation."""
        lines: List[str] = ["# Export", ""]

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    # Render as table
                    lines.append(f"## {key}", )
                    lines.extend(ExportManager._markdown_table(value))
                    lines.append("")
                else:
                    lines.append(f"**{key}**: {value}")
                    lines.append("")
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                lines.extend(ExportManager._markdown_table(data))
            else:
                for item in data:
                    lines.append(f"- {item}")
            lines.append("")
        else:
            lines.append(str(data))

        return "\n".join(lines)

    @staticmethod
    def _markdown_table(rows: List[Dict[str, Any]]) -> List[str]:
        """Render a list of dicts as a Markdown table."""
        if not rows:
            return []
        headers = sorted(rows[0].keys())
        lines: List[str] = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            cells = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    @staticmethod
    def _to_html(data: Any) -> str:
        """Convert data to a minimal HTML page."""
        import html as html_lib

        body_parts: List[str] = []

        if isinstance(data, dict):
            for key, value in data.items():
                escaped_key = html_lib.escape(str(key))
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    body_parts.append(f"<h2>{escaped_key}</h2>")
                    body_parts.append(ExportManager._html_table(value))
                else:
                    escaped_val = html_lib.escape(str(value))
                    body_parts.append(f"<p><strong>{escaped_key}:</strong> {escaped_val}</p>")
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                body_parts.append(ExportManager._html_table(data))
            else:
                body_parts.append("<ul>")
                for item in data:
                    body_parts.append(f"<li>{html_lib.escape(str(item))}</li>")
                body_parts.append("</ul>")
        else:
            body_parts.append(f"<p>{html_lib.escape(str(data))}</p>")

        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            "<meta charset='utf-8'>\n<title>Export</title>\n"
            "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
            "th,td{border:1px solid #ccc;padding:6px 10px}th{background:#f4f4f4}</style>\n"
            "</head>\n<body>\n" + "\n".join(body_parts) + "\n</body>\n</html>"
        )

    @staticmethod
    def _html_table(rows: List[Dict[str, Any]]) -> str:
        """Render a list of dicts as an HTML table."""
        import html as html_lib

        if not rows:
            return ""
        headers = sorted(rows[0].keys())
        parts: List[str] = ["<table>", "<tr>"]
        for h in headers:
            parts.append(f"<th>{html_lib.escape(str(h))}</th>")
        parts.append("</tr>")
        for row in rows:
            parts.append("<tr>")
            for h in headers:
                parts.append(f"<td>{html_lib.escape(str(row.get(h, '')))}</td>")
            parts.append("</tr>")
        parts.append("</table>")
        return "\n".join(parts)

    @staticmethod
    def _to_pdf_mock(data: Any) -> str:
        """Mock PDF export — returns a placeholder string.

        In a production system this would use a library like
        ``reportlab`` or ``weasyprint`` to generate a real PDF.
        """
        json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        return (
            "%PDF-1.4 Mock Export\n"
            "% This is a placeholder PDF generated by Nexus-LLM.\n"
            "% In production, a real PDF library would be used.\n"
            f"% Data length: {len(json_str)} characters\n"
            f"%\n{json_str}\n%%EOF"
        )
