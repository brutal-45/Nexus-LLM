"""Chat exporter for Nexus-LLM.

Exports conversations to Markdown, HTML, and JSON with beautiful
formatting, syntax highlighting for code blocks, and role-based
styling.
"""

import html as html_lib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Type for a conversation: list of message dicts
Conversation = List[Dict[str, Any]]


class ChatExporter:
    """Export chat conversations in multiple formats.

    Each message in a conversation is a dict with at least a ``"role"``
    key (``"user"``, ``"assistant"``, ``"system"``) and a ``"content"``
    key.

    Example::

        exporter = ChatExporter()
        conversation = [
            {"role": "user", "content": "Write a Python hello world"},
            {"role": "assistant", "content": "```python\\nprint('Hello, world!')\\n```"},
        ]
        md = exporter.to_markdown(conversation)
    """

    # ------------------------------------------------------------------
    # High-level export
    # ------------------------------------------------------------------

    def export_conversation(
        self,
        conversation: Conversation,
        format: str = "markdown",
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export a conversation to the specified format.

        Args:
            conversation: List of message dicts with ``role`` and ``content``.
            format: Output format — ``"markdown"``, ``"html"``, or ``"json"``.
            path: Optional file path.  If provided the result is written
                  to disk.

        Returns:
            The serialised string (or the path if written to disk).

        Raises:
            ValueError: If the format is unsupported.
        """
        format = format.lower()
        if format == "markdown":
            result = self.to_markdown(conversation)
        elif format == "html":
            result = self.to_html(conversation)
        elif format == "json":
            result = self.to_json(conversation)
        else:
            raise ValueError(
                f"Unsupported chat export format {format!r}. "
                f"Supported: markdown, html, json"
            )

        if path is not None:
            path = str(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(result)
            logger.info("Exported conversation to %s (format=%s)", path, format)
            return path

        return result

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def to_markdown(self, conversation: Conversation) -> str:
        """Render a conversation as Markdown with code-block highlighting.

        Returns:
            A Markdown-formatted string.
        """
        lines: List[str] = [
            "# Chat Export",
            f"*Exported at {datetime.now(timezone.utc).isoformat()}*",
            "",
        ]

        for msg in conversation:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            # Role header
            header = f"### {role}"
            if timestamp:
                header += f"  ·  {timestamp}"
            lines.append(header)
            lines.append("")

            # Content — code blocks are preserved
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def to_html(self, conversation: Conversation) -> str:
        """Render a conversation as a styled HTML page.

        Code blocks receive syntax-highlighting CSS classes
        (``language-<lang>``) compatible with highlight.js / Prism.

        Returns:
            An HTML string.
        """
        body_parts: List[str] = []

        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            role_class = f"msg-{role}"
            escaped_content = self._render_html_content(content)

            timestamp_html = (
                f'<span class="timestamp">{html_lib.escape(timestamp)}</span>'
                if timestamp
                else ""
            )

            body_parts.append(
                f'<div class="message {role_class}">'
                f'<div class="role">{html_lib.escape(role.capitalize())}'
                f"{timestamp_html}</div>"
                f'<div class="content">{escaped_content}</div>'
                f"</div>"
            )

        body = "\n".join(body_parts)

        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            "<meta charset='utf-8'>\n<title>Chat Export</title>\n"
            "<style>\n"
            "body{font-family:system-ui,-apple-system,sans-serif;max-width:800px;margin:2em auto;padding:0 1em;background:#fafafa}\n"
            ".message{margin:1em 0;padding:1em;border-radius:8px}\n"
            ".msg-user{background:#e3f2fd;border-left:4px solid #1976d2}\n"
            ".msg-assistant{background:#f1f8e9;border-left:4px solid #388e3c}\n"
            ".msg-system{background:#fff3e0;border-left:4px solid #f57c00}\n"
            ".role{font-weight:bold;margin-bottom:0.5em}\n"
            ".timestamp{margin-left:1em;font-size:0.85em;color:#888}\n"
            ".content{white-space:pre-wrap;line-height:1.6}\n"
            "pre{background:#263238;color:#eeffff;padding:1em;border-radius:4px;overflow-x:auto}\n"
            "code{font-family:'Fira Code',monospace}\n"
            "</style>\n"
            "</head>\n<body>\n<h1>Chat Export</h1>\n"
            + body
            + "\n</body>\n</html>"
        )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, conversation: Conversation) -> str:
        """Render a conversation as a JSON string.

        Returns:
            A JSON-formatted string with metadata.
        """
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "message_count": len(conversation),
            "messages": conversation,
        }
        return json.dumps(payload, indent=2, default=str, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_html_content(content: str) -> str:
        """Convert content with fenced code blocks to styled HTML.

        Preserves code blocks and applies language-specific CSS classes
        for syntax highlighters.
        """
        import re

        # Replace fenced code blocks with <pre><code>
        def _replace_code_block(match: re.Match) -> str:
            lang = match.group(1) or ""
            code = html_lib.escape(match.group(2))
            lang_class = f' class="language-{lang}"' if lang else ""
            return f"<pre><code{lang_class}>{code}</code></pre>"

        result = re.sub(
            r"```(\w*)\n(.*?)```",
            _replace_code_block,
            content,
            flags=re.DOTALL,
        )

        # Escape remaining non-code text
        # Split on <pre> blocks, escape the text parts, rejoin
        parts = result.split("<pre>")
        escaped_parts: List[str] = [html_lib.escape(parts[0]) if i == 0 else parts[0] for i, parts in enumerate([parts])]
        # Simplified: just handle the inline code case
        # Replace inline code `...`
        result = re.sub(
            r"`([^`]+)`",
            lambda m: f"<code>{html_lib.escape(m.group(1))}</code>",
            result,
        )

        # Escape remaining text (anything not already in an HTML tag)
        # For simplicity we do a basic pass — real world would use a proper parser
        lines = result.split("\n")
        processed: List[str] = []
        in_pre = False
        for line in lines:
            if "<pre>" in line:
                in_pre = True
                processed.append(line)
            elif "</pre>" in line:
                in_pre = False
                processed.append(line)
            elif in_pre:
                processed.append(line)
            else:
                # Escape and convert newlines to <br>
                if line.strip():
                    processed.append(html_lib.escape(line))
                else:
                    processed.append("<br>")

        return "\n".join(processed)
