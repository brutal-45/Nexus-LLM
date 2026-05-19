"""Nexus-LLM Response Parser.

Provides parsing capabilities for LLM responses, extracting structured
data from free-form text, JSON blocks, code blocks, and other formats.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParsedBlock:
    """A parsed content block from a response.

    Attributes:
        block_type: Type of block (text, code, json, list, etc.).
        content: The extracted content.
        language: Language identifier for code blocks.
        start_pos: Start position in the original text.
        end_pos: End position in the original text.
    """

    block_type: str = "text"
    content: Any = ""
    language: str = ""
    start_pos: int = 0
    end_pos: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type,
            "content": self.content,
            "language": self.language,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
        }


class ResponseParser:
    """Parses LLM responses into structured components.

    Extracts code blocks, JSON blocks, lists, and other structured
    content from mixed-format LLM responses.

    Example::

        parser = ResponseParser()
        blocks = parser.parse('''Here is some code:
        ```python
        print("hello")
        ```
        And a JSON: {"key": "value"}
        ''')
    """

    # Regex patterns
    CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    JSON_BLOCK_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
    LIST_PATTERN = re.compile(r"(?:^|\n)[-*]\s+(.*?)(?=\n[-*]|\n\n|$)", re.DOTALL)
    NUMBERED_LIST_PATTERN = re.compile(r"(?:^|\n)\d+\.\s+(.*?)(?=\n\d+|\n\n|$)", re.DOTALL)

    def __init__(self) -> None:
        logger.debug("ResponseParser initialized")

    def parse(self, response: str) -> List[ParsedBlock]:
        """Parse a response into structured blocks.

        Args:
            response: The LLM response text.

        Returns:
            List of ParsedBlock objects.
        """
        blocks: List[ParsedBlock] = []

        # Extract code blocks first
        code_blocks = self.extract_code_blocks(response)
        blocks.extend(code_blocks)

        # Extract JSON blocks
        json_blocks = self.extract_json_blocks(response)
        blocks.extend(json_blocks)

        # If no structured blocks found, treat as plain text
        if not blocks:
            blocks.append(ParsedBlock(block_type="text", content=response, start_pos=0, end_pos=len(response)))

        return blocks

    def extract_code_blocks(self, response: str) -> List[ParsedBlock]:
        """Extract fenced code blocks from a response.

        Args:
            response: The response text.

        Returns:
            List of code ParsedBlock objects.
        """
        blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(response):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            blocks.append(ParsedBlock(
                block_type="code",
                content=code,
                language=language,
                start_pos=match.start(),
                end_pos=match.end(),
            ))
        return blocks

    def extract_json_blocks(self, response: str) -> List[ParsedBlock]:
        """Extract JSON objects from a response.

        Args:
            response: The response text.

        Returns:
            List of JSON ParsedBlock objects.
        """
        blocks = []
        for match in self.JSON_BLOCK_PATTERN.finditer(response):
            text = match.group()
            try:
                parsed = json.loads(text)
                blocks.append(ParsedBlock(
                    block_type="json",
                    content=parsed,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
            except json.JSONDecodeError:
                continue
        return blocks

    def extract_list_items(self, response: str) -> List[str]:
        """Extract bullet list items from a response.

        Args:
            response: The response text.

        Returns:
            List of list item strings.
        """
        items = []
        for match in self.LIST_PATTERN.finditer(response):
            items.append(match.group(1).strip())
        return items

    def extract_numbered_items(self, response: str) -> List[Tuple[int, str]]:
        """Extract numbered list items from a response.

        Args:
            response: The response text.

        Returns:
            List of (number, text) tuples.
        """
        items = []
        for i, match in enumerate(self.NUMBERED_LIST_PATTERN.finditer(response), 1):
            items.append((i, match.group(1).strip()))
        return items

    def extract_text_only(self, response: str) -> str:
        """Extract only the plain text portions, removing code/JSON blocks.

        Args:
            response: The response text.

        Returns:
            Plain text with code/JSON blocks removed.
        """
        text = self.CODE_BLOCK_PATTERN.sub("", response)
        text = self.JSON_BLOCK_PATTERN.sub("", text)
        return text.strip()

    def get_first_code_block(self, response: str, language: str = "") -> Optional[str]:
        """Get the first code block, optionally filtered by language.

        Args:
            response: The response text.
            language: Optional language filter.

        Returns:
            The code block content, or None.
        """
        blocks = self.extract_code_blocks(response)
        for block in blocks:
            if not language or block.language == language:
                return block.content
        return None

    def get_first_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Get the first JSON object from a response.

        Args:
            response: The response text.

        Returns:
            The parsed JSON dictionary, or None.
        """
        blocks = self.extract_json_blocks(response)
        if blocks:
            return blocks[0].content
        return None
