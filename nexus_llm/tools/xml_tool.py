"""Nexus-LLM XML Processing Tool.

Provides XML parsing, querying, and transformation capabilities
using Python's built-in xml.etree module.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class XmlTool(BaseTool):
    """Tool for XML data processing.

    Supports parsing, querying with XPath, and transforming XML data.

    Example::

        tool = XmlTool()
        result = tool.run(operation="parse", data="<root><item>hello</item></root>")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="xml", description="Process XML data", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Operation to perform",
                         required=True, choices=["parse", "query", "to_dict", "validate"]),
            ToolParameter(name="data", type=ParameterType.STRING, description="XML data string", required=True),
            ToolParameter(name="xpath", type=ParameterType.STRING, description="XPath expression for queries", required=False),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation", "")
        data = kwargs.get("data", "")

        if not operation:
            return ToolResult(success=False, error="No operation specified")
        if not data:
            return ToolResult(success=False, error="No XML data provided")

        try:
            if operation == "parse":
                return self._parse(data)
            elif operation == "query":
                xpath = kwargs.get("xpath", "")
                return self._query(data, xpath)
            elif operation == "to_dict":
                return self._to_dict(data)
            elif operation == "validate":
                return self._validate(data)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _parse(self, data: str) -> ToolResult:
        """Parse XML and return element info."""
        root = ET.fromstring(data)
        return ToolResult(success=True, output={
            "root_tag": root.tag,
            "root_attrib": dict(root.attrib),
            "child_count": len(root),
        })

    def _query(self, data: str, xpath: str) -> ToolResult:
        """Query XML with XPath."""
        if not xpath:
            return ToolResult(success=False, error="No XPath expression provided")
        root = ET.fromstring(data)
        elements = root.findall(xpath)
        results = [{"tag": e.tag, "text": e.text, "attrib": dict(e.attrib)} for e in elements]
        return ToolResult(success=True, output=results, metadata={"match_count": len(results)})

    def _to_dict(self, data: str) -> ToolResult:
        """Convert XML to dictionary representation."""
        root = ET.fromstring(data)
        result = self._element_to_dict(root)
        return ToolResult(success=True, output=result)

    def _validate(self, data: str) -> ToolResult:
        """Validate XML well-formedness."""
        try:
            ET.fromstring(data)
            return ToolResult(success=True, output={"valid": True})
        except ET.ParseError as e:
            return ToolResult(success=True, output={"valid": False, "error": str(e)})

    def _element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Recursively convert an Element to a dictionary."""
        result: Dict[str, Any] = {}
        if element.attrib:
            result["@attributes"] = dict(element.attrib)
        if element.text and element.text.strip():
            result["#text"] = element.text.strip()
        for child in element:
            child_dict = self._element_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict
        return result
