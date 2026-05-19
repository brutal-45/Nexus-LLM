"""Test API documentation for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ParameterDoc:
    name: str
    type: str
    required: bool = True
    description: str = ""
    default: Any = None
    enum: List[str] = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type, "required": self.required}
        if self.description:
            d["description"] = self.description
        if self.default is not None:
            d["default"] = self.default
        if self.enum:
            d["enum"] = self.enum
        return d


@dataclass
class EndpointDoc:
    path: str
    method: str
    summary: str = ""
    description: str = ""
    parameters: List[ParameterDoc] = field(default_factory=list)
    request_body: Optional[Dict] = None
    responses: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "path": self.path,
            "method": self.method,
            "summary": self.summary,
        }
        if self.description:
            d["description"] = self.description
        if self.parameters:
            d["parameters"] = [p.to_dict() for p in self.parameters]
        if self.request_body:
            d["request_body"] = self.request_body
        if self.responses:
            d["responses"] = self.responses
        return d


class APIDocumentation:
    def __init__(self, title: str = "Nexus-LLM API", version: str = "1.0.0"):
        self.title = title
        self.version = version
        self._endpoints: Dict[str, EndpointDoc] = {}

    def add_endpoint(self, doc: EndpointDoc):
        key = f"{doc.method}:{doc.path}"
        self._endpoints[key] = doc

    def get_endpoint(self, method: str, path: str) -> Optional[EndpointDoc]:
        return self._endpoints.get(f"{method}:{path}")

    def list_endpoints(self) -> List[EndpointDoc]:
        return list(self._endpoints.values())

    def search_endpoints(self, query: str) -> List[EndpointDoc]:
        results = []
        query_lower = query.lower()
        for doc in self._endpoints.values():
            if (query_lower in doc.path.lower() or
                query_lower in doc.summary.lower() or
                query_lower in doc.description.lower()):
                results.append(doc)
        return results

    def to_openapi_spec(self) -> dict:
        paths = {}
        for doc in self._endpoints.values():
            if doc.path not in paths:
                paths[doc.path] = {}
            paths[doc.path][doc.method.lower()] = {
                "summary": doc.summary,
                "description": doc.description,
                "parameters": [p.to_dict() for p in doc.parameters],
                "responses": {str(code): {"description": desc} for code, desc in doc.responses.items()},
            }
        return {
            "openapi": "3.0.0",
            "info": {"title": self.title, "version": self.version},
            "paths": paths,
        }


class TestParameterDoc:
    def test_required_param(self):
        param = ParameterDoc(name="prompt", type="string", required=True, description="The input prompt")
        d = param.to_dict()
        assert d["name"] == "prompt"
        assert d["required"] is True

    def test_optional_param(self):
        param = ParameterDoc(name="temperature", type="float", required=False, default=0.7)
        d = param.to_dict()
        assert d["required"] is False
        assert d["default"] == 0.7

    def test_enum_param(self):
        param = ParameterDoc(name="model", type="string", required=True, enum=["gpt2", "llama"])
        d = param.to_dict()
        assert "enum" in d

    def test_minimal_dict(self):
        param = ParameterDoc(name="x", type="int")
        d = param.to_dict()
        assert "name" in d
        assert "default" not in d


class TestEndpointDoc:
    def test_creation(self):
        doc = EndpointDoc(path="/generate", method="POST", summary="Generate text")
        assert doc.path == "/generate"
        assert doc.method == "POST"

    def test_to_dict(self):
        doc = EndpointDoc(
            path="/generate", method="POST", summary="Generate",
            parameters=[ParameterDoc(name="prompt", type="string")],
            responses={200: "Success", 400: "Bad request"},
        )
        d = doc.to_dict()
        assert d["path"] == "/generate"
        assert len(d["parameters"]) == 1
        assert 200 in d["responses"]

    def test_empty_parameters(self):
        doc = EndpointDoc(path="/health", method="GET", summary="Health check")
        d = doc.to_dict()
        assert "parameters" not in d


class TestAPIDocumentation:
    def test_add_endpoint(self):
        api_doc = APIDocumentation()
        doc = EndpointDoc(path="/health", method="GET", summary="Health check")
        api_doc.add_endpoint(doc)
        assert len(api_doc.list_endpoints()) == 1

    def test_get_endpoint(self):
        api_doc = APIDocumentation()
        doc = EndpointDoc(path="/generate", method="POST", summary="Generate")
        api_doc.add_endpoint(doc)
        found = api_doc.get_endpoint("POST", "/generate")
        assert found is not None
        assert found.summary == "Generate"

    def test_get_nonexistent_endpoint(self):
        api_doc = APIDocumentation()
        assert api_doc.get_endpoint("GET", "/nonexistent") is None

    def test_list_endpoints(self):
        api_doc = APIDocumentation()
        api_doc.add_endpoint(EndpointDoc(path="/health", method="GET", summary="Health"))
        api_doc.add_endpoint(EndpointDoc(path="/generate", method="POST", summary="Generate"))
        assert len(api_doc.list_endpoints()) == 2

    def test_search_endpoints(self):
        api_doc = APIDocumentation()
        api_doc.add_endpoint(EndpointDoc(path="/generate", method="POST", summary="Generate text"))
        api_doc.add_endpoint(EndpointDoc(path="/health", method="GET", summary="Health check"))
        results = api_doc.search_endpoints("generate")
        assert len(results) == 1
        assert results[0].path == "/generate"

    def test_search_no_results(self):
        api_doc = APIDocumentation()
        assert api_doc.search_endpoints("nonexistent") == []

    def test_openapi_spec(self):
        api_doc = APIDocumentation()
        api_doc.add_endpoint(EndpointDoc(
            path="/generate", method="POST", summary="Generate",
            responses={200: "OK"},
        ))
        spec = api_doc.to_openapi_spec()
        assert spec["openapi"] == "3.0.0"
        assert "/generate" in spec["paths"]
        assert "post" in spec["paths"]["/generate"]

    def test_openapi_info(self):
        api_doc = APIDocumentation(title="Test API", version="2.0.0")
        spec = api_doc.to_openapi_spec()
        assert spec["info"]["title"] == "Test API"
        assert spec["info"]["version"] == "2.0.0"
