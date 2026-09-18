"""Test API middleware for Nexus-LLM."""
import time
import pytest
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class Request:
    method: str
    path: str
    headers: Dict[str, str] = None
    body: Any = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


@dataclass
class Response:
    status_code: int
    body: Any = None
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class MiddlewareChain:
    def __init__(self):
        self._middlewares: List[Callable] = []

    def add(self, middleware: Callable):
        self._middlewares.append(middleware)

    def execute(self, request: Request, handler: Callable) -> Response:
        def build_chain(middlewares, final_handler):
            if not middlewares:
                return final_handler
            middleware = middlewares[0]
            remaining = middlewares[1:]
            def chain(req):
                return middleware(req, build_chain(remaining, final_handler))
            return chain
        chain = build_chain(self._middlewares, handler)
        return chain(request)


def logging_middleware(request: Request, next_handler: Callable) -> Response:
    start = time.perf_counter()
    response = next_handler(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Response-Time"] = f"{elapsed:.4f}s"
    return response


def auth_middleware(request: Request, next_handler: Callable) -> Response:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return Response(status_code=401, body={"error": "Unauthorized"})
    return next_handler(request)


def content_type_middleware(request: Request, next_handler: Callable) -> Response:
    if request.method in ("POST", "PUT", "PATCH"):
        ct = request.headers.get("Content-Type", "")
        if not ct:
            return Response(status_code=400, body={"error": "Content-Type required"})
    return next_handler(request)


def request_id_middleware(request: Request, next_handler: Callable) -> Response:
    import uuid
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = next_handler(request)
    response.headers["X-Request-ID"] = req_id
    return response


def error_handler_middleware(request: Request, next_handler: Callable) -> Response:
    try:
        return next_handler(request)
    except Exception as e:
        return Response(status_code=500, body={"error": str(e)})


class TestMiddlewareChain:
    def test_no_middleware(self):
        chain = MiddlewareChain()
        handler = lambda req: Response(status_code=200, body={"ok": True})
        resp = chain.execute(Request(method="GET", path="/test"), handler)
        assert resp.status_code == 200

    def test_single_middleware(self):
        chain = MiddlewareChain()
        chain.add(logging_middleware)
        handler = lambda req: Response(status_code=200, body={"ok": True})
        resp = chain.execute(Request(method="GET", path="/test"), handler)
        assert "X-Response-Time" in resp.headers

    def test_multiple_middlewares(self):
        chain = MiddlewareChain()
        chain.add(request_id_middleware)
        chain.add(logging_middleware)
        handler = lambda req: Response(status_code=200, body={"ok": True})
        resp = chain.execute(Request(method="GET", path="/test"), handler)
        assert "X-Request-ID" in resp.headers
        assert "X-Response-Time" in resp.headers


class TestLoggingMiddleware:
    def test_adds_response_time(self):
        handler = lambda req: Response(status_code=200, body="ok")
        resp = logging_middleware(Request(method="GET", path="/"), handler)
        assert "X-Response-Time" in resp.headers

    def test_passes_through_response(self):
        handler = lambda req: Response(status_code=201, body="created")
        resp = logging_middleware(Request(method="POST", path="/"), handler)
        assert resp.status_code == 201


class TestAuthMiddleware:
    def test_with_auth(self):
        handler = lambda req: Response(status_code=200, body="ok")
        req = Request(method="GET", path="/", headers={"Authorization": "Bearer token"})
        resp = auth_middleware(req, handler)
        assert resp.status_code == 200

    def test_without_auth(self):
        handler = lambda req: Response(status_code=200, body="ok")
        req = Request(method="GET", path="/")
        resp = auth_middleware(req, handler)
        assert resp.status_code == 401


class TestContentTypeMiddleware:
    def test_post_with_content_type(self):
        handler = lambda req: Response(status_code=200, body="ok")
        req = Request(method="POST", path="/", headers={"Content-Type": "application/json"})
        resp = content_type_middleware(req, handler)
        assert resp.status_code == 200

    def test_post_without_content_type(self):
        handler = lambda req: Response(status_code=200, body="ok")
        req = Request(method="POST", path="/")
        resp = content_type_middleware(req, handler)
        assert resp.status_code == 400

    def test_get_skips_check(self):
        handler = lambda req: Response(status_code=200, body="ok")
        req = Request(method="GET", path="/")
        resp = content_type_middleware(req, handler)
        assert resp.status_code == 200


class TestRequestIdMiddleware:
    def test_adds_request_id(self):
        handler = lambda req: Response(status_code=200, body="ok")
        resp = request_id_middleware(Request(method="GET", path="/"), handler)
        assert "X-Request-ID" in resp.headers

    def test_uses_existing_request_id(self):
        handler = lambda req: Response(status_code=200, body="ok")
        req = Request(method="GET", path="/", headers={"X-Request-ID": "custom-id"})
        resp = request_id_middleware(req, handler)
        assert resp.headers["X-Request-ID"] == "custom-id"


class TestErrorHandlerMiddleware:
    def test_catches_exception(self):
        def bad_handler(req):
            raise ValueError("something broke")
        resp = error_handler_middleware(Request(method="GET", path="/"), bad_handler)
        assert resp.status_code == 500
        assert "something broke" in resp.body["error"]

    def test_passes_through_success(self):
        handler = lambda req: Response(status_code=200, body="ok")
        resp = error_handler_middleware(Request(method="GET", path="/"), handler)
        assert resp.status_code == 200
