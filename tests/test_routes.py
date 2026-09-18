"""Test API routes for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List


class RouteError(Exception):
    pass


@dataclass
class Route:
    path: str
    method: str
    handler: Callable
    name: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.method}_{self.path.replace('/', '_')}"


@dataclass
class Request:
    method: str
    path: str
    headers: Dict[str, str] = None
    body: Any = None
    params: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.params is None:
            self.params = {}


@dataclass
class Response:
    status_code: int
    body: Any = None
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

    @property
    def is_success(self):
        return 200 <= self.status_code < 300


class Router:
    def __init__(self):
        self._routes: Dict[str, Route] = {}

    def add_route(self, route: Route):
        key = f"{route.method}:{route.path}"
        if key in self._routes:
            raise RouteError(f"Route {key} already exists")
        self._routes[key] = route

    def remove_route(self, method: str, path: str):
        key = f"{method}:{path}"
        if key not in self._routes:
            raise RouteError(f"Route {key} not found")
        del self._routes[key]

    def match(self, method: str, path: str) -> Optional[Route]:
        key = f"{method}:{path}"
        return self._routes.get(key)

    def dispatch(self, request: Request) -> Response:
        route = self.match(request.method, request.path)
        if route is None:
            return Response(status_code=404, body={"error": "Not found"})
        try:
            result = route.handler(request)
            if isinstance(result, Response):
                return result
            return Response(status_code=200, body=result)
        except Exception as e:
            return Response(status_code=500, body={"error": str(e)})

    def list_routes(self) -> List[Route]:
        return list(self._routes.values())


# Handler functions
def health_handler(request: Request) -> Dict:
    return {"status": "healthy"}


def generate_handler(request: Request) -> Dict:
    body = request.body or {}
    prompt = body.get("prompt", "")
    if not prompt:
        return Response(status_code=400, body={"error": "prompt is required"})
    return {"generated_text": f"Response to: {prompt}"}


def models_handler(request: Request) -> Dict:
    return {"models": ["nexus-llm-base", "nexus-llm-chat", "nexus-llm-code"]}


class TestRoute:
    def test_creation(self):
        route = Route(path="/health", method="GET", handler=health_handler)
        assert route.path == "/health"
        assert route.method == "GET"

    def test_auto_name(self):
        route = Route(path="/api/v1/models", method="GET", handler=models_handler)
        assert "GET" in route.name

    def test_custom_name(self):
        route = Route(path="/health", method="GET", handler=health_handler, name="health_check")
        assert route.name == "health_check"


class TestRequest:
    def test_creation(self):
        req = Request(method="GET", path="/health")
        assert req.method == "GET"
        assert req.headers == {}

    def test_with_body(self):
        req = Request(method="POST", path="/generate", body={"prompt": "hello"})
        assert req.body["prompt"] == "hello"


class TestResponse:
    def test_success(self):
        resp = Response(status_code=200, body={"ok": True})
        assert resp.is_success is True

    def test_error(self):
        resp = Response(status_code=500, body={"error": "fail"})
        assert resp.is_success is False

    def test_not_found(self):
        resp = Response(status_code=404)
        assert resp.is_success is False

    def test_redirect(self):
        resp = Response(status_code=301)
        assert not resp.is_success


class TestRouter:
    def test_add_route(self):
        router = Router()
        route = Route(path="/health", method="GET", handler=health_handler)
        router.add_route(route)
        assert router.match("GET", "/health") is not None

    def test_add_duplicate_route(self):
        router = Router()
        route = Route(path="/health", method="GET", handler=health_handler)
        router.add_route(route)
        with pytest.raises(RouteError, match="already exists"):
            router.add_route(route)

    def test_remove_route(self):
        router = Router()
        route = Route(path="/health", method="GET", handler=health_handler)
        router.add_route(route)
        router.remove_route("GET", "/health")
        assert router.match("GET", "/health") is None

    def test_remove_nonexistent(self):
        router = Router()
        with pytest.raises(RouteError, match="not found"):
            router.remove_route("GET", "/nonexistent")

    def test_match_found(self):
        router = Router()
        router.add_route(Route(path="/models", method="GET", handler=models_handler))
        matched = router.match("GET", "/models")
        assert matched is not None
        assert matched.handler is models_handler

    def test_match_not_found(self):
        router = Router()
        assert router.match("GET", "/nonexistent") is None

    def test_dispatch_health(self):
        router = Router()
        router.add_route(Route(path="/health", method="GET", handler=health_handler))
        req = Request(method="GET", path="/health")
        resp = router.dispatch(req)
        assert resp.is_success
        assert resp.body["status"] == "healthy"

    def test_dispatch_not_found(self):
        router = Router()
        req = Request(method="GET", path="/nonexistent")
        resp = router.dispatch(req)
        assert resp.status_code == 404

    def test_dispatch_generate(self):
        router = Router()
        router.add_route(Route(path="/generate", method="POST", handler=generate_handler))
        req = Request(method="POST", path="/generate", body={"prompt": "hello"})
        resp = router.dispatch(req)
        assert resp.is_success
        assert "hello" in resp.body["generated_text"]

    def test_dispatch_generate_no_prompt(self):
        router = Router()
        router.add_route(Route(path="/generate", method="POST", handler=generate_handler))
        req = Request(method="POST", path="/generate", body={})
        resp = router.dispatch(req)
        assert resp.status_code == 400

    def test_dispatch_handler_error(self):
        def bad_handler(request):
            raise ValueError("oops")
        router = Router()
        router.add_route(Route(path="/bad", method="GET", handler=bad_handler))
        resp = router.dispatch(Request(method="GET", path="/bad"))
        assert resp.status_code == 500

    def test_list_routes(self):
        router = Router()
        router.add_route(Route(path="/health", method="GET", handler=health_handler))
        router.add_route(Route(path="/models", method="GET", handler=models_handler))
        routes = router.list_routes()
        assert len(routes) == 2
