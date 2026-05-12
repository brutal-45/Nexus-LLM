"""
Nexus LLM Deployment Module
=============================

Production-ready deployment tools for serving LLM inference including
HTTP server, request batching, caching, load balancing, monitoring,
and rate limiting.

All implementations use Python stdlib only (asyncio, json, time,
collections, hashlib, threading, queue, dataclasses, typing).

Classes:
    InferenceServer - Async HTTP inference server
    RequestHandler - Parse and validate incoming requests
    ModelWorker - Manage model loading and batched inference
    ResponseFormatter - Format responses as JSON, SSE, plain text
    ServerErrorHandler - Error codes, rate limits, graceful degradation
    ServerConfig - Server configuration dataclass

    RequestBatcher - Collect requests into batches
    ContinuousBatcher - Iteration-level batching (like vLLM)
    DynamicBatcher - Adaptive batch sizing
    BatchScheduler - Priority and fairness scheduling

    ResponseCache - LRU cache for completions
    SemanticCache - Cache semantically similar queries
    PrefixCache - Share KV cache prefixes
    CacheEvictionPolicy - LRU, LFU, TTL eviction

    LoadBalancer - Distribute across workers
    RoundRobinBalancer - Round-robin selection
    LeastConnectionsBalancer - Fewest active connections
    WeightedBalancer - Capacity-based selection

    MetricsCollector - Collect inference metrics
    AlertManager - Define and check alert rules
    HealthChecker - Periodic health checks
    DashboardMetrics - Format metrics for dashboards

    TokenBucketRateLimiter - Token bucket algorithm
    SlidingWindowRateLimiter - Sliding window counter
    AdaptiveRateLimiter - Load-adaptive limiting
    RateLimitMiddleware - Server integration

Usage:
    from nexus.deployment import InferenceServer, ServerConfig
    from nexus.deployment import RequestBatcher, ResponseCache
    from nexus.deployment import LoadBalancer, MetricsCollector
    from nexus.deployment import TokenBucketRateLimiter
"""

from nexus.deployment.server_core import (
    InferenceServer,
    RequestHandler,
    ModelWorker,
    ResponseFormatter,
    ServerErrorHandler,
    ServerConfig,
)

from nexus.deployment.batching import (
    RequestBatcher,
    ContinuousBatcher,
    DynamicBatcher,
    BatchScheduler,
)

from nexus.deployment.caching import (
    ResponseCache,
    SemanticCache,
    PrefixCache,
    CacheEvictionPolicy,
)

from nexus.deployment.load_balancer import (
    LoadBalancer,
    RoundRobinBalancer,
    LeastConnectionsBalancer,
    WeightedBalancer,
)

from nexus.deployment.monitoring import (
    MetricsCollector,
    AlertManager,
    HealthChecker,
    DashboardMetrics,
)

from nexus.deployment.rate_limiter import (
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    AdaptiveRateLimiter,
    RateLimitMiddleware,
)

__all__ = [
    # Server Core
    "InferenceServer",
    "RequestHandler",
    "ModelWorker",
    "ResponseFormatter",
    "ServerErrorHandler",
    "ServerConfig",
    # Batching
    "RequestBatcher",
    "ContinuousBatcher",
    "DynamicBatcher",
    "BatchScheduler",
    # Caching
    "ResponseCache",
    "SemanticCache",
    "PrefixCache",
    "CacheEvictionPolicy",
    # Load Balancing
    "LoadBalancer",
    "RoundRobinBalancer",
    "LeastConnectionsBalancer",
    "WeightedBalancer",
    # Monitoring
    "MetricsCollector",
    "AlertManager",
    "HealthChecker",
    "DashboardMetrics",
    # Rate Limiting
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "AdaptiveRateLimiter",
    "RateLimitMiddleware",
]

__version__ = "0.1.0"
__author__ = "Nexus LLM Team"
