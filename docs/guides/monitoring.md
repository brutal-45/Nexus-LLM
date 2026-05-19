# Monitoring Guide

This guide covers monitoring, dashboards, alerting, and metrics for Nexus-LLM deployments.

## Overview

Nexus-LLM includes a built-in monitoring stack that tracks server health, model performance, request throughput, and safety events. The monitoring system is designed to work with Prometheus, Grafana, and custom dashboards.

```
Nexus-LLM Server → [Metrics Collector] → [Prometheus] → [Grafana Dashboard]
        ↓                   ↓
   [Alert Engine] ←── [Threshold Checks]
        ↓
   [Notification Channels] (Email, Slack, Webhook)
```

---

## Quick Start

### Enable Monitoring

```bash
# Start server with monitoring enabled
nexus-llm serve --model meta-llama/Llama-3.1-8B-Instruct --monitoring enabled

# Or configure via environment
export NEXUS_MONITORING_ENABLED=true
export NEXUS_METRICS_PROMETHEUS=true
```

### Access the Metrics Endpoint

```bash
# Built-in metrics (JSON format)
curl http://localhost:8000/v1/metrics

# Prometheus-compatible metrics
curl http://localhost:8000/metrics
```

### Launch the Dashboard

```bash
# Start the built-in web dashboard
nexus-llm dashboard --port 8501

# Open in browser
open http://localhost:8501
```

---

## Dashboard

The built-in monitoring dashboard provides real-time visibility into your Nexus-LLM deployment.

### Dashboard Sections

#### 1. System Overview

| Metric | Description |
|---|---|
| Server Status | Running / Stopped / Degraded |
| Uptime | Time since server start |
| Version | Current Nexus-LLM version |
| Models Loaded | Number of active models |
| Active Connections | Current WebSocket connections |
| Requests/Second | Current request throughput |

#### 2. GPU Metrics

| Metric | Description |
|---|---|
| GPU Utilization | Compute usage percentage |
| GPU Memory Used | VRAM currently allocated |
| GPU Memory Total | Total VRAM available |
| GPU Temperature | Current GPU temperature |
| GPU Power Draw | Current power consumption |
| CUDA Version | CUDA runtime version |

#### 3. Model Performance

| Metric | Description |
|---|---|
| Inference Latency (P50/P95/P99) | Token generation latency |
| Time to First Token | Latency for first token in streaming |
| Tokens/Second | Generation throughput per model |
| Batch Size | Current batch size |
| Queue Depth | Pending requests in queue |
| KV Cache Utilization | Key-value cache memory usage |

#### 4. Request Metrics

| Metric | Description |
|---|---|
| Total Requests | Cumulative request count |
| Successful Requests | Requests returning 2xx |
| Failed Requests | Requests returning 4xx/5xx |
| Error Rate | Failed / Total percentage |
| Average Request Duration | Mean request processing time |
| Active Requests | Currently processing requests |

#### 5. Safety Metrics

| Metric | Description |
|---|---|
| Content Filtered | Requests blocked by content filter |
| Content Flagged | Requests flagged but not blocked |
| Moderation Triggers | Requests flagged by moderation model |
| Toxicity Score Average | Average toxicity score of outputs |
| Top Blocked Categories | Most common blocked categories |

### Custom Dashboard Configuration

Configure dashboard layout and refresh intervals in `monitoring_config.yaml`:

```yaml
dashboard:
  enabled: true
  host: "0.0.0.0"
  port: 8501
  refresh_interval: 5          # seconds
  theme: "dark"                # dark, light
  sections:
    system_overview: true
    gpu_metrics: true
    model_performance: true
    request_metrics: true
    safety_metrics: true
  custom_widgets: []
```

### Grafana Integration

For production deployments, use Grafana for advanced visualization:

1. **Install Grafana** and configure the Prometheus data source
2. **Import the Nexus-LLM dashboard** (dashboard ID available in the `/grafana` endpoint)
3. **Customize panels** for your specific needs

```bash
# Export Grafana dashboard JSON
nexus-llm dashboard export-grafana --output nexus-llm-grafana.json

# Import into Grafana
# Open Grafana → Dashboards → Import → Upload JSON file
```

---

## Metrics

### Prometheus Metrics

Nexus-LLM exposes the following Prometheus metrics at the `/metrics` endpoint:

#### Server Metrics

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `nexus_uptime_seconds` | Gauge | — | Server uptime in seconds |
| `nexus_info` | Gauge | `version`, `cuda_version` | Server version information |
| `nexus_loaded_models` | Gauge | — | Number of loaded models |
| `nexus_active_connections` | Gauge | — | Active connections |

#### Request Metrics

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `nexus_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests |
| `nexus_request_duration_seconds` | Histogram | `method`, `endpoint` | Request duration |
| `nexus_request_size_bytes` | Histogram | `method`, `endpoint` | Request body size |
| `nexus_response_size_bytes` | Histogram | `method`, `endpoint` | Response body size |
| `nexus_active_requests` | Gauge | `endpoint` | Currently processing requests |

#### Model Metrics

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `nexus_inference_duration_seconds` | Histogram | `model` | Inference time |
| `nexus_tokens_generated_total` | Counter | `model` | Total tokens generated |
| `nexus_tokens_input_total` | Counter | `model` | Total input tokens |
| `nexus_time_to_first_token_seconds` | Histogram | `model` | TTFT latency |
| `nexus_tokens_per_second` | Gauge | `model` | Generation throughput |
| `nexus_kv_cache_usage_ratio` | Gauge | `model` | KV cache utilization |
| `nexus_batch_size` | Gauge | `model` | Current batch size |
| `nexus_queue_depth` | Gauge | `model` | Pending requests |

#### GPU Metrics

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `nexus_gpu_utilization_ratio` | Gauge | `gpu_id` | GPU compute usage (0-1) |
| `nexus_gpu_memory_used_bytes` | Gauge | `gpu_id` | GPU memory used |
| `nexus_gpu_memory_total_bytes` | Gauge | `gpu_id` | GPU memory total |
| `nexus_gpu_temperature_celsius` | Gauge | `gpu_id` | GPU temperature |
| `nexus_gpu_power_watts` | Gauge | `gpu_id` | GPU power draw |
| `nexus_gpu_memory_utilization_ratio` | Gauge | `gpu_id` | GPU memory utilization (0-1) |

#### Safety Metrics

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `nexus_safety_checks_total` | Counter | `source`, `result` | Safety check results |
| `nexus_safety_blocked_total` | Counter | `source`, `category` | Blocked requests by category |
| `nexus_safety_flagged_total` | Counter | `source`, `category` | Flagged requests by category |
| `nexus_toxicity_score` | Histogram | `model` | Toxicity score distribution |

#### Rate Limiting Metrics

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `nexus_rate_limit_total` | Counter | `tier`, `result` | Rate limit check results |
| `nexus_rate_limit_remaining` | Gauge | `tier` | Remaining rate limit capacity |

### Custom Metrics

Register custom metrics in your application:

```python
from nexus_llm.monitoring import MetricsRegistry, Counter, Gauge, Histogram

registry = MetricsRegistry()

# Define custom metrics
custom_requests = Counter(
    name="my_app_requests_total",
    description="Total custom application requests",
    labels=["app", "method"]
)

custom_latency = Histogram(
    name="my_app_latency_seconds",
    description="Custom application latency",
    labels=["app"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

active_users = Gauge(
    name="my_app_active_users",
    description="Currently active users"
)

# Register and use
registry.register(custom_requests)
registry.register(custom_latency)
registry.register(active_users)

custom_requests.inc(labels={"app": "myapp", "method": "chat"})
custom_latency.observe(0.234, labels={"app": "myapp"})
active_users.set(42)
```

### Metric Collection Intervals

Configure how often different metrics are collected:

```yaml
metrics:
  enabled: true
  collect_interval: 10          # Default collection interval (seconds)
  prometheus:
    enabled: true
    endpoint: "/metrics"
    port: 8000

  # Per-category collection intervals
  intervals:
    server: 10
    gpu: 5                       # More frequent for GPU metrics
    model: 10
    safety: 30                   # Less frequent for safety metrics
    rate_limit: 10

  # Histogram bucket configuration
  histograms:
    request_duration:
      buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
    inference_duration:
      buckets: [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    time_to_first_token:
      buckets: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
```

---

## Alerts

The alerting system monitors metrics and triggers notifications when conditions are met.

### Alert Configuration

```yaml
alerts:
  enabled: true
  evaluation_interval: 30        # How often to check alert conditions (seconds)
  cooldown_period: 300           # Minimum time between repeated alerts (seconds)

  # Notification channels
  channels:
    - name: "slack"
      type: "slack"
      webhook_url: "https://hooks.slack.com/services/..."
      min_severity: "warning"    # warning, critical

    - name: "email"
      type: "email"
      recipients: ["ops@example.com"]
      smtp_host: "smtp.example.com"
      smtp_port: 587
      smtp_user: "alerts@example.com"
      smtp_password_env: "SMTP_PASSWORD"
      min_severity: "critical"

    - name: "webhook"
      type: "webhook"
      url: "https://api.example.com/alerts"
      headers:
        Authorization: "Bearer token123"
      min_severity: "warning"

    - name: "log"
      type: "log"
      level: "warning"
      min_severity: "warning"

  # Alert rules
  rules:
    # GPU Alerts
    - name: "gpu_memory_high"
      description: "GPU memory usage exceeds 90%"
      severity: "warning"
      condition: "nexus_gpu_memory_utilization_ratio > 0.9"
      for: "2m"
      channels: ["slack", "log"]

    - name: "gpu_memory_critical"
      description: "GPU memory usage exceeds 95%"
      severity: "critical"
      condition: "nexus_gpu_memory_utilization_ratio > 0.95"
      for: "30s"
      channels: ["slack", "email", "log"]

    - name: "gpu_temperature_high"
      description: "GPU temperature exceeds 85°C"
      severity: "warning"
      condition: "nexus_gpu_temperature_celsius > 85"
      for: "1m"
      channels: ["slack", "log"]

    # Inference Alerts
    - name: "high_latency"
      description: "P95 inference latency exceeds 5 seconds"
      severity: "warning"
      condition: "histogram_quantile(0.95, nexus_inference_duration_seconds) > 5.0"
      for: "5m"
      channels: ["slack", "log"]

    - name: "very_high_latency"
      description: "P95 inference latency exceeds 30 seconds"
      severity: "critical"
      condition: "histogram_quantile(0.95, nexus_inference_duration_seconds) > 30.0"
      for: "2m"
      channels: ["slack", "email", "log"]

    - name: "low_throughput"
      description: "Token generation throughput below 10 tokens/second"
      severity: "warning"
      condition: "nexus_tokens_per_second < 10"
      for: "5m"
      channels: ["slack", "log"]

    # Error Rate Alerts
    - name: "high_error_rate"
      description: "Error rate exceeds 5%"
      severity: "warning"
      condition: "rate(nexus_requests_total{status=~\"5..\"}[5m]) / rate(nexus_requests_total[5m]) > 0.05"
      for: "3m"
      channels: ["slack", "log"]

    - name: "critical_error_rate"
      description: "Error rate exceeds 20%"
      severity: "critical"
      condition: "rate(nexus_requests_total{status=~\"5..\"}[5m]) / rate(nexus_requests_total[5m]) > 0.20"
      for: "1m"
      channels: ["slack", "email", "log"]

    # Safety Alerts
    - name: "safety_block_spike"
      description: "Unusual spike in content filter blocks"
      severity: "warning"
      condition: "rate(nexus_safety_blocked_total[5m]) > 10"
      for: "2m"
      channels: ["slack", "log"]

    # Queue Alerts
    - name: "queue_backlog"
      description: "Request queue depth exceeds 100"
      severity: "warning"
      condition: "nexus_queue_depth > 100"
      for: "2m"
      channels: ["slack", "log"]

    # Server Health
    - name: "server_down"
      description: "Server is not responding"
      severity: "critical"
      condition: "up == 0"
      for: "30s"
      channels: ["slack", "email", "webhook"]
```

### Managing Alerts via CLI

```bash
# List all alert rules
nexus-llm alerts list

# View current alert state
nexus-llm alerts status

# Acknowledge an active alert
nexus-llm alerts acknowledge gpu_memory_high

# Silence alerts for a period
nexus-llm alerts silence --rule high_latency --duration 1h

# Test alert notifications
nexus-llm alerts test --channel slack
```

### Custom Alert Rules

Create custom alert rules via the API:

```bash
curl -X POST http://localhost:8000/v1/monitoring/alerts/rules \
  -H "Authorization: Bearer nxs_sk_abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_model_latency",
    "description": "Model latency exceeds threshold for my-app",
    "severity": "warning",
    "condition": "nexus_inference_duration_seconds{model=\"my-model\"} > 2.0",
    "for": "5m",
    "channels": ["slack"]
  }'
```

---

## Logging

### Structured Logging

Nexus-LLM uses structured JSON logging for all events:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "nexus_llm.api.routes",
  "message": "Chat completion request",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "input_tokens": 25,
  "output_tokens": 150,
  "duration_ms": 1234,
  "status": "success"
}
```

### Log Levels

| Level | Description |
|---|---|
| `DEBUG` | Detailed debug information |
| `INFO` | General operational events |
| `WARNING` | Potential issues that don't prevent operation |
| `ERROR` | Errors that affect individual requests |
| `CRITICAL` | Errors that affect the entire system |

### Configuring Log Output

```bash
# Set log level
nexus-llm serve --log-level DEBUG

# Log to file
nexus-llm serve --log-file /var/log/nexus-llm/server.log

# Log to stdout (for container deployments)
nexus-llm serve --log-format json --log-output stdout
```

### Log Aggregation

For production deployments, integrate with a log aggregation system:

```yaml
# In monitoring_config.yaml
logging:
  level: "info"
  format: "json"
  output: "stdout"

  # External log aggregation
  loki:
    enabled: false
    url: "http://loki:3100"
    labels:
      app: "nexus-llm"
      env: "production"

  elasticsearch:
    enabled: false
    url: "http://elasticsearch:9200"
    index: "nexus-llm-logs"
```

---

## Health Checks

### HTTP Health Endpoint

```bash
curl http://localhost:8000/v1/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "models_loaded": 1,
  "gpu_available": true,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_memory_total_mb": 81920.0,
  "gpu_memory_used_mb": 14200.5
}
```

### Readiness vs Liveness

For Kubernetes deployments, use the readiness and liveness endpoints:

```yaml
# Kubernetes probes
livenessProbe:
  httpGet:
    path: /v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /v1/health/ready
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 5
```

The `/v1/health/ready` endpoint returns 200 only when:
- At least one model is loaded
- GPU memory is not critically low
- No critical alerts are active

---

## Performance Monitoring

### Token Throughput

Monitor tokens per second across models:

```bash
nexus-llm metrics throughput --model meta-llama/Llama-3.1-8B-Instruct

# Output:
# Model: meta-llama/Llama-3.1-8B-Instruct
# Tokens/second: 85.2
# P50 latency: 12ms/token
# P95 latency: 25ms/token
# P99 latency: 45ms/token
# Time to first token: 180ms
```

### Benchmark Comparison

Compare model performance across different configurations:

```bash
# Run a benchmark
nexus-llm benchmark run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-length 128 \
  --max-tokens 256 \
  --iterations 50

# Compare benchmarks
nexus-llm benchmark compare \
  --baseline baseline_fp16.json \
  --current current_awq.json
```

---

## Production Deployment Checklist

- [ ] Enable Prometheus metrics endpoint
- [ ] Configure alert notification channels (Slack, email)
- [ ] Set up Grafana dashboards
- [ ] Configure log aggregation (Loki, Elasticsearch, or CloudWatch)
- [ ] Set up health check probes (readiness + liveness)
- [ ] Define alert rules for GPU memory, latency, error rate, and safety events
- [ ] Set up log rotation or retention policies
- [ ] Monitor disk usage for model cache and logs
- [ ] Document escalation procedures for critical alerts
- [ ] Test alert notifications end-to-end before going live

---

## Related Documentation

- [Deployment Guide](deployment.md) — Production deployment instructions
- [Safety Guide](safety.md) — Safety metrics and monitoring
- [Error Codes Reference](../api/errors.md) — API error documentation
