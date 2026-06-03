# Monitoring Setup Guide

This guide covers setting up monitoring, alerting, and observability for Nexus-LLM deployments.

## Overview

Nexus-LLM provides built-in monitoring through:
- **Metrics**: Prometheus-compatible metrics endpoint
- **Logging**: Structured JSON logging
- **Health checks**: Liveness and readiness endpoints
- **Dashboards**: Built-in monitoring UI or Grafana integration

## Quick Setup

### Enable Built-in Metrics

```python
from nexus_llm import InferenceEngine
from nexus_llm.monitoring import MetricsCollector

metrics = MetricsCollector(
    collect_interval=5,          # Seconds between metric collections
    retention_hours=24,
)

metrics.enable_default_metrics({
    "inference_latency": True,
    "tokens_generated": True,
    "request_count": True,
    "error_count": True,
    "gpu_utilization": True,
    "gpu_memory_used": True,
})

engine = InferenceEngine(
    model_name="nexus-7b-chat",
    metrics_collector=metrics,
)
```

### Start Prometheus Exporter

```python
from nexus_llm.monitoring import PrometheusExporter

prometheus = PrometheusExporter(
    port=9090,
    metrics=metrics,
    prefix="nexus_llm_",
)
prometheus.start()
```

Metrics are available at `http://localhost:9090/metrics`.

## Available Metrics

### Inference Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nexus_llm_inference_latency_p50` | gauge | P50 inference latency (seconds) |
| `nexus_llm_inference_latency_p95` | gauge | P95 inference latency (seconds) |
| `nexus_llm_inference_latency_p99` | gauge | P99 inference latency (seconds) |
| `nexus_llm_tokens_generated_total` | counter | Total tokens generated |
| `nexus_llm_tokens_per_second` | gauge | Generation throughput |
| `nexus_llm_time_to_first_token` | gauge | TTFT in seconds |
| `nexus_llm_batch_size` | gauge | Current batch size |

### Request Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nexus_llm_request_count_total` | counter | Total requests processed |
| `nexus_llm_error_count_total` | counter | Total errors |
| `nexus_llm_error_rate` | gauge | Error rate (0-1) |
| `nexus_llm_active_requests` | gauge | Currently processing requests |
| `nexus_llm_queue_length` | gauge | Requests waiting in queue |

### Resource Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nexus_llm_gpu_utilization` | gauge | GPU compute utilization (0-1) |
| `nexus_llm_gpu_memory_used_bytes` | gauge | GPU memory used |
| `nexus_llm_gpu_memory_total_bytes` | gauge | Total GPU memory |
| `nexus_llm_cpu_utilization` | gauge | CPU utilization (0-1) |
| `nexus_llm_memory_used_bytes` | gauge | System memory used |

### Safety Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `nexus_llm_safety_blocks_total` | counter | Requests blocked by safety filter |
| `nexus_llm_safety_warnings_total` | counter | Safety warnings issued |
| `nexus_llm_pii_redactions_total` | counter | PII redactions performed |

## Prometheus Configuration

### prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nexus-llm'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['nexus-llm:9090']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
```

### Alert Rules

Create `nexus_llm_alerts.yml`:

```yaml
groups:
  - name: nexus_llm_alerts
    rules:
      - alert: HighLatency
        expr: nexus_llm_inference_latency_p99 > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P99 inference latency"
          description: "P99 latency is {{ $value }}s (threshold: 5s)"

      - alert: CriticalLatency
        expr: nexus_llm_inference_latency_p99 > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical P99 inference latency"
          description: "P99 latency is {{ $value }}s (threshold: 10s)"

      - alert: HighErrorRate
        expr: nexus_llm_error_rate > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      - alert: GPUMemoryCritical
        expr: nexus_llm_gpu_memory_used_bytes / nexus_llm_gpu_memory_total_bytes > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory usage critical"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"

      - alert: HighSafetyBlocks
        expr: rate(nexus_llm_safety_blocks_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of safety blocks"
          description: "{{ $value }} safety blocks per second"
```

## Grafana Dashboard

### Import the Dashboard

1. Open Grafana and navigate to Dashboards → Import
2. Upload `monitoring/grafana/nexus-llm-dashboard.json`
3. Select your Prometheus data source
4. Click Import

### Key Dashboard Panels

- **Request Rate**: Requests per second over time
- **Latency Distribution**: P50/P95/P99 latency
- **Throughput**: Tokens generated per second
- **GPU Resources**: Memory and compute utilization
- **Error Rate**: Errors as percentage of total requests
- **Safety Events**: Blocks and warnings over time

## Alerting Configuration

### Slack Integration

```python
from nexus_llm.monitoring import AlertManager, AlertRule

alert_manager = AlertManager(
    notification_channels=[
        {
            "type": "webhook",
            "url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
        },
    ],
)

alert_manager.add_rule(AlertRule(
    name="high_latency",
    metric="inference_latency_p99",
    condition="greater_than",
    threshold=5.0,
    window_seconds=300,
    severity="warning",
    message="P99 inference latency exceeds 5s",
))
```

### Email Alerts

```python
alert_manager = AlertManager(
    notification_channels=[
        {
            "type": "email",
            "recipients": ["oncall@example.com", "ops@example.com"],
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_tls": True,
        },
    ],
)
```

## Health Checks

### Endpoints

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `GET /health` | Liveness check | No |
| `GET /ready` | Readiness check | No |

### Response Format

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "models_loaded": 2,
  "gpu_available": true,
  "gpu_memory_used_mb": 14000,
  "gpu_memory_total_mb": 24000,
  "active_requests": 5,
  "uptime_seconds": 86400
}
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

## Logging

### Structured JSON Logging

```yaml
logging:
  level: info
  format: json
  output: stdout
  fields:
    timestamp: true
    request_id: true
    user_id: true
    model: true
    latency: true
    token_count: true
```

### Example Log Entry

```json
{
  "timestamp": "2024-03-15T10:30:00.123Z",
  "level": "info",
  "request_id": "req-abc123",
  "user_id": "dev_001",
  "model": "nexus-7b-chat",
  "event": "inference_complete",
  "latency_seconds": 1.234,
  "prompt_tokens": 25,
  "completion_tokens": 42,
  "total_tokens": 67
}
```

## Monitoring Checklist

- [ ] Prometheus exporter is running on port 9090
- [ ] Grafana dashboard is imported and displaying data
- [ ] Alert rules are configured for latency, errors, and memory
- [ ] Health check endpoints are configured
- [ ] Structured logging is enabled
- [ ] Log aggregation (ELK, Loki, or CloudWatch) is set up
- [ ] Alert notifications are tested and reaching the right channels
- [ ] Dashboard is accessible to the on-call team
