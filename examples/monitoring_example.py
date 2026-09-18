#!/usr/bin/env python3
"""
Monitoring Setup Example - Nexus-LLM
======================================
Demonstrates how to set up monitoring, logging, and alerting
for Nexus-LLM deployments.
"""

from nexus_llm import InferenceEngine
from nexus_llm.monitoring import (
    MetricsCollector,
    PrometheusExporter,
    DashboardConfig,
    AlertManager,
    AlertRule,
    HealthChecker,
)


def main():
    # --- Set up metrics collection ---
    metrics = MetricsCollector(
        collect_interval=5,           # Seconds between metric collections
        retention_hours=24,           # How long to retain metrics in memory
        export_path="./metrics",      # Path for metric snapshots
    )

    # Enable built-in metrics
    metrics.enable_default_metrics({
        "inference_latency": True,
        "tokens_generated": True,
        "request_count": True,
        "error_count": True,
        "gpu_utilization": True,
        "gpu_memory_used": True,
        "queue_length": True,
        "batch_size": True,
    })

    # --- Configure Prometheus export ---
    prometheus = PrometheusExporter(
        port=9090,
        metrics=metrics,
        prefix="nexus_llm_",
    )
    prometheus.start()
    print("Prometheus exporter started on port 9090")

    # --- Set up alerting ---
    alert_manager = AlertManager(
        notification_channels=[
            {"type": "webhook", "url": "https://hooks.slack.com/services/XXX"},
            {"type": "email", "recipients": ["ops-team@example.com"]},
        ],
    )

    # Define alert rules
    alert_manager.add_rule(AlertRule(
        name="high_latency",
        metric="inference_latency_p99",
        condition="greater_than",
        threshold=5.0,               # 5 seconds
        window_seconds=300,          # Check over 5-minute window
        severity="warning",
        message="P99 inference latency exceeds 5s",
    ))

    alert_manager.add_rule(AlertRule(
        name="gpu_memory_critical",
        metric="gpu_memory_used_percent",
        condition="greater_than",
        threshold=90.0,              # 90% utilization
        window_seconds=60,
        severity="critical",
        message="GPU memory usage exceeds 90%",
    ))

    alert_manager.add_rule(AlertRule(
        name="high_error_rate",
        metric="error_rate",
        condition="greater_than",
        threshold=0.05,              # 5% error rate
        window_seconds=600,
        severity="critical",
        message="Error rate exceeds 5% over the last 10 minutes",
    ))

    # --- Configure health checks ---
    health_checker = HealthChecker(
        check_interval=30,           # Seconds between health checks
        alert_manager=alert_manager,
    )

    health_checker.add_check("model_loaded", lambda: engine.is_model_loaded())
    health_checker.add_check("gpu_available", lambda: engine.is_gpu_available())
    health_checker.add_check("memory_sufficient", lambda: engine.get_memory_usage() < 0.9)

    # --- Attach monitoring to the engine ---
    engine = InferenceEngine(
        model_name="nexus-7b-chat",
        device="auto",
        metrics_collector=metrics,
        health_checker=health_checker,
    )

    # --- Start monitoring dashboard ---
    dashboard = DashboardConfig(
        host="0.0.0.0",
        port=8080,
        refresh_interval=5,          # Dashboard refresh in seconds
        panels=[
            {
                "title": "Inference Latency",
                "type": "time_series",
                "metrics": ["inference_latency_p50", "inference_latency_p95", "inference_latency_p99"],
                "unit": "seconds",
            },
            {
                "title": "Throughput",
                "type": "time_series",
                "metrics": ["tokens_per_second", "requests_per_second"],
                "unit": "per_second",
            },
            {
                "title": "GPU Resources",
                "type": "gauge",
                "metrics": ["gpu_utilization", "gpu_memory_used_percent"],
                "unit": "percent",
            },
            {
                "title": "Error Rate",
                "type": "time_series",
                "metrics": ["error_rate"],
                "unit": "percent",
            },
        ],
    )

    print(f"Monitoring dashboard available at http://{dashboard.host}:{dashboard.port}")
    print(f"Prometheus metrics at http://localhost:{prometheus.port}/metrics")
    print(f"Health endpoint at http://localhost:8000/health")

    # --- Log custom events ---
    metrics.log_event("deployment_started", {"model": "nexus-7b-chat", "version": "2.1.0"})
    metrics.log_event("configuration_changed", {"parameter": "max_batch_size", "value": 16})

    # --- Run some inference to generate metrics ---
    from nexus_llm import Conversation
    conversation = Conversation(system_prompt="You are a helpful assistant.")

    for i in range(5):
        conversation.add_user_message(f"Tell me an interesting fact about number {i}.")
        response = engine.chat(conversation)
        print(f"Query {i+1}: Latency={response.elapsed_time:.3f}s, Tokens={response.token_count}")

    # Print current metrics
    print("\n--- Current Metrics Snapshot ---")
    snapshot = metrics.get_snapshot()
    for name, value in snapshot.items():
        print(f"  {name}: {value}")

    # Keep services running (in production, use a process manager)
    print("\nMonitoring services are running. Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down monitoring...")


if __name__ == "__main__":
    main()
