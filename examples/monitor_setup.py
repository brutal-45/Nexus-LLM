#!/usr/bin/env python3
"""Monitoring setup example."""
from nexus_llm.monitoring import MetricsCollector, AlertManager

collector = MetricsCollector()
alert_mgr = AlertManager()
alert_mgr.add_rule('high_latency', 'gt', 5.0, 'warning')

collector.record_metric('latency', 6.5)
alerts = alert_mgr.evaluate('latency', 6.5)
for a in alerts:
    print(f'Alert: {a}')
