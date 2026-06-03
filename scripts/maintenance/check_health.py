#!/usr/bin/env python3
"""Health check script."""
from nexus_llm.monitoring import HealthChecker
checker = HealthChecker()
report = checker.check_health()
print(report)
