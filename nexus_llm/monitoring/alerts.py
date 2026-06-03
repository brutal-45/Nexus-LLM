"""Alert management for Nexus-LLM monitoring."""

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class Severity(enum.Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """A rule that triggers alerts when a metric crosses a threshold."""
    name: str
    condition: str  # "gt", "lt", "gte", "lte", "eq"
    threshold: float
    severity: Severity
    description: str = ""
    cooldown_seconds: float = 300.0  # prevent alert spam


@dataclass
class Alert:
    """An active or historical alert."""
    alert_id: str
    rule_name: str
    severity: Severity
    message: str
    value: float
    threshold: float
    timestamp: float
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None


class AlertManager:
    """Manages alert rules and evaluates metrics against them.

    Supports adding rules with conditions and thresholds, evaluating
    metrics in real-time, and acknowledging alerts.
    """

    _CONDITION_OPS: Dict[str, Callable[[float, float], bool]] = {
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "eq": lambda v, t: v == t,
    }

    def __init__(self) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._last_fired: Dict[str, float] = {}  # rule_name -> last fire timestamp
        self._lock = threading.Lock()

    def add_rule(
        self,
        name: str,
        condition: str,
        threshold: float,
        severity: Severity = Severity.WARNING,
        description: str = "",
        cooldown_seconds: float = 300.0,
    ) -> None:
        """Add an alert rule.

        Args:
            name: Unique rule name (also the metric name by convention).
            condition: Comparison operator: "gt", "lt", "gte", "lte", "eq".
            threshold: Value threshold to compare against.
            severity: Alert severity level.
            description: Human-readable description.
            cooldown_seconds: Minimum time between repeated alerts.

        Raises:
            ValueError: If the condition is not a supported operator.
        """
        if condition not in self._CONDITION_OPS:
            raise ValueError(
                f"Unsupported condition '{condition}'. "
                f"Must be one of: {list(self._CONDITION_OPS.keys())}"
            )

        rule = AlertRule(
            name=name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            description=description,
            cooldown_seconds=cooldown_seconds,
        )
        with self._lock:
            self._rules[name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name."""
        with self._lock:
            self._rules.pop(name, None)
            self._last_fired.pop(name, None)

    def evaluate(self, metric_name: str, value: float) -> List[Alert]:
        """Evaluate a metric value against matching rules.

        If any rule is triggered (and not in cooldown), a new ``Alert``
        is created and added to active alerts.

        Args:
            metric_name: Name of the metric to evaluate.
            value: Current metric value.

        Returns:
            List of newly triggered alerts (may be empty).
        """
        triggered: List[Alert] = []
        now = time.time()

        with self._lock:
            for rule_name, rule in self._rules.items():
                if rule_name != metric_name:
                    continue

                op = self._CONDITION_OPS.get(rule.condition)
                if op is None:
                    continue

                if not op(value, rule.threshold):
                    continue

                # Cooldown check
                last = self._last_fired.get(rule_name, 0.0)
                if now - last < rule.cooldown_seconds:
                    continue

                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_name=rule_name,
                    severity=rule.severity,
                    message=(
                        f"Alert [{rule.severity.value}] '{rule_name}': "
                        f"value {value} {rule.condition} {rule.threshold}"
                    ),
                    value=value,
                    threshold=rule.threshold,
                    timestamp=now,
                )
                self._active_alerts[alert.alert_id] = alert
                self._alert_history.append(alert)
                self._last_fired[rule_name] = now
                triggered.append(alert)

        return triggered

    def get_active_alerts(self) -> List[Alert]:
        """Return all unacknowledged alerts sorted by severity and time."""
        with self._lock:
            alerts = [
                a for a in self._active_alerts.values()
                if not a.acknowledged
            ]
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.WARNING: 1,
            Severity.INFO: 2,
        }
        alerts.sort(key=lambda a: (severity_order.get(a.severity, 99), a.timestamp))
        return alerts

    def get_all_alerts(self) -> List[Alert]:
        """Return full alert history."""
        with self._lock:
            return list(self._alert_history)

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an active alert.

        Args:
            alert_id: The ID of the alert to acknowledge.

        Returns:
            ``True`` if the alert was found and acknowledged.
        """
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if alert is None:
                return False
            alert.acknowledged = True
            alert.acknowledged_at = time.time()
            return True

    def clear_alert(self, alert_id: str) -> bool:
        """Remove an alert from the active list entirely.

        Returns:
            ``True`` if the alert was found and removed.
        """
        with self._lock:
            return self._active_alerts.pop(alert_id, None) is not None

    def clear_all(self) -> None:
        """Remove all rules, active alerts, and history."""
        with self._lock:
            self._rules.clear()
            self._active_alerts.clear()
            self._alert_history.clear()
            self._last_fired.clear()

    def list_rules(self) -> List[AlertRule]:
        """Return all registered rules."""
        with self._lock:
            return list(self._rules.values())
