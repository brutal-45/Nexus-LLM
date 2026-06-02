"""Nexus-LLM Alert System for Threshold Monitoring.

Provides configurable alert management with threshold-based triggering,
severity levels, notification channels, and alert history tracking. 
"""

import json
import logging
import os
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """An alert event.

    Attributes:
        id: Unique alert identifier.
        name: Alert rule name that triggered.
        level: Alert severity level.
        message: Human-readable alert message.
        metric_name: Name of the metric that triggered the alert.
        current_value: Current metric value.
        threshold: Configured threshold value.
        comparison: Comparison operator (gt, lt, gte, lte, eq, neq).
        created_at: When the alert was triggered.
        acknowledged: Whether the alert has been acknowledged.
        resolved: Whether the alert condition has resolved.
        resolved_at: When the alert was resolved.
        metadata: Additional alert metadata.
    """

    id: str = ""
    name: str = ""
    level: AlertLevel = AlertLevel.WARNING
    message: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    comparison: str = "gt"
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class AlertRule:
    """A configurable alert rule.

    Attributes:
        name: Unique name for the alert rule.
        metric_name: Name of the metric to monitor.
        threshold: Threshold value for triggering.
        comparison: Comparison operator.
        level: Alert severity when triggered.
        cooldown_seconds: Minimum time between repeated alerts.
        message_template: Message template with {value} and {threshold} placeholders.
        enabled: Whether the rule is active.
    """

    name: str = ""
    metric_name: str = ""
    threshold: float = 0.0
    comparison: str = "gt"  # gt, lt, gte, lte, eq, neq
    level: AlertLevel = AlertLevel.WARNING
    cooldown_seconds: int = 300  # 5 minutes
    message_template: str = "{metric_name} is {value}, threshold is {threshold}"
    enabled: bool = True


class AlertManager:
    """Manages alert rules, evaluation, notification, and history.

    Monitors system metrics against configurable thresholds and triggers
    alerts when conditions are met. Supports multiple notification channels
    and cooldown periods to prevent alert fatigue.

    Attributes:
        rules: Dictionary of registered alert rules.
    """

    def __init__(self) -> None:
        """Initialize the alert manager."""
        self.rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._last_triggered: Dict[str, datetime] = {}
        self._notification_channels: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        self._max_history = 1000

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule.

        Args:
            rule: AlertRule to register.
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule '{rule.name}' for metric '{rule.metric_name}'")

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule.

        Args:
            name: Name of the rule to remove.

        Returns:
            True if the rule was removed.
        """
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Removed alert rule '{name}'")
            return True
        return False

    def add_notification_channel(self, channel: Callable[[Alert], None]) -> None:
        """Register a notification channel.

        Args:
            channel: Callable that accepts an Alert and sends a notification.
        """
        self._notification_channels.append(channel)
        logger.info(f"Added notification channel: {channel.__name__}")

    def evaluate(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate all alert rules against current metrics.

        Args:
            metrics: Dictionary mapping metric names to current values.

        Returns:
            List of newly triggered alerts.
        """
        new_alerts: List[Alert] = []

        with self._lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue

                value = metrics.get(rule.metric_name)
                if value is None:
                    continue

                if self._check_condition(value, rule.threshold, rule.comparison):
                    # Check cooldown
                    last_triggered = self._last_triggered.get(rule_name)
                    if last_triggered and rule.cooldown_seconds > 0:
                        elapsed = (datetime.now() - last_triggered).total_seconds()
                        if elapsed < rule.cooldown_seconds:
                            continue

                    alert = Alert(
                        name=rule.name,
                        level=rule.level,
                        message=rule.message_template.format(
                            metric_name=rule.metric_name,
                            value=value,
                            threshold=rule.threshold,
                        ),
                        metric_name=rule.metric_name,
                        current_value=value,
                        threshold=rule.threshold,
                        comparison=rule.comparison,
                    )

                    self._active_alerts[rule_name] = alert
                    self._last_triggered[rule_name] = datetime.now()
                    self._alert_history.append(alert)

                    if len(self._alert_history) > self._max_history:
                        self._alert_history = self._alert_history[-self._max_history:]

                    new_alerts.append(alert)

                    # Send notifications
                    self._notify(alert)
                else:
                    # Check if previously triggered alert should be resolved
                    if rule_name in self._active_alerts:
                        alert = self._active_alerts.pop(rule_name)
                        alert.resolved = True
                        alert.resolved_at = datetime.now()
                        logger.info(f"Alert '{rule_name}' resolved")

        return new_alerts

    def _check_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if a metric value meets the alert condition.

        Args:
            value: Current metric value.
            threshold: Threshold value.
            comparison: Comparison operator.

        Returns:
            True if the condition is met.
        """
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
            "neq": lambda v, t: v != t,
        }
        check = ops.get(comparison, ops["gt"])
        return check(value, threshold)

    def _notify(self, alert: Alert) -> None:
        """Send alert through all registered notification channels.

        Args:
            alert: The alert to send.
        """
        for channel in self._notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Notification channel error: {e}")

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an active alert.

        Args:
            alert_id: ID of the alert to acknowledge.

        Returns:
            True if the alert was found and acknowledged.
        """
        with self._lock:
            for alert in self._active_alerts.values():
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
        return False

    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get currently active (unresolved) alerts.

        Args:
            level: Filter by alert level.

        Returns:
            List of active Alert objects.
        """
        with self._lock:
            alerts = list(self._active_alerts.values())
        if level:
            alerts = [a for a in alerts if a.level == level]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_history(
        self,
        limit: int = 100,
        level: Optional[AlertLevel] = None,
        since: Optional[datetime] = None,
    ) -> List[Alert]:
        """Get alert history.

        Args:
            limit: Maximum alerts to return.
            level: Filter by alert level.
            since: Only return alerts after this time.

        Returns:
            List of Alert objects.
        """
        with self._lock:
            alerts = list(self._alert_history)

        if level:
            alerts = [a for a in alerts if a.level == level]
        if since:
            alerts = [a for a in alerts if a.created_at >= since]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics.

        Returns:
            Dictionary with alert counts by level and status.
        """
        with self._lock:
            active = list(self._active_alerts.values())
            history = list(self._alert_history)

        active_by_level = {}
        for alert in active:
            key = alert.level.value
            active_by_level[key] = active_by_level.get(key, 0) + 1

        return {
            "active_alerts": len(active),
            "total_alerts": len(history),
            "active_by_level": active_by_level,
            "rules_count": len(self.rules),
            "channels_count": len(self._notification_channels),
        }


def log_notification(alert: Alert) -> None:
    """Simple notification channel that logs alerts.

    Args:
        alert: The alert to log.
    """
    level_map = {
        AlertLevel.INFO: logging.INFO,
        AlertLevel.WARNING: logging.WARNING,
        AlertLevel.CRITICAL: logging.CRITICAL,
        AlertLevel.EMERGENCY: logging.CRITICAL,
    }
    log_level = level_map.get(alert.level, logging.WARNING)
    logger.log(log_level, f"[ALERT:{alert.level.value}] {alert.name}: {alert.message}")


def email_notification(
    smtp_host: str = "localhost",
    smtp_port: int = 587,
    sender: str = "alerts@nexus-llm.local",
    recipients: Optional[List[str]] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Callable[[Alert], None]:
    """Create an email notification channel.

    Args:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        sender: Sender email address.
        recipients: List of recipient email addresses.
        username: SMTP username.
        password: SMTP password.

    Returns:
        Callable notification channel function.
    """
    recipients = recipients or []

    def _send(alert: Alert) -> None:
        try:
            subject = f"[Nexus-LLM {alert.level.value.upper()}] {alert.name}"
            body = (
                f"Alert: {alert.name}\n"
                f"Level: {alert.level.value}\n"
                f"Message: {alert.message}\n"
                f"Metric: {alert.metric_name} = {alert.current_value} "
                f"(threshold: {alert.threshold})\n"
                f"Time: {alert.created_at.isoformat()}"
            )

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                server.sendmail(sender, recipients, msg.as_string())

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    return _send


def webhook_notification(url: str) -> Callable[[Alert], None]:
    """Create a webhook notification channel.

    Args:
        url: Webhook URL to POST alert data to.

    Returns:
        Callable notification channel function.
    """
    def _send(alert: Alert) -> None:
        try:
            import urllib.request
            data = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                logger.debug(f"Webhook notification sent: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    return _send
