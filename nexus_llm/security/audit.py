"""Nexus-LLM Security Audit Log.

Provides the AuditLogger for recording security-relevant events
such as authentication attempts, access control decisions, and
configuration changes.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """A single audit log entry.

    Attributes:
        timestamp: When the event occurred.
        level: Severity level.
        event_type: Category of the event.
        actor: Who or what triggered the event.
        action: What action was performed.
        resource: What resource was affected.
        outcome: Result of the action (success, failure, denied).
        details: Additional event details.
        source_ip: Optional source IP address.
        request_id: Optional request ID for tracing.
    """

    timestamp: float = field(default_factory=time.time)
    level: AuditLevel = AuditLevel.INFO
    event_type: str = ""
    actor: str = ""
    action: str = ""
    resource: str = ""
    outcome: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source_ip: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "event_type": self.event_type,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": self.details,
            "source_ip": self.source_ip,
            "request_id": self.request_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """Security audit logger for recording and querying events.

    The AuditLogger maintains an in-memory log and optionally
    persists entries to a file. It supports querying by level,
    event type, actor, and time range.

    Example::

        audit = AuditLogger(log_file="audit.jsonl")
        audit.log(AuditLevel.INFO, event_type="auth", actor="user1", action="login", outcome="success")
        entries = audit.query(event_type="auth")
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        max_entries: int = 10000,
        min_level: AuditLevel = AuditLevel.INFO,
    ) -> None:
        self._log_file = log_file
        self._max_entries = max_entries
        self._min_level = min_level
        self._entries: List[AuditEntry] = []
        self._level_order = {
            AuditLevel.DEBUG: 0,
            AuditLevel.INFO: 1,
            AuditLevel.WARNING: 2,
            AuditLevel.ERROR: 3,
            AuditLevel.CRITICAL: 4,
        }
        logger.debug("AuditLogger initialized (file=%s, max=%d)", log_file, max_entries)

    def log(
        self,
        level: AuditLevel = AuditLevel.INFO,
        event_type: str = "",
        actor: str = "",
        action: str = "",
        resource: str = "",
        outcome: str = "",
        details: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditEntry:
        """Record an audit event.

        Args:
            level: Severity level.
            event_type: Event category.
            actor: Who triggered the event.
            action: What was done.
            resource: What was affected.
            outcome: Result of the action.
            details: Additional details.
            source_ip: Optional source IP.
            request_id: Optional request ID.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            level=level,
            event_type=event_type,
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {},
            source_ip=source_ip,
            request_id=request_id,
        )

        # Filter by minimum level
        if self._level_order.get(level, 0) < self._level_order.get(self._min_level, 0):
            return entry

        self._entries.append(entry)

        # Trim if over max
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Persist to file
        if self._log_file:
            self._append_to_file(entry)

        logger.debug("Audit: [%s] %s/%s by %s - %s", level.value, event_type, action, actor, outcome)
        return entry

    def query(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        level: Optional[AuditLevel] = None,
        outcome: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Query audit entries with filters.

        Args:
            event_type: Filter by event type.
            actor: Filter by actor.
            level: Filter by minimum level.
            outcome: Filter by outcome.
            since: Filter entries after this timestamp.
            until: Filter entries before this timestamp.
            limit: Maximum entries to return.

        Returns:
            List of matching AuditEntry objects.
        """
        results = self._entries
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if actor:
            results = [e for e in results if e.actor == actor]
        if level:
            min_ord = self._level_order.get(level, 0)
            results = [e for e in results if self._level_order.get(e.level, 0) >= min_ord]
        if outcome:
            results = [e for e in results if e.outcome == outcome]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        if until is not None:
            results = [e for e in results if e.timestamp <= until]
        return results[-limit:]

    def count_by_type(self) -> Dict[str, int]:
        """Count entries by event type.

        Returns:
            Dictionary mapping event types to counts.
        """
        counts: Dict[str, int] = {}
        for entry in self._entries:
            counts[entry.event_type] = counts.get(entry.event_type, 0) + 1
        return counts

    def clear(self) -> None:
        """Clear all in-memory audit entries."""
        self._entries.clear()

    def _append_to_file(self, entry: AuditEntry) -> None:
        """Append an entry to the log file."""
        try:
            os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")
        except Exception as exc:
            logger.error("Failed to write audit entry to file: %s", exc)
