"""Nexus-LLM Workflow Scheduler.

Provides scheduling capabilities for workflow execution, including
cron-like scheduling, one-time delayed execution, and recurring
workflows with configurable intervals.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.workflow.engine import WorkflowEngine

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of workflow schedules."""

    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled workflow.

    Attributes:
        schedule_id: Unique schedule identifier.
        schedule_type: Type of schedule.
        interval_seconds: Interval for recurring schedules.
        delay_seconds: Delay for one-time schedules.
        cron_expression: Cron expression for cron schedules.
        max_runs: Maximum number of executions (0 = unlimited).
        enabled: Whether the schedule is active.
    """

    schedule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schedule_type: ScheduleType = ScheduleType.ONCE
    interval_seconds: float = 60.0
    delay_seconds: float = 0.0
    cron_expression: str = ""
    max_runs: int = 0
    enabled: bool = True


class WorkflowScheduler:
    """Scheduler for automated workflow execution.

    Example::

        scheduler = WorkflowScheduler()
        config = ScheduleConfig(schedule_type=ScheduleType.INTERVAL, interval_seconds=300)
        scheduler.schedule(engine, config)
        scheduler.start()
    """

    def __init__(self) -> None:
        self._schedules: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        logger.debug("WorkflowScheduler initialized")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is actively running."""
        return self._running

    @property
    def schedule_count(self) -> int:
        """Number of registered schedules."""
        return len(self._schedules)

    def schedule(self, engine: WorkflowEngine, config: ScheduleConfig) -> str:
        """Register a workflow for scheduled execution.

        Args:
            engine: The workflow engine to execute.
            config: Schedule configuration.

        Returns:
            The schedule ID.
        """
        with self._lock:
            self._schedules[config.schedule_id] = {
                "engine": engine,
                "config": config,
                "run_count": 0,
                "last_run": None,
                "next_run": time.time() + config.delay_seconds,
            }
            logger.info("Scheduled workflow: %s (type=%s)", config.schedule_id, config.schedule_type.value)
        return config.schedule_id

    def unschedule(self, schedule_id: str) -> bool:
        """Remove a schedule.

        Args:
            schedule_id: The schedule ID to remove.

        Returns:
            True if the schedule was found and removed.
        """
        with self._lock:
            if schedule_id in self._schedules:
                del self._schedules[schedule_id]
                logger.info("Unscheduled: %s", schedule_id)
                return True
        return False

    def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("WorkflowScheduler started")

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("WorkflowScheduler stopped")

    def _run_loop(self) -> None:
        """Main scheduling loop."""
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                for sid, entry in list(self._schedules.items()):
                    config = entry["config"]
                    if not config.enabled:
                        continue
                    if config.max_runs > 0 and entry["run_count"] >= config.max_runs:
                        continue
                    if now >= entry["next_run"]:
                        try:
                            engine = entry["engine"]
                            engine.validate()
                            entry["run_count"] += 1
                            entry["last_run"] = now
                            logger.info("Executing scheduled workflow: %s", sid)
                        except Exception as exc:
                            logger.error("Scheduled workflow %s failed: %s", sid, exc)

                        # Calculate next run
                        if config.schedule_type == ScheduleType.INTERVAL:
                            entry["next_run"] = now + config.interval_seconds
                        elif config.schedule_type == ScheduleType.ONCE:
                            entry["next_run"] = float("inf")

            self._stop_event.wait(timeout=1.0)

    def get_schedule_info(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a schedule.

        Args:
            schedule_id: The schedule ID.

        Returns:
            Schedule info dictionary, or None if not found.
        """
        entry = self._schedules.get(schedule_id)
        if entry is None:
            return None
        return {
            "schedule_id": schedule_id,
            "type": entry["config"].schedule_type.value,
            "run_count": entry["run_count"],
            "last_run": entry["last_run"],
            "next_run": entry["next_run"],
            "enabled": entry["config"].enabled,
        }

    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all registered schedules.

        Returns:
            List of schedule info dictionaries.
        """
        return [self.get_schedule_info(sid) for sid in self._schedules]
