"""Tests for nexus_llm.workflow.scheduler module."""

import pytest
import time
from nexus_llm.workflow.scheduler import WorkflowScheduler, ScheduleConfig, ScheduleType
from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType


class TestScheduleConfig:
    def test_default(self):
        config = ScheduleConfig()
        assert config.schedule_type == ScheduleType.ONCE
        assert config.enabled is True

    def test_custom(self):
        config = ScheduleConfig(
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=300,
            max_runs=10,
        )
        assert config.schedule_type == ScheduleType.INTERVAL
        assert config.interval_seconds == 300


class TestWorkflowScheduler:
    def test_init(self):
        scheduler = WorkflowScheduler()
        assert scheduler.is_running is False
        assert scheduler.schedule_count == 0

    def test_schedule(self):
        scheduler = WorkflowScheduler()
        engine = WorkflowEngine()
        config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
        schedule_id = scheduler.schedule(engine, config)
        assert scheduler.schedule_count == 1

    def test_unschedule(self):
        scheduler = WorkflowScheduler()
        engine = WorkflowEngine()
        config = ScheduleConfig()
        schedule_id = scheduler.schedule(engine, config)
        assert scheduler.unschedule(schedule_id) is True
        assert scheduler.schedule_count == 0

    def test_unschedule_missing(self):
        scheduler = WorkflowScheduler()
        assert scheduler.unschedule("missing") is False

    def test_get_schedule_info(self):
        scheduler = WorkflowScheduler()
        engine = WorkflowEngine()
        config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
        schedule_id = scheduler.schedule(engine, config)
        info = scheduler.get_schedule_info(schedule_id)
        assert info is not None
        assert info["schedule_id"] == schedule_id

    def test_list_schedules(self):
        scheduler = WorkflowScheduler()
        engine = WorkflowEngine()
        scheduler.schedule(engine, ScheduleConfig())
        scheduler.schedule(engine, ScheduleConfig())
        schedules = scheduler.list_schedules()
        assert len(schedules) == 2

    def test_start_stop(self):
        scheduler = WorkflowScheduler()
        scheduler.start()
        assert scheduler.is_running is True
        scheduler.stop()
        assert scheduler.is_running is False
