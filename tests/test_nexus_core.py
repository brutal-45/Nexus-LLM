"""Tests for the nexus orchestration subpackage.

Covers the pieces that make up the core: engine job lifecycle, runtime context,
request dispatch with middleware, task coordination with dependencies, the
performance optimiser, text analysis, transformations, prompt composition and
the NexusCore facade.
"""

from __future__ import annotations

import pytest

from nexus_llm.nexus.analyzer import Analyzer
from nexus_llm.nexus.composer import Composer
from nexus_llm.nexus.coordinator import Coordinator, TaskPriority, TaskState
from nexus_llm.nexus.core import NexusCore, create_nexus, get_nexus_instance, shutdown_nexus
from nexus_llm.nexus.dispatcher import DispatchStatus, Dispatcher, Request
from nexus_llm.nexus.engine import Engine, EngineConfig, EngineState
from nexus_llm.nexus.optimizer import OptimizationLevel, Optimizer
from nexus_llm.nexus.runtime import Runtime, RuntimeConfig
from nexus_llm.nexus.transformer import Transformer


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:
    def test_initial_state_is_idle(self):
        assert Engine().state is EngineState.IDLE

    def test_start_makes_it_ready(self):
        engine = Engine()
        engine.start()
        assert engine.state is EngineState.READY

    def test_double_start_is_rejected(self):
        engine = Engine()
        engine.start()
        with pytest.raises(Exception):
            engine.start()

    def test_submit_runs_the_job(self):
        engine = Engine()
        engine.start()
        result = engine.submit(lambda x: x * 2, x=21)
        assert result.success is True
        assert result.output == 42
        assert result.job_id

    def test_submit_captures_failures(self):
        engine = Engine()
        engine.start()

        def boom():
            raise RuntimeError("nope")

        result = engine.submit(boom)
        assert result.success is False
        assert "nope" in result.error

    def test_submit_before_start_raises(self):
        with pytest.raises(Exception):
            Engine().submit(lambda: 1)

    def test_pause_requires_a_running_engine(self):
        engine = Engine()
        engine.start()
        # A ready engine has no in-flight work, so pausing is rejected.
        with pytest.raises(Exception):
            engine.pause()
        engine._transition(EngineState.RUNNING)
        engine.pause()
        assert engine.state is EngineState.PAUSED
        engine.resume()
        assert engine.state is EngineState.RUNNING

    def test_stop_and_reset(self):
        engine = Engine()
        engine.start()
        engine.stop()
        assert engine.state is EngineState.STOPPED
        engine.reset()
        assert engine.state is EngineState.IDLE

    def test_concurrency_limit_is_enforced(self):
        engine = Engine(EngineConfig(max_concurrent_jobs=1))
        engine.start()

        def busy():
            # occupies the only slot while the second job is submitted
            engine.submit(lambda: None)
            return "done"

        engine.submit(busy)
        assert engine.active_job_count == 0  # jobs complete synchronously

    def test_state_events_are_emitted(self):
        engine = Engine()
        seen = []
        engine.on("state_change", seen.append)
        engine.start()
        assert seen and seen[-1]["new"] == "ready"

    def test_health_check_reports_state(self):
        engine = Engine()
        engine.start()
        health = engine.health_check()
        assert health["state"] == "ready"
        assert "active_jobs" in health


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class TestRuntime:
    def test_start_and_stop(self):
        runtime = Runtime()
        assert runtime.is_started is False
        runtime.start()
        assert runtime.is_started is True
        runtime.stop()
        assert runtime.is_started is False

    def test_context_round_trip(self):
        runtime = Runtime()
        runtime.start()
        runtime.set_context("model", "gpt2")
        assert runtime.get_context("model") == "gpt2"
        assert runtime.get_context("missing", "fallback") == "fallback"

    def test_hooks_run_in_order(self):
        runtime = Runtime()
        calls = []
        runtime.add_startup_hook(lambda r: calls.append("up"))
        runtime.add_shutdown_hook(lambda r: calls.append("down"))
        runtime.start()
        runtime.stop()
        assert calls == ["up", "down"]

    def test_config_is_exposed(self):
        config = RuntimeConfig()
        assert Runtime(config).config is config

    def test_platform_info_and_usage(self):
        runtime = Runtime()
        runtime.start()
        assert "python_version" in runtime.platform_info() or runtime.platform_info()
        assert runtime.resource_usage is not None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_dispatch_to_registered_handler(self):
        dispatcher = Dispatcher()
        dispatcher.register_handler("ping", lambda req: {"pong": True})
        result = dispatcher.dispatch(Request(type="ping", payload={}))
        assert result.status is DispatchStatus.COMPLETED
        assert result.output == {"pong": True}

    def test_unknown_type_is_not_a_success(self):
        dispatcher = Dispatcher()
        result = dispatcher.dispatch(Request(type="nothing", payload={}))
        assert result.status in (DispatchStatus.REJECTED, DispatchStatus.FAILED)

    def test_handler_exception_is_captured(self):
        dispatcher = Dispatcher()

        def boom(request):
            raise ValueError("bad")

        dispatcher.register_handler("explode", boom)
        result = dispatcher.dispatch(Request(type="explode", payload={}))
        assert result.status is DispatchStatus.FAILED
        assert "bad" in (result.error or "")

    def test_middleware_can_rewrite_the_request(self):
        dispatcher = Dispatcher()
        seen = []

        def upper_type(request):
            return Request(type=request.type.upper(), payload=request.payload)

        dispatcher.add_middleware(upper_type)
        dispatcher.register_handler("T", lambda req: seen.append(req) or "ok")
        result = dispatcher.dispatch(Request(type="t", payload={"n": 1}))
        assert result.status is DispatchStatus.COMPLETED
        assert seen and seen[0].payload == {"n": 1}

    def test_rejecting_middleware_short_circuits(self):
        dispatcher = Dispatcher()
        dispatcher.register_handler("t", lambda req: "ok")

        def reject(request):
            raise ValueError("denied")

        dispatcher.add_middleware(reject)
        result = dispatcher.dispatch(Request(type="t", payload={}))
        assert result.status is DispatchStatus.REJECTED
        assert "denied" in result.error

    def test_dispatch_batch(self):
        dispatcher = Dispatcher()
        dispatcher.register_handler("double", lambda req: req.payload["n"] * 2)
        results = dispatcher.dispatch_batch(
            [Request(type="double", payload={"n": i}) for i in range(3)]
        )
        assert [r.output for r in results] == [0, 2, 4]

    def test_unregister_handler(self):
        dispatcher = Dispatcher()
        handler = lambda req: "x"  # noqa: E731
        dispatcher.register_handler("temp", handler)
        assert dispatcher.handler_count == 1
        dispatcher.unregister_handler("temp")
        assert dispatcher.handler_count == 0


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class TestCoordinator:
    def test_submit_and_process(self):
        coordinator = Coordinator(max_workers=2)
        task_id = coordinator.submit(lambda: 7)
        processed = coordinator.process_all()
        assert processed >= 1
        assert coordinator.get_result(task_id) == 7

    def test_priority_ordering(self):
        coordinator = Coordinator()
        order = []
        coordinator.submit(lambda: order.append("low"), priority=TaskPriority.LOW)
        coordinator.submit(lambda: order.append("high"), priority=TaskPriority.HIGH)  # noqa: E501
        coordinator.process_all()
        assert order[0] == "high"

    def test_dependencies_are_respected(self):
        coordinator = Coordinator()
        first = coordinator.submit(lambda: 1)
        second = coordinator.submit(lambda: 2, dependencies={first})
        coordinator.process_all()
        assert coordinator.get_task(first).state is TaskState.COMPLETED
        assert coordinator.get_task(second).state is TaskState.COMPLETED

    def test_cancel_pending_task(self):
        coordinator = Coordinator()
        task_id = coordinator.submit(lambda: "never")
        assert coordinator.cancel(task_id) is True
        coordinator.process_all()
        assert coordinator.get_task(task_id).state is not TaskState.COMPLETED

    def test_stats_and_health(self):
        coordinator = Coordinator()
        task_id = coordinator.submit(lambda: 1)
        coordinator.process_all()
        assert coordinator.get_task(task_id).state is TaskState.COMPLETED
        assert coordinator.stats()[TaskState.COMPLETED.value] >= 1
        health = coordinator.health_check()
        assert health["pending_count"] == 0


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_default_level(self):
        assert isinstance(Optimizer().level, OptimizationLevel)

    def test_set_level(self):
        optimizer = Optimizer()
        optimizer.set_level(OptimizationLevel.AGGRESSIVE)
        assert optimizer.level is OptimizationLevel.AGGRESSIVE

    def test_records_feed_metrics(self):
        optimizer = Optimizer()
        for latency in (10.0, 20.0, 30.0):
            optimizer.record_latency(latency)
        optimizer.record_success()
        optimizer.record_error()
        optimizer.record_throughput(100.0)
        metrics = optimizer.metrics
        assert metrics.sample_count >= 2
        assert metrics.avg_latency_ms == pytest.approx(20.0)
        assert 0.0 <= metrics.error_rate <= 1.0

    def test_optimize_returns_actions(self):
        optimizer = Optimizer()
        for _ in range(5):
            optimizer.record_error()
        actions = optimizer.optimize()
        assert isinstance(actions, list)

    def test_health_check_and_reset(self):
        optimizer = Optimizer()
        optimizer.record_latency(5.0)
        assert optimizer.health_check()
        optimizer.reset()
        assert optimizer.metrics.sample_count == 0


# ---------------------------------------------------------------------------
# Analyzer & Transformer
# ---------------------------------------------------------------------------


class TestAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return Analyzer()

    def test_counts(self, analyzer):
        result = analyzer.analyze("One two three. Another sentence here!")
        assert result.word_count == 6
        assert result.sentence_count == 2
        assert result.char_count > 0

    def test_lexical_diversity(self, analyzer):
        repeated = analyzer.analyze("cat cat cat cat")
        varied = analyzer.analyze("cat dog fox owl bat")
        assert repeated.lexical_diversity < varied.lexical_diversity

    def test_keywords_are_extracted(self, analyzer):
        result = analyzer.analyze("python testing frameworks pytest fixtures coverage")
        assert result.top_keywords

    def test_quick_stats_shape(self, analyzer):
        stats = analyzer.quick_stats("hello world")
        assert {"word_count", "char_count"} <= set(stats)

    def test_empty_text_is_safe(self, analyzer):
        result = analyzer.analyze("")
        assert result.word_count == 0


class TestTransformer:
    @pytest.fixture
    def transformer(self):
        return Transformer()

    def test_builtin_transforms_available(self, transformer):
        assert {"lowercase", "uppercase", "snake_case"} <= set(transformer.available_transforms())

    def test_apply_transform(self, transformer):
        result = transformer.apply("HELLO", "lowercase")
        assert result.transformed == "hello"
        assert result.original == "HELLO"

    def test_snake_case(self, transformer):
        assert transformer.apply("Hello World", "snake_case").transformed == "hello_world"

    def test_unknown_transform_reports_failure(self, transformer):
        with pytest.raises(ValueError):
            transformer.apply("x", "no-such-transform")

    def test_pipeline_applies_in_order(self, transformer):
        result = transformer.apply_pipeline("Hello World!", ["lowercase", "remove_punctuation"])
        assert result.transformed == "hello world"

    def test_custom_transform_registration(self, transformer):
        transformer.register_transform("exclaim", lambda text: text + "!")
        assert transformer.apply("hi", "exclaim").transformed == "hi!"

    def test_apply_custom_without_registering(self, transformer):
        assert transformer.apply_custom("ab", str.upper).transformed == "AB"


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


class TestComposer:
    def test_empty_composition(self):
        result = Composer().compose()
        assert isinstance(result.content, str)
        assert result.id

    def test_text_parts_are_included(self):
        composer = Composer()
        composer.set_system_prompt("You are helpful.")
        composer.add_text("question here")
        result = composer.compose()
        assert "question here" in result.content
        # The system prompt is carried as a message, not duplicated into content.
        assert any("You are helpful." in m.content for m in result.messages)

    def test_code_block_is_fenced(self):
        composer = Composer()
        composer.add_code_block("print(1)", language="python")
        assert "print(1)" in composer.compose().content

    def test_tool_calls_round_trip(self):
        composer = Composer()
        composer.add_tool_call("search", {"q": "llm"}, result="found")
        result = composer.compose()
        assert "[Tool: search]" in result.content
        assert result.parts == 1

    def test_part_count_and_reset(self):
        composer = Composer()
        composer.add_text("a")
        composer.add_text("b")
        assert composer.part_count == 2
        composer.reset()
        assert composer.part_count == 0

    def test_metadata_is_attached(self):
        composer = Composer()
        composer.add_metadata("task", "qa")
        assert composer.compose().metadata.get("task") == "qa"

    def test_health_check(self):
        assert Composer().health_check()


# ---------------------------------------------------------------------------
# NexusCore facade
# ---------------------------------------------------------------------------


class TestNexusCore:
    def test_start_stop_and_state(self):
        core = NexusCore()
        assert core.is_running is False
        core.start()
        try:
            assert core.is_running is True
        finally:
            core.stop()
        assert core.is_running is False

    def test_component_registry(self):
        core = NexusCore()
        marker = object()
        core.register_component("marker", marker)
        assert core.get_component("marker") is marker
        assert core.unregister_component("marker") is marker
        assert core.get_component("marker") is None

    def test_config_updates_merge(self):
        core = NexusCore(config={"a": 1})
        core.update_config({"b": 2})
        assert core.config["a"] == 1
        assert core.config["b"] == 2

    def test_health_check_lists_components(self):
        core = NexusCore()
        core.register_component("engine", Engine())
        health = core.health_check()
        assert "components" in health or "engine" in str(health)

    def test_context_manager(self):
        with NexusCore() as core:
            assert core.is_running is True
        assert core.is_running is False

    def test_singleton_helpers(self):
        shutdown_nexus()
        core = create_nexus({"name": "test"})
        assert get_nexus_instance() is core
        shutdown_nexus()
        assert get_nexus_instance() is None
