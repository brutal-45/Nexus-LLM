"""Tests for the chains module.

Covers Chain, SequentialChain, ParallelChain, and ConditionalChain.
"""

from __future__ import annotations

import pytest

from nexus_llm.chains.chain import Chain
from nexus_llm.chains.sequential import SequentialChain, StepError
from nexus_llm.chains.parallel import ParallelChain
from nexus_llm.chains.conditional import ConditionalChain


# ---------------------------------------------------------------------------
# Concrete Chain subclass for basic testing
# ---------------------------------------------------------------------------

class SimpleChain(Chain):
    """Minimal concrete Chain for testing the base class."""

    def run(self, input_data=None):
        result = input_data
        for step in self._steps:
            result = step(result)
        return result


# ---------------------------------------------------------------------------
# Chain (base class)
# ---------------------------------------------------------------------------

class TestChain:
    """Tests for Chain base class."""

    def test_create_chain(self):
        chain = SimpleChain(name="test")
        assert chain.name == "test"
        assert chain.step_count == 0

    def test_add_step(self):
        chain = SimpleChain(name="test")
        chain.add_step(lambda x: x)
        assert chain.step_count == 1

    def test_add_step_returns_self(self):
        chain = SimpleChain(name="test")
        result = chain.add_step(lambda x: x)
        assert result is chain

    def test_add_non_callable_raises(self):
        chain = SimpleChain(name="test")
        with pytest.raises(TypeError, match="callable"):
            chain.add_step("not_callable")

    def test_remove_step(self):
        chain = SimpleChain(name="test")
        chain.add_step(lambda x: x)
        chain.add_step(lambda x: x * 2)
        chain.remove_step(0)
        assert chain.step_count == 1

    def test_remove_step_out_of_range(self):
        chain = SimpleChain(name="test")
        with pytest.raises(IndexError):
            chain.remove_step(0)

    def test_insert_step(self):
        chain = SimpleChain(name="test")
        chain.add_step(lambda x: x)
        chain.insert_step(0, lambda x: x + 1)
        assert chain.step_count == 2

    def test_validate_with_steps(self):
        chain = SimpleChain(name="test")
        chain.add_step(lambda x: x)
        assert chain.validate() is True

    def test_validate_empty(self):
        chain = SimpleChain(name="test")
        assert chain.validate() is False

    def test_steps_property(self):
        chain = SimpleChain(name="test")
        chain.add_step(lambda x: x)
        steps = chain.steps
        assert len(steps) == 1

    def test_len(self):
        chain = SimpleChain(name="test")
        chain.add_step(lambda x: x)
        chain.add_step(lambda x: x)
        assert len(chain) == 2

    def test_repr(self):
        chain = SimpleChain(name="test")
        r = repr(chain)
        assert "test" in r


# ---------------------------------------------------------------------------
# SequentialChain
# ---------------------------------------------------------------------------

class TestSequentialChain:
    """Tests for SequentialChain."""

    def test_simple_sequential(self):
        chain = SequentialChain(name="seq1")
        chain.add_step(lambda x: x + 1)
        chain.add_step(lambda x: x * 2)
        result = chain.run(5)
        assert result == 12  # (5+1)*2

    def test_pipe_output(self):
        chain = SequentialChain(name="pipe")
        chain.add_step(lambda x: x.upper())
        chain.add_step(lambda x: x + "!")
        result = chain.run("hello")
        assert result == "HELLO!"

    def test_empty_chain_raises(self):
        chain = SequentialChain(name="empty")
        with pytest.raises(StepError):
            chain.run("input")

    def test_failing_step(self):
        chain = SequentialChain(name="fail")
        chain.add_step(lambda x: 1 / 0)
        with pytest.raises(StepError):
            chain.run(None)

    def test_retry_on_failure(self):
        call_count = 0

        def flaky_step(x):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return x

        chain = SequentialChain(name="retry", max_retries=3, retry_delay=0.0)
        chain.add_step(flaky_step)
        result = chain.run("data")
        assert result == "data"
        assert call_count == 3

    def test_negative_retries_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SequentialChain(name="bad", max_retries=-1)

    def test_none_input(self):
        chain = SequentialChain(name="none")
        chain.add_step(lambda x: "default" if x is None else x)
        result = chain.run(None)
        assert result == "default"


# ---------------------------------------------------------------------------
# ParallelChain
# ---------------------------------------------------------------------------

class TestParallelChain:
    """Tests for ParallelChain."""

    def test_parallel_execution(self):
        chain = ParallelChain(name="par1")
        chain.add_step(lambda x: x + 1)
        chain.add_step(lambda x: x * 2)
        chain.add_step(lambda x: x ** 2)
        results = chain.run(5)
        assert results == [6, 10, 25]

    def test_empty_chain_raises(self):
        chain = ParallelChain(name="empty")
        with pytest.raises(RuntimeError):
            chain.run("input")

    def test_failing_step_raises(self):
        chain = ParallelChain(name="fail")
        chain.add_step(lambda x: x)
        chain.add_step(lambda x: 1 / 0)
        with pytest.raises(RuntimeError):
            chain.run(5)

    def test_custom_max_workers(self):
        chain = ParallelChain(name="workers", max_workers=2)
        chain.add_step(lambda x: x + 1)
        chain.add_step(lambda x: x + 2)
        results = chain.run(10)
        assert results == [11, 12]

    def test_single_step(self):
        chain = ParallelChain(name="single")
        chain.add_step(lambda x: x * 3)
        results = chain.run(7)
        assert results == [21]


# ---------------------------------------------------------------------------
# ConditionalChain
# ---------------------------------------------------------------------------

class TestConditionalChain:
    """Tests for ConditionalChain."""

    def test_matching_condition(self):
        cond = ConditionalChain(name="cond")
        sub_chain = SequentialChain(name="sub1")
        sub_chain.add_step(lambda x: x + 100)
        cond.add_condition(lambda x: x > 0, sub_chain)
        result = cond.run(5)
        assert result == 105

    def test_no_matching_condition_default(self):
        sub_default = SequentialChain(name="default")
        sub_default.add_step(lambda x: "default")
        cond = ConditionalChain(name="cond", default_chain=sub_default)
        cond.add_condition(lambda x: x > 100, SequentialChain(name="never"))
        result = cond.run(5)
        assert result == "default"

    def test_no_matching_no_default_raises(self):
        cond = ConditionalChain(name="cond")
        cond.add_condition(lambda x: x > 100, SequentialChain(name="never"))
        with pytest.raises(ValueError, match="No condition matched"):
            cond.run(5)

    def test_condition_ordering(self):
        cond = ConditionalChain(name="cond")
        sub1 = SequentialChain(name="first")
        sub1.add_step(lambda x: "first")
        sub2 = SequentialChain(name="second")
        sub2.add_step(lambda x: "second")
        cond.add_condition(lambda x: True, sub1)
        cond.add_condition(lambda x: True, sub2)
        result = cond.run(None)
        assert result == "first"  # first match wins

    def test_condition_exception_skipped(self):
        cond = ConditionalChain(name="cond")
        sub = SequentialChain(name="good")
        sub.add_step(lambda x: "good")
        cond.add_condition(lambda x: 1 / 0, SequentialChain(name="error"))
        cond.add_condition(lambda x: True, sub)
        result = cond.run(None)
        assert result == "good"

    def test_add_condition_type_check(self):
        cond = ConditionalChain(name="cond")
        with pytest.raises(TypeError, match="callable"):
            cond.add_condition("not_callable", SequentialChain(name="sub"))

    def test_add_condition_chain_type_check(self):
        cond = ConditionalChain(name="cond")
        with pytest.raises(TypeError, match="Chain"):
            cond.add_condition(lambda x: True, "not_a_chain")

    def test_remove_condition(self):
        cond = ConditionalChain(name="cond")
        sub = SequentialChain(name="sub")
        sub.add_step(lambda x: x)
        cond.add_condition(lambda x: True, sub)
        cond.remove_condition(0)
        assert len(cond.conditions) == 0

    def test_remove_condition_out_of_range(self):
        cond = ConditionalChain(name="cond")
        with pytest.raises(IndexError):
            cond.remove_condition(0)

    def test_validate_with_conditions(self):
        cond = ConditionalChain(name="cond")
        sub = SequentialChain(name="sub")
        sub.add_step(lambda x: x)
        cond.add_condition(lambda x: True, sub)
        assert cond.validate() is True

    def test_validate_with_default(self):
        sub = SequentialChain(name="default")
        sub.add_step(lambda x: x)
        cond = ConditionalChain(name="cond", default_chain=sub)
        assert cond.validate() is True

    def test_validate_empty(self):
        cond = ConditionalChain(name="cond")
        assert cond.validate() is False
