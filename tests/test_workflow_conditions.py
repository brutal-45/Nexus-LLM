"""Tests for nexus_llm.workflow.conditions module."""

import pytest
from nexus_llm.workflow.conditions import (
    field_equals,
    field_compare,
    field_in,
    field_exists,
    always,
    never,
    not_condition,
    and_conditions,
    or_conditions,
)


class TestFieldEquals:
    def test_match(self):
        cond = field_equals("status", "success")
        assert cond({"status": "success"}) is True

    def test_no_match(self):
        cond = field_equals("status", "success")
        assert cond({"status": "failed"}) is False

    def test_missing_field(self):
        cond = field_equals("status", "success")
        assert cond({}) is False


class TestFieldCompare:
    def test_equal(self):
        cond = field_compare("count", "==", 5)
        assert cond({"count": 5}) is True

    def test_greater_than(self):
        cond = field_compare("count", ">", 3)
        assert cond({"count": 5}) is True

    def test_less_than(self):
        cond = field_compare("count", "<", 10)
        assert cond({"count": 5}) is True

    def test_invalid_operator(self):
        with pytest.raises(ValueError):
            field_compare("count", "??", 5)


class TestFieldIn:
    def test_in_list(self):
        cond = field_in("status", ["running", "completed"])
        assert cond({"status": "running"}) is True

    def test_not_in_list(self):
        cond = field_in("status", ["running", "completed"])
        assert cond({"status": "failed"}) is False


class TestFieldExists:
    def test_exists(self):
        cond = field_exists("key")
        assert cond({"key": "value"}) is True

    def test_not_exists(self):
        cond = field_exists("key")
        assert cond({}) is False


class TestAlways:
    def test_always_true(self):
        cond = always()
        assert cond({}) is True
        assert cond({"any": "data"}) is True


class TestNever:
    def test_never_true(self):
        cond = never()
        assert cond({}) is False


class TestNotCondition:
    def test_negate_true(self):
        cond = not_condition(always())
        assert cond({}) is False

    def test_negate_false(self):
        cond = not_condition(never())
        assert cond({}) is True


class TestAndConditions:
    def test_all_true(self):
        cond = and_conditions(always(), always())
        assert cond({}) is True

    def test_one_false(self):
        cond = and_conditions(always(), never())
        assert cond({}) is False

    def test_all_false(self):
        cond = and_conditions(never(), never())
        assert cond({}) is False


class TestOrConditions:
    def test_all_true(self):
        cond = or_conditions(always(), always())
        assert cond({}) is True

    def test_one_true(self):
        cond = or_conditions(always(), never())
        assert cond({}) is True

    def test_all_false(self):
        cond = or_conditions(never(), never())
        assert cond({}) is False
