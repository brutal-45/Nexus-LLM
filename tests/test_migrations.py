"""Tests for the migrations module.

Covers MigrationManager, Migration, and MigrationHistory.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nexus_llm.migrations.migration import Migration
from nexus_llm.migrations.history import MigrationHistory
from nexus_llm.migrations.manager import MigrationManager


# ---------------------------------------------------------------------------
# Concrete Migration subclasses for testing
# ---------------------------------------------------------------------------

class V1_CreateUsers(Migration):
    version = "20250101_000001"
    description = "Create users table"

    def up(self, context):
        context["users"] = []

    def down(self, context):
        context.pop("users", None)


class V2_AddSettings(Migration):
    version = "20250101_000002"
    description = "Add settings table"

    def up(self, context):
        context["settings"] = {}

    def down(self, context):
        context.pop("settings", None)


class V3_AddRoles(Migration):
    version = "20250101_000003"
    description = "Add roles table"

    def up(self, context):
        context["roles"] = []

    def down(self, context):
        context.pop("roles", None)


class FailingMigration(Migration):
    version = "20250101_000099"
    description = "A failing migration"

    def up(self, context):
        raise RuntimeError("Migration failed!")

    def down(self, context):
        raise RuntimeError("Rollback failed!")


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

class TestMigration:
    """Tests for Migration base class."""

    def test_version_and_description(self):
        m = V1_CreateUsers()
        assert m.version == "20250101_000001"
        assert m.description == "Create users table"

    def test_migration_id(self):
        m = V1_CreateUsers()
        assert m.migration_id.startswith("20250101_000001_")

    def test_auto_generated_version(self):
        class NoVersion(Migration):
            def up(self, ctx): pass
            def down(self, ctx): pass
        m = NoVersion()
        assert m.version  # auto-generated

    def test_up(self):
        m = V1_CreateUsers()
        ctx = {}
        m.up(ctx)
        assert ctx["users"] == []

    def test_down(self):
        m = V1_CreateUsers()
        ctx = {}
        m.up(ctx)
        m.down(ctx)
        assert "users" not in ctx

    def test_repr(self):
        m = V1_CreateUsers()
        r = repr(m)
        assert "20250101_000001" in r
        assert "Create users table" in r

    def test_sorting(self):
        m1 = V1_CreateUsers()
        m2 = V2_AddSettings()
        assert m1 < m2

    def test_equality(self):
        m1 = V1_CreateUsers()
        m2 = V1_CreateUsers()
        assert m1 == m2

    def test_hash(self):
        m = V1_CreateUsers()
        assert hash(m) == hash("20250101_000001")


# ---------------------------------------------------------------------------
# MigrationHistory
# ---------------------------------------------------------------------------

class TestMigrationHistory:
    """Tests for MigrationHistory."""

    def test_record_and_is_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            hist.record("v1", direction="up")
            assert hist.is_applied("v1") is True

    def test_is_applied_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            assert hist.is_applied("nonexistent") is False

    def test_rollback_marks_not_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            hist.record("v1", direction="up")
            hist.record("v1", direction="down")
            assert hist.is_applied("v1") is False

    def test_get_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            hist.record("v1", direction="up")
            hist.record("v2", direction="up")
            applied = hist.get_applied()
            assert "v1" in applied
            assert "v2" in applied

    def test_get_last(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            hist.record("v1", direction="up")
            hist.record("v2", direction="up")
            assert hist.get_last() == "v2"

    def test_get_last_no_migrations_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            with pytest.raises(ValueError, match="No migrations"):
                hist.get_last()

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist1 = MigrationHistory(path=path)
            hist1.record("v1", direction="up")
            # Load a new history from same path
            hist2 = MigrationHistory(path=path)
            assert hist2.is_applied("v1") is True

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            r = repr(hist)
            assert "MigrationHistory" in r


# ---------------------------------------------------------------------------
# MigrationManager
# ---------------------------------------------------------------------------

class TestMigrationManager:
    """Tests for MigrationManager."""

    def test_register(self):
        mgr = MigrationManager()
        mgr.register(V1_CreateUsers())
        status = mgr.get_status()
        assert status["total"] == 1

    def test_register_duplicate_raises(self):
        mgr = MigrationManager()
        mgr.register(V1_CreateUsers())
        with pytest.raises(ValueError, match="already registered"):
            mgr.register(V1_CreateUsers())

    def test_run_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            mgr = MigrationManager(history=hist)
            mgr.register(V1_CreateUsers())
            mgr.register(V2_AddSettings())
            applied = mgr.run_pending()
            assert len(applied) == 2

    def test_run_pending_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            mgr = MigrationManager(history=hist)
            mgr.register(V1_CreateUsers())
            mgr.run_pending()
            applied = mgr.run_pending()
            assert len(applied) == 0

    def test_rollback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            ctx = {}
            mgr = MigrationManager(history=hist, context=ctx)
            mgr.register(V1_CreateUsers())
            mgr.register(V2_AddSettings())
            mgr.run_pending()
            assert "users" in ctx
            rolled_back = mgr.rollback(steps=1)
            assert len(rolled_back) == 1
            assert "settings" not in ctx  # V2 was rolled back

    def test_rollback_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            ctx = {}
            mgr = MigrationManager(history=hist, context=ctx)
            mgr.register(V1_CreateUsers())
            mgr.register(V2_AddSettings())
            mgr.run_pending()
            rolled_back = mgr.rollback(steps=2)
            assert len(rolled_back) == 2
            assert "users" not in ctx
            assert "settings" not in ctx

    def test_rollback_too_many_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            mgr = MigrationManager(history=hist)
            mgr.register(V1_CreateUsers())
            mgr.run_pending()
            with pytest.raises(ValueError, match="Cannot roll back"):
                mgr.rollback(steps=5)

    def test_rollback_invalid_steps(self):
        mgr = MigrationManager()
        with pytest.raises(ValueError, match="positive"):
            mgr.rollback(steps=0)

    def test_failed_migration_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            mgr = MigrationManager(history=hist)
            mgr.register(FailingMigration())
            with pytest.raises(RuntimeError, match="failed"):
                mgr.run_pending()

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            mgr = MigrationManager(history=hist)
            mgr.register(V1_CreateUsers())
            mgr.register(V2_AddSettings())
            mgr.run_pending()
            status = mgr.get_status()
            assert status["total"] == 2
            assert status["applied"] == 2
            assert status["pending"] == 0

    def test_get_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            hist = MigrationHistory(path=path)
            mgr = MigrationManager(history=hist)
            mgr.register(V1_CreateUsers())
            mgr.register(V2_AddSettings())
            mgr.run_pending()
            mgr.register(V3_AddRoles())
            pending = mgr.get_pending()
            assert pending == ["20250101_000003"]

    def test_repr(self):
        mgr = MigrationManager()
        r = repr(mgr)
        assert "MigrationManager" in r
