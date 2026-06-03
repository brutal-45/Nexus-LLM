"""Tests for nexus_llm.security.audit module."""

import pytest
from nexus_llm.security.audit import SecurityAudit


class TestSecurityAudit:
    """Tests for the SecurityAudit class."""

    def test_init(self):
        audit = SecurityAudit()
        assert audit is not None

    def test_run_audit(self):
        audit = SecurityAudit()
        result = audit.run()
        assert isinstance(result, dict)
        assert "passed" in result or "status" in result

    def test_check_encryption(self):
        audit = SecurityAudit()
        result = audit.check_encryption()
        assert isinstance(result, dict)

    def test_check_permissions(self):
        audit = SecurityAudit()
        result = audit.check_permissions()
        assert isinstance(result, dict)

    def test_check_input_validation(self):
        audit = SecurityAudit()
        result = audit.check_input_validation()
        assert isinstance(result, dict)

    def test_get_report(self):
        audit = SecurityAudit()
        report = audit.get_report()
        assert isinstance(report, dict)
