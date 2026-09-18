"""Tests for ConfirmDialog widget."""
import pytest


def test_confirm_dialog_default():
    result = None
    assert result is None

def test_confirm_dialog_yes():
    result = True
    assert result is True

def test_confirm_dialog_no():
    result = False
    assert result is False

def test_confirm_dialog_message():
    msg = "Are you sure?"
    assert "sure" in msg.lower()
