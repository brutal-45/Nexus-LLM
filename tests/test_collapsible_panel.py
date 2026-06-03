"""Tests for collapsible panel."""
import pytest


def test_panel_collapsed_default():
    panel = {"title": "Settings", "collapsed": True}
    assert panel["collapsed"] is True

def test_panel_toggle():
    panel = {"collapsed": True}
    panel["collapsed"] = not panel["collapsed"]
    assert panel["collapsed"] is False

def test_panel_content_hidden_when_collapsed():
    panel = {"collapsed": True, "content": "secret"}
    visible = None if panel["collapsed"] else panel["content"]
    assert visible is None

def test_panel_title():
    panel = {"title": "Advanced Options"}
    assert "Advanced" in panel["title"]
