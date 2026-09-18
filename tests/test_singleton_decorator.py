"""Tests for singleton decorator."""
import pytest
import functools


def test_singleton_creates_one_instance():
    instances = {}

    def singleton(cls):
        @functools.wraps(cls)
        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        return get_instance

    @singleton
    class Config:
        def __init__(self, value="default"):
            self.value = value

    c1 = Config("first")
    c2 = Config("second")
    assert c1 is c2
    assert c1.value == "first"

def test_singleton_identity():
    class Singleton:
        _instance = None
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    a = Singleton()
    b = Singleton()
    assert a is b
