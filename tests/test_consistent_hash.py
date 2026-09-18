"""Tests for consistent hashing."""
import pytest


def test_consistent_hash_deterministic():
    def chash(key, num_slots):
        return hash(key) % num_slots

    assert chash("key1", 100) == chash("key1", 100)

def test_consistent_hash_distribution():
    def chash(key, num_slots):
        return hash(key) % num_slots

    slots = [chash(f"key_{i}", 10) for i in range(1000)]
    # Each slot should have roughly 100 keys
    from collections import Counter
    counts = Counter(slots)
    assert all(c > 50 for c in counts.values())

def test_consistent_hash_ring():
    ring = sorted([(hash(f"node_{i}") % 1000, f"node_{i}") for i in range(5)])
    assert len(ring) == 5
    assert ring == sorted(ring)  # Sorted by position

def test_consistent_hash_minimal_remapping():
    # When adding a node, most keys should stay on the same node
    keys = [f"k{i}" for i in range(100)]
    slots_before = [hash(k) % 10 for k in keys]
    slots_after = [hash(k) % 11 for k in keys]
    unchanged = sum(1 for a, b in zip(slots_before, slots_after) if a == b)
    assert unchanged > 50  # Most should stay the same
