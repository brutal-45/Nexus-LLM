"""Tests for curriculum learning."""
import pytest


class CurriculumScheduler:
    """Simple curriculum learning scheduler for testing."""
    def __init__(self, stages):
        self.stages = stages  # List of (difficulty, num_steps)
        self.current_stage = 0
        self.current_step = 0

    @property
    def current_difficulty(self):
        if self.current_stage >= len(self.stages):
            return self.stages[-1][0]
        return self.stages[self.current_stage][0]

    def step(self):
        self.current_step += 1
        if self.current_stage < len(self.stages):
            _, stage_steps = self.stages[self.current_stage]
            if self.current_step >= stage_steps:
                self.current_stage += 1
                self.current_step = 0


@pytest.fixture
def curriculum():
    return CurriculumScheduler([
        (0.2, 100),  # Easy
        (0.5, 200),  # Medium
        (1.0, 300),  # Hard
    ])


def test_curriculum_starts_easy(curriculum):
    """Test that curriculum starts with easy difficulty."""
    assert curriculum.current_difficulty == 0.2


def test_curriculum_progression(curriculum):
    """Test curriculum difficulty progression."""
    for _ in range(100):
        curriculum.step()
    assert curriculum.current_difficulty == 0.5


def test_curriculum_final_difficulty(curriculum):
    """Test curriculum reaches final difficulty."""
    for _ in range(600):
        curriculum.step()
    assert curriculum.current_difficulty == 1.0


def test_curriculum_stays_at_max(curriculum):
    """Test that curriculum stays at max difficulty."""
    for _ in range(1000):
        curriculum.step()
    assert curriculum.current_difficulty == 1.0
