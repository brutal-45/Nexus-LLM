"""Tests for the training stack: collation, optimizers, schedulers, losses,
checkpointing and dataset loading.

A tiny randomly-initialised model is used throughout so these run offline in a
couple of seconds while still exercising real torch code paths.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from nexus_llm.training.checkpoint import CheckpointManager  # noqa: E402
from nexus_llm.training.collator import DataCollator, DynamicBatchCollator  # noqa: E402
from nexus_llm.training.dataset import DataFormat, DatasetLoader  # noqa: E402
from nexus_llm.training.loss import LabelSmoothingCrossEntropy  # noqa: E402
from nexus_llm.training.optimizer import (  # noqa: E402
    OptimizerConfig,
    build_optimizer,
    format_parameter_count,
    get_optimizer_lr,
    get_parameter_count,
    scale_optimizer_lr,
    set_optimizer_lr,
)
from nexus_llm.training.scheduler import (  # noqa: E402
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


@pytest.fixture
def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def make_features(lengths):
    return [{"input_ids": list(range(n)), "labels": list(range(n))} for n in lengths]


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


class TestDataCollator:
    def test_pads_to_longest(self):
        collator = DataCollator(padding=True, return_tensors="pt")
        batch = collator(make_features([3, 5]))
        assert batch["input_ids"].shape == (2, 5)

    def test_padding_positions_are_masked(self):
        collator = DataCollator(padding=True, return_tensors="pt")
        batch = collator(make_features([2, 4]))
        assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]
        assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1]

    def test_labels_padded_with_ignore_index(self):
        collator = DataCollator(padding=True, return_tensors="pt", label_pad_token_id=-100)
        batch = collator(make_features([2, 4]))
        assert batch["labels"][0, -2:].tolist() == [-100, -100]

    def test_max_length_truncation(self):
        collator = DataCollator(padding=True, truncation=True, max_length=3, return_tensors="pt")
        batch = collator(make_features([10]))
        assert batch["input_ids"].shape[-1] <= 3

    def test_pad_to_multiple_of(self):
        collator = DataCollator(padding=True, pad_to_multiple_of=4, return_tensors="pt")
        batch = collator(make_features([5]))
        assert batch["input_ids"].shape[-1] % 4 == 0

    def test_no_padding_keeps_ragged_when_single(self):
        collator = DataCollator(padding=False, return_tensors="pt")
        batch = collator(make_features([3]))
        assert batch["input_ids"].shape == (1, 3)

    def test_dynamic_batching_groups_by_token_budget(self):
        collator = DynamicBatchCollator(max_tokens_per_batch=12)
        groups = collator.group_into_batches(make_features([4, 4, 4, 4, 4]))
        assert len(groups) >= 2
        assert all(sum(len(f["input_ids"]) for f in g) <= 12 or len(g) == 1 for g in groups)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class TestOptimizers:
    def test_build_adamw(self, tiny_model):
        opt = build_optimizer(tiny_model, OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3))
        assert isinstance(opt, torch.optim.AdamW)
        assert get_optimizer_lr(opt) == pytest.approx(1e-3)

    def test_build_sgd(self, tiny_model):
        opt = build_optimizer(tiny_model, OptimizerConfig(optimizer_type="sgd", learning_rate=0.1))
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_optimizer_type_raises(self, tiny_model):
        with pytest.raises(ValueError):
            build_optimizer(tiny_model, OptimizerConfig(optimizer_type="not-an-optimizer"))

    def test_set_and_scale_learning_rate(self, tiny_model):
        opt = build_optimizer(tiny_model, OptimizerConfig(learning_rate=1e-3))
        set_optimizer_lr(opt, 5e-4)
        assert get_optimizer_lr(opt) == pytest.approx(5e-4)
        scale_optimizer_lr(opt, 2.0)
        assert get_optimizer_lr(opt) == pytest.approx(1e-3)

    def test_decay_groups_exclude_norm_and_bias(self, tiny_model):
        """LayerNorm weights and biases must not be weight-decayed."""
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        config = OptimizerConfig(separate_decay_groups=True, weight_decay=0.1)
        opt = build_optimizer(model, config)
        assert len(opt.param_groups) == 2
        decays = {g["weight_decay"] for g in opt.param_groups}
        assert decays == {0.0, 0.1}
        no_decay_params = [
            p for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]
        ]
        # LayerNorm.weight (1D) and the Linear bias (1D) land in the no-decay group.
        assert all(p.ndim == 1 for p in no_decay_params)

    def test_single_group_when_separation_disabled(self, tiny_model):
        config = OptimizerConfig(separate_decay_groups=False, weight_decay=0.1)
        opt = build_optimizer(tiny_model, config)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["weight_decay"] == 0.1

    def test_parameter_count_helpers(self, tiny_model):
        count = get_parameter_count(tiny_model)
        assert count == sum(p.numel() for p in tiny_model.parameters())
        assert format_parameter_count(1_234)
        assert "M" in format_parameter_count(2_500_000) or "2.5" in format_parameter_count(2_500_000)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


class TestSchedulers:
    @staticmethod
    def _opt(tiny_model):
        return build_optimizer(tiny_model, OptimizerConfig(learning_rate=1e-2))

    def _curve(self, scheduler, steps, optimizer=None):
        values = []
        for _ in range(steps):
            values.append(scheduler.get_last_lr()[0])
            if optimizer is not None:
                optimizer.step()  # torch wants this before scheduler.step()
            scheduler.step()
        return values

    def test_linear_warmup_then_decay(self, tiny_model):
        opt = self._opt(tiny_model)
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=6)
        curve = self._curve(sched, 6, opt)
        assert curve[0] < curve[1] < curve[2]  # warming up
        assert curve[-1] < curve[2]  # decaying afterwards

    def test_cosine_ends_near_zero(self, tiny_model):
        opt = self._opt(tiny_model)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=1, num_training_steps=8)
        curve = self._curve(sched, 8, opt)
        assert curve[-1] < curve[1]
        assert curve[-1] >= 0.0

    def test_constant_schedule_holds_after_warmup(self, tiny_model):
        opt = self._opt(tiny_model)
        sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=2)
        curve = self._curve(sched, 6, opt)
        assert len(set(curve[3:])) == 1

    def test_polynomial_decay_is_monotonic(self, tiny_model):
        opt = self._opt(tiny_model)
        sched = get_polynomial_decay_schedule_with_warmup(
            opt, num_warmup_steps=1, num_training_steps=6, power=1.0
        )
        curve = self._curve(sched, 6, opt)[2:]
        assert all(b <= a for a, b in zip(curve, curve[1:]))

    def test_warmup_only_step_zero_starts_low(self, tiny_model):
        opt = self._opt(tiny_model)
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=4, num_training_steps=10)
        assert sched.get_last_lr()[0] < 1e-2


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class TestLosses:
    def test_label_smoothing_matches_ce_when_smoothing_zero(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5)
        targets = torch.tensor([1, 3])
        loss = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, targets)
        reference = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.allclose(loss, reference, atol=1e-5)

    def test_smoothing_reduces_confidence_reward(self):
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        targets = torch.tensor([0])
        sharp = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, targets)
        smoothed = LabelSmoothingCrossEntropy(smoothing=0.2)(logits, targets)
        assert smoothed > sharp

    def test_ignore_index_skips_padded_positions(self):
        # (batch, seq, vocab) with three of four positions padded away.
        logits = torch.randn(1, 4, 6)
        targets = torch.tensor([[2, -100, -100, -100]])
        loss = LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=-100)(logits, targets)
        assert torch.isfinite(loss)
        # Only the single valid position contributes, so the mean is over one token.
        reference = LabelSmoothingCrossEntropy(smoothing=0.1)(
            logits[:, :1, :], torch.tensor([[2]])
        )
        assert torch.allclose(loss, reference, atol=1e-5)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_save_writes_a_numbered_checkpoint(self, tmp_dir):
        manager = CheckpointManager(output_dir=str(tmp_dir))
        path = manager.save({"model_state_dict": {"w": torch.zeros(2)}, "step": 1}, step=1)
        assert "checkpoint-1" in path
        assert any(p.name.startswith("checkpoint-1") for p in tmp_dir.iterdir())

    def test_round_trip_latest(self, tmp_dir):
        manager = CheckpointManager(output_dir=str(tmp_dir))
        state = {"model_state_dict": {"w": torch.ones(3)}, "epoch": 2}
        manager.save(state, step=5, metrics={"eval_loss": 0.5})
        loaded = manager.load_latest()
        assert loaded is not None
        assert loaded["metadata"]["epoch"] == 2
        assert loaded["metadata"]["step"] == 5
        assert torch.equal(loaded["model_state_dict"]["w"], torch.ones(3))

    def test_tracks_best_by_metric(self, tmp_dir):
        manager = CheckpointManager(output_dir=str(tmp_dir), metric_for_best_model="eval_loss")
        manager.save({"model_state_dict": {"w": torch.zeros(1)}}, step=1, metrics={"eval_loss": 0.9})
        manager.save({"model_state_dict": {"w": torch.ones(1)}}, step=2, metrics={"eval_loss": 0.1})
        assert manager.is_best({"eval_loss": 0.05}) is True
        assert manager.is_best({"eval_loss": 0.5}) is False
        best = manager.load_best()
        assert best is not None
        assert torch.equal(best["model_state_dict"]["w"], torch.ones(1))

    def test_prunes_old_checkpoints(self, tmp_dir):
        manager = CheckpointManager(output_dir=str(tmp_dir), save_total_limit=2)
        for step in range(1, 6):
            manager.save({"model_state_dict": {"w": torch.zeros(1)}}, step=step)
        assert manager.get_checkpoint_count() <= 2

    def test_list_checkpoints_sorted(self, tmp_dir):
        manager = CheckpointManager(output_dir=str(tmp_dir), save_total_limit=5)
        for step in (2, 1, 3):
            manager.save({"model_state_dict": {"w": torch.zeros(1)}}, step=step)
        steps = [entry["step"] for entry in manager.list_checkpoints()]
        assert steps == sorted(steps)

    def test_load_latest_when_empty(self, tmp_dir):
        manager = CheckpointManager(output_dir=str(tmp_dir / "missing"))
        assert manager.load_latest() is None
        assert manager.get_checkpoint_count() == 0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestDatasetLoader:
    def test_detects_jsonl(self, tmp_dir):
        path = tmp_dir / "train.jsonl"
        rows = [
            {"text": "the quick brown fox"},
            {"text": "jumps over the lazy dog"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        loader = DatasetLoader()
        train, val = loader.load(str(path), format="auto", val_split=0.5)
        assert len(train) + len(val) == 2
        # Plain-text rows normalise to a prompt/completion pair.
        assert loader.format == DataFormat.TEXT

    def test_alpaca_style_maps_to_text(self, tmp_dir):
        path = tmp_dir / "alpaca.jsonl"
        row = {"instruction": "Say hi", "input": "", "output": "hi"}
        path.write_text(json.dumps(row), encoding="utf-8")
        train, _ = DatasetLoader().load(str(path), format="auto", val_split=0.0)
        assert train
        assert any("hi" in str(value) for value in train[0].values())

    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises((FileNotFoundError, ValueError)):
            DatasetLoader().load(str(tmp_dir / "nope.jsonl"))

    def test_format_enum_members(self):
        names = {f.value for f in DataFormat}
        assert {"jsonl", "text"} <= names or len(names) >= 2
