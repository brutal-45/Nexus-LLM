"""Tests for the inference backend: sampling, logits processing, generation
configuration, KV caching, streaming decode and server-side metrics.
"""

from __future__ import annotations

import math
import threading

import pytest
import torch

from nexus_llm.backend.cache import PagedKVCache, compute_optimal_num_blocks, estimate_cache_size
from nexus_llm.backend.generation import GenerationConfig, GenerationPresets
from nexus_llm.backend.logits_process import (
    FrequencyPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsProcessor,
    TopKLogitsProcessor,
    TopPLogitsProcessor,
)
from nexus_llm.backend.metrics import BackendMetrics, Counter, Gauge, Histogram, MetricsRegistry
from nexus_llm.backend.sampling import (
    CombinedSampling,
    GreedySampling,
    MultinomialSampling,
    SamplingConfig,
    TopKSampling,
    TopPSampling,
    create_sampler,
)
from nexus_llm.backend.streamer import (
    CallbackStreamer,
    TextIteratorStreamer,
    create_streamer,
)


class FakeTokenizer:
    """Minimal tokenizer stand-in; tokens are already integer ids."""

    pad_token = "<pad>"
    eos_token = "</s>"

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        ids = [int(t) for t in torch.as_tensor(token_ids).flatten().tolist()]
        return " ".join(f"t{i}" for i in ids)

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        return [self.decode(s) for s in sequences]


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------


class TestSampling:
    def test_greedy_picks_argmax(self):
        logits = torch.tensor([[0.1, 0.2, 9.0, 0.3]])
        assert int(GreedySampling().sample(logits)[0]) == 2

    def test_top_k_keeps_only_k_candidates(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.05]])
        sampler = TopKSampling(SamplingConfig(top_k=2))
        picked = {int(sampler.sample(logits.clone())) for _ in range(40)}
        assert picked <= {2, 1}
        assert 4 not in picked and 0 not in picked

    def test_top_p_drops_low_probability_tail(self):
        logits = torch.tensor([[0.01, 0.02, 0.03, 9.0]])
        sampler = TopPSampling(SamplingConfig(top_p=0.5))
        picked = {int(sampler.sample(logits.clone())) for _ in range(30)}
        assert picked == {3}

    def test_temperature_zero_behaves_greedily(self):
        logits = torch.tensor([[0.5, 0.6, 5.0]])
        sampler = MultinomialSampling(SamplingConfig(temperature=1e-6))
        assert all(int(sampler.sample(logits.clone())) == 2 for _ in range(10))

    def test_sampling_is_reproducible_with_seed(self):
        logits = torch.randn(1, 50, generator=torch.Generator().manual_seed(0))
        cfg = SamplingConfig(temperature=1.5, top_k=10, seed=1234)
        a = [int(create_sampler(cfg).sample(logits.clone())) for _ in range(5)]
        b = [int(create_sampler(cfg).sample(logits.clone())) for _ in range(5)]
        assert a == b

    def test_create_sampler_picks_greedy_for_temperature_zero(self):
        assert isinstance(create_sampler(SamplingConfig(temperature=0.0)), GreedySampling)

    def test_combined_sampling_respects_top_k(self):
        logits = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0]])
        sampler = CombinedSampling(SamplingConfig(top_k=2, top_p=0.99, temperature=1.0))
        picked = {int(sampler.sample(logits.clone())) for _ in range(30)}
        assert picked <= {0, 1}

    def test_empty_logits_row_is_handled(self):
        sampler = GreedySampling()
        out = sampler.sample(torch.zeros(1, 1))
        assert int(out[0]) == 0


# ---------------------------------------------------------------------------
# Logits processors
# ---------------------------------------------------------------------------


class TestLogitsProcessors:
    def test_repetition_penalty_reduces_seen_tokens(self):
        proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
        ids = torch.tensor([[0, 2]])
        scores = torch.tensor([[2.0, 1.0, 3.0, 0.5]])
        out = proc(ids, scores.clone())
        assert out[0, 0] < scores[0, 0]
        assert out[0, 2] < scores[0, 2]
        assert math.isclose(out[0, 1].item(), 1.0)  # unseen token untouched

    def test_frequency_penalty_scales_with_count(self):
        proc = FrequencyPenaltyLogitsProcessor(penalty=0.5)
        scores = torch.tensor([[1.0, 1.0]])
        out = proc(torch.tensor([[0, 0, 0]]), scores.clone())
        assert out[0, 0] < out[0, 1]

    def test_temperature_sharpens_distribution(self):
        scores = torch.tensor([[0.2, 1.1, 2.4, 0.7]])
        cooled = TemperatureLogitsProcessor(temperature=0.5)(torch.tensor([[0]]), scores.clone())
        heated = TemperatureLogitsProcessor(temperature=2.0)(torch.tensor([[0]]), scores.clone())
        assert torch.softmax(cooled, -1).max() > torch.softmax(heated, -1).max()

    def test_temperature_one_is_identity(self):
        scores = torch.tensor([[0.3, 1.2, -0.4]])
        out = TemperatureLogitsProcessor(1.0)(torch.tensor([[0]]), scores.clone())
        assert torch.allclose(out, scores)

    def test_top_k_processor_masks_low_scores(self):
        scores = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
        out = TopKLogitsProcessor(top_k=2)(torch.tensor([[0]]), scores.clone())
        kept = (out > float("-inf")).sum().item()
        assert kept == 2
        assert out[0, 1] == 5.0

    def test_top_p_processor_keeps_nucleus(self):
        scores = torch.tensor([[0.1, 0.2, 6.0, 5.5]])
        out = TopPLogitsProcessor(top_p=0.5)(torch.tensor([[0]]), scores.clone())
        assert out[0, 2] > float("-inf") and out[0, 0] == float("-inf")

    def test_min_length_blocks_eos(self):
        eos_id = 9
        proc = MinLengthLogitsProcessor(min_length=5, eos_token_id=eos_id)
        scores = torch.zeros(1, 12)
        out = proc(torch.tensor([[1, 2]]), scores.clone())
        assert out[0, eos_id] == float("-inf")

    def test_min_length_allows_eos_after_limit(self):
        eos_id = 9
        proc = MinLengthLogitsProcessor(min_length=2, eos_token_id=eos_id)
        out = proc(torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 12))
        assert out[0, eos_id] == 0.0

    def test_reset_is_safe_and_results_are_deterministic(self):
        proc = FrequencyPenaltyLogitsProcessor(penalty=0.5)
        first = proc(torch.tensor([[1, 1, 1]]), torch.ones(1, 4))
        proc.reset()
        second = proc(torch.tensor([[1, 1, 1]]), torch.ones(1, 4))
        # penalties derive from the ids passed in, so they are stable per call
        assert torch.allclose(first, second)
        assert first[0, 1] < 1.0


# ---------------------------------------------------------------------------
# Generation config & presets
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    def test_round_trip_to_dict(self):
        cfg = GenerationConfig(max_new_tokens=64, temperature=0.3, top_p=0.8)
        restored = GenerationConfig(**cfg.to_dict())
        assert restored.max_new_tokens == 64
        assert restored.temperature == 0.3

    def test_to_json_is_parseable(self):
        payload = GenerationConfig(max_new_tokens=8).to_json()
        assert '"max_new_tokens": 8' in payload

    def test_validate_flags_bad_temperature(self):
        problems = GenerationConfig(temperature=-1.0).validate()
        assert problems and any("temperature" in p.lower() for p in problems)

    def test_validate_accepts_sane_config(self):
        assert GenerationConfig(temperature=0.7, top_p=0.9).validate() == []

    def test_merge_overlays_non_default_values(self):
        base = GenerationConfig(temperature=1.0, max_new_tokens=100)
        override = GenerationConfig(temperature=0.2)
        merged = base.merge(override)
        assert merged.temperature == 0.2
        assert merged.max_new_tokens == 100

    def test_presets_exist_and_validate(self):
        names = GenerationPresets.list_presets()
        assert len(names) >= 4
        for name in names:
            preset = GenerationPresets.get_preset(name)
            assert isinstance(preset, GenerationConfig)
            assert preset.validate() == []

    def test_greedy_preset_disables_sampling(self):
        assert GenerationPresets.greedy().do_sample is False

    def test_beam_preset_uses_beams(self):
        assert GenerationPresets.beam_search().num_beams > 1

    def test_unknown_preset_falls_back_to_balanced(self):
        # Documented behaviour: an unknown name yields the balanced preset, so a
        # typo in a config file is non-fatal.
        fallback = GenerationPresets.get_preset("definitely-not-a-preset")
        assert fallback.to_dict() == GenerationPresets.balanced().to_dict()


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class TestPagedKVCache:
    @staticmethod
    def _cache(**kwargs):
        defaults = dict(num_blocks=4, block_size=2, num_layers=1, num_heads=1, head_dim=4)
        defaults.update(kwargs)
        return PagedKVCache(**defaults)

    def test_starts_empty(self):
        cache = self._cache()
        assert cache.num_used_blocks == 0
        assert cache.num_free_blocks == 4
        assert cache.utilization == 0.0

    def test_allocate_blocks_consumes_free_blocks(self):
        cache = self._cache()
        allocated = cache.allocate_blocks("seq-a", num_tokens=3)
        assert allocated
        assert cache.num_used_blocks == len(allocated)
        assert cache.num_free_blocks == 4 - len(allocated)

    def test_free_returns_blocks_to_the_pool(self):
        cache = self._cache()
        cache.allocate_blocks("seq-a", num_tokens=4)
        used_before = cache.num_used_blocks
        assert cache.free_blocks("seq-a") == used_before
        assert cache.num_used_blocks == 0

    def test_can_allocate_reflects_capacity(self):
        cache = self._cache(num_blocks=1, block_size=2)
        assert cache.can_allocate(2) is True
        assert cache.can_allocate(500) is False

    def test_eviction_when_pool_is_exhausted(self):
        cache = self._cache(num_blocks=2, block_size=2)
        cache.allocate_blocks("seq-a", num_tokens=4)
        assert cache.allocate_blocks("seq-b", num_tokens=4)

    def test_clear_resets_everything(self):
        cache = self._cache()
        cache.allocate_blocks("seq-a", num_tokens=6)
        cache.clear()
        assert cache.num_used_blocks == 0
        assert cache.num_free_blocks == 4

    def test_memory_usage_is_positive(self):
        assert self._cache().get_total_memory_mb() > 0

    def test_estimate_cache_size_scales_with_sequence_length(self):
        small = estimate_cache_size(max_seq_len=32, num_layers=2, num_heads=2, head_dim=8)
        big = estimate_cache_size(max_seq_len=64, num_layers=2, num_heads=2, head_dim=8)
        assert big == 2 * small

    def test_estimate_cache_size_scales_with_batch(self):
        one = estimate_cache_size(max_seq_len=8, num_layers=2, num_heads=1, head_dim=4, batch_size=1)
        four = estimate_cache_size(max_seq_len=8, num_layers=2, num_heads=1, head_dim=4, batch_size=4)
        assert four == 4 * one

    def test_optimal_blocks_divides_available_memory(self):
        per_block = 2 * 4 * 1 * 8 * 2 * 2  # k+v, block_size, heads, dim... see impl
        assert compute_optimal_num_blocks(
            available_memory_bytes=per_block * 5,
            block_size=4,
            num_layers=2,
            num_heads=1,
            head_dim=8,
        ) == 5

    def test_optimal_blocks_with_zero_memory(self):
        assert compute_optimal_num_blocks(
            available_memory_bytes=0, block_size=4, num_layers=1, num_heads=1, head_dim=2
        ) == 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamers:
    @staticmethod
    def _drain(streamer):
        pieces: list[str] = []
        thread = threading.Thread(target=lambda: pieces.extend(streamer))
        thread.start()
        return pieces, thread

    def test_text_iterator_yields_decoded_text(self):
        streamer = TextIteratorStreamer(tokenizer=FakeTokenizer(), skip_prompt=True)
        pieces, thread = self._drain(streamer)
        streamer.put(torch.tensor([[1, 2, 3, 4]]))  # prompt -> skipped
        for token in (5, 6, 7, 8):
            streamer.put(torch.tensor([[token]]))
        streamer.end()
        thread.join(timeout=10)
        assert pieces, "streamer yielded nothing"
        assert "".join(pieces).split() == ["t5", "t6", "t7", "t8"]
        assert streamer.generated_tokens == 4

    def test_text_iterator_accepts_full_growing_sequence(self):
        """Callers that re-send the whole sequence each step also work."""
        streamer = TextIteratorStreamer(tokenizer=FakeTokenizer(), skip_prompt=True)
        pieces, thread = self._drain(streamer)
        prompt = [0, 0, 0]
        streamer.put(torch.tensor([prompt]))  # the prompt itself
        for n in range(1, 4):
            streamer.put(torch.tensor([prompt + list(range(10, 10 + n))]))
        streamer.end()
        thread.join(timeout=10)
        assert "".join(pieces).split() == ["t10", "t11", "t12"]

    def test_skip_prompt_false_emits_first_token(self):
        streamer = TextIteratorStreamer(tokenizer=FakeTokenizer(), skip_prompt=False)
        pieces, thread = self._drain(streamer)
        streamer.put(torch.tensor([[9]]))
        streamer.end()
        thread.join(timeout=10)
        assert "t9" in "".join(pieces)

    def test_streamer_is_reusable_after_end(self):
        """A streamer can drive a second generation without leaking state."""
        streamer = TextIteratorStreamer(tokenizer=FakeTokenizer(), skip_prompt=True)

        first, t1 = self._drain(streamer)
        streamer.put(torch.tensor([[1, 2]]))  # prompt
        streamer.put(torch.tensor([[3]]))
        streamer.end()
        t1.join(timeout=10)
        assert "".join(first).strip() == "t3"

        second, t2 = self._drain(streamer)
        streamer.put(torch.tensor([[4, 5]]))  # a fresh prompt
        streamer.put(torch.tensor([[77]]))
        streamer.end()
        t2.join(timeout=10)
        assert "".join(second).strip() == "t77"

    def test_callback_streamer_invokes_callback(self):
        seen: list[str] = []
        streamer = CallbackStreamer(tokenizer=FakeTokenizer(), callback=seen.append)
        streamer.put(torch.tensor([[1, 2, 3, 4, 5]]))  # prompt
        streamer.put(torch.tensor([[6]]))
        streamer.put(torch.tensor([[7]]))
        streamer.end()
        assert "".join(seen).split() == ["t6", "t7"]
        assert streamer.is_finished is True
        assert streamer.generated_tokens == 2

    def test_callback_streamer_reports_errors_to_handler(self):
        def boom(text):
            raise RuntimeError("sink down")

        errors: list[Exception] = []
        streamer = CallbackStreamer(
            tokenizer=FakeTokenizer(), callback=boom, on_error=errors.append, skip_prompt=False
        )
        streamer.put(torch.tensor([[5, 6]]))
        streamer.end()  # flush pushes the held-back text to the failing sink
        assert errors and isinstance(errors[0], RuntimeError)

    def test_create_streamer_factory(self):
        assert isinstance(create_streamer("iterator", tokenizer=FakeTokenizer()), TextIteratorStreamer)
        assert isinstance(
            create_streamer("callback", tokenizer=FakeTokenizer(), callback=lambda s: None),
            CallbackStreamer,
        )
        with pytest.raises(ValueError):
            create_streamer("callback", tokenizer=FakeTokenizer())
        with pytest.raises(ValueError):
            create_streamer("nonsense", tokenizer=FakeTokenizer())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_counter_is_monotonic(self):
        counter = Counter("nexus_requests_total")
        counter.inc()
        counter.inc(2.0)
        assert counter.get() == 3.0

    def test_counter_labels_are_separate_series(self):
        counter = Counter("nexus_requests_total", label_names=["model"])
        counter.inc(labels={"model": "gpt2"})
        counter.inc(5.0, labels={"model": "llama"})
        assert counter.get({"model": "gpt2"}) == 1.0
        assert counter.get({"model": "llama"}) == 5.0

    def test_gauge_set_and_dec(self):
        gauge = Gauge("nexus_gpu_mem")
        gauge.set(100.0)
        gauge.dec(30.0)
        assert gauge.get() == 70.0

    def test_histogram_stats_and_percentile(self):
        hist = Histogram("nexus_latency", buckets=(1.0, 5.0, 10.0))
        hist.observe_many([0.5, 2.0, 4.0, 9.0])
        stats = hist.get_stats()
        assert stats["count"] == 4
        assert stats["sum"] == pytest.approx(15.5)
        assert hist.get_percentile(0.5) > 0

    def test_prometheus_output_format(self):
        registry = MetricsRegistry()
        registry.counter("nexus_total").inc(3)
        body = registry.to_prometheus()
        assert "# TYPE nexus_total counter" in body
        assert "nexus_total 3.0" in body

    def test_record_request_updates_registry(self):
        metrics = BackendMetrics()
        metrics.record_request(model="gpt2", latency=0.25, tokens=12, status="success")
        payload = metrics.get_metrics()
        assert payload
        text = metrics.get_prometheus()
        assert "gpt2" in text or "nexus" in text
