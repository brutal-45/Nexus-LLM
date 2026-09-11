"""End-to-end inference test with a real (tiny, randomly initialised) model.

No weights are downloaded, so this runs offline and in CI: a 2-layer GPT-2 is
constructed from a config.  It exercises the full path the CLI depends on -
pipeline construction, tokenizer wrapping, generation and streaming - against
real torch tensors rather than mocks.
"""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nexus_llm.backend.generation import GenerationConfig  # noqa: E402
from nexus_llm.backend.pipeline import (  # noqa: E402
    PipelineConfig,
    PipelineType,
    TextGenerationPipeline,
    PipelineFactory,
)
from nexus_llm.backend.streamer import TextIteratorStreamer  # noqa: E402
from nexus_llm.backend.tokenizer_utils import TokenizerWrapper  # noqa: E402


class ToyTokenizer:
    """Callable tokenizer with the HF surface the wrapper relies on."""

    pad_token = None
    pad_token_id = None
    eos_token_id = 0
    chat_template = None

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        if isinstance(text, str):
            text = [text]
        rows = [[min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in t] for t in text]
        width = max(len(r) for r in rows)
        rows = [r + [0] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in [[min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in t] for t in text]]
        ids = torch.tensor(rows, dtype=torch.long)
        out = {"input_ids": ids, "attention_mask": torch.tensor(mask, dtype=torch.long)}
        if return_tensors is None:
            out["input_ids"] = [r for r in rows]
            out["attention_mask"] = [r for r in mask]
        return out

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        vals = [int(i) for i in (ids.tolist() if torch.is_tensor(ids) else list(ids))]
        return "".join(chr((v % self.vocab_size) + 32) if v else "" for v in vals)

    def batch_decode(self, seqs, skip_special_tokens=True, **kwargs):
        return [self.decode(s, skip_special_tokens) for s in seqs]




@pytest.fixture(scope="module")
def tiny_model():
    vocab = 96
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=vocab,
            n_positions=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=0,
            eos_token_id=0,
        )
    )
    model.eval()
    return model, ToyTokenizer(vocab)


@pytest.fixture
def pipeline(tiny_model):
    model, raw_tokenizer = tiny_model
    tokenizer = TokenizerWrapper(raw_tokenizer)
    config = PipelineConfig(
        pipeline_type=PipelineType.TEXT_GENERATION,
        generation_config=GenerationConfig(max_new_tokens=8, do_sample=False),
    )
    return model, tokenizer, PipelineFactory.create(model, tokenizer, PipelineType.TEXT_GENERATION, config), config




class TestEndToEndGeneration:
    def test_pipeline_generates_text(self, pipeline):
        model, tokenizer, pipe, config = pipeline
        out = pipe("hello world")
        assert isinstance(out, list) and out
        assert "generated_text" in out[0]

    def test_wrapper_encode_decode_roundtrip(self, pipeline):
        model, tokenizer, pipe, config = pipeline
        encoded = tokenizer.encode("abc", return_tensors="pt")
        assert tuple(encoded.shape) == (1, 3)
        assert isinstance(tokenizer.decode(encoded[0]), str)

    def test_batch_padding_produces_uniform_width(self, pipeline):
        model, tokenizer, pipe, config = pipeline
        batch = tokenizer.encode_with_padding(["hi", "hello there"], return_tensors="pt")
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape[0] == 2

    def test_real_model_streams_tokens(self, pipeline):
        """model.generate(streamer=...) must actually yield text."""
        model, tokenizer, pipe, config = pipeline
        streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True)
        inputs = tokenizer(["hello world"], return_tensors="pt")
        pieces: list[str] = []

        def run():
            model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                streamer=streamer,
            )

        thread = threading.Thread(target=run)
        thread.start()
        try:
            for piece in streamer:
                pieces.append(piece)
        finally:
            thread.join(timeout=120)

        assert pieces, "streaming produced no text"
        assert streamer.generated_tokens > 0

    def test_text_generation_pipeline_call(self, pipeline):
        model, tokenizer, pipe, config = pipeline
        gen = TextGenerationPipeline(model, tokenizer, config)
        out = gen("abc")
        assert isinstance(out, list) and "generated_text" in out[0]
