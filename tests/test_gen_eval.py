"""Test generation quality for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class GenerationSample:
    prompt: str
    generated: str
    reference: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def repetition_rate(text: str, n: int = 3) -> float:
    if len(text.split()) < n:
        return 0.0
    words = text.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - unique / len(ngrams)


def distinct_ngrams(text: str, n: int = 2) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def average_length(generations: List[str]) -> float:
    if not generations:
        return 0.0
    return sum(len(g.split()) for g in generations) / len(generations)


def length_variance(generations: List[str]) -> float:
    if len(generations) < 2:
        return 0.0
    lengths = [len(g.split()) for g in generations]
    avg = sum(lengths) / len(lengths)
    return sum((l - avg) ** 2 for l in lengths) / len(lengths)


def fluency_score(text: str) -> float:
    sentences = [s for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0
    total_words = len(text.split())
    avg_sentence_len = total_words / len(sentences) if sentences else 0
    if avg_sentence_len < 3 or avg_sentence_len > 40:
        return 0.3
    if avg_sentence_len < 5 or avg_sentence_len > 30:
        return 0.6
    return 0.9


def diversity_score(generations: List[str]) -> float:
    if not generations:
        return 0.0
    unique = len(set(generations))
    return unique / len(generations)


class GenerationEvaluator:
    def __init__(self):
        self._samples: List[GenerationSample] = []

    def add_sample(self, sample: GenerationSample):
        self._samples.append(sample)

    def evaluate(self) -> Dict[str, float]:
        if not self._samples:
            return {}
        generations = [s.generated for s in self._samples]
        return {
            "repetition_rate": sum(repetition_rate(g) for g in generations) / len(generations),
            "distinct_2grams": sum(distinct_ngrams(g, 2) for g in generations) / len(generations),
            "avg_length": average_length(generations),
            "diversity": diversity_score(generations),
            "avg_fluency": sum(fluency_score(g) for g in generations) / len(generations),
        }

    def get_samples(self) -> List[GenerationSample]:
        return list(self._samples)

    def clear(self):
        self._samples.clear()


class TestRepetitionRate:
    def test_no_repetition(self):
        text = "the cat sat on the mat and ran"
        rate = repetition_rate(text, n=3)
        assert rate >= 0.0

    def test_high_repetition(self):
        text = "hello hello hello hello hello"
        rate = repetition_rate(text, n=2)
        assert rate > 0.0

    def test_short_text(self):
        assert repetition_rate("hi", n=3) == 0.0


class TestDistinctNgrams:
    def test_all_distinct(self):
        text = "the cat sat on a mat"
        score = distinct_ngrams(text, 2)
        assert score == 1.0

    def test_some_repeat(self):
        text = "the cat the cat"
        score = distinct_ngrams(text, 2)
        assert score < 1.0

    def test_short_text(self):
        assert distinct_ngrams("hi", 2) == 0.0


class TestAverageLength:
    def test_calculation(self):
        gens = ["hello world", "foo bar baz"]
        assert average_length(gens) == 2.5

    def test_empty(self):
        assert average_length([]) == 0.0


class TestLengthVariance:
    def test_zero_variance(self):
        gens = ["a b c", "d e f"]
        assert length_variance(gens) == 0.0

    def test_some_variance(self):
        gens = ["short", "a longer sentence here"]
        assert length_variance(gens) > 0

    def test_single(self):
        assert length_variance(["hello"]) == 0.0


class TestFluencyScore:
    def test_good_text(self):
        score = fluency_score("This is a well-formed sentence. Another good sentence follows.")
        assert score >= 0.5

    def test_terse_text(self):
        score = fluency_score("Hi. Ok.")
        assert score < 0.9

    def test_empty(self):
        assert fluency_score("") == 0.0


class TestDiversityScore:
    def test_all_unique(self):
        gens = ["a", "b", "c"]
        assert diversity_score(gens) == 1.0

    def test_all_same(self):
        gens = ["same", "same", "same"]
        assert diversity_score(gens) == pytest.approx(1/3)

    def test_empty(self):
        assert diversity_score([]) == 0.0


class TestGenerationEvaluator:
    def test_evaluate(self):
        evaluator = GenerationEvaluator()
        evaluator.add_sample(GenerationSample(prompt="test", generated="Hello world this is a test"))
        evaluator.add_sample(GenerationSample(prompt="test", generated="Another generated output here"))
        results = evaluator.evaluate()
        assert "repetition_rate" in results
        assert "distinct_2grams" in results
        assert "avg_length" in results

    def test_evaluate_empty(self):
        evaluator = GenerationEvaluator()
        assert evaluator.evaluate() == {}

    def test_get_samples(self):
        evaluator = GenerationEvaluator()
        evaluator.add_sample(GenerationSample(prompt="p", generated="g"))
        assert len(evaluator.get_samples()) == 1

    def test_clear(self):
        evaluator = GenerationEvaluator()
        evaluator.add_sample(GenerationSample(prompt="p", generated="g"))
        evaluator.clear()
        assert len(evaluator.get_samples()) == 0
