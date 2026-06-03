"""Speed benchmark for Nexus-LLM.

Measures inference latency, streaming throughput, model loading time,
and tokenization performance.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Type alias — a model-like object expected to have .generate() / .stream()
ModelLike = Any
TokenizerLike = Any


class SpeedBenchmark:
    """Benchmark suite focused on speed and throughput metrics.

    Example::

        sb = SpeedBenchmark()
        results = sb.benchmark_inference_speed(model, "Hello world", n_runs=10)
        print(results)
    """

    # ------------------------------------------------------------------
    # Inference speed
    # ------------------------------------------------------------------

    def benchmark_inference_speed(
        self,
        model: ModelLike,
        prompt: str,
        n_runs: int = 5,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Measure batch inference latency and throughput.

        Args:
            model: A model-like object with a ``generate(text, max_tokens)``
                   method.  If the object lacks this method a small sleep
                   is used as a placeholder.
            prompt: The input prompt to benchmark.
            n_runs: Number of inference iterations.
            max_tokens: Maximum tokens to generate per run.

        Returns:
            Dict with ``avg_time``, ``min_time``, ``max_time``,
            ``tokens_per_sec``, ``total_tokens``, and ``n_runs``.
        """
        latencies: List[float] = []
        total_tokens = 0

        generate_fn = getattr(model, "generate", None)

        for i in range(n_runs):
            start = time.perf_counter()
            if generate_fn is not None and callable(generate_fn):
                output = generate_fn(prompt, max_tokens=max_tokens)
                if isinstance(output, dict):
                    total_tokens += output.get("tokens_generated", max_tokens)
                elif isinstance(output, str):
                    total_tokens += len(output.split())
                else:
                    total_tokens += max_tokens
            else:
                # Placeholder when no real model is available
                time.sleep(0.005)
                total_tokens += max_tokens

            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            logger.debug("Run %d/%d — %.4fs", i + 1, n_runs, elapsed)

        avg_time = sum(latencies) / len(latencies)
        tokens_per_sec = total_tokens / sum(latencies) if sum(latencies) > 0 else 0.0

        result = {
            "avg_time": round(avg_time, 6),
            "min_time": round(min(latencies), 6),
            "max_time": round(max(latencies), 6),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_tokens": total_tokens,
            "n_runs": n_runs,
        }
        logger.info(
            "Inference speed: avg=%.4fs, tokens/sec=%.1f", avg_time, tokens_per_sec
        )
        return result

    # ------------------------------------------------------------------
    # Streaming speed
    # ------------------------------------------------------------------

    def benchmark_streaming_speed(
        self,
        model: ModelLike,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Measure streaming (token-by-token) generation throughput.

        Args:
            model: A model-like object with a ``stream(text, max_tokens)``
                   method yielding token chunks.
            prompt: The input prompt.
            max_tokens: Maximum tokens to stream.

        Returns:
            Dict with ``total_time``, ``tokens_per_sec``,
            ``first_token_latency``, ``total_tokens``.
        """
        stream_fn = getattr(model, "stream", None)

        first_token_latency: Optional[float] = None
        total_tokens = 0
        start = time.perf_counter()

        if stream_fn is not None and callable(stream_fn):
            for chunk in stream_fn(prompt, max_tokens=max_tokens):
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - start
                if isinstance(chunk, dict):
                    total_tokens += len(chunk.get("text", "").split())
                else:
                    total_tokens += 1
        else:
            # Placeholder
            time.sleep(0.01)
            total_tokens = max_tokens
            first_token_latency = 0.005

        total_time = time.perf_counter() - start
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

        result = {
            "total_time": round(total_time, 6),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "first_token_latency": round(first_token_latency or 0.0, 6),
            "total_tokens": total_tokens,
        }
        logger.info(
            "Streaming speed: %.4fs, %.1f tokens/sec, first=%.4fs",
            total_time,
            tokens_per_sec,
            first_token_latency or 0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def benchmark_model_loading(
        self,
        model_name: str,
        load_fn: Optional[Callable[..., ModelLike]] = None,
    ) -> Dict[str, Any]:
        """Measure model loading time and memory usage.

        Args:
            model_name: Name or path of the model to load.
            load_fn: Optional callable that accepts *model_name* and returns
                     a loaded model.  If ``None`` a placeholder is used.

        Returns:
            Dict with ``load_time``, ``memory_used_mb``, and ``model_name``.
        """
        import sys

        mem_before = sys.getsizeof(0)  # baseline
        start = time.perf_counter()

        if load_fn is not None and callable(load_fn):
            model = load_fn(model_name)
            mem_after = sys.getsizeof(model)
            memory_used = max(0, mem_after - mem_before) / (1024 * 1024)
        else:
            # Placeholder loading simulation
            time.sleep(0.05)
            _ = {"placeholder": model_name}  # noqa: F841
            memory_used = 0.0

        load_time = time.perf_counter() - start

        result = {
            "load_time": round(load_time, 6),
            "memory_used_mb": round(memory_used, 4),
            "model_name": model_name,
        }
        logger.info(
            "Model loading: %.4fs, %.2f MB for %s",
            load_time,
            memory_used,
            model_name,
        )
        return result

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def benchmark_tokenization(
        self,
        tokenizer: TokenizerLike,
        texts: List[str],
    ) -> Dict[str, Any]:
        """Measure tokenization throughput.

        Args:
            tokenizer: A tokenizer-like object with an ``encode(text)``
                       or ``tokenize(text)`` method.
            texts: List of strings to tokenize.

        Returns:
            Dict with ``avg_time``, ``total_time``, ``total_tokens``,
            ``texts_per_sec``, and ``tokens_per_sec``.
        """
        encode_fn = getattr(tokenizer, "encode", None) or getattr(
            tokenizer, "tokenize", None
        )

        total_tokens = 0
        latencies: List[float] = []
        start = time.perf_counter()

        for text in texts:
            t0 = time.perf_counter()
            if encode_fn is not None and callable(encode_fn):
                tokens = encode_fn(text)
                total_tokens += len(tokens) if hasattr(tokens, "__len__") else 0
            else:
                # Fallback: whitespace tokenization
                total_tokens += len(text.split())
            latencies.append(time.perf_counter() - t0)

        total_time = time.perf_counter() - start
        avg_time = sum(latencies) / len(latencies) if latencies else 0.0

        result = {
            "avg_time": round(avg_time, 6),
            "total_time": round(total_time, 6),
            "total_tokens": total_tokens,
            "texts_per_sec": round(len(texts) / total_time, 2) if total_time > 0 else 0.0,
            "tokens_per_sec": round(total_tokens / total_time, 2) if total_time > 0 else 0.0,
        }
        logger.info(
            "Tokenization: %d texts, %d tokens in %.4fs",
            len(texts),
            total_tokens,
            total_time,
        )
        return result
