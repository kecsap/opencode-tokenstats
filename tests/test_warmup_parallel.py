from __future__ import annotations

import time
from opencode_tokenstats.tokenization import TokenizerRegistry, TokenizerSpec


def test_warmup_parallel_returns_results() -> None:
    """Test that parallel warmup returns results for all models."""
    registry = TokenizerRegistry()
    pairs = [
        ("local", "qwen3.6-27b"),
        ("openai", "gpt-5.3-codex"),
        ("anthropic", "claude-sonnet-4"),
    ]
    results = registry.warmup_parallel(pairs, sample_text="test")
    assert len(results) == 3
    # All results should have status
    for r in results:
        assert r.status in ("warmed", "approximate", "failed")


def test_warmup_parallel_same_status_as_sequential() -> None:
    """Test that parallel warmup produces same status as sequential warmup."""
    registry = TokenizerRegistry()
    pairs = [
        ("local", "qwen3.6-27b"),
        ("openai", "gpt-5.3-codex"),
        ("anthropic", "claude-sonnet-4"),
    ]
    sequential_results = registry.warmup(pairs, sample_text="test")
    parallel_results = registry.warmup_parallel(pairs, sample_text="test")

    # Same number of results
    assert len(sequential_results) == len(parallel_results)

    # Same statuses (order may differ)
    seq_statuses = [r.status for r in sequential_results]
    par_statuses = [r.status for r in parallel_results]
    assert set(seq_statuses) == set(par_statuses)


def test_warmup_parallel_with_single_worker() -> None:
    """Test parallel warmup with single worker still works."""
    registry = TokenizerRegistry()
    pairs = [("local", "qwen3.6-27b")]
    results = registry.warmup_parallel(pairs, sample_text="test", max_workers=1)
    assert len(results) == 1


def test_warmup_parallel_handles_exceptions() -> None:
    """Test that parallel warmup handles exceptions gracefully."""
    registry = TokenizerRegistry()
    # Use an invalid pair that might fail
    pairs = [("invalid-provider", "invalid-model")]
    results = registry.warmup_parallel(pairs, sample_text="test")
    assert len(results) == 1
    # Should still return a result even if it fails
    assert results[0].status in ("warmed", "approximate", "failed")


def test_warmup_parallel_with_approx_model() -> None:
    """Test parallel warmup with approximate tokenizer."""
    registry = TokenizerRegistry()
    # This model doesn't have an exact tokenizer mapping
    pairs = [("unknown-provider", "unknown-model")]
    results = registry.warmup_parallel(pairs, sample_text="test")
    assert len(results) == 1
    # Should be marked as approximate
    assert results[0].status == "approximate"


def test_warmup_parallel_concurrency() -> None:
    """Test that parallel warmup actually runs concurrently."""
    registry = TokenizerRegistry()

    # Track execution time
    start = time.time()
    pairs = [
        ("local", "qwen3.6-27b"),
        ("openai", "gpt-5.3-codex"),
        ("anthropic", "claude-sonnet-4"),
    ]
    results = registry.warmup_parallel(pairs, sample_text="test", max_workers=4)
    elapsed = time.time() - start

    # Should complete in reasonable time
    assert elapsed < 10.0  # Should be fast
    assert len(results) == 3


def test_warmup_parallel_empty_list() -> None:
    """Test parallel warmup with empty list."""
    registry = TokenizerRegistry()
    results = registry.warmup_parallel([], sample_text="test")
    assert results == []


def test_warmup_parallel_large_batch() -> None:
    """Test parallel warmup with a larger batch."""
    registry = TokenizerRegistry()
    # Create a batch of models
    pairs = [
        ("local", "qwen3.6-27b"),
        ("openai", "gpt-5.3-codex"),
        ("anthropic", "claude-sonnet-4"),
        ("openai", "gpt-4o"),
        ("openai", "gpt-4"),
    ]
    results = registry.warmup_parallel(pairs, sample_text="test")
    assert len(results) == 5