from __future__ import annotations

import pytest
from datetime import datetime

from opencode_tokenstats.pricing import (
    ContextPricing,
    ModelPricing,
    PricingLookup,
    canonical_model_keys,
    estimate_session_cost_usd,
    load_pricing_lookup,
)
from opencode_tokenstats.pricing import _parse_pricing_history


def test_build_lookup_key_normalization() -> None:
    assert PricingLookup.build_lookup_key("openai", "gpt-4o") == "openai/gpt-4o"
    assert PricingLookup.build_lookup_key("openai", "openai/gpt-4o") == "openai/gpt-4o"
    assert PricingLookup.build_lookup_key(None, "gpt-4o") == "gpt-4o"


def test_load_pricing_history_package_data() -> None:
    lookup = load_pricing_lookup()

    terra = lookup.get_pricing("openai/gpt-5.6-terra")
    terra_fast = lookup.get_pricing("openai/gpt-5.6-terra-fast")
    luna_fast = lookup.get_pricing("openai/gpt-5.6-luna-fast")

    assert (terra.input, terra.output, terra.cache_read, terra.cache_write) == (2.0, 0.2, 2.5, 12.0)
    assert (terra_fast.input, terra_fast.output, terra_fast.cache_read, terra_fast.cache_write) == (4.0, 0.4, 5.0, 24.0)
    assert (luna_fast.input, luna_fast.output, luna_fast.cache_read, luna_fast.cache_write) == (0.4, 0.04, 0.5, 2.4)
    assert lookup.history[0].effective_from == "2026-09-18T00:00:00Z"
    assert lookup.history[0].confidence == "observed/inferred"


def test_pricing_history_rejects_invalid_metadata_and_tiers() -> None:
    payload = {
        "schema_version": 1,
        "unit": "USD per 1M tokens",
        "records": [{
            "provider": "openai",
            "model": "test",
            "aliases": ["test"],
            "service_profile": "standard",
            "context": "short",
            "effective_from": "2026-09-18T00:00:00Z",
            "effective_to": None,
            "status": "active",
            "confidence": "observed",
            "source": {"url": "https://example.test", "retrieved_at": "not-a-date"},
            "rates": {"input": 1, "output": 1, "cacheRead": 1, "tiers": [{"input": -1, "output": 1, "threshold": 200000}]},
        }],
    }

    with pytest.raises(ValueError):
        _parse_pricing_history(payload)


def test_explicit_flat_pricing_file_overrides_history(tmp_path, monkeypatch) -> None:
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text('{"openai/gpt-5.6-terra": {"input": 9, "output": 8, "cacheRead": 7}}')
    monkeypatch.setenv("OPENCODE_MODEL_PRICING_FILE", str(pricing_file))

    pricing = load_pricing_lookup().get_pricing("openai/gpt-5.6-terra")
    assert (pricing.input, pricing.output, pricing.cache_read) == (9, 8, 7)


def test_pricing_exact_normalized_and_prefix_fallback() -> None:
    lookup = PricingLookup(
        {
            "openai/gpt-4o": ModelPricing(input=2, output=8, cache_read=0.5),
            "gpt-4o": ModelPricing(input=3, output=9, cache_read=0.5),
            "claude-sonnet": ModelPricing(input=4, output=12, cache_read=0),
            "default": ModelPricing(input=1, output=3, cache_read=0),
        }
    )

    exact = lookup.get_pricing("openai/gpt-4o")
    normalized = lookup.get_pricing("provider/gpt-4o")
    prefix = lookup.get_pricing("claude-sonnet-4-20250514")
    fallback = lookup.get_pricing("totally-unknown")

    assert exact.input == 2
    assert normalized.input == 3
    assert prefix.input == 4
    assert fallback.input == 1


def test_pricing_prefix_match_requires_version_boundary() -> None:
    lookup = PricingLookup(
        {
            "claude-sonnet": ModelPricing(input=4, output=12, cache_read=0),
            "default": ModelPricing(input=1, output=3, cache_read=0),
        }
    )

    assert lookup.get_pricing("claude-sonnet-4-20250514").input == 4
    assert lookup.get_pricing("claude-sonnetx").input == 1


def test_estimate_session_cost_uses_reasoning_and_cache_components() -> None:
    pricing = ModelPricing(
        input=2.0,
        output=8.0,
        cache_read=0.5,
        cache_write=2.5,
        web_search=0.01,
    )
    cost = estimate_session_cost_usd(
        pricing,
        input_tokens=1_000_000,
        output_tokens=500_000,
        reasoning_tokens=500_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        web_search_requests=2,
    )
    assert cost == 13.02


def test_estimate_session_cost_uses_context_tier_when_threshold_crossed() -> None:
    pricing = ModelPricing(
        input=1.0,
        output=1.0,
        cache_read=0.0,
        cache_write=0.0,
        context_over_200k=ContextPricing(input=2.0, output=2.0, cache_read=0.0, cache_write=0.0, threshold=200_000),
    )
    cost = estimate_session_cost_usd(
        pricing,
        input_tokens=300_000,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        context_tokens=300_000,
    )
    assert cost == 0.6


def test_estimate_session_cost_uses_highest_matching_context_tier() -> None:
    pricing = ModelPricing(
        input=1.0,
        output=1.0,
        cache_read=0.5,
        cache_write=0.25,
        tiers=(
            ContextPricing(input=2.0, output=2.0, cache_read=0.0, cache_write=0.0, threshold=100_000),
            ContextPricing(input=3.0, output=4.0, cache_read=0.0, cache_write=0.0, threshold=200_000),
        ),
        context_over_200k=ContextPricing(input=9.0, output=9.0, cache_read=9.0, cache_write=9.0, threshold=200_000),
    )
    cost = estimate_session_cost_usd(
        pricing,
        input_tokens=300_000,
        output_tokens=100_000,
        reasoning_tokens=0,
        cache_read_tokens=50_000,
        cache_write_tokens=25_000,
        context_tokens=375_000,
    )
    assert cost == pytest.approx(1.3)


def test_canonical_model_keys_match_converter_style() -> None:
    assert canonical_model_keys("gpt-5.3-codex") == ["gpt-5.3-codex", "openai/gpt-5.3-codex", "azure/gpt-5.3-codex"]
    assert canonical_model_keys("openai/gpt-5.3-codex") == ["openai/gpt-5.3-codex", "gpt-5.3-codex"]


def test_load_model_aliases_empty() -> None:
    from opencode_tokenstats.pricing import load_model_aliases
    # Pass nonexistent file path directly, should return empty dict
    result = load_model_aliases(file_path="/nonexistent/path")
    assert result == {}


def test_load_model_aliases_from_file(tmp_path) -> None:
    from opencode_tokenstats.pricing import load_model_aliases
    import os

    conf = tmp_path / "models.conf"
    conf.write_text(
        "# Comment\n"
        "gpt-unified = azure/gpt-5.4 openai/gpt-5.4\n"
        "claude-pro = anthropic/claude-sonnet-4\n"
    )
    old_env = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE")
    os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = str(conf)
    try:
        result = load_model_aliases()
        assert result["azure/gpt-5.4"] == "gpt-unified"
        assert result["openai/gpt-5.4"] == "gpt-unified"
        assert result["anthropic/claude-sonnet-4"] == "claude-pro"
    finally:
        if old_env is None:
            os.environ.pop("OPTOKEN_MODEL_ALIAS_FILE", None)
        else:
            os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = old_env


def test_load_local_model_patterns(tmp_path) -> None:
    from opencode_tokenstats.pricing import load_local_model_patterns
    import os

    conf = tmp_path / "models.conf"
    conf.write_text(
        "# Comment\n"
        "gpt-unified = azure/gpt-5.4 openai/gpt-5.4\n"
        "@local myollama/* myllamacpp/*\n"
        "@local *qwen36*\n"
    )
    old_env = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE")
    os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = str(conf)
    try:
        result = load_local_model_patterns()
        assert "myollama/*" in result
        assert "myllamacpp/*" in result
        assert "*qwen36*" in result
    finally:
        if old_env is None:
            os.environ.pop("OPTOKEN_MODEL_ALIAS_FILE", None)
        else:
            os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = old_env


def test_load_model_aliases_wildcard(tmp_path) -> None:
    from opencode_tokenstats.pricing import load_model_aliases
    import os

    conf = tmp_path / "models.conf"
    conf.write_text(
        "@local local/qwen = myollama/qwen*\n"
        "@local local/llama = myllama/*\n"
        "exact-alias = openai/gpt-4o\n"
    )
    old_env = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE")
    os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = str(conf)
    try:
        result = load_model_aliases()
        # Wildcard patterns stored with * marker
        assert "*myollama/qwen*" in result
        assert "*myllama/*" in result
        # Exact matches stored normally
        assert result["openai/gpt-4o"] == "exact-alias"
    finally:
        if old_env is None:
            os.environ.pop("OPTOKEN_MODEL_ALIAS_FILE", None)
        else:
            os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = old_env


def test_resolve_alias_exact_match() -> None:
    from opencode_tokenstats.pricing import resolve_alias

    aliases = {
        "openai/gpt-4o": "gpt-unified",
        "anthropic/claude": "claude-pro",
    }

    assert resolve_alias("openai/gpt-4o", aliases) == "gpt-unified"
    assert resolve_alias("anthropic/claude", aliases) == "claude-pro"
    assert resolve_alias("unknown/model", aliases) == "unknown/model"


def test_resolve_alias_wildcard_suffix() -> None:
    from opencode_tokenstats.pricing import resolve_alias

    aliases = {
        "*myollama/qwen*": "local/qwen",
        "*myllama/*": "local/llama",
    }

    assert resolve_alias("myollama/qwen3.6-27b", aliases) == "local/qwen"
    assert resolve_alias("myllama/llama3.1", aliases) == "local/llama"
    # No match
    assert resolve_alias("openai/gpt-4o", aliases) == "openai/gpt-4o"


def test_resolve_alias_exact_precedes_wildcard() -> None:
    from opencode_tokenstats.pricing import resolve_alias

    aliases = {
        "myollama/qwen3.6-27b": "exact-alias",
        "*myollama/qwen*": "local/qwen",
    }

    # Exact match takes precedence
    assert resolve_alias("myollama/qwen3.6-27b", aliases) == "exact-alias"
    # Wildcard matches other variants
    assert resolve_alias("myollama/qwen3.5-9b", aliases) == "local/qwen"


def test_resolve_alias_wildcard_provider_and_model_prefix() -> None:
    from opencode_tokenstats.pricing import resolve_alias

    aliases = {
        "*llamacpp_qwen36*/qwen3.6-27b*": "local/qwen3.6-27b",
    }

    assert resolve_alias("llamacpp_qwen36_gpu/qwen3.6-27b1", aliases) == "local/qwen3.6-27b"
    assert resolve_alias("llamacpp_qwen36/qwen3.6-27b", aliases) == "local/qwen3.6-27b"
    assert resolve_alias("myllamacpp/qwen3.6-27b", aliases) == "myllamacpp/qwen3.6-27b"


def _two_period_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "unit": "USD per 1M tokens",
        "records": [
            {
                "provider": "openai",
                "model": "gpt-x",
                "aliases": ["openai/gpt-x", "gpt-x"],
                "service_profile": "standard",
                "context": "short",
                "effective_from": "2026-01-01T00:00:00Z",
                "effective_to": "2026-06-01T00:00:00Z",
                "status": "active",
                "confidence": "observed",
                "source": {"url": "https://example.com/old", "retrieved_at": "2026-01-01T00:00:00Z"},
                "rates": {"input": 1.0, "output": 2.0},
            },
            {
                "provider": "openai",
                "model": "gpt-x",
                "aliases": ["openai/gpt-x", "gpt-x"],
                "service_profile": "standard",
                "context": "short",
                "effective_from": "2026-06-01T00:00:00Z",
                "effective_to": None,
                "status": "active",
                "confidence": "observed",
                "source": {"url": "https://example.com/new", "retrieved_at": "2026-06-01T00:00:00Z"},
                "rates": {"input": 4.0, "output": 8.0},
            },
            {
                "provider": "openai",
                "model": "gpt-y",
                "aliases": ["openai/gpt-y", "gpt-y"],
                "service_profile": "standard",
                "context": "short",
                "effective_from": "2026-01-01T00:00:00Z",
                "effective_to": "2026-02-01T00:00:00Z",
                "status": "retired",
                "confidence": "observed",
                "source": {"url": "https://example.com/y", "retrieved_at": "2026-01-01T00:00:00Z"},
                "rates": {"input": 9.0, "output": 9.0},
            },
        ]
    }


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def test_resolve_call_pricing_uses_active_period_rate() -> None:
    data, history = _parse_pricing_history(_two_period_payload())
    lookup = PricingLookup(data, history, flat_keys=frozenset())

    old = lookup.resolve_call_pricing("openai/gpt-x", _ms("2026-03-15T00:00:00Z"))
    assert old.status == "active"
    assert old.pricing is not None
    assert old.pricing.input == 1.0
    assert old.provenance == "https://example.com/old"
    assert old.effective_from == "2026-01-01T00:00:00Z"
    assert old.effective_to == "2026-06-01T00:00:00Z"

    new = lookup.resolve_call_pricing("gpt-x", _ms("2026-07-15T00:00:00Z"))
    assert new.status == "active"
    assert new.pricing is not None
    assert new.pricing.input == 4.0
    assert new.provenance == "https://example.com/new"
    assert new.effective_to is None


def test_resolve_call_pricing_before_first_period_is_future_fallback() -> None:
    data, history = _parse_pricing_history(_two_period_payload())
    lookup = PricingLookup(data, history, flat_keys=frozenset())

    resolution = lookup.resolve_call_pricing("openai/gpt-x", _ms("2025-12-01T00:00:00Z"))
    assert resolution.status == "future_fallback"
    assert resolution.pricing is not None
    assert resolution.pricing.input == 1.0
    assert resolution.provenance == "https://example.com/old"


def test_resolve_call_pricing_unknown_retired_and_missing_timestamp_are_unpriced() -> None:
    data, history = _parse_pricing_history(_two_period_payload())
    lookup = PricingLookup(data, history, flat_keys=frozenset())

    unknown = lookup.resolve_call_pricing("openai/never-seen", _ms("2026-03-15T00:00:00Z"))
    assert unknown.status == "unpriced"
    assert unknown.pricing is None

    retired = lookup.resolve_call_pricing("openai/gpt-y", _ms("2026-03-15T00:00:00Z"))
    assert retired.status == "unpriced"
    assert retired.pricing is None

    no_timestamp = lookup.resolve_call_pricing("openai/gpt-x", None)
    assert no_timestamp.status == "unpriced"
    assert no_timestamp.pricing is None


def test_resolve_call_pricing_flat_override_applies_without_timestamp() -> None:
    flat = ModelPricing(input=7.0, output=7.0, cache_read=0.0, cache_write=0.0, web_search=0.0)
    data, history = _parse_pricing_history(_two_period_payload())
    lookup = PricingLookup({"openai/gpt-z": flat}, history, flat_keys=frozenset({"openai/gpt-z"}))

    override = lookup.resolve_call_pricing("openai/gpt-z", None)
    assert override.status == "flat_override"
    assert override.pricing is flat
    assert override.provenance == "flat:openai/gpt-z"

    with_timestamp = lookup.resolve_call_pricing("openai/gpt-z", _ms("2026-03-15T00:00:00Z"))
    assert with_timestamp.status == "flat_override"
    assert with_timestamp.pricing is flat


def test_resolve_call_pricing_history_wins_over_derived_flat_snapshot() -> None:
    data, history = _parse_pricing_history(_two_period_payload())
    # pricing_data carries the latest record under aliases, but only explicit
    # flat-file keys may act as overrides when history cannot resolve a call.
    lookup = PricingLookup(data, history, flat_keys=frozenset())

    resolution = lookup.resolve_call_pricing("openai/gpt-x", _ms("2026-03-15T00:00:00Z"))
    assert resolution.status == "active"
    assert resolution.provenance == "https://example.com/old"


def test_resolve_call_pricing_keeps_standard_and_fast_rates_separate() -> None:
    lookup = load_pricing_lookup()

    standard = lookup.resolve_call_pricing("openai/gpt-5.6-terra", _ms("2026-09-19T00:00:00Z"))
    fast = lookup.resolve_call_pricing("openai/gpt-5.6-terra-fast", _ms("2026-09-19T00:00:00Z"))
    assert standard.status == "active"
    assert fast.status == "active"
    assert standard.pricing is not None and fast.pricing is not None
    assert standard.pricing.input == 2.0
    assert fast.pricing.input == 4.0
    assert standard.pricing.output == 0.2
    assert fast.pricing.output == 0.4
