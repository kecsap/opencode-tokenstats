from __future__ import annotations

from opencode_tokenstats.pricing import (
    ModelPricing,
    PricingLookup,
    canonical_model_keys,
    estimate_session_cost_usd,
)


def test_build_lookup_key_normalization() -> None:
    assert PricingLookup.build_lookup_key("openai", "gpt-4o") == "openai/gpt-4o"
    assert PricingLookup.build_lookup_key("openai", "openai/gpt-4o") == "openai/gpt-4o"
    assert PricingLookup.build_lookup_key(None, "gpt-4o") == "gpt-4o"


def test_pricing_exact_normalized_and_prefix_fallback() -> None:
    lookup = PricingLookup(
        {
            "openai/gpt-4o": ModelPricing(input=2, output=8, cache_write=1, cache_read=0.5),
            "gpt-4o": ModelPricing(input=3, output=9, cache_write=1, cache_read=0.5),
            "claude-sonnet": ModelPricing(input=4, output=12, cache_write=0, cache_read=0),
            "default": ModelPricing(input=1, output=3, cache_write=0, cache_read=0),
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


def test_estimate_session_cost_uses_reasoning_and_cache_components() -> None:
    pricing = ModelPricing(input=2.0, output=8.0, cache_write=1.0, cache_read=0.5)
    cost = estimate_session_cost_usd(
        pricing,
        input_tokens=1_000_000,
        output_tokens=500_000,
        reasoning_tokens=500_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == 11.5


def test_canonical_model_keys_match_converter_style() -> None:
    assert canonical_model_keys("gpt-5.3-codex") == ["gpt-5.3-codex", "openai/gpt-5.3-codex", "azure/gpt-5.3-codex"]
    assert canonical_model_keys("openai/gpt-5.3-codex") == ["openai/gpt-5.3-codex", "gpt-5.3-codex"]
