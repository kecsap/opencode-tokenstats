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
    load_model_equivalence_rules,
    reset_pricing_lookup_cache,
    resolve_market_model,
    tier_applicability,
)
from opencode_tokenstats.pricing import PricingRecord, _parse_pricing_history


def test_build_lookup_key_normalization() -> None:
    assert PricingLookup.build_lookup_key("openai", "gpt-4o") == "openai/gpt-4o"
    assert PricingLookup.build_lookup_key("openai", "openai/gpt-4o") == "openai/gpt-4o"
    assert PricingLookup.build_lookup_key(None, "gpt-4o") == "gpt-4o"


def test_load_pricing_history_package_data() -> None:
    lookup = load_pricing_lookup()

    terra = lookup.get_pricing("openai/gpt-5.6-terra")
    terra_fast = lookup.get_pricing("openai/gpt-5.6-terra-fast")
    luna_fast = lookup.get_pricing("openai/gpt-5.6-luna-fast")
    official_terra = next(
        record
        for record in lookup.history
        if record.model == "gpt-5.6-terra"
        and record.source_kind == "provider_official"
        and record.source_url == "https://openai.com/api/pricing/"
    )

    assert (terra.input, terra.output, terra.cache_read, terra.cache_write) == (2.0, 0.2, 2.5, 12.0)
    assert (terra_fast.input, terra_fast.output, terra_fast.cache_read, terra_fast.cache_write) == (4.0, 0.4, 5.0, 24.0)
    assert (luna_fast.input, luna_fast.output, luna_fast.cache_read, luna_fast.cache_write) == (0.4, 0.04, 0.5, 2.4)
    assert official_terra.effective_from == "2026-09-18T00:00:00Z"
    assert official_terra.confidence == "observed/inferred"
    assert official_terra.billing_channel == "direct_api"
    assert official_terra.source_revision == "openai-pricing-2026-09-18"


def test_official_correction_supersedes_catalog_record() -> None:
    payload = {
        "schema_version": 2,
        "unit": "USD per 1M tokens",
        "records": [
            {
                "provider": "openai", "model": "gpt-x", "aliases": ["gpt-x"],
                "billing_channel": "direct_api", "service_profile": "standard", "context": "short",
                "effective_from": "2026-01-01T00:00:00Z", "effective_to": None,
                "status": "active", "confidence": "observed",
                "source": {"url": "catalog", "retrieved_at": "2026-01-01T00:00:00Z", "kind": "catalog"},
                "rates": {"input": 1, "output": 2},
            },
            {
                "provider": "openai", "model": "gpt-x", "aliases": ["gpt-x"],
                "billing_channel": "direct_api", "service_profile": "standard", "context": "short",
                "effective_from": "2026-01-01T00:00:00Z", "effective_to": None,
                "status": "active", "confidence": "official",
                "source": {"url": "official", "retrieved_at": "2026-01-02T00:00:00Z", "kind": "provider_official"},
                "rates": {"input": 3, "output": 4},
            },
        ],
    }
    data, history = _parse_pricing_history(payload)
    resolution = PricingLookup(data, history, flat_keys=frozenset()).resolve_call_pricing(
        "gpt-x", 1770000000000
    )
    assert resolution.pricing is not None
    assert resolution.pricing.input == 3
    assert resolution.provenance == "official"


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


def test_pricing_lookup_is_cached_until_reset(tmp_path, monkeypatch) -> None:
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text('{"openai/gpt-5.6-terra": {"input": 9, "output": 8}}')
    monkeypatch.setenv("OPENCODE_MODEL_PRICING_FILE", str(pricing_file))
    reset_pricing_lookup_cache()

    first = load_pricing_lookup()
    pricing_file.write_text('{"openai/gpt-5.6-terra": {"input": 1, "output": 2}}')
    assert load_pricing_lookup() is first
    assert load_pricing_lookup().get_pricing("openai/gpt-5.6-terra").input == 9

    reset_pricing_lookup_cache()
    assert load_pricing_lookup().get_pricing("openai/gpt-5.6-terra").input == 1


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


def test_tier_applicability_is_conservative_for_missing_context() -> None:
    pricing = ModelPricing(
        input=1.0,
        output=1.0,
        cache_read=0.0,
        tiers=(ContextPricing(input=2.0, output=2.0, cache_read=0.0, threshold=100_000),),
    )

    assert tier_applicability(pricing, 200_000) == "applied"
    assert tier_applicability(pricing, 100_000) == "base"
    assert tier_applicability(pricing, None) == "unknown"


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


def test_load_model_equivalence_rules_and_precedence(tmp_path) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text(
        "@market-model qwen3.8-27b* = catalog/old\n"
        "@market-model qwen3.8* = catalog/wild\n"
        "@market-model qwen3.8-27b = catalog/exact\n"
        "@cloud-equivalent qwen3.8-27b catalog/cloud\n"
    )
    rules = load_model_equivalence_rules(str(conf))
    assert len(rules) == 4
    assert rules[0].targets == ("catalog/old",)
    assert rules[2].source == "qwen3.8-27b"


def test_resolve_market_model_matches_normalized_model_and_aliases(tmp_path) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text("@market-model qwen3.8-27b = catalog/qwen3.8-27b\n")
    record = PricingRecord(
        provider="catalog",
        model="QWEN-3.8-27B-whatever",
        service_profile="standard",
        context="short",
        effective_from="2026-01-01",
        effective_to=None,
        status="active",
        confidence="observed",
        source_url="",
        retrieved_at="",
        pricing=ModelPricing(1, 1, 0),
        aliases=("catalog/qwen_3_8_27b",),
    )
    rules = load_model_equivalence_rules(str(conf))
    assert resolve_market_model("local/QWEN_3.8_27B", (record,), rules) is record


def test_unavailable_market_rule_uses_cloud_equivalent_not_automatic_match(tmp_path) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text(
        "@market-model qwen3.8-27b = catalog/missing\n"
        "@cloud-equivalent qwen3.8-27b = catalog/cloud\n"
    )
    automatic = PricingRecord(
        provider="catalog",
        model="qwen3.8-27b",
        service_profile="standard",
        context="short",
        effective_from="2026-01-01",
        effective_to=None,
        status="active",
        confidence="observed",
        source_url="",
        retrieved_at="",
        pricing=ModelPricing(1, 1, 0),
    )
    cloud = PricingRecord(
        provider="catalog",
        model="cloud-model",
        service_profile="standard",
        context="short",
        effective_from="2026-01-01",
        effective_to=None,
        status="active",
        confidence="observed",
        source_url="",
        retrieved_at="",
        pricing=ModelPricing(1, 1, 0),
        aliases=("catalog/cloud",),
    )
    rules = load_model_equivalence_rules(str(conf))
    assert resolve_market_model("local/qwen3.8-27b", (automatic, cloud), rules) is cloud
    assert resolve_market_model("local/qwen3.8-27b", (automatic,), rules) is None


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


def test_local_market_averages_paid_providers_once_and_keeps_basis(tmp_path) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text("@local local/*\n@market-model qwen = catalog/qwen\n")
    records = tuple(
        PricingRecord(
            provider=provider,
            model=model,
            service_profile="standard",
            context="short",
            effective_from="2026-01-01T00:00:00Z",
            effective_to=None,
            status="active",
            confidence="observed",
            source_url=provider,
            retrieved_at="2026-01-01T00:00:00Z",
            pricing=pricing,
            aliases=("catalog/qwen",),
        )
        for provider, model, pricing in (
            ("alpha", "qwen-a", ModelPricing(2, 4, 1, reasoning=6)),
            ("alpha", "qwen-a-fast", ModelPricing(100, 100, 100)),
            ("beta", "qwen-b", ModelPricing(4, 6, 3, reasoning=8)),
            ("free", "qwen-free", ModelPricing(0, 0, 0)),
        )
    )
    lookup = PricingLookup({"default": ModelPricing(1, 1, 0)}, records, equivalence_rules=load_model_equivalence_rules(str(conf)))
    result = lookup.resolve_local_call_pricing("local/qwen", _ms("2026-06-01T00:00:00Z"))
    assert result.pricing == ModelPricing(3, 5, 2, reasoning=7)
    assert result.provider_count == 2
    assert result.cost_basis == "market"
    assert result.market_status == "active"


def test_local_market_candidates_are_cached_per_model_and_rule(monkeypatch, tmp_path) -> None:
    import opencode_tokenstats.pricing as pricing_module

    conf = tmp_path / "models.conf"
    conf.write_text("@market-model qwen = catalog/qwen\n")
    record = PricingRecord(
        provider="catalog",
        model="qwen-3.8-27b",
        service_profile="standard",
        context="short",
        effective_from="2026-01-01T00:00:00Z",
        effective_to=None,
        status="active",
        confidence="observed",
        source_url="catalog",
        retrieved_at="2026-01-01T00:00:00Z",
        pricing=ModelPricing(1, 1, 0),
        aliases=("catalog/qwen",),
    )
    lookup = PricingLookup(
        {"default": ModelPricing(1, 1, 0)},
        (record,),
        equivalence_rules=load_model_equivalence_rules(str(conf)),
    )
    original = pricing_module._market_candidates
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pricing_module, "_market_candidates", counted)
    timestamp = _ms("2026-06-01T00:00:00Z")
    first = lookup.resolve_local_call_pricing("local/QWEN-3.8_27B", timestamp)
    second = lookup.resolve_local_call_pricing("local/qwen-3.8-27b", timestamp)

    assert first == second
    assert first.cost_basis == "market"
    assert calls == 1


@pytest.mark.parametrize(
    ("direct_start", "timestamp", "expected_status"),
    (
        ("2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z", "active"),
        ("2027-01-01T00:00:00Z", "2026-06-01T00:00:00Z", "future_fallback"),
    ),
)
def test_direct_historical_pricing_precedes_explicit_market_mapping(
    tmp_path, direct_start, timestamp, expected_status
) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text("@market-model qwen = catalog/qwen\n")
    direct = PricingRecord(
        provider="local",
        model="local/qwen",
        service_profile="standard",
        context="short",
        effective_from=direct_start,
        effective_to=None,
        status="active",
        confidence="observed",
        source_url="local",
        retrieved_at=direct_start,
        pricing=ModelPricing(9, 9, 0),
    )
    hosted = PricingRecord(
        provider="hosted",
        model="qwen",
        service_profile="standard",
        context="short",
        effective_from="2026-01-01T00:00:00Z",
        effective_to=None,
        status="active",
        confidence="observed",
        source_url="hosted",
        retrieved_at="2026-01-01T00:00:00Z",
        pricing=ModelPricing(2, 2, 0),
        aliases=("catalog/qwen",),
    )
    lookup = PricingLookup(
        {"default": ModelPricing(1, 1, 0)},
        (direct, hosted),
        equivalence_rules=load_model_equivalence_rules(str(conf)),
    )

    result = lookup.resolve_local_call_pricing("local/qwen", _ms(timestamp))

    assert result.status == expected_status
    assert result.pricing == ModelPricing(9, 9, 0)


def test_local_market_uses_earliest_future_date(tmp_path) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text("@market-model qwen = catalog/qwen\n")
    record = PricingRecord(
        provider="alpha", model="qwen", service_profile="standard", context="short",
        effective_from="2027-01-01T00:00:00Z", effective_to=None, status="active",
        confidence="observed", source_url="alpha", retrieved_at="2027-01-01T00:00:00Z",
        pricing=ModelPricing(2, 3, 0), aliases=("catalog/qwen",),
    )
    lookup = PricingLookup({"default": ModelPricing(1, 1, 0)}, (record,), equivalence_rules=load_model_equivalence_rules(str(conf)))
    result = lookup.resolve_local_call_pricing("local/qwen", _ms("2026-06-01T00:00:00Z"))
    assert result.status == "future_fallback"
    assert result.market_status == "future"
    assert result.rate_date == "2027-01-01T00:00:00Z"


def test_local_market_selects_per_provider_history_and_models_dev_future(tmp_path) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text("@market-model qwen = catalog/qwen\n")
    records = (
        PricingRecord(
            provider="alpha", model="qwen-new", service_profile="standard", context="short",
            effective_from="2027-01-01T00:00:00Z", effective_to=None,
            status="active", confidence="observed", source_url="alpha-new", retrieved_at="",
            pricing=ModelPricing(4, 4, 0), aliases=("catalog/qwen",), source_kind="models.dev",
        ),
        PricingRecord(
            provider="beta", model="qwen-beta", service_profile="standard", context="short",
            effective_from="2025-06-01T00:00:00Z", effective_to=None,
            status="active", confidence="observed", source_url="beta", retrieved_at="",
            pricing=ModelPricing(6, 6, 0), aliases=("catalog/qwen",),
        ),
    )
    lookup = PricingLookup(
        {"default": ModelPricing(1, 3, 0)}, records,
        equivalence_rules=load_model_equivalence_rules(str(conf)),
    )

    result = lookup.resolve_local_call_pricing("local/qwen", _ms("2026-06-01T00:00:00Z"))

    assert result.pricing == ModelPricing(5, 5, 0)
    assert result.status == "future_fallback"
    assert result.market_status == "future"
    assert result.provider_count == 2


def test_local_market_timestamp_less_uses_latest_rate_per_provider(tmp_path) -> None:
    conf = tmp_path / "models.conf"
    conf.write_text("@market-model qwen = catalog/qwen\n")
    records = tuple(
        PricingRecord(
            provider=provider, model=model, service_profile="standard", context="short",
            effective_from=effective_from, effective_to=None, status="active", confidence="observed",
            source_url=provider, retrieved_at="", pricing=ModelPricing(input_rate, input_rate, 0),
            aliases=("catalog/qwen",), source_kind=source_kind,
        )
        for provider, model, effective_from, input_rate, source_kind in (
            ("alpha", "qwen-old", "2025-01-01T00:00:00Z", 2, "catalog"),
            ("alpha", "qwen-new", "2026-01-01T00:00:00Z", 4, "models.dev"),
            ("beta", "qwen", "2025-06-01T00:00:00Z", 6, "catalog"),
        )
    )
    lookup = PricingLookup(
        {"default": ModelPricing(1, 3, 0)}, records,
        equivalence_rules=load_model_equivalence_rules(str(conf)),
    )

    result = lookup.resolve_local_call_pricing("local/qwen")

    assert result.pricing == ModelPricing(5, 5, 0)
    assert result.status == "future_fallback"
    assert result.market_status == "future"
    assert result.provider_count == 2


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


def test_resolve_call_pricing_projects_same_provider_market_rate() -> None:
    record = PricingRecord(
        provider="anthropic", model="claude-sonnet", service_profile="standard", context="short",
        effective_from="2026-01-01T00:00:00Z", effective_to=None, status="active", confidence="observed",
        source_url="anthropic", retrieved_at="", pricing=ModelPricing(2, 4, 0), aliases=("claude-sonnet",),
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, (record,), flat_keys=frozenset())

    result = lookup.resolve_call_pricing("anthropic/claude-sonnet-v2", _ms("2025-12-01T00:00:00Z"))

    assert result.pricing == record.pricing
    assert result.status == "future_fallback"
    assert result.provider_count == 1


def test_local_resolution_does_not_suppress_reentrant_market_resolution() -> None:
    record = PricingRecord(
        provider="market", model="qwen", service_profile="standard", context="short",
        effective_from="2026-01-01T00:00:00Z", effective_to=None, status="active", confidence="observed",
        source_url="market", retrieved_at="", pricing=ModelPricing(2, 4, 0), aliases=("qwen",),
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, (record,), flat_keys=frozenset())
    original_match = lookup._match_history_record
    nested: list[object] = []

    def match(raw_name: str, timestamp_ms: int | None, exact_only: bool = False):
        if not nested:
            nested.append(None)
            nested[0] = lookup.resolve_call_pricing("local/qwen-v2", timestamp_ms)
        return original_match(raw_name, timestamp_ms, exact_only)

    lookup._match_history_record = match  # type: ignore[method-assign]
    lookup.resolve_local_call_pricing("local/qwen-v2", _ms("2026-06-01T00:00:00Z"))

    assert nested[0].status == "future_fallback"


def test_resolve_call_pricing_cross_provider_market_uses_one_vote_per_provider() -> None:
    records = tuple(
        PricingRecord(
            provider=provider, model="claude-sonnet", service_profile="standard", context="short",
            effective_from=effective_from, effective_to=None, status="active", confidence="observed",
            source_url=provider, retrieved_at="", pricing=ModelPricing(rate, rate, 0),
            aliases=("claude-sonnet",),
        )
        for provider, effective_from, rate in (
            ("alpha", "2025-01-01T00:00:00Z", 2),
            ("alpha", "2026-01-01T00:00:00Z", 6),
            ("beta", "2025-01-01T00:00:00Z", 10),
        )
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, records, flat_keys=frozenset())

    result = lookup.resolve_call_pricing("target/claude-sonnet-v2", _ms("2026-06-01T00:00:00Z"))

    assert result.pricing == ModelPricing(8, 8, 0)
    assert result.status == "future_fallback"
    assert result.market_status == "future"
    assert result.provider_count == 2


def test_resolve_call_pricing_does_not_substitute_openai_gpt_variant() -> None:
    record = PricingRecord(
        provider="openai", model="gpt-4o", service_profile="standard", context="short",
        effective_from="2026-01-01T00:00:00Z", effective_to=None, status="active", confidence="observed",
        source_url="openai", retrieved_at="", pricing=ModelPricing(2, 4, 0), aliases=("gpt-4o",),
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, (record,), flat_keys=frozenset())

    result = lookup.resolve_call_pricing("openai/gpt-4o-mini", _ms("2026-06-01T00:00:00Z"))

    assert result.status == "default_fallback"
    assert result.pricing == ModelPricing(1, 3, 0)


def test_nonlocal_market_fallback_uses_models_dev_and_latest_paid_records() -> None:
    records = (
        PricingRecord(
            provider="openai", model="gpt-5", service_profile="standard", context="short",
            effective_from="2027-01-01T00:00:00Z", effective_to=None, status="active",
            confidence="observed", source_url="models.dev/openai", retrieved_at="",
            pricing=ModelPricing(4, 8, 0), aliases=("gpt-5",), source_kind="models.dev",
        ),
        PricingRecord(
            provider="other", model="gpt-6", service_profile="standard", context="short",
            effective_from="2027-01-01T00:00:00Z", effective_to=None, status="active",
            confidence="observed", source_url="models.dev/other", retrieved_at="",
            pricing=ModelPricing(6, 10, 0), aliases=("gpt-6",), source_kind="models.dev",
        ),
        PricingRecord(
            provider="other", model="free-model", service_profile="standard", context="short",
            effective_from="2026-01-01T00:00:00Z", effective_to=None, status="active",
            confidence="observed", source_url="models.dev/free", retrieved_at="",
            pricing=ModelPricing(0, 0, 0), aliases=("free-model",), source_kind="models.dev",
        ),
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, records, flat_keys=frozenset())

    projected = lookup.resolve_call_pricing("openai/gpt-5", _ms("2026-06-01T00:00:00Z"))
    assert projected.pricing == records[0].pricing
    assert projected.status == "future_fallback"
    assert projected.market_status == "future"

    latest = lookup.resolve_call_pricing("openai/gpt-5")
    assert latest.pricing == ModelPricing(4, 8, 0)
    assert latest.status == "future_fallback"

    cross_provider = lookup.resolve_call_pricing("missing/gpt-6", _ms("2026-06-01T00:00:00Z"))
    assert cross_provider.pricing == records[1].pricing
    assert cross_provider.market_status == "future"

    free = lookup.resolve_call_pricing("other/free-model-v2", _ms("2026-06-01T00:00:00Z"))
    assert free.status == "default_fallback"

    variant = lookup.resolve_call_pricing("openai/gpt-5-mini", _ms("2026-06-01T00:00:00Z"))
    assert variant.status == "default_fallback"


def test_nonlocal_models_dev_active_rate_is_unmarked() -> None:
    record = PricingRecord(
        provider="openai", model="gpt-5", service_profile="standard", context="short",
        effective_from="2026-01-01T00:00:00Z", effective_to=None, status="active",
        confidence="observed", source_url="models.dev/openai", retrieved_at="",
        pricing=ModelPricing(4, 8, 0), aliases=("gpt-5",), source_kind="models.dev",
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, (record,), flat_keys=frozenset())

    result = lookup.resolve_call_pricing("openai/gpt-5", _ms("2026-06-01T00:00:00Z"))

    assert result.status == "active"
    assert result.market_status == ""
    assert result.pricing == record.pricing


def test_nonlocal_models_dev_after_last_observation_is_projected() -> None:
    record = PricingRecord(
        provider="openai", model="gpt-5", service_profile="standard", context="short",
        effective_from="2025-01-01T00:00:00Z", effective_to="2026-01-01T00:00:00Z",
        status="active", confidence="observed", source_url="models.dev/openai", retrieved_at="",
        pricing=ModelPricing(4, 8, 0), aliases=("gpt-5",), source_kind="models.dev",
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, (record,), flat_keys=frozenset())

    result = lookup.resolve_call_pricing("openai/gpt-5", _ms("2026-06-01T00:00:00Z"))

    assert result.status == "future_fallback"
    assert result.market_status == "future"
    assert result.pricing == record.pricing


def test_market_candidate_cache_keeps_openai_strict_matching_provider_isolated() -> None:
    records = tuple(
        PricingRecord(
            provider=provider, model="gpt-4o", service_profile="standard", context="short",
            effective_from="2026-01-01T00:00:00Z", effective_to=None, status="active", confidence="observed",
            source_url=provider, retrieved_at="", pricing=ModelPricing(rate, rate, 0), aliases=("gpt-4o",),
        )
        for provider, rate in (("anthropic", 3), ("openai", 2))
    )
    lookup = PricingLookup({"default": ModelPricing(1, 3, 0)}, records, flat_keys=frozenset())
    timestamp = _ms("2026-06-01T00:00:00Z")

    fuzzy = lookup.resolve_call_pricing("anthropic/gpt-4o-mini", timestamp)
    strict = lookup.resolve_call_pricing("openai/gpt-4o-mini", timestamp)

    assert fuzzy.pricing == ModelPricing(3, 3, 0)
    assert strict.status == "default_fallback"


def test_resolve_call_pricing_unknown_retired_and_missing_timestamp_use_default_fallback() -> None:
    data, history = _parse_pricing_history(_two_period_payload())
    lookup = PricingLookup(data, history, flat_keys=frozenset())

    unknown = lookup.resolve_call_pricing("openai/never-seen", _ms("2026-03-15T00:00:00Z"))
    assert unknown.status == "default_fallback"
    assert unknown.pricing is not None
    assert unknown.pricing.input == 1.0
    assert unknown.pricing.output == 3.0
    assert unknown.provenance == "default"

    retired = lookup.resolve_call_pricing("openai/gpt-y", _ms("2026-03-15T00:00:00Z"))
    assert retired.status == "default_fallback"
    assert retired.pricing is not None

    no_timestamp = lookup.resolve_call_pricing("openai/gpt-x", None)
    assert no_timestamp.status == "future_fallback"
    assert no_timestamp.pricing is not None
    assert no_timestamp.pricing.input == 4.0



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


def test_resolve_call_pricing_index_matches_repeated_alias_and_provider_lookups() -> None:
    payload = _two_period_payload()
    payload["records"].extend([
        {
            "provider": "openai", "model": "gpt-official", "aliases": ["gpt-official"],
            "service_profile": "standard", "context": "short",
            "effective_from": "2026-01-01T00:00:00Z", "effective_to": None,
            "status": "active", "confidence": "observed",
            "source": {"url": "catalog-official", "retrieved_at": "2026-01-01T00:00:00Z"},
            "rates": {"input": 1.0, "output": 2.0},
        },
        {
            "provider": "openai", "model": "gpt-official", "aliases": ["gpt-official"],
            "service_profile": "standard", "context": "short",
            "effective_from": "2026-01-01T00:00:00Z", "effective_to": None,
            "status": "active", "confidence": "official",
            "source": {
                "url": "provider-official", "retrieved_at": "2026-01-02T00:00:00Z",
                "kind": "provider_official",
            },
            "rates": {"input": 3.0, "output": 4.0},
        },
        {
            "provider": "openai", "model": "gpt-tiered", "aliases": ["gpt-tiered"],
            "service_profile": "standard", "context": "short",
            "effective_from": "2026-01-01T00:00:00Z", "effective_to": None,
            "status": "active", "confidence": "observed",
            "source": {"url": "tiered", "retrieved_at": "2026-01-01T00:00:00Z"},
            "rates": {
                "input": 5.0, "output": 6.0, "cacheRead": 1.0,
                "tiers": [{"input": 7.0, "output": 8.0, "cacheRead": 2.0, "cacheWrite": 3.0, "threshold": 200000}],
            },
        },
    ])
    data, history = _parse_pricing_history(payload)
    data["openai/gpt-prefix"] = ModelPricing(input=9.0, output=10.0, cache_read=0.0)
    lookup = PricingLookup(data, history, flat_keys=frozenset({"openai/gpt-prefix"}))

    def assert_same_resolution(model_names: tuple[str, ...], timestamp: int | None) -> None:
        expected = lookup.resolve_call_pricing(model_names[0], timestamp)
        for model_name in model_names[1:]:
            assert lookup.resolve_call_pricing(model_name, timestamp) == expected

    historical = lookup.resolve_call_pricing("openai/gpt-x", _ms("2026-03-15T00:00:00Z"))
    assert_same_resolution(("openai/gpt-x", "gpt-x", "azure/gpt-x"), _ms("2026-03-15T00:00:00Z"))
    assert historical.status == "active"
    assert historical.provenance == "https://example.com/old"
    assert (historical.effective_from, historical.effective_to) == (
        "2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z"
    )
    assert historical.pricing is not None and historical.pricing.input == 1.0

    prefix = lookup.resolve_call_pricing("openai/gpt-prefix-v2", _ms("2026-03-15T00:00:00Z"))
    assert_same_resolution(("openai/gpt-prefix-v2",), _ms("2026-03-15T00:00:00Z"))
    assert prefix.status == "flat_override"
    assert prefix.provenance == "flat:openai/gpt-prefix"

    future = lookup.resolve_call_pricing("gpt-x", _ms("2025-12-01T00:00:00Z"))
    assert_same_resolution(("gpt-x", "openai/gpt-x"), _ms("2025-12-01T00:00:00Z"))
    assert future.status == "future_fallback"
    assert future.provenance == "https://example.com/old"
    assert (future.effective_from, future.effective_to) == (
        "2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z"
    )

    retired = lookup.resolve_call_pricing("openai/gpt-y", _ms("2026-03-15T00:00:00Z"))
    assert_same_resolution(("openai/gpt-y", "gpt-y"), _ms("2026-03-15T00:00:00Z"))
    assert (retired.status, retired.provenance, retired.effective_from, retired.effective_to) == (
        "default_fallback", "default", None, None
    )

    official = lookup.resolve_call_pricing("gpt-official", _ms("2026-03-15T00:00:00Z"))
    assert_same_resolution(("gpt-official", "openai/gpt-official"), _ms("2026-03-15T00:00:00Z"))
    assert official.status == "active"
    assert official.provenance == "provider-official"
    assert official.effective_from == "2026-01-01T00:00:00Z"
    assert official.effective_to is None
    assert official.pricing is not None and official.pricing.input == 3.0

    tiered = lookup.resolve_call_pricing("gpt-tiered", _ms("2026-03-15T00:00:00Z"))
    assert_same_resolution(("gpt-tiered", "openai/gpt-tiered"), _ms("2026-03-15T00:00:00Z"))
    assert tiered.status == "active"
    assert tiered.provenance == "tiered"
    assert tiered.pricing is not None
    assert (tiered.pricing.input, tiered.pricing.output) == (5.0, 6.0)
    assert tiered.pricing.tiers == (ContextPricing(input=7.0, output=8.0, cache_read=2.0, cache_write=3.0, threshold=200000),)


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
