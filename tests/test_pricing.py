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


def test_canonical_model_keys_match_converter_style() -> None:
    assert canonical_model_keys("gpt-5.3-codex") == ["gpt-5.3-codex", "openai/gpt-5.3-codex", "azure/gpt-5.3-codex"]
    assert canonical_model_keys("openai/gpt-5.3-codex") == ["openai/gpt-5.3-codex", "gpt-5.3-codex"]


def test_load_model_aliases_empty() -> None:
    from opencode_tokenstats.pricing import load_model_aliases
    # No file exists, should return empty dict
    import os
    old_env = os.environ.get("OPENCODE_MODEL_ALIASES_FILE")
    os.environ["OPENCODE_MODEL_ALIASES_FILE"] = "/nonexistent/path"
    try:
        result = load_model_aliases()
        assert result == {}
    finally:
        if old_env is None:
            os.environ.pop("OPENCODE_MODEL_ALIASES_FILE", None)
        else:
            os.environ["OPENCODE_MODEL_ALIASES_FILE"] = old_env


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
