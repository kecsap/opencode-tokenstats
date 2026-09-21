from __future__ import annotations

import pytest

from opencode_tokenstats.canonical_metrics import build_canonical_metrics, _build_component_family_rows, _pricing_warnings
from opencode_tokenstats.content_attribution import collect_content_attribution


def test_build_canonical_metrics_basic_semantics() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {
                    "input": 100,
                    "output": 20,
                    "reasoning": 5,
                    "cache": {"read": 10, "write": 0},
                },
                "cost": 0.5,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_search",
                    "state": {"status": "completed", "output": "abc"},
                },
            ],
        }
    ]

    out = build_canonical_metrics("s1", messages)
    assert out.session_id == "s1"
    assert out.model == "gpt-5.3-codex"
    assert out.api_calls == 1
    assert out.input_tokens == 100
    assert len(out.tool_rows) == 1
    assert out.tool_rows[0]["tool"] == "lean-ctx_ctx_search"
    assert out.component_rows[0]["component_group"] == "lean-ctx"
    assert out.mcp_rows[0]["name"] == "lean-ctx"


def test_canonical_metrics_resolves_each_call_once(monkeypatch) -> None:
    from opencode_tokenstats.pricing import PricingLookup

    calls = 0
    original = PricingLookup.resolve_call_pricing

    def resolve_once(self, model_name, timestamp_ms=None, *, include_market=True):
        nonlocal calls
        calls += 1
        return original(self, model_name, timestamp_ms, include_market=include_market)

    monkeypatch.setattr(PricingLookup, "resolve_call_pricing", resolve_once)
    out = build_canonical_metrics("s-reuse", [{
        "role": "assistant",
        "info": {
            "providerID": "openai",
            "modelID": "gpt-5.3-codex",
            "tokens": {"input": 10, "output": 5, "reasoning": 1, "cache": {"read": 2, "write": 0}},
            "cost": 0.0,
        },
        "parts": [{"type": "text", "text": "ok"}],
    }])

    assert calls == 1
    assert out.estimated_cost_usd == pytest.approx(out.per_model_costs[0]["estimated_cost"])
    assert out.activity_rows[0]["estimated_cost"] == out.estimated_cost_usd


def test_canonical_metrics_uses_local_market_rules_for_estimates(monkeypatch, tmp_path) -> None:
    from opencode_tokenstats.pricing import (
        ModelPricing,
        PricingLookup,
        PricingRecord,
    )

    conf = tmp_path / "models.conf"
    conf.write_text(
        "@local *\n"
        "@market-model qwen3.8-27b* = catalog/missing\n"
        "@market-model qwen3.8* = catalog/wild\n"
        "@market-model qwen3.8-27b = catalog/exact\n"
        "@cloud-equivalent qwen3.8-27b = catalog/cloud\n"
    )
    rate = ModelPricing(input=10.0, output=0.0, cache_read=0.0)
    history = tuple(
        PricingRecord(
            provider="catalog",
            model=model,
            service_profile="standard",
            context="short",
            effective_from="2026-01-01T00:00:00Z",
            effective_to=None,
            status="active",
            confidence="observed",
            source_url="",
            retrieved_at="",
            pricing=ModelPricing(input=input_rate, output=0.0, cache_read=0.0),
            aliases=aliases,
        )
        for model, input_rate, aliases in (
            ("wild", 2.0, ("catalog/wild",)),
            ("cloud-model", 3.0, ("catalog/cloud",)),
            ("QWEN-3.8-27B-whatever", 4.0, ("catalog/exact",)),
            ("qwen3.8-27b", 5.0, ()),
        )
    )
    monkeypatch.setenv("OPTOKEN_MODEL_ALIAS_FILE", str(conf))
    lookup = PricingLookup({"default": rate}, history)
    assert lookup.resolve_local_call_pricing("local/QWEN_3.8_27B", 1773878400000).pricing.input == 4.0
    assert lookup.resolve_local_call_pricing("local/local/QWEN_3.8_27B", 1773878400000).pricing.input == 4.0
    monkeypatch.setattr("opencode_tokenstats.canonical_metrics.build_default_pricing_lookup", lambda: lookup)
    monkeypatch.setattr("opencode_tokenstats.canonical_metrics._is_local_model", lambda model: True)

    out = build_canonical_metrics("s-market", [{
        "role": "assistant",
        "info": {
            "providerID": "local",
            "modelID": "local/QWEN_3.8_27B",
            "time": {"completed": 1773878400000},
            "tokens": {"input": 1_000_000, "output": 0, "cache": {"read": 0, "write": 0}},
            "cost": 0.0,
        },
        "parts": [{"type": "text", "text": "ok"}],
    }])

    assert out.estimated_cost_usd == pytest.approx(4.0)

    monkeypatch.setenv("OPTOKEN_MODEL_ALIAS_FILE", str(tmp_path / "unavailable.conf"))
    (tmp_path / "unavailable.conf").write_text(
        "@local *local-model\n"
        "@market-model local-model = catalog/missing\n"
        "@cloud-equivalent local-model = catalog/cloud\n"
    )
    fallback_lookup = PricingLookup({"default": rate}, history)
    monkeypatch.setattr(
        "opencode_tokenstats.canonical_metrics.build_default_pricing_lookup", lambda: fallback_lookup
    )
    fallback = build_canonical_metrics("s-market-fallback", [{
        "role": "assistant",
        "info": {
            "providerID": "local",
            "modelID": "local-model",
            "time": {"completed": 1773878400000},
            "tokens": {"input": 1_000_000, "output": 0, "cache": {"read": 0, "write": 0}},
            "cost": 0.0,
        },
        "parts": [{"type": "text", "text": "ok"}],
    }])

    assert fallback.estimated_cost_usd == pytest.approx(3.0)


def test_canonical_metrics_exposes_tier_coverage(monkeypatch, tmp_path) -> None:
    import json

    pricing_path = tmp_path / "models.json"
    pricing_path.write_text(json.dumps({
        "openai/tiered": {
            "input": 1.0,
            "output": 1.0,
            "tiers": [{"input": 2.0, "output": 2.0, "threshold": 100000}],
        }
    }))
    monkeypatch.setenv("OPENCODE_MODEL_PRICING_FILE", str(pricing_path))

    out = build_canonical_metrics("s-tier", [{
        "role": "assistant",
        "info": {
            "providerID": "openai",
            "modelID": "tiered",
            "tokens": {"input": 200000, "output": 0, "cache": {"read": 0, "write": 0}},
            "cost": 0.0,
        },
        "parts": [{"type": "text", "text": "ok"}],
    }])

    row = out.per_model_costs[0]
    assert row["tier_applied_calls"] == 1
    assert row["base_rate_calls"] == 0
    assert row["context_tokens"] == 200000
    assert row["context_token_source"] == "input_plus_cache"
    assert out.pricing_coverage["tier_applied_calls"] == 1


def test_canonical_metrics_aggregates_each_cost_basis(monkeypatch) -> None:
    from opencode_tokenstats.pricing import ModelPricing, PricingResolution

    resolutions = {
        "test/direct": PricingResolution(
            ModelPricing(1.0, 0.0, 0.0), "active", cost_basis="direct", billing_channel="direct_api"
        ),
        "test/generic": PricingResolution(
            ModelPricing(2.0, 0.0, 0.0), "default_fallback", cost_basis="generic"
        ),
        "local/active": PricingResolution(
            ModelPricing(3.0, 0.0, 0.0), "active", cost_basis="market", billing_channel="market",
            provider_count=2, rate_date="2026-01-01", market_status="active",
        ),
        "local/future": PricingResolution(
            ModelPricing(4.0, 0.0, 0.0), "future_fallback", cost_basis="market", billing_channel="market",
            provider_count=3, rate_date="2027-01-01", market_status="future",
        ),
        "local/cloud": PricingResolution(
            ModelPricing(5.0, 0.0, 0.0), "active", cost_basis="cloud_equivalent",
            billing_channel="cloud_equivalent", provider_count=1, rate_date="2026-01-01",
            market_status="active",
        ),
    }

    class Lookup:
        def resolve_call_pricing(self, model_name, _timestamp_ms=None):
            return resolutions[model_name]

        def resolve_local_call_pricing(self, model_name, _timestamp_ms=None):
            return resolutions[model_name]

    monkeypatch.setattr("opencode_tokenstats.canonical_metrics.build_default_pricing_lookup", Lookup)
    monkeypatch.setattr(
        "opencode_tokenstats.canonical_metrics._is_local_model",
        lambda model: model.startswith("local/"),
    )
    messages = [
        {
            "role": "assistant",
            "info": {
                "providerID": provider,
                "modelID": model,
                "tokens": {"input": 1_000_000, "output": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.0,
            },
            "parts": [{"type": "text", "text": "ok"}],
        }
        for provider, model in (
            ("test", "direct"),
            ("test", "generic"),
            ("local", "active"),
            ("local", "future"),
            ("local", "cloud"),
        )
    ]

    rows = {row["model"]: row for row in build_canonical_metrics("s-bases", messages).per_model_costs}

    assert [rows[model][field] for model, field in (
        ("test/direct", "estimated_direct_cost"),
        ("test/generic", "estimated_generic_cost"),
        ("local/active", "estimated_market_cost"),
        ("local/future", "estimated_future_market_cost"),
        ("local/cloud", "estimated_cloud_equivalent_cost"),
    )] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [rows[model][field] for model, field in (
        ("test/direct", "direct_estimate_calls"),
        ("test/generic", "generic_estimate_calls"),
        ("local/active", "market_estimate_calls"),
        ("local/future", "future_market_estimate_calls"),
        ("local/cloud", "cloud_equivalent_estimate_calls"),
    )] == [1, 1, 1, 1, 1]
    assert rows["local/active"]["market_provider_count"] == 2
    assert rows["local/active"]["market_status"] == "active"
    assert rows["local/future"]["market_rate_date"] == "2027-01-01"
    assert rows["local/future"]["market_status"] == "future"
    assert rows["local/cloud"]["cost_basis"] == "cloud_equivalent"


def test_canonical_metrics_marks_incomplete_context_as_tier_unknown(monkeypatch, tmp_path) -> None:
    import json

    pricing_path = tmp_path / "models.json"
    pricing_path.write_text(json.dumps({
        "openai/tiered": {
            "input": 1.0,
            "output": 1.0,
            "tiers": [{"input": 2.0, "output": 2.0, "threshold": 100000}],
        }
    }))
    monkeypatch.setenv("OPENCODE_MODEL_PRICING_FILE", str(pricing_path))

    out = build_canonical_metrics("s-tier-incomplete", [{
        "role": "assistant",
        "info": {
            "providerID": "openai",
            "modelID": "tiered",
            "tokens": {"input": 200000, "output": 0},
            "cost": 0.0,
        },
        "parts": [{"type": "text", "text": "ok"}],
    }])

    row = out.per_model_costs[0]
    assert row["tier_unknown_calls"] == 1
    assert row["tier_applied_calls"] == 0
    assert row["base_rate_calls"] == 0
    assert row["context_tokens"] == 0
    assert row["context_token_source"] == "incomplete"
    assert out.pricing_coverage["tier_unknown_calls"] == 1


def test_future_fallback_pricing_does_not_emit_warning() -> None:
    assert _pricing_warnings([{"model": "openai/gpt-x", "future_fallback_calls": 1}]) == []


def test_canonical_metrics_extracts_skill_and_subagent_components() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": """
<available_skills>
  <skill>
    <name>caveman</name>
    <description>Ultra-compressed communication mode.</description>
  </skill>
</available_skills>
- explore: Fast agent specialized for exploring codebases.
- general: General-purpose agent for researching complex questions.
""",
            },
            "parts": [{"type": "text", "text": "ok"}],
        }
    ]

    out = build_canonical_metrics("s2", messages)
    # explore and general are core subagents, classified as "core" type
    names = {(r["component_type"], r["component_name"]) for r in out.component_rows}
    assert ("skill", "caveman") in names
    assert ("core", "explore") in names
    assert ("core", "general") in names

    # skill/subagent estimates stay as single observed catalog/context burden
    for row in out.component_rows:
        if row["component_type"] in {"skill", "core"}:
            assert row["estimated_session_tokens"] == row["tokens"]


def test_model_includes_provider_prefix() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "providerID": "azure",
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
            },
            "parts": [{"type": "text", "text": "ok"}],
        }
    ]

    out = build_canonical_metrics("s3", messages)
    assert out.model == "azure/gpt-5.3-codex"


def test_canonical_metrics_reads_data_and_part_model_shapes() -> None:
    messages = [
        {
            "role": "assistant",
            "data": {
                "providerID": "openai",
                "modelID": "gpt-a",
                "tokens": {"input": 4, "output": 2, "reasoning": 1, "cache": {"read": 0, "write": 0}},
                "cost": 0.0,
            },
            "parts": [
                {
                    "type": "step-finish",
                    "model": {"providerID": "openai", "modelID": "gpt-a"},
                    "tokens": {"input": 4, "output": 2, "reasoning": 1, "cache": {"read": 0, "write": 0}},
                    "cost": 0.0,
                }
            ],
        }
    ]

    out = build_canonical_metrics("s-data", messages)
    assert out.model == "openai/gpt-a"
    assert out.per_model_costs[0]["model"] == "openai/gpt-a"
    # Row tokens include reasoning, same as session/activity totals
    assert out.per_model_costs[0]["tokens"] == 7


def test_local_model_has_zero_api_cost(tmp_path) -> None:
    from opencode_tokenstats.canonical_metrics import _is_local_model
    import os

    conf = tmp_path / "models.conf"
    conf.write_text("@local myollama/* myllamacpp/* *qwen36*\n")
    old_env = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE")
    os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = str(conf)
    try:
        # Local patterns
        assert _is_local_model("myollama/qwen3.6:35b-yarn")
        assert _is_local_model("myllamacpp/qwen3.6-27b1")
        assert _is_local_model("llamacpp_qwen36_gpu/qwen3.6-27b1")

        # Non-local patterns
        assert not _is_local_model("azure/gpt-5.4")
        assert not _is_local_model("openai/gpt-5.3-codex")
        assert not _is_local_model("anthropic/claude-sonnet-4")
    finally:
        if old_env is None:
            os.environ.pop("OPTOKEN_MODEL_ALIAS_FILE", None)
        else:
            os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = old_env


def test_local_model_cost_is_zero(tmp_path) -> None:
    import os

    conf = tmp_path / "models.conf"
    conf.write_text("@local myollama/*\n")
    old_env = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE")
    os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = str(conf)
    try:
        messages = [
            {
                "role": "assistant",
                "info": {
                    "providerID": "myollama",
                    "modelID": "qwen3.6:35b-yarn",
                    "tokens": {"input": 100, "output": 50, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                    "cost": 18.78,  # Cost in telemetry, but should be ignored for local models
                },
                "parts": [{"type": "text", "text": "ok"}],
            }
        ]

        out = build_canonical_metrics("s-local", messages)
        assert out.model == "myollama/qwen3.6:35b-yarn"
        assert out.actual_cost_usd == 0.0  # API cost should be 0 for local models
        assert out.estimated_cost_usd == 0.0
        assert out.pricing_coverage["future_fallback_calls"] == 1
    finally:
        if old_env is None:
            os.environ.pop("OPTOKEN_MODEL_ALIAS_FILE", None)
        else:
            os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = old_env


def test_estimated_cost_uses_per_call_models(tmp_path) -> None:
    import json
    import os

    pricing_path = tmp_path / "models.json"
    pricing_path.write_text(
        json.dumps(
            {
                "openai/gpt-a": {
                    "input": 2.0,
                    "output": 8.0,
                    "cacheRead": 0.5,
                    "cacheWrite": 2.5,
                    "webSearch": 0.01,
                },
                "openai/gpt-b": {
                    "input": 1.0,
                    "output": 4.0,
                    "cacheRead": 0.0,
                    "cacheWrite": 1.0,
                    "webSearch": 0.02,
                },
                "default": {"input": 1.0, "output": 3.0, "cacheRead": 0.0},
            }
        )
    )

    old_pricing_env = os.environ.get("OPENCODE_MODEL_PRICING_FILE")
    os.environ["OPENCODE_MODEL_PRICING_FILE"] = str(pricing_path)
    try:
        messages = [
            {
                "role": "assistant",
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-a",
                    "tokens": {
                        "input": 1_000_000,
                        "output": 500_000,
                        "reasoning": 500_000,
                        "cache": {"read": 1_000_000, "write": 1_000_000},
                        "server_tool_use": {"web_search_requests": 1},
                    },
                    "cost": 0.0,
                },
                "parts": [{"type": "text", "text": "a"}],
            },
            {
                "role": "assistant",
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-b",
                    "tokens": {
                        "input": 1_000_000,
                        "output": 500_000,
                        "reasoning": 500_000,
                        "cache": {"read": 0, "write": 1_000_000},
                        "server_tool_use": {"web_search_requests": 2},
                    },
                    "cost": 0.0,
                },
                "parts": [{"type": "text", "text": "b"}],
            },
        ]
        out = build_canonical_metrics("s-mixed", messages)
        assert out.actual_cost_usd == 0.0
        assert out.estimated_cost_usd == 19.05
        assert {row["model"] for row in out.per_model_costs} == {"openai/gpt-a", "openai/gpt-b"}
    finally:
        if old_pricing_env is None:
            os.environ.pop("OPENCODE_MODEL_PRICING_FILE", None)
        else:
            os.environ["OPENCODE_MODEL_PRICING_FILE"] = old_pricing_env


def test_per_model_costs_keep_api_only_for_trusted_billed_model_rows(tmp_path, monkeypatch) -> None:
    import json

    pricing_path = tmp_path / "models.json"
    pricing_path.write_text(
        json.dumps(
            {
                "openai/gpt-5.4-mini-fast": {"input": 1.0, "output": 3.0},
                "openai/gpt-5.4": {"input": 1.0, "output": 3.0},
            }
        )
    )
    monkeypatch.setenv("OPENCODE_MODEL_PRICING_FILE", str(pricing_path))

    messages = [
        {
            "role": "assistant",
            "info": {
                "providerID": "openai",
                "modelID": "gpt-5.4-mini-fast",
                "tokens": {
                    "input": 100,
                    "output": 50,
                    "reasoning": 10,
                    "cache": {"read": 20, "write": 0},
                },
                "cost": 0.0,
            },
            "parts": [{"type": "text", "text": "fast"}],
        },
        {
            "role": "assistant",
            "info": {
                "providerID": "openai",
                "modelID": "gpt-5.4",
                "tokens": {
                    "input": 200,
                    "output": 100,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 0},
                },
                "cost": 12.34,
            },
            "parts": [{"type": "text", "text": "main"}],
        },
    ]

    out = build_canonical_metrics("s-actual-mixed", messages)

    assert out.actual_cost_usd == 12.34
    model_rows = {row["model"]: row for row in out.per_model_costs}
    assert model_rows["openai/gpt-5.4-mini-fast"]["api_cost"] == 0.0
    assert model_rows["openai/gpt-5.4-mini-fast"]["cost"] == model_rows["openai/gpt-5.4-mini-fast"]["estimated_cost"]
    assert model_rows["openai/gpt-5.4-mini-fast"]["estimated_cost"] > 0
    assert model_rows["openai/gpt-5.4"]["api_cost"] == 12.34
    assert model_rows["openai/gpt-5.4"]["estimated_cost"] == 0.0
    assert model_rows["openai/gpt-5.4"]["cost"] == 12.34
    assert round(out.estimated_cost_usd, 6) == model_rows["openai/gpt-5.4-mini-fast"]["estimated_cost"]
    assert sum(row["api_cost"] for row in out.activity_rows) == 12.34
    assert round(sum(row["estimated_cost"] for row in out.activity_rows), 6) == round(out.estimated_cost_usd, 6)


def test_per_model_costs_exclude_zero_usage_plugin_calls() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "providerID": "magic-context",
                "modelID": "magic-context",
                "tokens": {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.0,
            },
            "parts": [{"type": "text", "text": "plugin"}],
        },
        {
            "role": "assistant",
            "info": {
                "providerID": "llamacpp-plant",
                "modelID": "qwen3.8-27b",
                "tokens": {"input": 100, "output": 10, "reasoning": 5, "cache": {"read": 0, "write": 0}},
                "cost": 0.0,
            },
            "parts": [{"type": "text", "text": "model"}],
        },
    ]

    out = build_canonical_metrics("s-plugin", messages)

    assert [row["model"] for row in out.per_model_costs] == ["llamacpp-plant/qwen3.8-27b"]
    assert out.per_model_costs[0]["reasoning_percent"] == 33.33


def test_build_canonical_metrics_adds_session_aggregate_and_revert_warnings() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "providerID": "openai",
                "modelID": "gpt-5.3-codex",
            },
            "parts": [
                {
                    "type": "step-finish",
                    "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                    "cost": 0.1,
                }
            ],
        }
    ]

    out = build_canonical_metrics(
        "s-warn",
        messages,
        session_info={
            "data": {
                "tokens": {"input": 12, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.2,
                "revert": {"messageID": "msg_123"},
            }
        },
    )

    assert any("session aggregate mismatch" in warning for warning in out.warnings)
    assert any("active revert" in warning for warning in out.warnings)


def test_component_family_rows_aggregate_by_group() -> None:
    component_rows = [
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 100, "estimated_session_tokens": 100, "calls": 5},
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 200, "estimated_session_tokens": 200, "calls": 3},
        {"component_type": "tool", "component_group": "jcodemunch", "tokens": 50, "estimated_session_tokens": 50, "calls": 2},
        {"component_type": "skill", "component_group": "caveman", "tokens": 30, "estimated_session_tokens": 30, "calls": 0},
    ]
    family = _build_component_family_rows(component_rows)

    assert len(family) == 3
    assert family[0]["component_group"] == "lean-ctx"
    assert family[0]["tokens"] == 300
    assert family[0]["estimated_session_tokens"] == 300
    assert family[0]["calls"] == 8
    assert family[1]["component_group"] == "jcodemunch"
    assert family[1]["tokens"] == 50
    assert family[1]["calls"] == 2
    assert family[2]["component_group"] == "caveman"
    assert family[2]["component_type"] == "skill"


def test_component_family_rows_in_canonical_metrics() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {"type": "tool", "tool": "lean-ctx_ctx_read", "state": {"status": "completed", "output": "a"}},
                {"type": "tool", "tool": "lean-ctx_ctx_search", "state": {"status": "completed", "output": "b"}},
            ],
        }
    ]
    out = build_canonical_metrics("s-family", messages)

    lean_ctx_family = [r for r in out.component_family_rows if r["component_group"] == "lean-ctx"]
    assert len(lean_ctx_family) == 1
    assert lean_ctx_family[0]["calls"] == 2
    assert lean_ctx_family[0]["tokens"] > 0


def test_skill_call_attribution() -> None:
    """Test that skill tool calls are attributed to the correct skill component."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "caveman"},
                        "output": "skill loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "impeccable"},
                        "output": "skill loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-skill", messages)

    # Skill calls should be attributed with component_type "skill"
    skill_rows = [r for r in out.component_rows if r["component_type"] == "skill"]
    skill_names = {r["component_name"] for r in skill_rows}
    assert "caveman" in skill_names
    assert "impeccable" in skill_names

    # Tool calls should be attributed with component_type "tool"
    tool_rows = [r for r in out.component_rows if r["component_type"] == "tool"]
    assert any(r["component_name"] == "lean-ctx_ctx_read" for r in tool_rows)


def test_skill_call_grouping_in_family() -> None:
    """Test that skill calls are grouped with tools in the same family."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "svelte"},
                        "output": "svelte skill",
                    },
                },
                {
                    "type": "tool",
                    "tool": "svelte_list-sections",
                    "state": {"status": "completed", "output": "sections"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-svelte", messages)

    # Both skill and tool should be grouped under "svelte"
    svelte_family = [r for r in out.component_family_rows if r["component_group"] == "svelte"]
    assert len(svelte_family) == 1
    assert svelte_family[0]["calls"] == 2


def test_component_family_mixed_type_when_group_has_multiple_types() -> None:
    """Test that a group with multiple types shows 'mixed' instead of a single type."""
    component_rows = [
        {"component_type": "tool", "component_group": "svelte", "tokens": 100, "estimated_session_tokens": 100, "calls": 2},
        {"component_type": "skill", "component_group": "svelte", "tokens": 50, "estimated_session_tokens": 50, "calls": 1},
    ]
    family = _build_component_family_rows(component_rows)

    assert len(family) == 1
    assert family[0]["component_group"] == "svelte"
    assert family[0]["component_type"] == "mixed"
    assert family[0]["tokens"] == 150


def test_component_family_single_type_preserved() -> None:
    """Test that a group with a single type keeps that type."""
    component_rows = [
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 100, "estimated_session_tokens": 100, "calls": 3},
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 50, "estimated_session_tokens": 50, "calls": 2},
    ]
    family = _build_component_family_rows(component_rows)

    assert len(family) == 1
    assert family[0]["component_group"] == "lean-ctx"
    assert family[0]["component_type"] == "tool"
    assert family[0]["tokens"] == 150


def test_skill_family_preserves_single_hyphenated_skill_name() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "implement-slice"},
                        "output": "skill loaded",
                    },
                },
            ],
        }
    ]

    out = build_canonical_metrics("s-skill-hyphen-single", messages)

    family_names = {r["component_group"] for r in out.component_family_rows}
    assert "implement-slice" in family_names
    assert "implement" not in family_names


def test_skill_family_merges_multiple_hyphenated_skills_with_same_prefix() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "implement-slice"},
                        "output": "slice loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "implement-plan"},
                        "output": "plan loaded",
                    },
                },
            ],
        }
    ]

    out = build_canonical_metrics("s-skill-hyphen-multi", messages)

    implement_family = [r for r in out.component_family_rows if r["component_group"] == "implement"]
    assert len(implement_family) == 1
    assert implement_family[0]["calls"] == 2


def test_skill_calls_excluded_from_mcp_servers() -> None:
    """Test that skill calls are not included in MCP Servers."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "caveman"},
                        "output": "skill loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-mcp", messages)

    # MCP rows should only contain lean-ctx, not caveman skill
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "lean-ctx" in mcp_names
    assert "caveman" not in mcp_names


def test_subagent_call_attribution() -> None:
    """Test that task tool calls are attributed to the correct subagent component."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "task",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "explore", "prompt": "find files"},
                        "output": "subagent result",
                    },
                },
                {
                    "type": "tool",
                    "tool": "task",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "general", "prompt": "research"},
                        "output": "general result",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-subagent", messages)

    # explore and general are core subagents, attributed with component_type "core"
    core_rows = [r for r in out.component_rows if r["component_type"] == "core"]
    core_names = {r["component_name"] for r in core_rows}
    assert "explore" in core_names
    assert "general" in core_names

    # Tool calls should be attributed with component_type "tool"
    tool_rows = [r for r in out.component_rows if r["component_type"] == "tool"]
    assert any(r["component_name"] == "lean-ctx_ctx_read" for r in tool_rows)


def test_subagent_calls_excluded_from_mcp_servers() -> None:
    """Test that subagent calls are not included in MCP Servers."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "task",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "explore", "prompt": "find files"},
                        "output": "subagent result",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-mcp-subagent", messages)
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "lean-ctx" in mcp_names
    assert "explore" not in mcp_names


def test_edit_tool_excluded_from_mcp_servers() -> None:
    """Test that the 'edit' core tool is not included in MCP Servers."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "edit",
                    "state": {
                        "status": "completed",
                        "output": "file modified",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-edit", messages)
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "edit" not in mcp_names
    assert "lean-ctx" in mcp_names
    core_names = {r["component_name"] for r in out.core_rows}
    assert "edit" in core_names


def test_question_tool_excluded_from_mcp_servers() -> None:
    """Test that the 'question' core tool is not included in MCP Servers."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "question",
                    "state": {
                        "status": "completed",
                        "output": "user answered",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-question", messages)
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "question" not in mcp_names
    assert "lean-ctx" in mcp_names
    core_names = {r["component_name"] for r in out.core_rows}
    assert "question" in core_names


def test_compress_tool_excluded_from_mcp_servers() -> None:
    """Test that the 'compress' core tool is not included in MCP Servers."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "compress",
                    "state": {
                        "status": "completed",
                        "output": "context compressed",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-compress", messages)
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "compress" not in mcp_names
    assert "lean-ctx" in mcp_names
    core_names = {r["component_name"] for r in out.core_rows}
    assert "compress" in core_names


def test_write_tool_excluded_from_mcp_servers() -> None:
    """Test that the 'write' core tool is not included in MCP Servers."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "write",
                    "state": {
                        "status": "completed",
                        "output": "file written",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-write", messages)
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "write" not in mcp_names
    assert "lean-ctx" in mcp_names
    core_names = {r["component_name"] for r in out.core_rows}
    assert "write" in core_names


def test_additional_core_components_classification() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": """
<available_skills>
  <skill><name>plan</name><description>Planner skill.</description></skill>
  <skill><name>implement</name><description>Implementor skill.</description></skill>
</available_skills>
""",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {"type": "tool", "tool": "webfetch", "state": {"status": "completed", "output": "html"}},
                {"type": "tool", "tool": "invalid", "state": {"status": "completed", "output": "oops"}},
            ],
        }
    ]
    out = build_canonical_metrics("s-core-extra", messages)

    core_names = {r["component_name"] for r in out.core_rows}
    assert "webfetch" in core_names
    assert "general" in core_names
    assert "invalid" not in core_names
    assert "plan" in core_names
    assert "implement" in core_names

    family = [r for r in out.component_family_rows if r["component_group"] == "opencode-core"]
    assert len(family) == 1
    assert family[0]["component_type"] == "core"

    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "webfetch" not in mcp_names
    assert "invalid" not in mcp_names


def _two_period_history_lookup():
    from opencode_tokenstats.pricing import PricingLookup, _parse_pricing_history

    payload = {
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
        ]
    }
    data, history = _parse_pricing_history(payload)
    return PricingLookup(data, history, flat_keys=frozenset())


def test_estimated_cost_uses_call_time_pricing_periods(monkeypatch) -> None:
    import pytest

    from opencode_tokenstats import canonical_metrics

    monkeypatch.setattr(canonical_metrics, "build_default_pricing_lookup", _two_period_history_lookup)

    def step_finish(timestamp: int) -> dict[str, object]:
        return {
            "type": "step-finish",
            "tokens": {"input": 1000, "output": 500, "reasoning": 0, "cache": {"read": 0, "write": 0}},
            "cost": 0,
            "timestamp": timestamp,
        }

    messages = [
        {
            "role": "assistant",
            "info": {"providerID": "openai", "modelID": "gpt-x"},
            "parts": [step_finish(1772668800000)],  # 2026-03-05, first period
        },
        {
            "role": "assistant",
            "info": {"providerID": "openai", "modelID": "gpt-x"},
            "parts": [step_finish(1783209600000)],  # 2026-07-05, second period
        },
    ]

    out = build_canonical_metrics("s-periods", messages)
    # First period: (1000*1 + 500*2) / 1M = 0.002; second: (1000*4 + 500*8) / 1M = 0.008
    assert out.estimated_cost_usd == pytest.approx(0.01)
    row = out.per_model_costs[0]
    assert row["model"] == "openai/gpt-x"
    assert row["priced_calls"] == 2
    assert row["unpriced_calls"] == 0
    assert row["pricing_provenance"] == "https://example.com/old; https://example.com/new"
    assert out.pricing_coverage["calls"] == 2
    assert out.pricing_coverage["priced_calls"] == 2
    assert out.pricing_coverage["coverage_percent"] == 100.0


def test_unknown_model_usage_uses_default_fallback() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-zz-never-seen",
                "tokens": {"input": 100, "output": 50, "reasoning": 5, "cache": {"read": 10, "write": 0}},
                "cost": 0,
            },
            "parts": [{"type": "text", "text": "ok"}],
        }
    ]

    out = build_canonical_metrics("s-unpriced", messages)
    assert out.actual_cost_usd == 0.0
    assert out.estimated_cost_usd == 0.000265
    row = out.per_model_costs[0]
    assert row["priced_calls"] == 1
    assert row["default_fallback_calls"] == 1
    assert row["unpriced_calls"] == 0
    assert row["pricing_provenance"] == "default"
    assert not any("unpriced" in warning for warning in out.warnings)
    assert out.pricing_coverage == {
        "calls": 1,
        "priced_calls": 1,
        "future_fallback_calls": 0,
        "default_fallback_calls": 1,
        "unpriced_calls": 0,
        "coverage_percent": 100.0,
        "tier_applied_calls": 0,
        "base_rate_calls": 1,
        "tier_unknown_calls": 0,
    }


def test_canonical_metrics_uses_nonlocal_models_dev_market_estimate(monkeypatch) -> None:
    from opencode_tokenstats.pricing import ModelPricing, PricingLookup, PricingRecord

    record = PricingRecord(
        provider="openai", model="gpt-5", service_profile="standard", context="short",
        effective_from="2027-01-01T00:00:00Z", effective_to=None, status="active",
        confidence="observed", source_url="models.dev", retrieved_at="",
        pricing=ModelPricing(4.0, 8.0, 0.0), aliases=("gpt-5",), source_kind="models.dev",
    )
    lookup = PricingLookup({"default": ModelPricing(1.0, 3.0, 0.0)}, (record,), flat_keys=frozenset())
    monkeypatch.setattr("opencode_tokenstats.canonical_metrics.build_default_pricing_lookup", lambda: lookup)

    message = {
        "role": "assistant",
        "info": {"providerID": "openai", "modelID": "gpt-5"},
        "parts": [{
            "type": "step-finish", "timestamp": 1767225600000, "cost": 2.0,
            "tokens": {"input": 1_000_000, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
        }],
    }
    estimated_message = {
        **message,
        "parts": [{**message["parts"][0], "cost": 0.0}],
    }
    out = build_canonical_metrics("s-models-dev", [message, estimated_message])

    row = out.per_model_costs[0]
    assert out.estimated_cost_usd == pytest.approx(4.0)
    assert out.actual_cost_usd == pytest.approx(2.0)
    assert row["estimated_future_market_cost"] == pytest.approx(4.0)
    assert row["cost"] == pytest.approx(2.0)
    assert row["market_status"] == "future"
