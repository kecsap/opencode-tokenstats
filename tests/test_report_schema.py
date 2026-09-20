from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from opencode_tokenstats.canonical_metrics import CanonicalMetrics
from opencode_tokenstats.report_schema import build_report_schema, report_to_markdown


def _metric() -> CanonicalMetrics:
    return CanonicalMetrics(
        session_id="s1",
        model="gpt-5.3-codex",
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=1,
        cache_read_tokens=2,
        session_total_tokens=19,
        api_calls=1,
        actual_cost_usd=0.01,
        estimated_cost_usd=0.02,
        token_composition={"input": 10, "cache_read": 2, "output": 5, "reasoning": 1},
        component_rows=[
            {
                "component_type": "tool",
                "component_group": "lean",
                "component_name": "lean-ctx_ctx_search",
                "tokens": 4,
                "estimated_session_tokens": 4,
                "calls": 2,
            }
        ],
        component_family_rows=[
            {
                "component_type": "tool",
                "component_group": "lean",
                "tokens": 4,
                "estimated_session_tokens": 4,
                "calls": 2,
                "percent": 100.0,
            }
        ],
        core_rows=[],
        tool_rows=[{"tool": "lean-ctx_ctx_search", "tokens": 4, "percent": 100.0, "calls": 2}],
        mcp_rows=[{"name": "lean", "tokens": 4, "calls": 2, "tokens_per_call": 2.0, "percent": 100.0}],
        per_model_costs=[{"model": "gpt-5.3-codex", "tokens": 19, "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 1, "generated_tokens": 6, "reasoning_percent": 16.67, "api_cost": 0.01, "estimated_cost": 0.0, "cost": 0.01}],
    )


def test_build_report_schema_blocks() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(period="daily", mode="local", start=start, end=end, session_metrics=[_metric()])
    for key in [
        "overview",
        "tokens",
        "tools",
        "skills",
        "subagents",
        "context_estimates",
        "warnings",
        "pricing",
        "period_series",
        "projects",
        "models",
        "trends",
    ]:
        assert key in report


def test_report_to_markdown() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(period="daily", mode="local", start=start, end=end, session_metrics=[_metric()])
    md = report_to_markdown(report)
    assert "## Overview" in md
    assert "## Top Tools" in md
    assert "## Top Models" in md


def test_model_costs_use_api_when_model_row_has_trusted_billed_cost() -> None:
    """Model row uses API cost when billed calls are attributed to that model."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(
        period="daily", mode="local", start=start, end=end, session_metrics=[_metric()]
    )
    models = report["models"]
    assert len(models) == 1
    model = models[0]
    assert model["model"] == "gpt-5.3-codex"
    assert model["api_cost"] == 0.01
    assert model["estimated_cost"] == 0.0
    assert model["cost"] == 0.01
    assert model["tokens"] == 19
    assert model["input_percent"] == 52.63
    assert model["output_percent"] == 26.32
    assert model["reasoning_tokens"] == 1
    assert model["reasoning_percent"] == 16.67


def test_reasoning_stats_are_in_activity_and_session_rows() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(period="daily", mode="local", start=start, end=end, session_metrics=[_metric()])

    activity = report["by_activity"][0]
    session = report["top_sessions"][0]
    assert activity["reasoning_tokens"] == 1
    assert activity["input_percent"] == 52.63
    assert activity["output_percent"] == 26.32
    assert activity["reasoning_percent"] == 16.67
    assert session["reasoning_tokens"] == 1
    assert session["input_percent"] == 52.63
    assert session["output_percent"] == 26.32
    assert session["reasoning_percent"] == 16.67


def test_activity_rows_are_aggregated_by_turn_category() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    metric = replace(
        _metric(),
        api_calls=2,
        activity_rows=[
            {"category": "exploration", "tokens": 7, "input_tokens": 5, "output_tokens": 2, "reasoning_tokens": 0, "generated_tokens": 2, "calls": 1, "api_cost": 0.004, "estimated_cost": 0.0},
            {"category": "feature", "tokens": 12, "input_tokens": 5, "output_tokens": 3, "reasoning_tokens": 1, "generated_tokens": 4, "calls": 1, "api_cost": 0.006, "estimated_cost": 0.0},
        ],
    )

    report = build_report_schema(period="daily", mode="local", start=start, end=end, session_metrics=[metric])

    assert {row["category"] for row in report["by_activity"]} == {"exploration", "feature"}
    assert sum(row["tokens"] for row in report["by_activity"]) == 19
    assert sum(row["calls"] for row in report["by_activity"]) == 2


def test_model_costs_uses_estimated_when_no_api() -> None:
    """Test that estimated cost is used as primary when API cost is 0."""
    metric = _metric()
    metric_zero = CanonicalMetrics(
        session_id="s2",
        model="gpt-5.3-codex",
        input_tokens=metric.input_tokens,
        output_tokens=metric.output_tokens,
        reasoning_tokens=metric.reasoning_tokens,
        cache_read_tokens=metric.cache_read_tokens,
        session_total_tokens=metric.session_total_tokens,
        api_calls=metric.api_calls,
        actual_cost_usd=0.0,  # No API cost
        estimated_cost_usd=0.05,
        token_composition=metric.token_composition,
        component_rows=metric.component_rows,
        component_family_rows=metric.component_family_rows,
        core_rows=metric.core_rows,
        tool_rows=metric.tool_rows,
        mcp_rows=metric.mcp_rows,
        per_model_costs=[{"model": "gpt-5.3-codex", "tokens": 19, "api_cost": 0.0, "estimated_cost": 0.05, "cost": 0.05}],
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(
        period="daily", mode="local", start=start, end=end, session_metrics=[metric_zero]
    )
    models = report["models"]
    assert len(models) == 1
    model = models[0]
    assert model["api_cost"] == 0.0
    assert model["estimated_cost"] == 0.05
    assert model["cost"] == 0.05
    assert model["reasoning_tokens"] == 0


def test_model_costs_merge_same_model_into_one_api_row_when_any_billed_cost_exists() -> None:
    metric_a = _metric()
    metric_b = CanonicalMetrics(
        session_id="s2",
        model="gpt-5.3-codex",
        input_tokens=metric_a.input_tokens,
        output_tokens=metric_a.output_tokens,
        reasoning_tokens=metric_a.reasoning_tokens,
        cache_read_tokens=metric_a.cache_read_tokens,
        session_total_tokens=metric_a.session_total_tokens,
        api_calls=metric_a.api_calls,
        actual_cost_usd=0.0,
        estimated_cost_usd=0.05,
        token_composition=metric_a.token_composition,
        component_rows=metric_a.component_rows,
        component_family_rows=metric_a.component_family_rows,
        core_rows=metric_a.core_rows,
        tool_rows=metric_a.tool_rows,
        mcp_rows=metric_a.mcp_rows,
        per_model_costs=[{"model": "gpt-5.3-codex", "tokens": 19, "api_cost": 0.0, "estimated_cost": 0.05, "cost": 0.05}],
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(
        period="daily", mode="local", start=start, end=end, session_metrics=[metric_a, metric_b]
    )

    models = report["models"]
    assert len(models) == 1
    model = models[0]
    assert model["model"] == "gpt-5.3-codex"
    assert model["api_cost"] == 0.01
    assert model["estimated_cost"] == 0.0
    assert model["cost"] == 0.01
    assert model["tokens"] == 38
    assert model["reasoning_tokens"] == 1
    # reasoning_percent is reasoning / generated (output + reasoning) across the merge:
    # 1 / (5 + 1) = 16.67 (metric_b's row carries no generated tokens)
    assert model["reasoning_percent"] == 16.67


def test_model_costs_use_estimated_for_unbilled_model_rows_in_mixed_sessions(tmp_path, monkeypatch) -> None:
    # Isolate from any developer-local models.conf (alias mapping would rename
    # the model key and change the aggregation under test).
    alias_file = tmp_path / "models.conf"
    alias_file.write_text("# no aliases\n", encoding="utf-8")
    monkeypatch.setenv("OPTOKEN_MODEL_ALIAS_FILE", str(alias_file))

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(
        period="daily",
        mode="local",
        start=start,
        end=end,
        session_metrics=[
            CanonicalMetrics(
                session_id="s-mixed",
                model="openai/gpt-5.4",
                input_tokens=300,
                output_tokens=150,
                reasoning_tokens=10,
                cache_read_tokens=20,
                session_total_tokens=480,
                api_calls=2,
                actual_cost_usd=12.34,
                estimated_cost_usd=0.25,
                token_composition={"input": 300, "cache_read": 20, "output": 150, "reasoning": 10},
                component_rows=[],
                component_family_rows=[],
                core_rows=[],
                tool_rows=[],
                mcp_rows=[],
                per_model_costs=[
                    {"model": "openai/gpt-5.4-mini-fast", "tokens": 180, "api_cost": 0.0, "estimated_cost": 0.08, "cost": 0.08},
                    {"model": "openai/gpt-5.4", "tokens": 300, "api_cost": 12.34, "estimated_cost": 0.0, "cost": 12.34},
                ],
            )
        ],
    )

    models = {row["model"]: row for row in report["models"]}
    assert models["openai/gpt-5.4-mini-fast"]["api_cost"] == 0.0
    assert models["openai/gpt-5.4-mini-fast"]["estimated_cost"] == 0.08
    assert models["openai/gpt-5.4-mini-fast"]["cost"] == 0.08
    assert models["openai/gpt-5.4"]["api_cost"] == 12.34
    assert models["openai/gpt-5.4"]["estimated_cost"] == 0.0
    assert models["openai/gpt-5.4"]["cost"] == 12.34


def test_report_exposes_pricing_coverage_status_and_provenance(tmp_path, monkeypatch) -> None:
    alias_file = tmp_path / "models.conf"
    alias_file.write_text("# no aliases\n", encoding="utf-8")
    monkeypatch.setenv("OPTOKEN_MODEL_ALIAS_FILE", str(alias_file))

    metric = CanonicalMetrics(
        session_id="s-pricing",
        model="openai/gpt-x",
        input_tokens=100,
        output_tokens=50,
        reasoning_tokens=0,
        cache_read_tokens=0,
        session_total_tokens=150,
        api_calls=3,
        actual_cost_usd=0.0,
        estimated_cost_usd=0.005,
        token_composition={"input": 100, "cache_read": 0, "output": 50, "reasoning": 0},
        component_rows=[],
        component_family_rows=[],
        core_rows=[],
        tool_rows=[],
        mcp_rows=[],
        per_model_costs=[
            {
                "model": "openai/gpt-x",
                "tokens": 150,
                "input_tokens": 100,
                "output_tokens": 50,
                "reasoning_tokens": 0,
                "generated_tokens": 50,
                "api_cost": 0.0,
                "estimated_cost": 0.005,
                "cost": 0.005,
                "priced_calls": 2,
                "future_fallback_calls": 1,
                "unpriced_calls": 1,
                "pricing_provenance": "https://example.com/old; https://example.com/new",
            }
        ],
        pricing_coverage={
            "calls": 3,
            "priced_calls": 2,
            "future_fallback_calls": 1,
            "unpriced_calls": 1,
            "coverage_percent": 66.67,
        },
    )

    report = build_report_schema(
        period="daily",
        mode="local",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        session_metrics=[metric],
    )

    assert report["pricing"] == {
        "calls": 3,
        "priced_calls": 2,
        "future_fallback_calls": 1,
        "default_fallback_calls": 0,
            "unpriced_calls": 1,
            "coverage_percent": 66.67,
            "tier_applied_calls": 0,
            "base_rate_calls": 0,
            "tier_unknown_calls": 0,
        }

    model = {row["model"]: row for row in report["models"]}["openai/gpt-x"]
    assert model["priced_calls"] == 2
    assert model["future_fallback_calls"] == 1
    assert model["unpriced_calls"] == 1
    assert model["pricing_provenance"] == "https://example.com/old; https://example.com/new"
    assert model["pricing_status"] == "future_fallback"

    session_row = report["top_sessions"][0]
    assert session_row["pricing_coverage"]["unpriced_calls"] == 1
    assert session_row["pricing_coverage"]["coverage_percent"] == 66.67


def test_report_exposes_per_model_tier_and_context_metadata() -> None:
    metric = replace(
        _metric(),
        per_model_costs=[
            {
                "model": "gpt-5.3-codex",
                "tokens": 19,
                "api_cost": 0.01,
                "estimated_cost": 0.0,
                "cost": 0.01,
                "tier_applied_calls": 2,
                "base_rate_calls": 1,
                "tier_unknown_calls": 1,
                "context_tokens": 200000,
                "context_token_source": "input_plus_cache; incomplete",
            }
        ],
    )
    report = build_report_schema(
        period="daily",
        mode="local",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        session_metrics=[metric],
    )

    model = report["models"][0]
    assert model["tier_applied_calls"] == 2
    assert model["base_rate_calls"] == 1
    assert model["tier_unknown_calls"] == 1
    assert model["context_tokens"] == 200000
    assert model["context_token_source"] == "input_plus_cache; incomplete"


def test_report_aggregates_market_metadata_across_alias_rows(tmp_path) -> None:
    alias_file = tmp_path / "models.conf"
    alias_file.write_text("shared = provider/active provider/future\n", encoding="utf-8")
    metric = replace(
        _metric(),
        actual_cost_usd=0.0,
        estimated_cost_usd=0.03,
        per_model_costs=[
            {
                "model": "provider/active",
                "tokens": 10,
                "input_tokens": 10,
                "estimated_cost": 0.01,
                "estimated_market_cost": 0.01,
                "market_provider_count": 2,
                "market_rate_date": "2026-01-01",
                "market_status": "active",
                "cost_basis": "market",
            },
            {
                "model": "provider/future",
                "tokens": 20,
                "input_tokens": 20,
                "estimated_cost": 0.02,
                "estimated_future_market_cost": 0.02,
                "market_provider_count": 3,
                "market_rate_date": "2026-02-01",
                "market_status": "future",
                "cost_basis": "market",
            },
        ],
    )

    report = build_report_schema(
        period="daily",
        mode="local",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        session_metrics=[metric],
        model_alias_file=str(alias_file),
    )

    model = report["models"][0]
    assert model["model"] == "shared"
    assert model["market_provider_count"] == 3
    assert model["market_rate_date"] == "2026-01-01; 2026-02-01"
    assert model["market_status"] == "active; future"
    assert model["cost_basis"] == "market"
