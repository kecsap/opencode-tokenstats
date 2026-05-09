from __future__ import annotations

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
        contributor_rows=[{"name": "lean-ctx_ctx_search", "tokens": 4, "percent": 100.0}],
        tool_rows=[{"tool": "lean-ctx_ctx_search", "tokens": 4, "percent": 100.0, "calls": 2}],
        mcp_rows=[{"name": "lean", "tokens": 4, "calls": 2, "tokens_per_call": 2.0, "percent": 100.0}],
    )


def test_build_report_schema_blocks() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    report = build_report_schema(period="daily", mode="local", start=start, end=end, session_metrics=[_metric()])
    for key in [
        "overview",
        "tokens",
        "tools",
        "contributors",
        "skills",
        "subagents",
        "context_estimates",
        "warnings",
        "period_series",
        "projects",
        "models",
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


def test_model_costs_separate_api_and_estimated() -> None:
    """Test that API costs and estimated costs are tracked separately."""
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
    assert model["estimated_cost"] == 0.02
    # Primary cost should be API cost when available
    assert model["cost"] == 0.01


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
        contributor_rows=metric.contributor_rows,
        tool_rows=metric.tool_rows,
        mcp_rows=metric.mcp_rows,
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
    # Primary cost should be estimated when API cost is 0
    assert model["cost"] == 0.05
