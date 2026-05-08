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
        token_composition={"input": 10, "output": 5, "reasoning": 1, "cache_read": 2, "tool_output": 4},
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
