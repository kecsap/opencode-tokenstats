from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from opencode_tokenstats.canonical_metrics import build_canonical_metrics
from opencode_tokenstats.report_schema import build_report_schema


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_multi_provider_fixture_coverage() -> None:
    m1 = build_canonical_metrics("s-openai", _load("session_messages_api_sample.json"))
    m2 = build_canonical_metrics("s-anthropic", _load("session_messages_api_sample_anthropic.json"))
    m3 = build_canonical_metrics("s-openai-zero", _load("session_messages_api_sample_openai_zero_cost.json"))

    assert m1.model == "openai/gpt-5.3-codex"
    assert m2.model == "anthropic/claude-sonnet-4"
    assert m3.model == "openai/gpt-5.3-codex"

    assert any(r["component_type"] == "skill" for r in m2.component_rows)
    assert any(r["component_type"] == "subagent" for r in m2.component_rows)

    assert m3.actual_cost_usd == 0.0
    assert m3.estimated_cost_usd > 0.0


def test_period_aggregation_two_sessions() -> None:
    metrics = [
        build_canonical_metrics("s1", _load("session_messages_api_sample.json")),
        build_canonical_metrics("s2", _load("session_messages_api_sample_anthropic.json")),
    ]
    report = build_report_schema(
        period="weekly",
        mode="local",
        start=datetime(2026, 5, 1, tzinfo=UTC),
        end=datetime(2026, 5, 8, tzinfo=UTC),
        session_metrics=metrics,
    )

    assert report["overview"]["sessions"] == 2
    assert report["overview"]["api_calls"] == 2
    assert len(report["models"]) >= 2
    assert len(report["tools"]) >= 2


def test_mcp_grouping_fixture_coverage() -> None:
    metric = build_canonical_metrics("s-anthropic", _load("session_messages_api_sample_anthropic.json"))
    mcp_names = [row["name"] for row in metric.mcp_rows]
    # codegraphcontext_analyze_code_relationships -> codegraphcontext
    assert "codegraphcontext" in mcp_names
