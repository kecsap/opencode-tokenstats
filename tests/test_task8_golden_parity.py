from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from opencode_tokenstats.canonical_metrics import build_canonical_metrics
from opencode_tokenstats.report_schema import build_report_schema


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_golden_report_schema_parity() -> None:
    messages = _load_json("session_messages_api_sample.json")
    golden = _load_json("golden_report_daily.json")

    metric = build_canonical_metrics("ses_fixture_1", messages)
    report = build_report_schema(
        period="daily",
        mode="local",
        start=datetime(2026, 5, 1, tzinfo=UTC),
        end=datetime(2026, 5, 2, tzinfo=UTC),
        session_metrics=[metric],
    )

    assert report == golden


def test_pipeline_is_not_text_report_dependent() -> None:
    """Regression guard: metrics come from structured messages, not txt parsing."""
    messages = _load_json("session_messages_api_sample.json")
    metric = build_canonical_metrics("ses_fixture_2", messages)

    # If this pipeline depended on TokenScope text formatting, this test would
    # need textual sections. It does not: structured message payload is enough.
    assert metric.api_calls == 1
    assert metric.session_total_tokens == 184
    assert metric.model == "gpt-5.3-codex"
