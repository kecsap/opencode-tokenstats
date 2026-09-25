from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
from pathlib import Path
import sys
from datetime import UTC, datetime
from click.testing import CliRunner
import pytest

from opencode_tokenstats import cli
from opencode_tokenstats import pricing
from opencode_tokenstats.canonical_metrics import CanonicalMetrics


@pytest.fixture(autouse=True)
def _mock_local_period_messages(monkeypatch):
    """Keep CLI unit tests independent from the developer's local OpenCode DB."""
    def get_period_messages(_service, _start, _end, *, session_ids=None):
        sessions = cli._list_sessions({})
        return {
            str(session["id"]): cli._get_messages({}, str(session["id"]))
            for session in sessions
            if isinstance(session.get("id"), str)
            and (session_ids is None or str(session["id"]) in session_ids)
        }

    monkeypatch.setattr(cli.LocalSessionService, "get_period_messages", get_period_messages)


def _sessions():
    return [
        {"id": "s1", "time_created": 1_700_000_000_000},
        {"id": "s2", "time_created": 1_700_010_000_000},
    ]


def _messages(_sid: str):
    return [
        {
            "role": "assistant",
            "info": {
                "tokens": {
                    "input": 10,
                    "output": 5,
                    "reasoning": 1,
                    "cache": {"read": 2, "write": 3},
                },
                "cost": 0.01,
            },
        }
    ]


def test_status_command(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    runner = CliRunner()
    result = runner.invoke(cli.main, ["status"])
    assert result.exit_code == 0
    assert "Status" in result.output
    assert "Sessions" in result.output
    assert "2" in result.output


def test_session_command(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["session", "--session-id", "s1"])
    assert result.exit_code == 0
    assert "Session" in result.output
    assert "s1" in result.output
    assert "API calls" in result.output


def test_json_command(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["json", "--period", "month"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["overview"]["period"] == "month"
    assert "tokens" in payload
    assert "tools" in payload


def test_json_command_markdown(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["json", "--period", "daily", "--format", "md"])
    assert result.exit_code == 0
    assert "## Overview" in result.output
    assert "## Top Tools" in result.output


def test_json_command_exposes_pricing_coverage(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["json", "--period", "daily"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    # No model ids in fixture: every call uses legacy default fallback.
    assert payload["pricing"]["calls"] == 2
    assert payload["pricing"]["priced_calls"] == 2
    assert payload["pricing"]["default_fallback_calls"] == 2
    assert payload["pricing"]["unpriced_calls"] == 0
    assert payload["pricing"]["coverage_percent"] == 100.0


def test_json_command_preserves_matched_session_count_after_filter(monkeypatch) -> None:
    sessions = [
        {"id": "s1", "directory": "/tmp/alpha"},
        {"id": "s2", "directory": "/tmp/beta"},
    ]
    metrics = cli._PeriodSessionMetrics()
    for session_id in ("s1", "s2"):
        metrics.append(
            CanonicalMetrics(
                session_id=session_id, model="unknown", input_tokens=0, output_tokens=0,
                reasoning_tokens=0, cache_read_tokens=0, session_total_tokens=0,
                api_calls=0, actual_cost_usd=0.0, estimated_cost_usd=0.0,
                token_composition={}, component_rows=[], component_family_rows=[],
                core_rows=[], tool_rows=[], mcp_rows=[], per_model_costs=[],
            )
        )
    metrics.matched_session_count = 2
    monkeypatch.setattr(cli, "_list_sessions", lambda _options: sessions)
    monkeypatch.setattr(cli, "_collect_period_session_metrics", lambda *args, **kwargs: metrics)

    result = CliRunner().invoke(cli.main, ["--no-warmup", "--session-filter", "alpha", "json", "--period", "daily"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["overview"]["sessions"] == 2
    assert payload["period_series"][0]["sessions"] == 2


def test_json_command_preserves_cost_basis_amounts_and_metadata(monkeypatch) -> None:
    rows = [
        {"model": "shared", "tokens": 1, "estimated_cost": 1.0, "estimated_direct_cost": 1.0, "direct_estimate_calls": 1, "cost_basis": "direct"},
        {"model": "shared", "tokens": 1, "estimated_cost": 2.0, "estimated_generic_cost": 2.0, "generic_estimate_calls": 1, "cost_basis": "generic"},
        {"model": "shared", "tokens": 1, "estimated_cost": 3.0, "estimated_market_cost": 3.0, "market_estimate_calls": 1, "market_provider_count": 2, "market_rate_date": "2026-01-01", "market_status": "active", "cost_basis": "market"},
        {"model": "shared", "tokens": 1, "estimated_cost": 4.0, "estimated_future_market_cost": 4.0, "future_market_estimate_calls": 1, "market_provider_count": 3, "market_rate_date": "2027-01-01", "market_status": "future", "cost_basis": "market"},
        {"model": "shared", "tokens": 1, "estimated_cost": 5.0, "estimated_cloud_equivalent_cost": 5.0, "cloud_equivalent_estimate_calls": 1, "cost_basis": "cloud_equivalent"},
    ]
    metric = CanonicalMetrics(
        session_id="s-bases", model="shared", input_tokens=5, output_tokens=0, reasoning_tokens=0,
        cache_read_tokens=0, session_total_tokens=5, api_calls=5, actual_cost_usd=0.0,
        estimated_cost_usd=15.0, token_composition={}, component_rows=[], component_family_rows=[],
        core_rows=[], tool_rows=[], mcp_rows=[], per_model_costs=rows,
        pricing_coverage={"calls": 5, "priced_calls": 5},
    )
    monkeypatch.setattr(cli, "_collect_period_session_metrics", lambda *args, **kwargs: [metric])
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())

    result = CliRunner().invoke(cli.main, ["json", "--period", "daily"])
    assert result.exit_code == 0
    model = json.loads(result.output)["models"][0]
    for field, value in {
        "estimated_direct_cost": 1.0,
        "estimated_generic_cost": 2.0,
        "estimated_market_cost": 3.0,
        "estimated_future_market_cost": 4.0,
        "estimated_cloud_equivalent_cost": 5.0,
        "direct_estimate_calls": 1,
        "generic_estimate_calls": 1,
        "market_estimate_calls": 1,
        "future_market_estimate_calls": 1,
        "cloud_equivalent_estimate_calls": 1,
    }.items():
        assert model[field] == value
    assert model["market_provider_count"] == 3
    assert model["market_rate_date"] == "2026-01-01; 2027-01-01"
    assert model["market_status"] == "active; future"
    assert model["cost_basis"] == "direct; generic; market; cloud_equivalent"


def test_negative_model_filter_rejects_assistant_without_model_identity() -> None:
    assert not cli._session_matches_model_filter(
        [{"role": "assistant", "info": {}}],
        {"model_filter": ("!denied",)},
        aliases={},
    )


def test_negative_model_filter_rejects_provider_only_telemetry() -> None:
    assert not cli._session_matches_model_filter(
        [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "step-finish",
                        "providerID": "openai",
                        "tokens": {"input": 1, "output": 1},
                    }
                ],
            }
        ],
        {"model_filter": ("!denied",)},
        aliases={},
    )


def test_session_command_shows_pricing_coverage_and_warnings(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["session", "--session-id", "s1"])
    assert result.exit_code == 0
    assert "Pricing" in result.output
    assert "unpriced" in result.output


def test_period_report_shows_pricing_coverage_and_warnings(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["daily"])
    assert result.exit_code == 0
    assert "Pricing" in result.output
    assert "unpriced" in result.output


def test_model_cost_alias_aggregation_preserves_market_metadata() -> None:
    rows = cli._accumulate_model_cost_rows(
        [
            {
                "model": "provider/active",
                "tokens": 10,
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
                "estimated_cost": 0.02,
                "estimated_future_market_cost": 0.02,
                "market_provider_count": 3,
                "market_rate_date": "2026-02-01",
                "market_status": "future",
                "cost_basis": "market",
            },
        ],
        {"provider/active": "shared", "provider/future": "shared"},
    )

    model = cli._finalize_model_costs(rows)[0]
    assert model["model"] == "shared"
    assert model["market_provider_count"] == 3
    assert model["market_rate_date"] == "2026-01-01; 2026-02-01"
    assert model["market_status"] == "active; future"
    assert model["cost_basis"] == "market"


def test_local_collection_reports_discovery_and_processing_stages(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    updates = []

    metrics = cli._collect_period_session_metrics(
        {"mode": "local", "db_path": None, "model_alias_file": None, "session_filter": None, "model_filter": None},
        datetime.fromtimestamp(1_699_000_000, UTC),
        datetime.fromtimestamp(1_701_000_000, UTC),
        progress_callback=lambda *update: updates.append(update),
    )

    assert len(metrics) == 2
    assert updates[:3] == [
        (0, 0, "Finding in-period sessions"),
        (0, 0, "Retrieving period messages"),
        (0, 2, "Processing session metrics"),
    ]
    assert updates[-1] == (2, 2)


def test_api_collection_reports_fetch_and_processing_progress(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    updates = []

    metrics = cli._collect_period_session_metrics(
        {"mode": "api", "model_alias_file": None, "session_filter": None, "model_filter": None},
        datetime.fromtimestamp(1_699_000_000, UTC),
        datetime.fromtimestamp(1_701_000_000, UTC),
        progress_callback=lambda *update: updates.append(update),
    )

    assert len(metrics) == 2
    assert updates[0] == (0, 0, "Finding in-period sessions")
    assert (0, 2, "Retrieving session messages") in updates
    assert (1, 2) in updates
    assert (2, 2) in updates
    assert (0, 2, "Processing session metrics") in updates
    assert updates[-1] == (2, 2)


def test_period_session_count_uses_model_identity_without_telemetry(monkeypatch) -> None:
    sessions = [
        {"id": "s1", "time_created": 1_700_000_000_000, "directory": "/tmp/alpha"},
        {"id": "s2", "time_created": 1_700_000_000_000, "directory": "/tmp/beta"},
        {"id": "s3", "time_created": 1_700_000_000_000, "directory": "/tmp/alpha"},
    ]
    messages = {
        "s1": [{"role": "assistant", "info": {"providerID": "openai", "modelID": "allowed"}}],
        "s2": [{"role": "assistant", "info": {"providerID": "openai", "modelID": "denied"}}],
        "s3": [
            {"role": "assistant", "info": {"providerID": "openai", "modelID": "denied"}},
            {"role": "assistant", "info": {"providerID": "openai", "modelID": "allowed"}},
        ],
    }
    monkeypatch.setattr(cli, "_list_sessions", lambda _options: sessions)
    monkeypatch.setattr(cli.LocalSessionService, "find_database_path", lambda _path: None)
    monkeypatch.setattr(
        cli.LocalSessionService,
        "get_period_messages_bucketed",
        lambda self, *_args, **_kwargs: messages,
    )
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 1)

    options = {
        "mode": "local",
        "db_path": None,
        "model_alias_file": None,
        "session_filter": ("alpha",),
        "model_filter": ("openai/allowed",),
    }
    metrics = cli._collect_period_session_metrics(
        options,
        datetime.fromtimestamp(1_699_000_000, UTC),
        datetime.fromtimestamp(1_701_000_000, UTC),
    )

    assert metrics.matched_session_count == 2
    assert {metric.session_id for metric in metrics} == {"s1", "s3"}

    options["model_filter"] = None
    metrics = cli._collect_period_session_metrics(
        options,
        datetime.fromtimestamp(1_699_000_000, UTC),
        datetime.fromtimestamp(1_701_000_000, UTC),
    )
    assert metrics.matched_session_count == 2


@pytest.mark.skipif(sys.platform != "linux", reason="fork preload coverage is Linux-specific")
def test_period_collection_preloads_pricing_before_fork_and_reuses_lookup(monkeypatch, tmp_path) -> None:
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text('{"default": {"input": 1, "output": 3}}', encoding="utf-8")
    monkeypatch.setenv("OPENCODE_MODEL_PRICING_FILE", str(pricing_file))
    pricing.reset_pricing_lookup_cache()

    parse_count = mp.Value("i", 0)
    original_parse = pricing._parse_flat_pricing

    def counted_parse(payload):
        with parse_count.get_lock():
            parse_count.value += 1
        return original_parse(payload)

    monkeypatch.setattr(pricing, "_parse_flat_pricing", counted_parse)
    sessions = _sessions()
    monkeypatch.setattr(cli, "_list_sessions", lambda _options: sessions)
    monkeypatch.setattr(cli.LocalSessionService, "find_database_path", lambda _path: tmp_path / "db.sqlite")
    monkeypatch.setattr(
        cli.LocalSessionService,
        "get_period_messages",
        lambda _service, _start, _end, *, session_ids=None: {
            session["id"]: _messages(session["id"])
            for session in sessions
            if session_ids is None or session["id"] in session_ids
        },
    )
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 2)

    metrics = cli._collect_period_session_metrics(
        {"mode": "local", "db_path": None, "model_alias_file": None, "session_filter": None, "model_filter": None},
        datetime.fromtimestamp(1_699_000_000, UTC),
        datetime.fromtimestamp(1_701_000_000, UTC),
    )

    assert parse_count.value == 1
    assert sorted(metric.session_id for metric in metrics) == ["s1", "s2"]
    assert [
        (
            metric.api_calls,
            metric.input_tokens,
            metric.output_tokens,
            metric.reasoning_tokens,
            metric.cache_read_tokens,
            metric.actual_cost_usd,
            metric.pricing_coverage["calls"],
        )
        for metric in sorted(metrics, key=lambda item: item.session_id)
    ] == [(1, 10, 5, 1, 2, 0.01, 1)] * 2


@pytest.mark.skipif(sys.platform != "linux", reason="fork alias handoff is Linux-specific")
def test_period_collection_fork_workers_receive_component_aliases(monkeypatch, tmp_path) -> None:
    alias_file = tmp_path / "components.conf"
    alias_file.write_text("deep-code-review = deep\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_list_sessions", lambda _options: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _options, _sid: _deep_alias_messages(_sid))
    monkeypatch.setattr(cli.LocalSessionService, "find_database_path", lambda _path: tmp_path / "db.sqlite")
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 2)

    options = {
        "mode": "local",
        "db_path": None,
        "model_alias_file": None,
        "session_filter": None,
        "model_filter": None,
        "component_alias_file": str(alias_file),
    }
    metrics = cli._collect_period_session_metrics(
        options,
        datetime.fromtimestamp(1_699_000_000, UTC),
        datetime.fromtimestamp(1_701_000_000, UTC),
    )

    assert sorted(metric.session_id for metric in metrics) == ["s1", "s2"]
    for metric in metrics:
        groups = {row["component_group"]: row for row in metric.component_family_rows}
        assert "deep-code-review" in groups
        assert "deep" not in groups
        assert groups["deep-code-review"]["calls"] == 1


def _deep_alias_messages(_sid: str):
    return [
        {
            "role": "assistant",
            "info": {
                "tokens": {"input": 10, "output": 5, "reasoning": 1, "cache": {"read": 2, "write": 3}},
                "cost": 0.01,
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {"type": "tool", "tool": "deep", "state": {"status": "completed", "output": "review"}},
            ],
        }
    ]


def test_lifetime_command(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["lifetime"])
    assert result.exit_code == 0
    assert "Period Summary" in result.output
    assert "lifetime" in result.output


def test_json_lifetime_period(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["json", "--period", "lifetime"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["overview"]["period"] == "lifetime"


def test_month_command_default(monkeypatch) -> None:
    """month command without argument shows last 30 days."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["month"])
    assert result.exit_code == 0
    assert "Period Summary" in result.output
    assert "month" in result.output


def test_month_command_with_name(monkeypatch) -> None:
    """month command with month name shows that specific month."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["month", "may"])
    assert result.exit_code == 0
    assert "Period Summary" in result.output
    assert "month" in result.output


def test_month_command_with_number(monkeypatch) -> None:
    """month command with month number shows that specific month."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["month", "05"])
    assert result.exit_code == 0
    assert "Period Summary" in result.output
    assert "month" in result.output


def test_month_command_invalid_name(monkeypatch) -> None:
    """month command with invalid month name fails."""
    runner = CliRunner()
    result = runner.invoke(cli.main, ["month", "foobar"])
    assert result.exit_code != 0
    assert "Invalid month" in result.output


def test_month_command_invalid_number(monkeypatch) -> None:
    """month command with invalid month number fails."""
    runner = CliRunner()
    result = runner.invoke(cli.main, ["month", "13"])
    assert result.exit_code != 0
    assert "Invalid month" in result.output


def test_json_command_includes_by_activity_and_top_sessions(monkeypatch) -> None:
    """JSON output includes by_activity and top_sessions fields."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["json", "--period", "daily"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "by_activity" in payload
    assert "top_sessions" in payload
    # by_activity is a list
    assert isinstance(payload["by_activity"], list)
    # top_sessions is a list
    assert isinstance(payload["top_sessions"], list)


def _sessions_with_dirs():
    """Return sessions with directory fields for session filter testing."""
    from datetime import datetime, UTC
    now = datetime.now(UTC)
    base = int(now.timestamp() * 1000)
    return [
        {"id": "s1", "time_created": base - 3600000, "directory": "/home/user/project-alpha"},
        {"id": "s2", "time_created": base - 7200000, "directory": "/home/user/project-beta"},
        {"id": "s3", "time_created": base - 10800000, "directory": "/home/user/project-alpha"},
    ]


def _messages_with_cost(_sid: str):
    """Return messages with cost info for session filter testing."""
    return [
        {
            "role": "assistant",
            "info": {
                "tokens": {
                    "input": 10,
                    "output": 5,
                    "reasoning": 1,
                    "cache": {"read": 2, "write": 3},
                },
                "cost": 0.01,
            },
        }
    ]


def test_session_filter_single_match(monkeypatch) -> None:
    """Session filter with single value matches only matching sessions."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["-sf", "project-alpha", "json", "--period", "daily"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    # Only sessions from project-alpha should be included (s1 and s3)
    assert payload["overview"]["sessions"] == 2
    # top_sessions lists individual sessions (not aggregated by root_dir in json schema)
    assert len(payload["top_sessions"]) == 2
    assert all(s["root_dir"] == "project-alpha" for s in payload["top_sessions"])


def test_session_filter_multiple_values(monkeypatch) -> None:
    """Session filter with comma-separated values matches multiple projects."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["-sf", "project-alpha,project-beta", "json", "--period", "daily"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    # All sessions should be included
    assert payload["overview"]["sessions"] == 3
    assert len(payload["top_sessions"]) == 3


def test_session_filter_no_match(monkeypatch) -> None:
    """Session filter with no matching sessions returns empty report."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["-sf", "nonexistent", "json", "--period", "daily"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["overview"]["sessions"] == 0
    assert len(payload["top_sessions"]) == 0


def test_session_filter_short_flag(monkeypatch) -> None:
    """Session filter with short flag -sf works."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["-sf", "project-beta", "json", "--period", "daily"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["overview"]["sessions"] == 1
    assert len(payload["top_sessions"]) == 1
    assert payload["top_sessions"][0]["root_dir"] == "project-beta"


def test_session_filter_does_not_affect_session_command(monkeypatch) -> None:
    """Session filter should not affect the session command (single session lookup)."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    # Even with filter for project-alpha, session s2 (project-beta) should still be accessible
    result = runner.invoke(cli.main, ["-sf", "project-alpha", "session", "--session-id", "s2"])
    assert result.exit_code == 0
    assert "s2" in result.output


def test_session_filter_in_period_report(monkeypatch) -> None:
    """Session filter applies to period report (daily command)."""
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["-sf", "project-alpha", "daily"])
    assert result.exit_code == 0
    assert "Period Summary" in result.output
    # Should only show 2 sessions (s1 and s3 from project-alpha)
    assert "2" in result.output


def test_export_session_list_daily(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["-esl", "sessions.txt", "daily"])
        assert result.exit_code == 0
        content = open("sessions.txt", "r", encoding="utf-8").read().strip().splitlines()
        assert set(content) == {"s1", "s2", "s3"}


def test_export_session_list_with_filter(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["-esl", "sessions.txt", "-sf", "project-alpha", "daily"])
        assert result.exit_code == 0
        content = open("sessions.txt", "r", encoding="utf-8").read().strip().splitlines()
        assert set(content) == {"s1", "s3"}


def test_session_output_dir_exports_transcripts(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["-o", "transcripts", "daily"])
        assert result.exit_code == 0
        files = sorted(p.name for p in Path("transcripts").glob("*.txt"))
        assert files == ["s1.txt", "s2.txt", "s3.txt"]
        content = open("transcripts/s1.txt", "r", encoding="utf-8").read()
        assert "# session: s1" in content


def test_export_session_list_fails_when_parent_dir_missing(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions_with_dirs())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages_with_cost(_sid))
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["-esl", "missing/sessions.txt", "daily"])
        assert result.exit_code != 0
        assert "parent directory does not exist" in result.output


def test_include_in_top_tools_excludes_core_skill_subagent() -> None:
    assert cli._include_in_top_tools({"is_core": True, "is_skill": False, "is_subagent": False}) is False
    assert cli._include_in_top_tools({"is_core": False, "is_skill": True, "is_subagent": False}) is False
    assert cli._include_in_top_tools({"is_core": False, "is_skill": False, "is_subagent": True}) is False
    assert cli._include_in_top_tools({"is_core": False, "is_skill": False, "is_subagent": False}) is True


def test_max_ext_tools_option_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_list_sessions", lambda _opts: _sessions())
    monkeypatch.setattr(cli, "_get_messages", lambda _opts, _sid: _messages(_sid))
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--max-ext-tools", "20", "json", "--period", "daily"])
    assert result.exit_code == 0


def test_max_ext_tools_helper_defaults_to_24() -> None:
    assert cli._max_ext_tools({}) == 24


def test_local_query_workers_option_is_parsed() -> None:
    assert cli._parse_local_query_workers("auto") == "auto"
    assert cli._parse_local_query_workers("6") == 6
    assert cli._local_query_worker_count({"local_query_workers": 6}, 2000) == 6
    assert cli._local_query_worker_count({"local_query_workers": "auto"}, 100) == 1
    assert cli._local_query_worker_count({"local_query_workers": "auto"}, 2000) == 4


def test_local_query_benchmark_reports_strategies_without_writes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli.LocalSessionService, "find_database_path", lambda _path: tmp_path / "db.sqlite")
    monkeypatch.setattr(cli.LocalSessionService, "list_sessions", lambda _service: _sessions())
    monkeypatch.setattr(cli, "_list_sessions", lambda _options: _sessions())
    monkeypatch.setattr(
        cli.LocalSessionService,
        "get_period_messages",
        lambda _service, _start, _end, *, session_ids=None: {
            sid: [] for sid in (session_ids or set())
        },
    )
    before = sorted(path.name for path in tmp_path.iterdir())
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--no-warmup", "local-query-benchmark", "--period", "daily"])
    assert result.exit_code == 0
    assert all(f"workers={workers}" in result.output for workers in (1, 2, 4, 8))
    assert "recommendation:" in result.output
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_local_query_benchmark_live_fixture_does_not_mutate_database(monkeypatch, tmp_path) -> None:
    monkeypatch.undo()
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE session (id TEXT PRIMARY KEY, title TEXT, parent_id TEXT, time_created INTEGER, directory TEXT);
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT, time_created INTEGER);
        """
    )
    session_ids = [f"s{index:04d}" for index in range(901)]
    conn.executemany(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?)",
        [(sid, sid, None, 100, "/repo") for sid in session_ids],
    )
    conn.executemany(
        "INSERT INTO message VALUES (?, ?, ?, ?)",
        [(f"m{sid}", sid, json.dumps({"role": "assistant"}), 150) for sid in session_ids],
    )
    conn.executemany(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
        [(f"p{sid}", f"m{sid}", sid, json.dumps({"type": "tool"}), 150) for sid in session_ids],
    )
    conn.commit()
    conn.close()
    before = db.read_bytes()

    result = CliRunner().invoke(
        cli.main,
        ["--no-warmup", "--db-path", str(db), "local-query-benchmark", "--period", "lifetime"],
    )

    assert result.exit_code == 0
    assert all(f"workers={workers}" in result.output for workers in (1, 2, 4, 8))
    assert "recommendation:" in result.output
    assert db.read_bytes() == before


def test_period_component_stats_merge_aliased_groups_across_sessions() -> None:
    # Mirrors _build_period_report's component_map keys (type|group|name),
    # built from aliased canonical rows of two different sessions.
    component_map = {
        "tool|make-suite|make_test": {"tokens": 4.0, "calls": 1.0},
        "skill|make-suite|make-plan": {"tokens": 6.0, "calls": 1.0},
        "skill|make-suite|make-implement": {"tokens": 2.0, "calls": 1.0},
        "tool|lean-ctx|lean-ctx_ctx_search": {"tokens": 3.0, "calls": 1.0},
    }
    stats = cli._finalize_component_stats_canonical(component_map)
    rows = {r["component_group"]: r for r in stats["rows"]}
    merged = rows["make-suite"]
    assert merged["component_type"] == "mixed"
    assert merged["tokens"] == 12
    assert merged["calls"] == 3
    assert rows["lean-ctx"]["component_type"] == "tool"


def _component_metric(session_id: str, rows: list) -> CanonicalMetrics:
    return CanonicalMetrics(
        session_id=session_id, model="unknown", input_tokens=0, output_tokens=0,
        reasoning_tokens=0, cache_read_tokens=0, session_total_tokens=0,
        api_calls=0, actual_cost_usd=0.0, estimated_cost_usd=0.0,
        token_composition={}, component_rows=rows, component_family_rows=[],
        core_rows=[], tool_rows=[], mcp_rows=[], per_model_costs=[],
    )


def _cross_session_metrics_fixture(monkeypatch, tmp_path, s2_directory: str) -> str:
    rows_a = [
        {"component_type": "subagent", "component_group": "deep", "component_name": "deep-code-reviewer-qwen38", "tokens": 50, "estimated_session_tokens": 50, "calls": 3},
    ]
    rows_b = [
        {"component_type": "skill", "component_group": "deep-code-reviewer", "component_name": "deep-code-reviewer", "tokens": 20, "estimated_session_tokens": 20, "calls": 0},
    ]
    metrics = cli._PeriodSessionMetrics()
    for sid, rows in (("s1", rows_a), ("s2", rows_b)):
        metrics.append(_component_metric(sid, rows))
    metrics.matched_session_count = 2
    monkeypatch.setattr(cli, "_list_sessions", lambda _options: [
        {"id": "s1", "directory": "/tmp/alpha"},
        {"id": "s2", "directory": s2_directory},
    ])
    monkeypatch.setattr(cli, "_collect_period_session_metrics", lambda *args, **kwargs: metrics)
    return str(tmp_path / "no-aliases.conf")


def test_period_report_matches_subagent_to_skill_in_other_session(monkeypatch, tmp_path) -> None:
    alias_file = _cross_session_metrics_fixture(monkeypatch, tmp_path, "/tmp/alpha")
    options = {
        "mode": "local",
        "db_path": None,
        "model_alias_file": None,
        "component_alias_file": alias_file,
        "session_filter": None,
        "model_filter": None,
        "export_session_list": None,
        "session_output_dir": None,
    }
    report = cli._build_period_report(
        options,
        datetime.fromtimestamp(1_700_000_000, UTC),
        datetime.fromtimestamp(1_700_100_000, UTC),
    )
    rows = {r["component_group"]: r for r in report["component_stats"]["rows"]}
    merged = rows["deep-code-reviewer"]
    assert merged["component_type"] == "mixed"
    assert merged["tokens"] == 70
    assert merged["calls"] == 3


def test_json_report_matches_subagent_to_skill_in_other_session(monkeypatch, tmp_path) -> None:
    alias_file = _cross_session_metrics_fixture(monkeypatch, tmp_path, "/tmp/alpha")
    result = CliRunner().invoke(
        cli.main,
        ["--no-warmup", "-caf", alias_file, "json", "--period", "daily"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    components = {
        (c["component_type"], c["component_group"], c["component_name"]): c
        for c in payload["context_estimates"]["components"]
    }
    assert ("subagent", "deep-code-reviewer", "deep-code-reviewer-qwen38") in components
    assert ("skill", "deep-code-reviewer", "deep-code-reviewer") in components


def test_json_report_cross_session_matching_respects_roots_and_session_filter(monkeypatch, tmp_path) -> None:
    alias_file = _cross_session_metrics_fixture(monkeypatch, tmp_path, "/tmp/beta")
    result = CliRunner().invoke(
        cli.main,
        ["--no-warmup", "-caf", alias_file, "-sf", "alpha", "json", "--period", "daily"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    components = {
        (c["component_type"], c["component_group"], c["component_name"]): c
        for c in payload["context_estimates"]["components"]
    }
    # s2 is filtered out (and in another root), so the subagent stays unmatched.
    subagent_row = components[("subagent", "deep", "deep-code-reviewer-qwen38")]
    assert subagent_row["estimated_session_tokens"] == 50
    assert not any(group == "deep-code-reviewer" for (_, group, _name) in components)


def _aliased_skill_cross_session_fixture(monkeypatch, tmp_path) -> str:
    # The skill row already carries its canonical group, as
    # build_canonical_metrics applies aliases before cross-session matching.
    rows_a = [
        {"component_type": "subagent", "component_group": "make", "component_name": "make-code-fixes-coder", "tokens": 50, "estimated_session_tokens": 50, "calls": 3},
    ]
    rows_b = [
        {"component_type": "skill", "component_group": "make-code-changes", "component_name": "make-code-fixes", "tokens": 20, "estimated_session_tokens": 20, "calls": 0},
    ]
    alias_file = tmp_path / "components.conf"
    alias_file.write_text("make-code-changes = make-code-fixes\n", encoding="utf-8")
    metrics = cli._PeriodSessionMetrics()
    metrics.append(_component_metric("s1", rows_a))
    metrics.append(_component_metric("s2", rows_b))
    metrics.matched_session_count = 2
    monkeypatch.setattr(cli, "_list_sessions", lambda _options: [
        {"id": "s1", "directory": "/tmp/alpha"},
        {"id": "s2", "directory": "/tmp/alpha"},
    ])
    monkeypatch.setattr(cli, "_collect_period_session_metrics", lambda *args, **kwargs: metrics)
    return str(alias_file)


def test_period_report_propagates_aliased_skill_group_to_subagent(monkeypatch, tmp_path) -> None:
    alias_file = _aliased_skill_cross_session_fixture(monkeypatch, tmp_path)
    options = {
        "mode": "local",
        "db_path": None,
        "model_alias_file": None,
        "component_alias_file": alias_file,
        "session_filter": None,
        "model_filter": None,
        "export_session_list": None,
        "session_output_dir": None,
    }
    report = cli._build_period_report(
        options,
        datetime.fromtimestamp(1_700_000_000, UTC),
        datetime.fromtimestamp(1_700_100_000, UTC),
    )
    rows = {r["component_group"]: r for r in report["component_stats"]["rows"]}
    merged = rows["make-code-changes"]
    assert merged["component_type"] == "mixed"
    assert merged["tokens"] == 70
    assert merged["calls"] == 3


def test_json_report_propagates_aliased_skill_group_to_subagent(monkeypatch, tmp_path) -> None:
    alias_file = _aliased_skill_cross_session_fixture(monkeypatch, tmp_path)
    result = CliRunner().invoke(
        cli.main,
        ["--no-warmup", "-caf", alias_file, "json", "--period", "daily"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    components = {
        (c["component_type"], c["component_group"], c["component_name"]): c
        for c in payload["context_estimates"]["components"]
    }
    assert ("subagent", "make-code-changes", "make-code-fixes-coder") in components
    assert ("skill", "make-code-changes", "make-code-fixes") in components
