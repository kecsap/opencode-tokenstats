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
