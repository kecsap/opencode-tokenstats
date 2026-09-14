from __future__ import annotations

import json
from pathlib import Path
from click.testing import CliRunner
import pytest

from opencode_tokenstats import cli


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
