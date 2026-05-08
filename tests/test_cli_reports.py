from __future__ import annotations

import json
from click.testing import CliRunner

from opencode_tokenstats import cli


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
    assert "contributors" in payload


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
