from __future__ import annotations

from click.testing import CliRunner

from opencode_tokenstats import cli


def test_main_help_lists_core_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["health", "session", "status", "daily", "weekly", "month", "lifetime", "range", "json", "tokenizer-warmup"]:
        assert cmd in result.output


def test_main_help_shows_priority_command_order() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0

    daily_idx = result.output.index("daily")
    weekly_idx = result.output.index("weekly")
    month_idx = result.output.index("month")
    range_idx = result.output.index("range")
    lifetime_idx = result.output.index("lifetime")
    health_idx = result.output.index("health")

    assert daily_idx < weekly_idx < month_idx < range_idx < lifetime_idx < health_idx


def test_health_help_lists_new_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["health", "--help"])
    assert result.exit_code == 0
    assert "--check-tokenizer" in result.output
    assert "--compat-mode" in result.output
    assert "--compat-source" in result.output
    assert "--compat-session-id" in result.output


def test_json_help_lists_format_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["json", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
    assert "lifetime" in result.output


def test_range_help_has_examples() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["range", "--help"])
    assert result.exit_code == 0
    assert "2026-05-01" in result.output
    assert "2026-05-07" in result.output
