from __future__ import annotations

from click.testing import CliRunner

from opencode_tokenstats import cli


def test_main_help_lists_core_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["doctor", "session", "status", "daily", "weekly", "month", "range", "json", "tokenizer-warmup"]:
        assert cmd in result.output


def test_doctor_help_lists_new_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["doctor", "--help"])
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
