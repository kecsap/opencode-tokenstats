from __future__ import annotations

from opencode_tokenstats import renderer


def test_period_report_formats_numbers_and_local_timestamps(monkeypatch, capsys) -> None:
    monkeypatch.setattr(renderer, "RICH_AVAILABLE", False)

    renderer.print_period_report(
        "daily",
        {
            "sessions": 1222332,
            "api_calls": 4567,
            "tokens": 1222332,
            "api_cost": 12.3456,
            "from": "2026-05-01T11:57:15.228518+00:00",
            "to": "2026-05-02T11:57:15.228518+00:00",
        },
    )

    out = capsys.readouterr().out
    assert "1 222 332" in out
    assert "T11:57:15" not in out
    assert "+00:00" not in out


def test_session_report_formats_fractions(monkeypatch, capsys) -> None:
    monkeypatch.setattr(renderer, "RICH_AVAILABLE", False)

    renderer.print_session_report(
        "s1",
        api_calls=4,
        tokens=1000,
        api_cost=0.123456,
    )

    out = capsys.readouterr().out


def test_composition_table_tokens_column_no_custom_style() -> None:
    """Token Composition Tokens/% columns must not carry custom style (e.g. dim).

    They should use the default style so terminal colors are respected,
    matching every other numeric column in the dashboard."""
    table = renderer._build_composition_table(
        {"input": 100, "output": 200, "reasoning": 50},
        350,
    )
    columns = table.columns
    tokens_col = columns[2]
    pct_col = columns[3]

    assert not tokens_col.style, "Tokens column must not have a custom style"
    assert not pct_col.style, "% column must not have a custom style"
