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


def test_period_report_includes_by_activity_and_top_sessions(monkeypatch, capsys) -> None:
    """Period report fallback output includes by_activity and top_sessions."""
    monkeypatch.setattr(renderer, "RICH_AVAILABLE", False)

    renderer.print_period_report(
        "daily",
        {
            "sessions": 5,
            "api_calls": 10,
            "tokens": 1000,
            "api_cost": 0.5,
            "from": "2026-05-01T00:00:00+00:00",
            "to": "2026-05-02T00:00:00+00:00",
            "by_activity": [
                {"category": "coding", "label": "Coding", "tokens": 600, "calls": 5, "api_cost": 0.2, "estimated_cost": 0.3},
                {"category": "exploration", "label": "Exploration", "tokens": 400, "calls": 5, "api_cost": 0.1, "estimated_cost": 0.2},
            ],
            "top_sessions": [
                {"root_dir": "eju", "tokens": 500, "api_cost": 0.15, "estimated_cost": 0.25},
                {"root_dir": "other", "tokens": 500, "api_cost": 0.1, "estimated_cost": 0.25},
            ],
        },
    )

    out = capsys.readouterr().out
    assert "Session Categories" in out
    assert "Top Sessions" in out
