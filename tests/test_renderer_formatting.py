from __future__ import annotations

from opencode_tokenstats import renderer


def test_trend_ratio_footer_uses_delta_loc(monkeypatch) -> None:
    summaries = []
    monkeypatch.setattr(renderer, "_build_trend_chart", lambda *args: summaries.append(args[3]) or None)
    monkeypatch.setattr(renderer, "Columns", lambda items, **_kwargs: items)

    class Console:
        width = 80

        def print(self, _value) -> None:
            pass

    renderer._print_trends(
        Console(),
        {
            "points": [{"date": "2026-01-01T00:00:00+00:00"}],
            "category_tokens_per_loc": {
                "input_tokens": 1,
                "cache_read_tokens": 2,
                "output_tokens": 3,
            },
            "outliers": [],
        },
    )

    assert summaries[3:6] == [
        "period 1 tok/ΔLOC",
        "period 2 tok/ΔLOC",
        "period 3 tok/ΔLOC",
    ]


def test_trend_sessions_chart_matches_lines_chart(monkeypatch) -> None:
    charts = []
    renderables = []

    def build_chart(*args):
        charts.append(args)
        return renderer.Panel(args[0], title=args[0])

    monkeypatch.setattr(renderer, "_build_trend_chart", build_chart)

    class Console:
        width = 80

        def print(self, value) -> None:
            renderables.append(value)

    renderer._print_trends(
        Console(),
        {
            "points": [
                {"date": "2026-01-01T00:00:00+00:00", "churn_loc": 3, "sessions": 2},
                {"date": "2026-01-02T00:00:00+00:00", "churn_loc": 5, "sessions": 4},
                {"date": "2026-01-03T00:00:00+00:00", "churn_loc": 7, "sessions": 6},
            ],
            "included_sessions": 12,
            "churn_loc": 15,
            "added_loc": 0,
            "deleted_loc": 0,
            "net_loc": 0,
            "outliers": [
                {
                    "date": "2026-01-02T00:00:00+00:00",
                    "end_date": "2026-01-02T00:00:00+00:00",
                    "churn_loc": 10,
                    "added_loc": 10,
                    "deleted_loc": 0,
                    "triggers": {},
                    "commits": [],
                }
            ],
        },
    )

    lines, sessions = charts[-2:]
    assert lines[:4] == ("Lines Changed", [3, 5, 7], renderer.COL_ORANGE, "changed 15  +0  -0  net +0")
    assert sessions[:4] == ("Sessions", [2, 4, 6], renderer.COL_CYAN, "overall 12 sessions")
    assert lines[4:7] == sessions[4:7] == ("Jan 01", "Jan 02", "Jan 03")
    trend_row = renderables[2]
    for row in renderables[:3]:
        assert isinstance(row, renderer.Table)
        assert [column.ratio for column in row.columns] == [1, 1, 1]
        assert len(row.rows) == 1
    assert isinstance(renderables[3], renderer.Panel)
    assert renderables[3].title == "[bold]Chart Outliers[/bold]"


def test_sessions_chart_renders_in_cache_read_column() -> None:
    console = renderer.Console(width=200, record=True, color_system=None, force_terminal=False)
    renderer._print_trends(
        console,
        {
            "points": [
                {"date": "2026-01-01T00:00:00+00:00", "input_tokens": 10,
                 "cache_read_tokens": 12, "output_tokens": 8, "churn_loc": 2, "sessions": 1},
                {"date": "2026-01-02T00:00:00+00:00", "input_tokens": 12,
                 "cache_read_tokens": 14, "output_tokens": 9, "churn_loc": 3, "sessions": 2},
                {"date": "2026-01-03T00:00:00+00:00", "input_tokens": 14,
                 "cache_read_tokens": 16, "output_tokens": 10, "churn_loc": 4, "sessions": 3},
            ],
            "included_sessions": 3,
            "input_tokens": 36,
            "cache_read_tokens": 42,
            "output_tokens": 27,
            "category_tokens_per_loc": {},
            "churn_loc": 9,
            "added_loc": 9,
            "deleted_loc": 0,
            "net_loc": 9,
            "outliers": [],
        },
    )

    rendered_lines = console.export_text().splitlines()

    def panel_column(title: str) -> int:
        return next(line.index("╭") for line in rendered_lines if title in line)

    cache_column = panel_column("Cache Read Tokens")
    assert panel_column("Cache Read Tokens / ΔLOC") == cache_column
    assert panel_column("Sessions") == cache_column


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


def test_model_cost_markers_and_footer(monkeypatch, capsys) -> None:
    monkeypatch.setattr(renderer, "RICH_AVAILABLE", False)
    renderer.print_period_report(
        "daily",
        {
            "sessions": 1,
            "api_calls": 1,
            "tokens": 10,
            "from": "2026-05-01T00:00:00+00:00",
            "to": "2026-05-02T00:00:00+00:00",
            "model_costs": [
                {"model": "generic", "estimated_cost": 1.0, "estimated_generic_cost": 1.0},
                {"model": "future", "estimated_cost": 2.0, "estimated_future_market_cost": 2.0},
                {"model": "active", "estimated_cost": 4.0, "estimated_market_cost": 4.0, "market_provider_count": 2, "market_status": "active"},
                {"model": "mixed", "estimated_cost": 3.0, "estimated_generic_cost": 1.0},
            ],
        },
    )
    out = capsys.readouterr().out
    assert "1.00*" in out
    assert "2.00†" in out
    assert "4.00†" in out
    assert "3.00*" not in out
    assert "* generic fallback estimate   † hosted-market estimation" in out
    assert "counterfactual, not spend" not in out


def test_rich_model_cost_notes_are_final_styled_table_row(monkeypatch) -> None:
    captured = []

    class CaptureConsole:
        def print(self, renderable) -> None:
            captured.append(renderable)

    monkeypatch.setattr(renderer, "RICH_AVAILABLE", True)
    monkeypatch.setattr(renderer, "Console", CaptureConsole)

    renderer.print_session_report(
        "s1",
        api_calls=1,
        tokens=10,
        api_cost=0,
        model_costs=[
            {"model": "generic", "estimated_cost": 1, "estimated_generic_cost": 1},
            {
                "model": "hosted",
                "estimated_cost": 2,
                "estimated_future_market_cost": 2,
            },
        ],
    )

    table = captured[-1].renderable
    note_cells = [column._cells[-1] for column in table.columns]
    note = note_cells[0]
    assert note.plain == "* generic fallback estimate   † hosted-market estimation"
    assert note.style == renderer.COL_AXIS
    assert "bold" not in note.style
    assert all(not cell for cell in note_cells[1:])


def test_rich_model_cost_notes_omitted_without_markers(monkeypatch) -> None:
    captured = []

    class CaptureConsole:
        def print(self, renderable) -> None:
            captured.append(renderable)

    monkeypatch.setattr(renderer, "RICH_AVAILABLE", True)
    monkeypatch.setattr(renderer, "Console", CaptureConsole)

    renderer.print_session_report(
        "s1",
        api_calls=1,
        tokens=10,
        api_cost=0,
        model_costs=[{"model": "direct", "estimated_cost": 1, "estimated_direct_cost": 1}],
    )

    table = captured[-1].renderable
    assert len(table.rows) == 1


def test_rich_model_cost_notes_support_single_markers(monkeypatch) -> None:
    captured = []

    class CaptureConsole:
        def print(self, renderable) -> None:
            captured.append(renderable)

    monkeypatch.setattr(renderer, "RICH_AVAILABLE", True)
    monkeypatch.setattr(renderer, "Console", CaptureConsole)

    for model_cost, expected in [
        ({"estimated_generic_cost": 1}, "* generic fallback estimate"),
        ({"estimated_future_market_cost": 1}, "† hosted-market estimation"),
    ]:
        captured.clear()
        renderer.print_session_report(
            "s1",
            api_calls=1,
            tokens=10,
            api_cost=0,
            model_costs=[{"model": "model", "estimated_cost": 1, **model_cost}],
        )

        table = captured[-1].renderable
        note_cells = [column._cells[-1] for column in table.columns]
        assert note_cells[0].plain == expected
        assert note_cells[0].style == renderer.COL_AXIS
        assert "bold" not in note_cells[0].style
        assert all(not cell for cell in note_cells[1:])


def test_rich_period_model_cost_notes_are_final_table_row(monkeypatch) -> None:
    captured = []

    class CaptureConsole:
        def print(self, renderable) -> None:
            captured.append(renderable)

    monkeypatch.setattr(renderer, "RICH_AVAILABLE", True)
    monkeypatch.setattr(renderer, "Console", CaptureConsole)

    renderer.print_period_report(
        "daily",
        {
            "sessions": 1,
            "api_calls": 1,
            "tokens": 10,
            "from": "2026-05-01T00:00:00+00:00",
            "to": "2026-05-02T00:00:00+00:00",
            "model_costs": [
                {"model": "hosted", "estimated_cost": 1, "estimated_future_market_cost": 1},
            ],
        },
    )

    model_costs_panel = captured[0].renderables[1]
    assert isinstance(model_costs_panel, renderer.Panel)
    table = model_costs_panel.renderable
    note_cells = [column._cells[-1] for column in table.columns]
    assert note_cells[0].plain == "† hosted-market estimation"
    assert note_cells[0].style == renderer.COL_AXIS
    assert "bold" not in note_cells[0].style
    assert all(not cell for cell in note_cells[1:])
    assert not any(
        getattr(renderable, "plain", "") == "† hosted-market estimation"
        for renderable in captured
    )


def test_model_cost_currency_display_preserves_json_values(monkeypatch, capsys) -> None:
    monkeypatch.setattr(renderer, "RICH_AVAILABLE", False)
    renderer.print_period_report(
        "daily",
        {
            "sessions": 1,
            "api_calls": 1,
            "tokens": 1,
            "from": "2026-05-01T00:00:00+00:00",
            "to": "2026-05-02T00:00:00+00:00",
            "model_costs": [
                {"model": "zero", "api_cost": 0.0, "estimated_cost": 0.0},
                {"model": "small", "api_cost": 0.001, "estimated_cost": 0.001},
            ],
        },
    )
    out = capsys.readouterr().out
    assert "'api_cost': '-'" in out
    assert "'estimated_cost': '-'" in out
    assert "'api_cost': '<0.01'" in out
    assert "'estimated_cost': '<0.01'" in out


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
    assert "Activity by Turn" in out
    assert "Top Sessions" in out
