from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from rich.console import Console, Group
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    RICH_AVAILABLE = False

# Codeburn dashboard palette
COL_ORANGE = "#FF8C42"
COL_BLUE = "#5B9EF5"
COL_GREEN = "#5BF5A0"
COL_RED = "#F55B5B"
COL_PURPLE = "#E05BF5"
COL_YELLOW = "#F5C85B"
COL_CYAN = "#5BF5E0"
COL_MAGENTA = "#F55BE0"
COL_GOLD = "#FFD700"
COL_DIM = "#555555"
COL_BAR_EMPTY = "#333333"

COL_INPUT = COL_BLUE
COL_OUTPUT = COL_GREEN
COL_REASONING = COL_PURPLE
COL_CACHE_READ = COL_YELLOW
COL_TOTAL = COL_ORANGE


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return str(value)


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ts_local(value: Any) -> str:
    if not isinstance(value, str):
        return str(value)
    raw = value.strip()
    if not raw:
        return raw
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _bar(value: int, max_value: int, width: int = 12) -> str:
    """Render a horizontal bar using block characters."""
    if max_value == 0:
        return "\u2591" * width  # ░ (light shade)
    filled = max(1, round(value / max_value * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _to_hex(r: float, g: float, b: float) -> str:
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _gradient_color(pct: float) -> str:
    if pct <= 0.33:
        t = pct / 0.33 if pct > 0 else 0.0
        return _to_hex(_lerp(91, 245, t), _lerp(158, 200, t), _lerp(245, 91, t))
    if pct <= 0.66:
        t = (pct - 0.33) / 0.33
        return _to_hex(_lerp(245, 255, t), _lerp(200, 140, t), _lerp(91, 66, t))
    t = (pct - 0.66) / 0.34
    return _to_hex(_lerp(255, 245, t), _lerp(140, 91, t), _lerp(66, 91, t))


def _color_bar(value: int, max_value: int, color: str, width: int = 12) -> Text:
    """Render a colored horizontal bar."""
    if max_value == 0:
        return Text("\u2591" * width, style=COL_DIM)
    filled = max(1, round(value / max_value * width))
    bar = Text()
    for i in range(min(filled, width)):
        pct = i / max(width, 1)
        block_color = _gradient_color(pct)
        bar.append("\u2588", style=f"bold {block_color}")
    bar.append("\u2591" * (width - filled), style=COL_BAR_EMPTY)
    return bar


def _build_composition_table(token_composition: dict[str, int], total_tokens: int) -> Table:
    """Build a Token Composition table with bars and colors."""
    comp = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
    comp.add_column("", style="bold")
    comp.add_column("", justify="left")
    comp.add_column("Tokens", justify="right", style="dim")
    comp.add_column("%", justify="right", style="dim")

    # Color mapping for components
    color_map = {
        "input": COL_INPUT,
        "cache_read": COL_CACHE_READ,
        "output": COL_OUTPUT,
        "reasoning": COL_REASONING,
    }

    excluded = {"cache_write", "web_search_requests"}
    displayed = {k: v for k, v in token_composition.items() if k not in excluded}
    max_val = max(displayed.values()) if displayed else 1

    for key, value in displayed.items():
        if key in excluded:
            continue
        color = color_map.get(key, COL_TOTAL)
        bar_text = _color_bar(value, max_val, color, width=10)
        pct = value / total_tokens * 100 if total_tokens else 0
        comp.add_row(key, bar_text, _fmt_int(value), f"{pct:.1f}")

    # Add total row (no bar)
    comp.add_row("", "", "", "")
    comp.add_row("total", "", _fmt_int(total_tokens), "100.0")

    return comp


def print_status_report(mode: str, sessions: list[dict[str, object]]) -> None:
    latest = sessions[0].get("id") if sessions else "-"
    if not RICH_AVAILABLE:
        print(f"Mode: {mode}")
        print(f"Sessions: {_fmt_int(len(sessions))}")
        print(f"Latest Session: {latest}")
        return

    console = Console()
    table = Table(show_header=False, box=None)
    table.add_row("Mode", str(mode))
    table.add_row("Sessions", _fmt_int(len(sessions)))
    table.add_row("Latest Session", str(latest))
    console.print(Panel(table, title="Status", border_style=COL_CYAN))


def print_session_report(
    session_id: str,
    api_calls: int,
    tokens: int,
    api_cost: float,
    *,
    token_composition: dict[str, int] | None = None,
    top_tools: list[dict[str, Any]] | None = None,
    mcp_stats: dict[str, Any] | None = None,
    component_stats: dict[str, Any] | None = None,
    contributor_stats: dict[str, Any] | None = None,
    model_costs: list[dict[str, Any]] | None = None,
) -> None:
    if not RICH_AVAILABLE:
        print(f"Session: {session_id}")
        print(f"API calls: {_fmt_int(api_calls)}")
        print(f"Tokens: {_fmt_int(tokens)}")
        print(f"Cost (API): {_fmt_float(api_cost)}")
        if token_composition:
            print(f"Token Composition: {token_composition}")
        if top_tools:
            print(f"Top Tools: {top_tools}")
        if model_costs:
            print(f"Model Costs: {model_costs}")
        if mcp_stats:
            print(f"MCP Stats: {mcp_stats}")
        if component_stats:
            print(f"Components: {component_stats}")
        if contributor_stats:
            print(f"Contributors: {contributor_stats}")
        return

    console = Console()
    table = Table(show_header=False, box=None)
    table.add_row("Session", session_id)
    table.add_row("API calls", _fmt_int(api_calls))
    table.add_row("Tokens", _fmt_int(tokens))
    table.add_row("Cost (API)", _fmt_float(api_cost))
    console.print(Panel(table, title="Session", border_style=COL_GREEN))

    if token_composition:
        comp = _build_composition_table(token_composition, tokens)
        console.print(Panel(comp, title="Token Composition", border_style=COL_BLUE))

    if top_tools:
        tt = Table(show_header=True, box=None)
        tt.add_column("Tool")
        tt.add_column("Output Tokens", justify="right")
        tt.add_column("Calls", justify="right")
        for item in top_tools:
            tt.add_row(str(item["name"]), _fmt_int(item["output_tokens"]), _fmt_int(item["call_count"]))
        console.print(Panel(tt, title="Top Tools", border_style=COL_GOLD))

    if model_costs:
        mt = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        mt.add_column("", style="bold", justify="left")
        mt.add_column("", justify="left")
        mt.add_column("API", justify="right")
        mt.add_column("Estimated", justify="right")
        max_api_cost = max((float(item.get("api_cost", 0)) for item in model_costs), default=1) or 1
        for item in model_costs[:12]:
            api_cost = float(item.get("api_cost", 0))
            bar_text = _color_bar(api_cost, max_api_cost, COL_GREEN, width=10)
            mt.add_row(
                str(item.get("model")),
                bar_text,
                _fmt_float(api_cost),
                _fmt_float(item.get("estimated_cost")),
            )
        console.print(Panel(mt, title="Model Costs", border_style=COL_GREEN))

    if mcp_stats and mcp_stats.get("rows"):
        mcp = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        mcp.add_column("", style="bold")
        mcp.add_column("", justify="left")
        mcp.add_column("Tokens", justify="right")
        mcp.add_column("%", justify="right")
        mcp.add_column("Calls", justify="right")
        mcp.add_column("Tok/Call", justify="right")
        mcp_filtered = [r for r in mcp_stats["rows"] if r.get("name") not in {"grep", "invalid", "webfetch"}]
        max_tokens = max((r.get("tokens", 0) for r in mcp_filtered), default=1) or 1
        for row in mcp_filtered:
            tokens = int(row.get("tokens", 0))
            bar_text = _color_bar(tokens, max_tokens, COL_CYAN, width=10)
            mcp.add_row(
                str(row.get("name")),
                bar_text,
                _fmt_int(tokens),
                _fmt_float(row.get("percent")),
                _fmt_int(row.get("calls")),
                _fmt_float(row.get("tokens_per_call")),
            )
        console.print(Panel(mcp, title="MCP Insights", border_style=COL_CYAN))

    if component_stats and component_stats.get("rows"):
        ct = Table(show_header=True, box=None)
        ct.add_column("Type")
        ct.add_column("Group")
        ct.add_column("Name")
        ct.add_column("Tokens", justify="right")
        ct.add_column("Est.Session", justify="right")
        ct.add_column("Calls", justify="right")
        ct.add_column("%", justify="right")
        for row in component_stats["rows"]:
            ct.add_row(
                str(row.get("component_type")),
                str(row.get("component_group")),
                str(row.get("component_name")),
                _fmt_int(row.get("tokens")),
                _fmt_int(row.get("estimated_session_tokens")),
                _fmt_int(row.get("calls", 0)),
                _fmt_float(row.get("percent")),
            )
        console.print(Panel(ct, title="Component Contribution", border_style=COL_MAGENTA))

    if contributor_stats and contributor_stats.get("rows"):
        cc = Table(show_header=True, box=None)
        cc.add_column("Contributor")
        cc.add_column("Tokens", justify="right")
        cc.add_column("%", justify="right")
        for row in contributor_stats["rows"]:
            cc.add_row(str(row.get("name")), _fmt_int(row.get("tokens")), _fmt_float(row.get("percent")))
        console.print(Panel(cc, title="Top Contributors", border_style=COL_BLUE))


def print_period_report(label: str, report: dict[str, Any]) -> None:
    if not RICH_AVAILABLE:
        print(f"Period: {label}")
        print(f"Sessions: {_fmt_int(report['sessions'])}")
        print(f"API calls: {_fmt_int(report['api_calls'])}")
        print(f"Tokens: {_fmt_int(report['tokens'])}")
        print(f"From: {_fmt_ts_local(report.get('from'))}")
        print(f"To: {_fmt_ts_local(report.get('to'))}")
        if report.get("token_composition"):
            print(f"Token Composition: {report['token_composition']}")
        if report.get("top_tools"):
            print(f"Top Tools: {report['top_tools']}")
        if report.get("model_costs"):
            print(f"Model Costs: {report['model_costs']}")
        if report.get("mcp_stats"):
            print(f"MCP Stats: {report['mcp_stats']}")
        if report.get("component_stats"):
            print(f"Components: {report['component_stats']}")
        if report.get("contributor_stats"):
            print(f"Contributors: {report['contributor_stats']}")
        return

    console = Console()
    total_tokens = report.get("tokens", 0)
    token_composition = report.get("token_composition")

    # Build Period Summary table (without Tokens - moved to composition)
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column(style="bold", justify="left")
    summary_table.add_column(justify="left")
    summary_table.add_row("Window", label)
    summary_table.add_row("Sessions", _fmt_int(report["sessions"]))
    summary_table.add_row("API calls", _fmt_int(report["api_calls"]))
    summary_table.add_row("From", _fmt_ts_local(report["from"]))
    summary_table.add_row("To", _fmt_ts_local(report["to"]))

    # Build Token Composition table with bars
    if token_composition and isinstance(token_composition, dict):
        comp_table = _build_composition_table(token_composition, total_tokens)
    else:
        comp_table = Table(show_header=False, box=None)
        comp_table.add_row("No token composition data")

    # Build Model Costs panel
    model_costs = report.get("model_costs")
    if isinstance(model_costs, list) and model_costs:
        mt = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        mt.add_column("", style="bold", justify="left")
        mt.add_column("", justify="left")
        mt.add_column("API", justify="right")
        mt.add_column("Estimated", justify="right")
        max_api_cost = max((float(item.get("api_cost", 0)) for item in model_costs), default=1) or 1
        for item in model_costs[:12]:
            api_cost = float(item.get("api_cost", 0))
            bar_text = _color_bar(api_cost, max_api_cost, COL_GREEN, width=10)
            mt.add_row(
                str(item.get("model")),
                bar_text,
                _fmt_float(api_cost),
                _fmt_float(item.get("estimated_cost")),
            )
        model_costs_panel = Panel(mt, title="Model Costs", border_style=COL_GREEN)
    else:
        model_costs_panel = None

    # Layout: left column = Period Summary + Token Composition (stacked), right column = Model Costs
    summary_panel = Panel(summary_table, title="Period Summary", border_style=COL_MAGENTA)
    comp_panel = Panel(comp_table, title="Token Composition", border_style=COL_BLUE)
    left_group = Group(summary_panel, comp_panel)
    if model_costs_panel:
        console.print(Columns([left_group, model_costs_panel], equal=False, padding=0))
    else:
        console.print(left_group)

    top_tools = report.get("top_tools")
    mcp_stats = report.get("mcp_stats")
    component_stats = report.get("component_stats")

    # Build panels for MCP Insights, Component Contribution, Top Tools
    panels = []

    if isinstance(mcp_stats, dict) and mcp_stats.get("rows"):
        mcp = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        mcp.add_column("", style="bold")
        mcp.add_column("", justify="left")
        mcp.add_column("Tokens", justify="right")
        mcp.add_column("%", justify="right")
        mcp.add_column("Calls", justify="right")
        mcp.add_column("Tok/Call", justify="right")
        mcp_filtered = [r for r in mcp_stats["rows"] if r.get("name") not in {"grep", "invalid", "webfetch"}]
        max_tokens = max((r.get("tokens", 0) for r in mcp_filtered), default=1) or 1
        for row in mcp_filtered:
            tokens = int(row.get("tokens", 0))
            bar_text = _color_bar(tokens, max_tokens, COL_CYAN, width=10)
            mcp.add_row(
                str(row.get("name")),
                bar_text,
                _fmt_int(tokens),
                _fmt_float(row.get("percent")),
                _fmt_int(row.get("calls")),
                _fmt_float(row.get("tokens_per_call")),
            )
        panels.append(Panel(mcp, title="MCP Insights", border_style=COL_CYAN))

    if isinstance(component_stats, dict) and component_stats.get("rows"):
        ct = Table(show_header=True, box=None)
        ct.add_column("Type")
        ct.add_column("Group")
        ct.add_column("Name")
        ct.add_column("Tokens", justify="right")
        ct.add_column("Est.Session", justify="right")
        ct.add_column("Calls", justify="right")
        ct.add_column("%", justify="right")
        for row in component_stats["rows"]:
            ct.add_row(
                str(row.get("component_type")),
                str(row.get("component_group")),
                str(row.get("component_name")),
                _fmt_int(row.get("tokens")),
                _fmt_int(row.get("estimated_session_tokens")),
                _fmt_int(row.get("calls", 0)),
                _fmt_float(row.get("percent")),
            )
        panels.append(Panel(ct, title="Component Contribution", border_style=COL_MAGENTA))

    if isinstance(top_tools, list) and top_tools:
        tt = Table(show_header=True, box=None)
        tt.add_column("Tool")
        tt.add_column("Output Tokens", justify="right")
        tt.add_column("Calls", justify="right")
        for item in top_tools:
            tt.add_row(str(item.get("name")), _fmt_int(item.get("output_tokens")), _fmt_int(item.get("call_count")))
        panels.append(Panel(tt, title="Top Tools", border_style=COL_GOLD))

    # Print all three panels in one row
    if panels:
        console.print(Columns(panels, equal=False, padding=0))

    contributor_stats = report.get("contributor_stats")
    if isinstance(contributor_stats, dict) and contributor_stats.get("rows"):
        cc = Table(show_header=True, box=None)
        cc.add_column("Contributor")
        cc.add_column("Tokens", justify="right")
        cc.add_column("%", justify="right")
        for row in contributor_stats["rows"]:
            cc.add_row(str(row.get("name")), _fmt_int(row.get("tokens")), _fmt_float(row.get("percent")))
        console.print(Panel(cc, title="Top Contributors", border_style=COL_BLUE))
