from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from rich.console import Console
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    RICH_AVAILABLE = False

# Codeburn-inspired color palette (lighter, visible on dark terminals)
COL_INPUT = "#5B9EF5"    # blue
COL_OUTPUT = "#5BF5A0"   # green
COL_REASONING = "#E05BF5"  # purple
COL_CACHE_READ = "#F5C85B"  # yellow
COL_TOTAL = "#FF8C42"    # orange
COL_BAR_EMPTY = "#333333"  # dim gray


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


def _color_bar(value: int, max_value: int, color: str, width: int = 12) -> Text:
    """Render a colored horizontal bar."""
    if max_value == 0:
        return Text("\u2591" * width, style="dim")
    filled = max(1, round(value / max_value * width))
    filled_chars = "\u2588" * filled
    empty_chars = "\u2591" * (width - filled)
    bar = Text()
    bar.append(filled_chars, style=f"bold {color}")
    bar.append(empty_chars, style="dim")
    return bar


def _build_composition_table(token_composition: dict[str, int], total_tokens: int) -> Table:
    """Build a Token Composition table with bars and colors."""
    comp = Table(show_header=False, box=None, padding=(0, 0, 0, 1))
    comp.add_column("Component", style="bold")
    comp.add_column("Bar", justify="left")
    comp.add_column("Tokens", justify="right", style="dim")

    # Color mapping for components
    color_map = {
        "input": COL_INPUT,
        "cache_read": COL_CACHE_READ,
        "output": COL_OUTPUT,
        "reasoning": COL_REASONING,
    }

    max_val = max(token_composition.values()) if token_composition else 1

    for key, value in token_composition.items():
        color = color_map.get(key, COL_TOTAL)
        bar_text = _color_bar(value, max_val, color, width=10)
        comp.add_row(key, bar_text, _fmt_int(value))

    # Add total row
    comp.add_row("", "", "")
    comp.add_row("Total", _color_bar(total_tokens, total_tokens, COL_TOTAL, width=10), _fmt_int(total_tokens))

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
    console.print(Panel(table, title="Status", border_style="cyan"))


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
    console.print(Panel(table, title="Session", border_style="green"))

    if token_composition:
        comp = _build_composition_table(token_composition, tokens)
        console.print(Panel(comp, title="Token Composition", border_style="blue"))

    if top_tools:
        tt = Table(show_header=True, box=None)
        tt.add_column("Tool")
        tt.add_column("Output Tokens", justify="right")
        tt.add_column("Calls", justify="right")
        for item in top_tools:
            tt.add_row(str(item["name"]), _fmt_int(item["output_tokens"]), _fmt_int(item["call_count"]))
        console.print(Panel(tt, title="Top Tools", border_style="yellow"))

    if model_costs:
        mt = Table(show_header=True, box=None)
        mt.add_column("Model")
        mt.add_column("Cost (API)", justify="right")
        for item in model_costs:
            mt.add_row(str(item.get("model")), _fmt_float(item.get("cost")))
        console.print(Panel(mt, title="Model Costs", border_style="green"))

    if mcp_stats and mcp_stats.get("rows"):
        mcp = Table(show_header=True, box=None)
        mcp.add_column("MCP")
        mcp.add_column("Tokens", justify="right")
        mcp.add_column("Calls", justify="right")
        mcp.add_column("Tok/Call", justify="right")
        mcp.add_column("%", justify="right")
        for row in mcp_stats["rows"]:
            mcp.add_row(
                str(row.get("name")),
                _fmt_int(row.get("tokens")),
                _fmt_int(row.get("calls")),
                _fmt_float(row.get("tokens_per_call")),
                _fmt_float(row.get("percent")),
            )
        console.print(Panel(mcp, title="MCP Insights", border_style="cyan"))

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
        console.print(Panel(ct, title="Component Contribution", border_style="magenta"))

    if contributor_stats and contributor_stats.get("rows"):
        cc = Table(show_header=True, box=None)
        cc.add_column("Contributor")
        cc.add_column("Tokens", justify="right")
        cc.add_column("%", justify="right")
        for row in contributor_stats["rows"]:
            cc.add_row(str(row.get("name")), _fmt_int(row.get("tokens")), _fmt_float(row.get("percent")))
        console.print(Panel(cc, title="Top Contributors", border_style="bright_blue"))


def print_period_report(label: str, report: dict[str, Any]) -> None:
    if not RICH_AVAILABLE:
        print(f"Period: {label}")
        print(f"Sessions: {_fmt_int(report['sessions'])}")
        print(f"API calls: {_fmt_int(report['api_calls'])}")
        print(f"Tokens: {_fmt_int(report['tokens'])}")
        print(f"Cost (API): {_fmt_float(report['api_cost'])}")
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
    summary_table.add_row("Window", label)
    summary_table.add_row("Sessions", _fmt_int(report["sessions"]))
    summary_table.add_row("API calls", _fmt_int(report["api_calls"]))
    summary_table.add_row("Cost (API)", _fmt_float(report["api_cost"]))
    summary_table.add_row("From", _fmt_ts_local(report["from"]))
    summary_table.add_row("To", _fmt_ts_local(report["to"]))

    # Build Token Composition table with bars
    if token_composition and isinstance(token_composition, dict):
        comp_table = _build_composition_table(token_composition, total_tokens)
    else:
        comp_table = Table(show_header=False, box=None)
        comp_table.add_row("No token composition data")

    # Print side by side using Columns (compact, content-sized)
    summary_panel = Panel(summary_table, title="Period Summary", border_style="magenta")
    comp_panel = Panel(comp_table, title="Token Composition", border_style="blue")
    console.print(Columns([summary_panel, comp_panel], equal=True, padding=0))

    top_tools = report.get("top_tools")
    if isinstance(top_tools, list) and top_tools:
        tt = Table(show_header=True, box=None)
        tt.add_column("Tool")
        tt.add_column("Output Tokens", justify="right")
        tt.add_column("Calls", justify="right")
        for item in top_tools:
            tt.add_row(str(item.get("name")), _fmt_int(item.get("output_tokens")), _fmt_int(item.get("call_count")))
        console.print(Panel(tt, title="Top Tools", border_style="yellow"))

    model_costs = report.get("model_costs")
    if isinstance(model_costs, list) and model_costs:
        mt = Table(show_header=True, box=None)
        mt.add_column("Model")
        mt.add_column("Cost (API)", justify="right")
        for item in model_costs:
            mt.add_row(str(item.get("model")), _fmt_float(item.get("cost")))
        console.print(Panel(mt, title="Model Costs", border_style="green"))

    mcp_stats = report.get("mcp_stats")
    if isinstance(mcp_stats, dict) and mcp_stats.get("rows"):
        mcp = Table(show_header=True, box=None)
        mcp.add_column("MCP")
        mcp.add_column("Tokens", justify="right")
        mcp.add_column("Calls", justify="right")
        mcp.add_column("Tok/Call", justify="right")
        mcp.add_column("%", justify="right")
        for row in mcp_stats["rows"]:
            mcp.add_row(
                str(row.get("name")),
                _fmt_int(row.get("tokens")),
                _fmt_int(row.get("calls")),
                _fmt_float(row.get("tokens_per_call")),
                _fmt_float(row.get("percent")),
            )
        console.print(Panel(mcp, title="MCP Insights", border_style="cyan"))

    component_stats = report.get("component_stats")
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
        console.print(Panel(ct, title="Component Contribution", border_style="magenta"))

    contributor_stats = report.get("contributor_stats")
    if isinstance(contributor_stats, dict) and contributor_stats.get("rows"):
        cc = Table(show_header=True, box=None)
        cc.add_column("Contributor")
        cc.add_column("Tokens", justify="right")
        cc.add_column("%", justify="right")
        for row in contributor_stats["rows"]:
            cc.add_row(str(row.get("name")), _fmt_int(row.get("tokens")), _fmt_float(row.get("percent")))
        console.print(Panel(cc, title="Top Contributors", border_style="bright_blue"))
