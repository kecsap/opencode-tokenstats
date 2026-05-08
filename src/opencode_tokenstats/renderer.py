from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    RICH_AVAILABLE = False


def print_status_report(mode: str, sessions: list[dict[str, object]]) -> None:
    latest = sessions[0].get("id") if sessions else "-"
    if not RICH_AVAILABLE:
        print(f"Mode: {mode}")
        print(f"Sessions: {len(sessions)}")
        print(f"Latest Session: {latest}")
        return

    console = Console()
    table = Table(show_header=False, box=None)
    table.add_row("Mode", str(mode))
    table.add_row("Sessions", str(len(sessions)))
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
        print(f"API calls: {api_calls}")
        print(f"Tokens: {tokens}")
        print(f"Cost (API): {api_cost:.6f}")
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
    table.add_row("API calls", str(api_calls))
    table.add_row("Tokens", str(tokens))
    table.add_row("Cost (API)", f"{api_cost:.6f}")
    console.print(Panel(table, title="Session", border_style="green"))

    if token_composition:
        comp = Table(show_header=True, box=None)
        comp.add_column("Component")
        comp.add_column("Tokens", justify="right")
        for key, value in token_composition.items():
            comp.add_row(key, str(value))
        console.print(Panel(comp, title="Token Composition", border_style="blue"))

    if top_tools:
        tt = Table(show_header=True, box=None)
        tt.add_column("Tool")
        tt.add_column("Output Tokens", justify="right")
        tt.add_column("Calls", justify="right")
        for item in top_tools:
            tt.add_row(str(item["name"]), str(item["output_tokens"]), str(item["call_count"]))
        console.print(Panel(tt, title="Top Tools", border_style="yellow"))

    if model_costs:
        mt = Table(show_header=True, box=None)
        mt.add_column("Model")
        mt.add_column("Cost (API)", justify="right")
        for item in model_costs:
            mt.add_row(str(item.get("model")), str(item.get("cost")))
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
                str(row.get("tokens")),
                str(row.get("calls")),
                str(row.get("tokens_per_call")),
                str(row.get("percent")),
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
                str(row.get("tokens")),
                str(row.get("estimated_session_tokens")),
                str(row.get("calls", 0)),
                str(row.get("percent")),
            )
        console.print(Panel(ct, title="Component Contribution", border_style="magenta"))

    if contributor_stats and contributor_stats.get("rows"):
        cc = Table(show_header=True, box=None)
        cc.add_column("Contributor")
        cc.add_column("Tokens", justify="right")
        cc.add_column("%", justify="right")
        for row in contributor_stats["rows"]:
            cc.add_row(str(row.get("name")), str(row.get("tokens")), str(row.get("percent")))
        console.print(Panel(cc, title="Top Contributors", border_style="bright_blue"))


def print_period_report(label: str, report: dict[str, Any]) -> None:
    if not RICH_AVAILABLE:
        print(f"Period: {label}")
        print(f"Sessions: {report['sessions']}")
        print(f"API calls: {report['api_calls']}")
        print(f"Tokens: {report['tokens']}")
        print(f"Cost (API): {report['api_cost']}")
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
    table = Table(show_header=False, box=None)
    table.add_row("Window", label)
    table.add_row("Sessions", str(report["sessions"]))
    table.add_row("API calls", str(report["api_calls"]))
    table.add_row("Tokens", str(report["tokens"]))
    table.add_row("Cost (API)", str(report["api_cost"]))
    table.add_row("From", str(report["from"]))
    table.add_row("To", str(report["to"]))
    console.print(Panel(table, title="Period Summary", border_style="magenta"))

    token_composition = report.get("token_composition")
    if isinstance(token_composition, dict):
        comp = Table(show_header=True, box=None)
        comp.add_column("Component")
        comp.add_column("Tokens", justify="right")
        for key, value in token_composition.items():
            comp.add_row(str(key), str(value))
        console.print(Panel(comp, title="Token Composition", border_style="blue"))

    top_tools = report.get("top_tools")
    if isinstance(top_tools, list) and top_tools:
        tt = Table(show_header=True, box=None)
        tt.add_column("Tool")
        tt.add_column("Output Tokens", justify="right")
        tt.add_column("Calls", justify="right")
        for item in top_tools:
            tt.add_row(str(item.get("name")), str(item.get("output_tokens")), str(item.get("call_count")))
        console.print(Panel(tt, title="Top Tools", border_style="yellow"))

    model_costs = report.get("model_costs")
    if isinstance(model_costs, list) and model_costs:
        mt = Table(show_header=True, box=None)
        mt.add_column("Model")
        mt.add_column("Cost (API)", justify="right")
        for item in model_costs:
            mt.add_row(str(item.get("model")), str(item.get("cost")))
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
                str(row.get("tokens")),
                str(row.get("calls")),
                str(row.get("tokens_per_call")),
                str(row.get("percent")),
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
                str(row.get("tokens")),
                str(row.get("estimated_session_tokens")),
                str(row.get("calls", 0)),
                str(row.get("percent")),
            )
        console.print(Panel(ct, title="Component Contribution", border_style="magenta"))

    contributor_stats = report.get("contributor_stats")
    if isinstance(contributor_stats, dict) and contributor_stats.get("rows"):
        cc = Table(show_header=True, box=None)
        cc.add_column("Contributor")
        cc.add_column("Tokens", justify="right")
        cc.add_column("%", justify="right")
        for row in contributor_stats["rows"]:
            cc.add_row(str(row.get("name")), str(row.get("tokens")), str(row.get("percent")))
        console.print(Panel(cc, title="Top Contributors", border_style="bright_blue"))
