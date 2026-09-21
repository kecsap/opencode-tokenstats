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
COL_AXIS = "#B0B0B0"
COL_BAR_EMPTY = "#333333"

COL_INPUT = COL_BLUE
COL_OUTPUT = COL_GREEN
COL_REASONING = COL_PURPLE
COL_CACHE_READ = COL_YELLOW
COL_TOTAL = COL_ORANGE

TOP_TOOLS_CHUNK_SIZE = 8


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


def _fmt_currency(value: Any) -> str:
    try:
        amount = float(value)
        if amount == 0:
            return "-"
        if 0 < amount < 0.01:
            return "<0.01"
        return f"{amount:.2f}"
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


def _build_composition_table(
    token_composition: dict[str, int],
    total_tokens: int,
    category_ratios: dict[str, Any] | None = None,
    total_ratio: Any = None,
) -> Table:
    """Build a Token Composition table with bars and colors."""
    comp = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
    comp.add_column("", style="bold")
    comp.add_column("", justify="left")
    comp.add_column("Tokens", justify="right")
    comp.add_column("%", justify="right")
    comp.add_column("Tokens/ΔLOC", justify="right")

    # Color mapping for components
    color_map = {
        "input": COL_INPUT,
        "cache_read": COL_CACHE_READ,
        "output": COL_OUTPUT,
        "reasoning": COL_REASONING,
    }

    excluded = {"cache_write", "web_search_requests", "reasoning"}
    displayed = {k: v for k, v in token_composition.items() if k not in excluded}
    composition_total = sum(
        int(token_composition.get(key, 0)) for key in ("input", "cache_read", "output")
    )
    max_val = max(displayed.values()) if displayed else 1

    for key, value in displayed.items():
        if key in excluded:
            continue
        color = color_map.get(key, COL_TOTAL)
        bar_text = _color_bar(value, max_val, color, width=6)
        pct = value / composition_total * 100 if composition_total else 0
        ratio_key = {
            "input": "input_tokens",
            "cache_read": "cache_read_tokens",
            "output": "output_tokens",
        }.get(key, key)
        ratio = category_ratios.get(ratio_key) if category_ratios else None
        label = "input (cached)" if key == "cache_read" else key
        comp.add_row(label, bar_text, _fmt_int(value), f"{pct:.1f}", _trend_value(ratio))

    reasoning = int(token_composition.get("reasoning", 0))
    if reasoning:
        comp.add_row("(reasoning)", "", f"({_fmt_int(reasoning)})", "—", "—")

    # Add total row (no bar)
    comp.add_row("", "", "", "", "")
    comp.add_row("total", "", _fmt_int(composition_total), "100.0", _trend_value(total_ratio))

    return comp


def _build_top_tools_columns(top_tools: list[dict[str, Any]]):
    displayed = top_tools
    total_tt_tokens = sum(int(item.get("output_tokens", 0)) for item in displayed) or 1
    max_tokens = max((int(item.get("output_tokens", 0)) for item in displayed), default=1) or 1

    panels = []
    for idx in range(0, len(displayed), TOP_TOOLS_CHUNK_SIZE):
        chunk = displayed[idx: idx + TOP_TOOLS_CHUNK_SIZE]
        tt = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        tt.add_column("", style="bold")
        tt.add_column("", justify="left")
        tt.add_column("Tokens", justify="right")
        tt.add_column("%", justify="right")
        tt.add_column("Calls", justify="right")
        tt.add_column("Tok/Call", justify="right")
        for item in chunk:
            tokens = int(item.get("output_tokens", 0))
            calls = int(item.get("call_count", 0))
            pct = tokens / total_tt_tokens * 100
            bar_text = _color_bar(tokens, max_tokens, COL_GOLD, width=6)
            tt.add_row(
                str(item["name"]),
                bar_text,
                _fmt_int(tokens),
                f"{pct:.1f}",
                _fmt_int(calls),
                _fmt_float(tokens / calls if calls else 0),
            )

        start = idx + 1
        end = idx + len(chunk)
        panels.append(
            Panel(
                tt,
                title=f"[bold]External Tools ({start}-{end})[/bold]",
                border_style=COL_GOLD,
                expand=False,
            )
        )

    return Columns(panels, equal=False, padding=0)


def _trend_value(value: Any) -> str:
    if value is None:
        return "—"
    return _fmt_int(value)


def _trend_axis_value(value: Any) -> str:
    """Compact chart axis labels while keeping report/table values exact."""
    number = float(value or 0)
    absolute = abs(number)
    suffix = ""
    divisor = 1.0
    if absolute >= 1_000_000_000:
        suffix, divisor = "B", 1_000_000_000.0
    elif absolute >= 1_000_000:
        suffix, divisor = "M", 1_000_000.0
    elif absolute >= 1_000:
        suffix, divisor = "K", 1_000.0
    if not suffix:
        return _fmt_int(int(number))
    compact = number / divisor
    rendered = f"{compact:.2f}".rstrip("0").rstrip(".")
    return f"{rendered}{suffix}"


def _build_trend_chart(
    title: str,
    values: list[Any],
    color: str,
    summary: str,
    start_label: str,
    midpoint_label: str,
    end_label: str,
    width: int,
) -> Panel:
    height = 6
    numeric = [float(value) for value in values if value is not None]
    maximum = max(numeric, default=0.0)
    partial_glyphs = " ▁▂▃▄▅▆▇"
    body = Text()
    for row in range(height, 0, -1):
        axis_label = ""
        if row == height:
            axis_label = _trend_axis_value(maximum)
        elif row == (height + 1) // 2:
            axis_label = _trend_axis_value(maximum / 2)
        body.append(f"{axis_label:>7} ", style=COL_AXIS)
        body.append("┤ ", style=COL_AXIS)
        for value in values[:width]:
            if value is None:
                body.append(" ")
                continue
            levels = round(float(value) / maximum * height * 8) if maximum else 0
            full_cells, partial_level = divmod(levels, 8)
            # Match table bars: low values are blue and high values move
            # through yellow toward orange/red. The partial cap uses the same
            # value-based color as the full cells below it.
            pct = min(1.0, max(0.0, float(value) / maximum)) if maximum else 0.0
            bar_color = f"bold {_gradient_color(pct)}"
            if full_cells >= row:
                body.append("█", style=bar_color)
            elif partial_level and full_cells + 1 == row:
                body.append(partial_glyphs[partial_level], style=bar_color)
            elif levels == 0 and row == 1 and float(value) > 0:
                body.append("·", style=COL_DIM)
            else:
                body.append("░", style=COL_BAR_EMPTY)
        body.append("\n")
    axis = ["─"] * width
    # Major ticks anchor the sparse start/middle/end labels. Three minor ticks
    # between each major interval add temporal reference points without adding
    # more date labels.
    for position in (
        width // 8,
        (2 * width) // 8,
        (3 * width) // 8,
        width // 2,
        (5 * width) // 8,
        (6 * width) // 8,
        (7 * width) // 8,
        width - 1,
    ):
        axis[position] = "┴"
    body.append(f"{'0':>7} └{''.join(axis)}", style=COL_AXIS)
    labels = [" "] * width
    label_positions = [(0, start_label), (max(0, width - len(end_label)), end_label)]
    if width >= 24:
        label_positions.insert(1, (max(0, width // 2 - len(midpoint_label) // 2), midpoint_label))
    for position, label in label_positions:
        for index, character in enumerate(label):
            target = position + index
            if target < width:
                labels[target] = character
    body.append(f"\n{'':>9}{''.join(labels)}", style=COL_AXIS)
    body.append(f"\n{'':>9}{summary}", style=COL_AXIS)
    return Panel(body, title=f"[bold]{title}[/bold]", border_style=color, expand=False)


def _print_trends(console: Console, trends: dict[str, Any]) -> None:
    points = trends.get("points")
    if not isinstance(points, list) or not points:
        return
    def short_date(point: dict[str, Any]) -> str:
        raw = str(point.get("date", ""))
        try:
            return datetime.fromisoformat(raw).strftime("%b %d")
        except ValueError:
            return raw[:10]

    start_label = short_date(points[0])
    midpoint_label = short_date(points[len(points) // 2])
    end_label = short_date(points[-1])
    width = min(48, max(12, (console.width - 8) // 3))

    def fit(values: list[Any]) -> list[Any]:
        if len(values) <= width:
            return values
        return [values[round(index * (len(values) - 1) / (width - 1))] for index in range(width)]

    categories = [
        ("Input Tokens", "input_tokens", COL_BLUE),
        ("Cache Read Tokens", "cache_read_tokens", COL_CYAN),
        ("Output Tokens", "output_tokens", COL_PURPLE),
    ]
    token_charts = [
        _build_trend_chart(
            title,
            fit([p.get(key, 0) for p in points]),
            color,
            f"total {_fmt_int(trends.get(key, 0))}",
            start_label,
            midpoint_label,
            end_label,
            width,
        )
        for title, key, color in categories
    ]
    ratio_charts = [
        _build_trend_chart(
            f"{title} / ΔLOC",
            fit([p.get(f"{key}_per_loc") for p in points]),
            color,
            f"period {_trend_value(trends.get('category_tokens_per_loc', {}).get(key))} tok/LOC",
            start_label,
            midpoint_label,
            end_label,
            width,
        )
        for title, key, color in categories
    ]
    console.print(Columns(token_charts, equal=True, expand=True, padding=1))
    console.print(Columns(ratio_charts, equal=True, expand=True, padding=1))
    git_chart = _build_trend_chart(
        "Lines Changed",
        fit([p.get("churn_loc", 0) for p in points]),
        COL_ORANGE,
        f"changed {_fmt_int(trends.get('churn_loc', 0))}  +{_fmt_int(trends.get('added_loc', 0))}  -{_fmt_int(trends.get('deleted_loc', 0))}  net {int(trends.get('net_loc', 0)):+d}",
        start_label,
        midpoint_label,
        end_label,
        width,
    )
    console.print(git_chart)
    outliers = trends.get("outliers", [])
    if outliers:
        outlier_table = Table(show_header=True, box=None, expand=False)
        outlier_table.add_column("Bucket", style=COL_AXIS)
        outlier_table.add_column("ΔLOC", justify="right")
        outlier_table.add_column("+", justify="right")
        outlier_table.add_column("-", justify="right")
        outlier_table.add_column("Trigger")
        outlier_table.add_column("Largest commits")
        for outlier in outliers:
            causes = []
            for commit in outlier.get("commits", []):
                causes.append(f"{commit['sha'][:7]}  {_fmt_int(commit.get('delta_loc', 0))} ΔLOC  {commit['subject']}")
            remainder = int(outlier.get("other_commits_delta_loc", 0))
            if remainder:
                causes.append(f"Other commits  {_fmt_int(remainder)} ΔLOC")
            outlier_table.add_row(
                f"{str(outlier.get('date', ''))[5:16].replace('T', ' ')}–\n{str(outlier.get('end_date', ''))[5:16].replace('T', ' ')} UTC",
                _fmt_int(outlier.get("churn_loc", 0)),
                _fmt_int(outlier.get("added_loc", 0)),
                _fmt_int(outlier.get("deleted_loc", 0)),
                "\n".join(
                    f"{key.replace('_tokens_per_loc', '')} {_trend_value(value.get('ratio'))} (>{_trend_value(value.get('threshold'))})"
                    for key, value in outlier.get("triggers", {}).items()
                ) or "—",
                "\n".join(causes) or "—",
            )
        console.print(Panel(outlier_table, title="[bold]Chart Outliers[/bold]", border_style=COL_ORANGE, expand=False))



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
    console.print(Panel(table, title="[bold]Status[/bold]", border_style=COL_CYAN))


def _format_pricing_coverage(coverage: dict[str, Any] | None) -> str | None:
    if not isinstance(coverage, dict):
        return None
    total = int(coverage.get("calls", 0) or 0)
    if total <= 0:
        return None
    priced = int(coverage.get("priced_calls", 0) or 0)
    unpriced = int(coverage.get("unpriced_calls", 0) or 0)
    future_fallback = int(coverage.get("future_fallback_calls", 0) or 0)
    percent = float(coverage.get("coverage_percent", 0.0) or 0.0)
    return f"{percent:.2f}% ({priced} priced, {unpriced} unpriced, {future_fallback} future-fallback)"


def _model_estimate_label(item: dict[str, Any]) -> str:
    value = _fmt_currency(item.get("estimated_cost"))
    if float(item.get("estimated_future_market_cost", 0) or 0) > 0:
        return value + "†"
    if (
        float(item.get("estimated_generic_cost", 0) or 0) > 0
        and float(item.get("estimated_cost", 0) or 0)
        == float(item.get("estimated_generic_cost", 0) or 0)
        and not float(item.get("estimated_direct_cost", 0) or 0)
        and not float(item.get("estimated_market_cost", 0) or 0)
        and not float(item.get("estimated_cloud_equivalent_cost", 0) or 0)
    ):
        return value + "*"
    if (
        float(item.get("estimated_market_cost", 0) or 0) > 0
        and int(item.get("market_provider_count", 0) or 0) > 1
    ):
        return value + "†"
    return value


def _model_costs_footer(model_costs: list[dict[str, Any]]) -> Text:
    has_generic = any(
        float(row.get("estimated_generic_cost", 0) or 0) > 0
        and float(row.get("estimated_cost", 0) or 0)
        == float(row.get("estimated_generic_cost", 0) or 0)
        and float(row.get("estimated_future_market_cost", 0) or 0) <= 0
        and not float(row.get("estimated_direct_cost", 0) or 0)
        and not float(row.get("estimated_market_cost", 0) or 0)
        and not float(row.get("estimated_cloud_equivalent_cost", 0) or 0)
        for row in model_costs
    )
    has_projected_market = any(
        float(row.get("estimated_future_market_cost", 0) or 0) > 0
        or (
            float(row.get("estimated_market_cost", 0) or 0) > 0
            and int(row.get("market_provider_count", 0) or 0) > 1
        )
        for row in model_costs
    )
    markers = []
    if has_generic:
        markers.append("* generic fallback estimate")
    if has_projected_market:
        markers.append("† hosted-market estimation")
    if not markers:
        return Text("")
    return Text("   ".join(markers), style=COL_AXIS)


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
    core_stats: dict[str, Any] | None = None,
    model_costs: list[dict[str, Any]] | None = None,
    pricing_coverage: dict[str, Any] | None = None,
    pricing_warnings: list[str] | None = None,
) -> None:
    coverage_text = _format_pricing_coverage(pricing_coverage)
    if not RICH_AVAILABLE:
        print(f"Session: {session_id}")
        print(f"API calls: {_fmt_int(api_calls)}")
        print(f"Tokens: {_fmt_int(tokens)}")
        print(f"Cost (API): {_fmt_float(api_cost)}")
        if coverage_text:
            print(f"Pricing coverage: {coverage_text}")
        for warning in pricing_warnings or []:
            print(f"WARNING: {warning}")
        if token_composition:
            print(f"Token Composition: {token_composition}")
        if top_tools:
            print(f"External Tools: {top_tools}")
        if model_costs:
            displayed = [
                dict(
                    item,
                    api_cost=_fmt_currency(item.get("api_cost")),
                    estimated_cost=_model_estimate_label(item),
                )
                for item in model_costs
            ]
            print(f"Model Costs: {displayed}")
            footer = _model_costs_footer(model_costs)
            if footer.plain:
                print(footer.plain)
        if mcp_stats:
            print(f"MCP Stats: {mcp_stats}")
        if core_stats:
            print(f"OpenCode Core: {core_stats}")
        if component_stats:
            print(f"Components: {component_stats}")
        return

    console = Console()
    table = Table(show_header=False, box=None)
    table.add_row("Session", session_id)
    table.add_row("API calls", _fmt_int(api_calls))
    table.add_row("Tokens", _fmt_int(tokens))
    table.add_row("Cost (API)", _fmt_float(api_cost))
    if coverage_text:
        table.add_row("Pricing", coverage_text)
    console.print(Panel(table, title="[bold]Session[/bold]", border_style=COL_GREEN))
    for warning in pricing_warnings or []:
        console.print(Text(f"WARNING: {warning}", style=COL_DIM))

    if token_composition:
        comp = _build_composition_table(token_composition, tokens)
        console.print(Panel(comp, title="[bold]Token Composition[/bold]", border_style=COL_BLUE))

    if top_tools:
        console.print(_build_top_tools_columns(top_tools))

    if model_costs:
        mt = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        mt.add_column("", style="bold", justify="left")
        mt.add_column("", justify="left")
        mt.add_column("Tokens", justify="right")
        mt.add_column("Input (%)", justify="center")
        mt.add_column("Output (%)", justify="center")
        mt.add_column("Reasoning (%)", justify="center")
        mt.add_column("API", justify="right")
        mt.add_column("Est.", justify="right")
        max_cost = max((float(item.get("api_cost", 0)) or float(item.get("estimated_cost", 0)) for item in model_costs), default=1) or 1
        shown_models = model_costs[:15]
        has_api_costs = any(float(item.get("api_cost", 0)) > 0 for item in shown_models)
        separator_added = False
        for item in shown_models:
            api_cost = float(item.get("api_cost", 0))
            if has_api_costs and not api_cost and not separator_added:
                mt.add_row(*[Text("─" * 6, style=COL_DIM) for _ in range(8)])
                separator_added = True
            primary_cost = api_cost or float(item.get("estimated_cost", 0))
            bar_text = _color_bar(primary_cost, max_cost, COL_GREEN, width=6)
            mt.add_row(
                str(item.get("model")),
                bar_text,
                _fmt_int(item.get("tokens", 0)),
                f"{float(item.get('input_percent', 0)):.2f}",
                f"{float(item.get('output_percent', 0)):.2f}",
                f"{float(item.get('reasoning_percent', 0)):.2f}",
                _fmt_currency(api_cost),
                _model_estimate_label(item),
            )
        footer = _model_costs_footer(shown_models)
        if footer.plain:
            mt.add_row(footer, *([""] * 7))
        model_costs_panel = Panel(mt, title="[bold]Model Costs[/bold]", border_style=COL_GREEN)
        console.print(model_costs_panel)

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
        for row in mcp_filtered[:15]:
            tokens = int(row.get("tokens", 0))
            bar_text = _color_bar(tokens, max_tokens, COL_CYAN, width=6)
            mcp.add_row(
                str(row.get("name")),
                bar_text,
                _fmt_int(tokens),
                f"{row.get('percent', 0):.1f}",
                _fmt_int(row.get("calls")),
                _fmt_float(row.get("tokens_per_call")),
            )
        console.print(Panel(mcp, title="[bold]MCP Servers[/bold]", border_style=COL_CYAN))

    if core_stats and core_stats.get("rows"):
        oc = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        oc.add_column("", style="bold")
        oc.add_column("", justify="left")
        oc.add_column("Tokens", justify="right")
        oc.add_column("%", justify="right")
        oc.add_column("Calls", justify="right")
        oc.add_column("Tok/Call", justify="right")
        total_core_tokens = sum(int(row.get("tokens", 0)) for row in core_stats["rows"]) or 1
        max_tokens = max((int(row.get("tokens", 0)) for row in core_stats["rows"]), default=1) or 1
        for row in core_stats["rows"][:15]:
            tokens = int(row.get("tokens", 0))
            calls = int(row.get("calls", 0))
            pct = tokens / total_core_tokens * 100
            bar_text = _color_bar(tokens, max_tokens, COL_ORANGE, width=6)
            oc.add_row(
                str(row.get("component_name")),
                bar_text,
                _fmt_int(row.get("tokens")),
                f"{pct:.1f}",
                _fmt_int(calls),
                _fmt_float(tokens / calls if calls else 0),
            )
        console.print(Panel(oc, title="[bold]OpenCode Contribution[/bold]", border_style=COL_ORANGE))

    if component_stats and component_stats.get("rows"):
        ct = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        ct.add_column("Type")
        ct.add_column("", style="bold")
        ct.add_column("", justify="left")
        ct.add_column("Tokens", justify="right")
        ct.add_column("%", justify="right")
        ct.add_column("Calls", justify="right")
        ct.add_column("Tok/Call", justify="right")
        max_tokens = max((int(row.get("tokens", 0)) for row in component_stats["rows"]), default=1) or 1
        for row in component_stats["rows"]:
            tokens = int(row.get("tokens", 0))
            calls = int(row.get("calls", 0))
            bar_text = _color_bar(tokens, max_tokens, COL_MAGENTA, width=6)
            ct.add_row(
                str(row.get("component_type")),
                str(row.get("component_group")),
                bar_text,
                _fmt_int(row.get("tokens")),
                f"{row.get('percent', 0):.1f}",
                _fmt_int(calls),
                _fmt_float(tokens / calls if calls else 0),
            )
        console.print(Panel(ct, title="[bold]Components Contribution[/bold]", border_style=COL_MAGENTA))


def print_period_report(label: str, report: dict[str, Any]) -> None:
    if not RICH_AVAILABLE:
        print(f"Period: {label}")
        print(f"Sessions: {_fmt_int(report['sessions'])}")
        print(f"API calls: {_fmt_int(report['api_calls'])}")
        print(f"Tokens: {_fmt_int(report['tokens'])}")
        print(f"From: {_fmt_ts_local(report.get('from'))}")
        print(f"To: {_fmt_ts_local(report.get('to'))}")
        coverage_text = _format_pricing_coverage(report.get("pricing"))
        if coverage_text:
            print(f"Pricing coverage: {coverage_text}")
        for warning in report.get("warnings") or []:
            print(f"WARNING: {warning}")
        if report.get("token_composition"):
            print(f"Token Composition: {report['token_composition']}")
        if report.get("top_tools"):
            print(f"External Tools: {report['top_tools']}")
        if report.get("model_costs"):
            displayed = [
                dict(
                    item,
                    api_cost=_fmt_currency(item.get("api_cost")),
                    estimated_cost=_model_estimate_label(item),
                )
                for item in report["model_costs"]
            ]
            print(f"Model Costs: {displayed}")
            footer = _model_costs_footer(report["model_costs"])
            if footer.plain:
                print(footer.plain)
        if report.get("mcp_stats"):
            print(f"MCP Stats: {report['mcp_stats']}")
        if report.get("component_stats"):
            print(f"Components: {report['component_stats']}")
        if report.get("by_activity"):
            print(f"Activity by Turn: {report['by_activity']}")
        if report.get("top_sessions"):
            print(f"Top Sessions: {report['top_sessions']}")
        if report.get("trends"):
            trends = report["trends"]
            print(f"Period Trends: {trends.get('tokens', 0)} tokens, {trends.get('churn_loc', 0)} ΔLOC")
        else:
            print("Period Trends unavailable: no session has both Git history and timestamped token data.")
        return

    console = Console()
    total_tokens = report.get("tokens", 0)
    token_composition = report.get("token_composition")
    trends = report.get("trends")
    trend_ratios = trends.get("category_tokens_per_loc", {}) if isinstance(trends, dict) else None
    trend_total_ratio = trends.get("tokens_per_loc") if isinstance(trends, dict) else None

    # Build Period Summary table (without Tokens - moved to composition)
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column(style="bold", justify="left")
    summary_table.add_column(justify="left")
    summary_table.add_row("Window", label)
    summary_table.add_row("Sessions", _fmt_int(report["sessions"]))
    summary_table.add_row("API calls", _fmt_int(report["api_calls"]))
    summary_table.add_row("From", _fmt_ts_local(report["from"]))
    summary_table.add_row("To", _fmt_ts_local(report["to"]))
    coverage_text = _format_pricing_coverage(report.get("pricing"))
    if coverage_text:
        summary_table.add_row("Pricing", coverage_text)

    # Build Token Composition table with bars
    if token_composition and isinstance(token_composition, dict):
        comp_table = _build_composition_table(
            token_composition, total_tokens, trend_ratios, trend_total_ratio
        )
    else:
        comp_table = Table(show_header=False, box=None)
        comp_table.add_row("No token composition data")

    # Build Model Costs panel
    model_costs = report.get("model_costs")
    if isinstance(model_costs, list) and model_costs:
        mt = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        mt.add_column("", style="bold", justify="left")
        mt.add_column("", justify="left")
        mt.add_column("Tokens", justify="right")
        mt.add_column("Input (%)", justify="center")
        mt.add_column("Output (%)", justify="center")
        mt.add_column("Reasoning (%)", justify="center")
        mt.add_column("API", justify="right")
        mt.add_column("Est.", justify="right")
        max_cost = max((float(item.get("api_cost", 0)) or float(item.get("estimated_cost", 0)) for item in model_costs), default=1) or 1
        shown_models = model_costs[:15]
        has_api_costs = any(float(item.get("api_cost", 0)) > 0 for item in shown_models)
        separator_added = False
        for item in shown_models:
            api_cost = float(item.get("api_cost", 0))
            if has_api_costs and not api_cost and not separator_added:
                mt.add_row(*[Text("─" * 6, style=COL_DIM) for _ in range(8)])
                separator_added = True
            primary_cost = api_cost or float(item.get("estimated_cost", 0))
            bar_text = _color_bar(primary_cost, max_cost, COL_GREEN, width=6)
            mt.add_row(
                str(item.get("model")),
                bar_text,
                _fmt_int(item.get("tokens", 0)),
                f"{float(item.get('input_percent', 0)):.2f}",
                f"{float(item.get('output_percent', 0)):.2f}",
                f"{float(item.get('reasoning_percent', 0)):.2f}",
                _fmt_currency(api_cost),
                _model_estimate_label(item),
            )
        footer = _model_costs_footer(shown_models)
        if footer.plain:
            mt.add_row(footer, *([""] * 7))
        model_costs_panel = Panel(mt, title="[bold]Model Costs[/bold]", border_style=COL_GREEN, expand=False)
    else:
        model_costs_panel = None

    # Build Activity by Turn panel
    by_activity = report.get("by_activity")
    act_panel = None
    if isinstance(by_activity, list) and by_activity:
        act = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        act.add_column("", style="bold")
        act.add_column("", justify="left")
        act.add_column("Tokens", justify="right")
        act.add_column("%", justify="right")
        act.add_column("Input (%)", justify="center")
        act.add_column("Output (%)", justify="center")
        act.add_column("Reasoning (%)", justify="center")
        act.add_column("Calls", justify="right")
        act.add_column("API", justify="right")
        act.add_column("Est.", justify="right")
        cat_rows = sorted(by_activity, key=lambda x: int(x.get("tokens", 0)), reverse=True)[:10]
        total_cat_tokens = sum(int(row.get("tokens", 0)) for row in cat_rows) or 1
        max_tokens = max((int(row.get("tokens", 0)) for row in cat_rows), default=1) or 1
        for row in cat_rows:
            tokens = int(row.get("tokens", 0))
            pct = tokens / total_cat_tokens * 100
            bar_text = _color_bar(tokens, max_tokens, COL_GREEN, width=6)
            act.add_row(
                str(row.get("label", row.get("category", "?"))),
                bar_text,
                _fmt_int(tokens),
                f"{pct:.1f}",
                f"{float(row.get('input_percent', 0)):.2f}",
                f"{float(row.get('output_percent', 0)):.2f}",
                f"{float(row.get('reasoning_percent', 0)):.2f}",
                _fmt_int(row.get("calls", 0)),
                _fmt_float(row.get("api_cost", 0)),
                _fmt_float(row.get("estimated_cost", 0)),
            )
        act_panel = Panel(act, title="[bold]Activity by Turn[/bold]", border_style=COL_GREEN, expand=False)

    # Build Top Sessions panel
    top_sess_panel = None
    top_sessions = report.get("top_sessions")
    if isinstance(top_sessions, list) and top_sessions:
        ts = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        ts.add_column("", style="bold")
        ts.add_column("", justify="left")
        ts.add_column("Tokens", justify="right")
        ts.add_column("%", justify="right")
        ts.add_column("Input (%)", justify="center")
        ts.add_column("Output (%)", justify="center")
        ts.add_column("Reasoning (%)", justify="center")
        ts.add_column("API", justify="right")
        ts.add_column("Est.", justify="right")
        sess_rows = sorted(top_sessions, key=lambda x: int(x.get("tokens", 0)), reverse=True)[:10]
        total_sess_tokens = sum(int(row.get("tokens", 0)) for row in sess_rows) or 1
        max_tokens = max((int(row.get("tokens", 0)) for row in sess_rows), default=1) or 1
        for row in sess_rows:
            tokens = int(row.get("tokens", 0))
            pct = tokens / total_sess_tokens * 100
            bar_text = _color_bar(tokens, max_tokens, COL_ORANGE, width=6)
            ts.add_row(
                str(row.get("root_dir", "-")),
                bar_text,
                _fmt_int(tokens),
                f"{pct:.1f}",
                f"{float(row.get('input_percent', 0)):.2f}",
                f"{float(row.get('output_percent', 0)):.2f}",
                f"{float(row.get('reasoning_percent', 0)):.2f}",
                _fmt_float(row.get("api_cost", 0)),
                _fmt_float(row.get("estimated_cost", 0)),
            )
        top_sess_panel = Panel(ts, title="[bold]Top Sessions[/bold]", border_style=COL_ORANGE, expand=False)

    # Build External Tools panel
    top_tools = report.get("top_tools")
    top_tools_renderable = None
    if isinstance(top_tools, list) and top_tools:
        top_tools_renderable = _build_top_tools_columns(top_tools)

    summary_panel = Panel(summary_table, title="[bold]Period Summary[/bold]", border_style=COL_MAGENTA, expand=False)
    comp_panel = Panel(comp_table, title="[bold]Token Summary[/bold]", border_style=COL_BLUE, expand=False)

    left_top = Columns([summary_panel, comp_panel], equal=False, padding=0)
    left_column = Group(left_top, model_costs_panel) if model_costs_panel else left_top
    right_panels = [panel for panel in (act_panel, top_sess_panel) if panel]
    right_column = Group(*right_panels) if right_panels else None
    if right_column:
        upper_grid = Table.grid(expand=True, padding=(0, 1))
        upper_grid.add_column(ratio=1)
        upper_grid.add_column(ratio=1)
        upper_grid.add_row(left_column, right_column)
        console.print(upper_grid)
    else:
        console.print(left_column)

    mcp_stats = report.get("mcp_stats")
    core_stats = report.get("core_stats")
    component_stats = report.get("component_stats")

    # Build panels for Component Contribution, OpenCode Contribution, MCP Servers
    panels = []

    if isinstance(component_stats, dict) and component_stats.get("rows"):
        ct = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        ct.add_column("Type")
        ct.add_column("", style="bold")
        ct.add_column("", justify="left")
        ct.add_column("Tokens", justify="right")
        ct.add_column("%", justify="right")
        ct.add_column("Calls", justify="right")
        ct.add_column("Tok/Call", justify="right")
        max_tokens = max((int(row.get("tokens", 0)) for row in component_stats["rows"]), default=1) or 1
        for row in component_stats["rows"]:
            tokens = int(row.get("tokens", 0))
            calls = int(row.get("calls", 0))
            bar_text = _color_bar(tokens, max_tokens, COL_MAGENTA, width=6)
            ct.add_row(
                str(row.get("component_type")),
                str(row.get("component_group")),
                bar_text,
                _fmt_int(row.get("tokens")),
                f"{row.get('percent', 0):.1f}",
                _fmt_int(calls),
                _fmt_float(tokens / calls if calls else 0),
            )
        panels.append(Panel(ct, title="[bold]Components Contribution[/bold]", border_style=COL_MAGENTA))

    if isinstance(core_stats, dict) and core_stats.get("rows"):
        oc = Table(show_header=True, box=None, padding=(0, 0, 0, 1))
        oc.add_column("", style="bold")
        oc.add_column("", justify="left")
        oc.add_column("Tokens", justify="right")
        oc.add_column("%", justify="right")
        oc.add_column("Calls", justify="right")
        oc.add_column("Tok/Call", justify="right")
        total_core_tokens = sum(int(row.get("tokens", 0)) for row in core_stats["rows"]) or 1
        max_tokens = max((int(row.get("tokens", 0)) for row in core_stats["rows"]), default=1) or 1
        for row in core_stats["rows"][:15]:
            tokens = int(row.get("tokens", 0))
            calls = int(row.get("calls", 0))
            pct = tokens / total_core_tokens * 100
            bar_text = _color_bar(tokens, max_tokens, COL_ORANGE, width=6)
            oc.add_row(
                str(row.get("component_name")),
                bar_text,
                _fmt_int(row.get("tokens")),
                f"{pct:.1f}",
                _fmt_int(calls),
                _fmt_float(tokens / calls if calls else 0),
            )
        panels.append(Panel(oc, title="[bold]OpenCode Contribution[/bold]", border_style=COL_ORANGE))

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
        for row in mcp_filtered[:15]:
            tokens = int(row.get("tokens", 0))
            bar_text = _color_bar(tokens, max_tokens, COL_CYAN, width=6)
            mcp.add_row(
                str(row.get("name")),
                bar_text,
                _fmt_int(tokens),
                f"{row.get('percent', 0):.1f}",
                _fmt_int(row.get("calls")),
                _fmt_float(row.get("tokens_per_call")),
            )
        panels.append(Panel(mcp, title="[bold]MCP Servers[/bold]", border_style=COL_CYAN))

    if panels:
        console.print(Columns(panels, equal=False, padding=0))

    # External Tools at the bottom (not spanning full width)
    if top_tools_renderable:
        console.print(top_tools_renderable)

    for warning in report.get("warnings") or []:
        console.print(Text(f"WARNING: {warning}", style=COL_DIM))

    if isinstance(trends, dict):
        _print_trends(console, trends)
    else:
        console.print(Text("Period Trends unavailable: no session has both Git history and timestamped token data.", style=COL_DIM))
