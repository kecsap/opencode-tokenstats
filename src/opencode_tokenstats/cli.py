from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from collections import defaultdict
import re

import click

from .client import ApiClientError, OpencodeApiClient
from .canonical_metrics import build_canonical_metrics
from .compatibility import CompatMode, analyze_context_compatibility
from .content_attribution import collect_content_attribution
from .local_session_service import LocalSessionService, LocalStorageError
from .renderer import print_period_report, print_session_report, print_status_report
from .report_schema import build_report_schema, report_to_markdown
from .session_service import SessionService
from .telemetry import collect_telemetry_calls, summarize_telemetry
from .tokenization import TokenizerRegistry


@click.group()
@click.option("--base-url", default="http://127.0.0.1:4096", show_default=True)
@click.option("--username", default=None)
@click.option("--password", default=None)
@click.option("--timeout", default=10.0, show_default=True, type=float)
@click.option("--retries", default=2, show_default=True, type=int)
@click.option("--mode", type=click.Choice(["local", "api"]), default="local", show_default=True)
@click.option("--db-path", default=None)
@click.pass_context
def main(
    ctx: click.Context,
    base_url: str,
    username: str | None,
    password: str | None,
    timeout: float,
    retries: int,
    mode: str,
    db_path: str | None,
) -> None:
    """OpenCode TokenStats CLI."""
    ctx.obj = {
        "base_url": base_url,
        "username": username,
        "password": password,
        "timeout": timeout,
        "retries": retries,
        "mode": mode,
        "db_path": db_path,
    }


@main.command()
@click.option("--check-tokenizer", is_flag=True, help="Check tokenizer resolution and mode")
@click.option("--provider-id", default="local", show_default=True)
@click.option("--model-id", default="qwen3.6-27b", show_default=True)
@click.option("--sample-text", default="hello world", show_default=True)
@click.option(
    "--compat-mode",
    type=click.Choice(["strict_local", "strict_api", "tokenscope_compat"]),
    default=None,
)
@click.option("--compat-source", type=click.Choice(["auto", "local", "api"]), default="auto", show_default=True)
@click.option("--compat-session-id", default=None)
@click.pass_context
def doctor(
    ctx: click.Context,
    check_tokenizer: bool,
    provider_id: str,
    model_id: str,
    sample_text: str,
    compat_mode: str | None,
    compat_source: str,
    compat_session_id: str | None,
) -> None:
    """Check local OpenCode storage or API session endpoints."""
    options = ctx.obj
    if options["mode"] == "local":
        try:
            db_path = LocalSessionService.find_database_path(options.get("db_path"))
            service = LocalSessionService(db_path=db_path)
            sessions = service.list_sessions()
            click.echo("OpenCode Local Storage: OK")
            click.echo(f"SQLite DB: {db_path}")
            click.echo(f"Session DB: OK (list_sessions returned {len(sessions)} entries)")
            if check_tokenizer:
                _print_tokenizer_check(provider_id, model_id, sample_text)
            if compat_mode:
                session_id = compat_session_id or _pick_latest_session_id(sessions)
                if not session_id:
                    raise click.ClickException("No sessions available for compatibility check.")
                messages = service.get_messages(session_id)
                _print_compatibility_check(
                    messages,
                    mode=compat_mode,
                    source=_effective_source(compat_source, options["mode"]),
                    session_id=session_id,
                )
            return
        except LocalStorageError as exc:
            raise click.ClickException(str(exc)) from exc

    try:
        with OpencodeApiClient(
            base_url=options["base_url"],
            username=options["username"],
            password=options["password"],
            timeout=options["timeout"],
            retries=options["retries"],
        ) as client:
            service = SessionService(client)
            sessions = service.list_sessions()
            click.echo("OpenCode API: OK")
            click.echo(f"Session API: OK (list_sessions returned {len(sessions)} entries)")
            if check_tokenizer:
                _print_tokenizer_check(provider_id, model_id, sample_text)
            if compat_mode:
                session_id = compat_session_id or _pick_latest_session_id(sessions)
                if not session_id:
                    raise click.ClickException("No sessions available for compatibility check.")
                messages = service.get_messages(session_id)
                _print_compatibility_check(
                    messages,
                    mode=compat_mode,
                    source=_effective_source(compat_source, options["mode"]),
                    session_id=session_id,
                )
    except ApiClientError as exc:
        raise click.ClickException(str(exc)) from exc


def _print_tokenizer_check(provider_id: str, model_id: str, sample_text: str) -> None:
    registry = TokenizerRegistry()
    resolved = registry.resolve_model(provider_id, model_id)
    result = registry.count(sample_text, resolved.tokenizer)
    mode = "approximate" if result.approximate else "exact"
    click.echo(
        f"Tokenizer Check: {mode} (provider={resolved.provider_id}, model={resolved.model_id}, "
        f"kind={resolved.tokenizer.kind}, value={resolved.tokenizer.value})"
    )
    if result.warning:
        click.echo(f"Tokenizer Warning: {result.warning}")


@main.command()
@click.option("--session-id", default=None)
@click.pass_context
def session(ctx: click.Context, session_id: str | None) -> None:
    """Show telemetry summary for one session."""
    options = ctx.obj
    sessions = _list_sessions(options)
    sid = session_id or _pick_latest_session_id(sessions)
    if not sid:
        raise click.ClickException("No sessions available.")
    messages = _get_messages(options, sid)
    canonical = build_canonical_metrics(sid, messages)
    top_tools = [
        {
            "name": t["tool"],
            "output_tokens": t["tokens"],
            "call_count": t["calls"],
        }
        for t in canonical.tool_rows[:5]
    ]
    mcp_stats = {"rows": canonical.mcp_rows, "total_tokens": sum(r["tokens"] for r in canonical.mcp_rows)}
    component_stats = {"rows": canonical.component_rows, "total_tokens": sum(r["tokens"] for r in canonical.component_rows)}
    contributor_stats = {
        "rows": canonical.contributor_rows,
        "total_tokens": sum(r["tokens"] for r in canonical.contributor_rows),
    }
    model_costs = [{"model": canonical.model, "cost": round(canonical.estimated_cost_usd, 6)}]
    print_session_report(
        sid,
        canonical.api_calls,
        canonical.session_total_tokens,
        canonical.estimated_cost_usd,
        token_composition=canonical.token_composition,
        top_tools=top_tools,
        mcp_stats=mcp_stats,
        component_stats=component_stats,
        contributor_stats=contributor_stats,
        model_costs=model_costs,
    )


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show quick status for current data source."""
    options = ctx.obj
    sessions = _list_sessions(options)
    print_status_report(str(options["mode"]), sessions)


@main.command()
@click.pass_context
def daily(ctx: click.Context) -> None:
    """Show last 1 day aggregate."""
    _print_period_report(ctx.obj, days=1, label="daily")


@main.command()
@click.pass_context
def weekly(ctx: click.Context) -> None:
    """Show last 7 days aggregate."""
    _print_period_report(ctx.obj, days=7, label="weekly")


@main.command(name="month")
@click.pass_context
def month_cmd(ctx: click.Context) -> None:
    """Show last 30 days aggregate."""
    _print_period_report(ctx.obj, days=30, label="month")


@main.command()
@click.option("--from-date", required=True, help="YYYY-MM-DD")
@click.option("--to-date", required=True, help="YYYY-MM-DD")
@click.pass_context
def range(ctx: click.Context, from_date: str, to_date: str) -> None:
    """Show aggregate for explicit date range."""
    start = _parse_date(from_date)
    end = _parse_date(to_date) + timedelta(days=1)
    report = _build_period_report(ctx.obj, start, end)
    _print_report("range", report)


@main.command(name="json")
@click.option(
    "--period",
    type=click.Choice(["daily", "weekly", "month"]),
    default="daily",
    show_default=True,
)
@click.option("--format", "output_format", type=click.Choice(["json", "md"]), default="json", show_default=True)
@click.pass_context
def json_cmd(ctx: click.Context, period: str, output_format: str) -> None:
    """Emit aggregate report as JSON."""
    days = {"daily": 1, "weekly": 7, "month": 30}[period]
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    session_metrics = _collect_period_session_metrics(ctx.obj, start, end)
    payload = build_report_schema(
        period=period,
        mode=str(ctx.obj["mode"]),
        start=start,
        end=end,
        session_metrics=session_metrics,
    )
    if output_format == "md":
        click.echo(report_to_markdown(payload))
    else:
        click.echo(json.dumps(payload))


def _pick_latest_session_id(sessions: list[dict[str, object]]) -> str | None:
    if not sessions:
        return None
    first = sessions[0]
    sid = first.get("id")
    if isinstance(sid, str) and sid:
        return sid
    return None


def _effective_source(compat_source: str, mode: str) -> str:
    if compat_source == "auto":
        return mode
    return compat_source


def _print_compatibility_check(
    messages: list[dict[str, object]],
    *,
    mode: str,
    source: str,
    session_id: str,
) -> None:
    result = analyze_context_compatibility(
        messages, mode=mode, source=source  # type: ignore[arg-type]
    )
    click.echo(
        f"Compatibility Check: mode={result.mode}, source={source}, session={session_id}, observed_tools_only={result.observed_tools_only}"
    )
    for warning in result.warnings:
        click.echo(f"Compatibility Warning: {warning}")
    if result.tool_schema_estimates:
        top = result.tool_schema_estimates[:5]
        for est in top:
            click.echo(
                f"Tool Estimate: {est.name} tokens={est.estimated_tokens} args={est.argument_count} complex={est.has_complex_args}"
            )


def _print_period_report(options: dict[str, object], *, days: int, label: str) -> None:
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    report = _build_period_report(options, start, end)
    _print_report(label, report)


def _build_period_report(
    options: dict[str, object], start: datetime, end: datetime
) -> dict[str, object]:
    session_metrics = _collect_period_session_metrics(options, start, end)
    sessions = _list_sessions(options)
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    used = 0
    token_composition = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0, "tool_output": 0}
    tool_map: dict[str, dict[str, int]] = defaultdict(lambda: {"output_tokens": 0, "call_count": 0})
    mcp_map: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "calls": 0.0})
    component_map: dict[str, float] = defaultdict(float)
    contributor_map: dict[str, float] = defaultdict(float)
    model_map: dict[str, float] = defaultdict(float)

    for canonical in session_metrics:
        used += 1
        total_calls += canonical.api_calls
        total_tokens += canonical.session_total_tokens
        total_cost += canonical.estimated_cost_usd
        for k in token_composition.keys():
            token_composition[k] += int(canonical.token_composition.get(k, 0))
        for t in canonical.tool_rows:
            tool_map[str(t["tool"])]["output_tokens"] += int(t["tokens"])
            tool_map[str(t["tool"])]["call_count"] += int(t["calls"])
        for r in canonical.mcp_rows:
            mcp_map[str(r["name"])]["tokens"] += float(r["tokens"])
            mcp_map[str(r["name"])]["calls"] += float(r["calls"])
        for r in canonical.component_rows:
            component_map[f"{r['component_type']}|{r['component_group']}|{r['component_name']}"] += float(r["tokens"])
        for row in canonical.contributor_rows:
            contributor_map[row["name"]] += float(row["tokens"])
        model_map[canonical.model] += float(canonical.estimated_cost_usd)

    top_tools = sorted(
        [
            {
                "name": name,
                "output_tokens": data["output_tokens"],
                "call_count": data["call_count"],
            }
            for name, data in tool_map.items()
        ],
        key=lambda x: (x["output_tokens"], x["call_count"]),
        reverse=True,
    )[:5]

    return {
        "sessions": used,
        "api_calls": total_calls,
        "tokens": total_tokens,
        "api_cost": round(total_cost, 6),
        "from": start.isoformat(),
        "to": end.isoformat(),
        "token_composition": token_composition,
        "top_tools": top_tools,
        "mcp_stats": _finalize_mcp_stats(mcp_map),
        "component_stats": _finalize_component_stats_canonical(component_map),
        "contributor_stats": _finalize_contributor_stats(contributor_map),
        "model_costs": _finalize_model_costs(model_map),
    }


def _collect_period_session_metrics(
    options: dict[str, object], start: datetime, end: datetime
) -> list[object]:
    sessions = _list_sessions(options)
    out = []
    for sess in sessions:
        created = _session_created_at(sess)
        if created is None or created < start or created >= end:
            continue
        sid = sess.get("id")
        if not isinstance(sid, str) or not sid:
            continue
        messages = _get_messages(options, sid)
        out.append(build_canonical_metrics(sid, messages))
    return out


def _print_report(label: str, report: dict[str, object]) -> None:
    print_period_report(label, report)


def _list_sessions(options: dict[str, object]) -> list[dict[str, object]]:
    if options["mode"] == "local":
        db_path = LocalSessionService.find_database_path(options.get("db_path"))
        service = LocalSessionService(db_path=db_path)
        return service.list_sessions()

    with OpencodeApiClient(
        base_url=options["base_url"],
        username=options["username"],
        password=options["password"],
        timeout=options["timeout"],
        retries=options["retries"],
    ) as client:
        service = SessionService(client)
        return service.list_sessions()


def _get_messages(options: dict[str, object], session_id: str) -> list[dict[str, object]]:
    if options["mode"] == "local":
        db_path = LocalSessionService.find_database_path(options.get("db_path"))
        service = LocalSessionService(db_path=db_path)
        return service.get_messages(session_id)

    with OpencodeApiClient(
        base_url=options["base_url"],
        username=options["username"],
        password=options["password"],
        timeout=options["timeout"],
        retries=options["retries"],
    ) as client:
        service = SessionService(client)
        return service.get_messages(session_id)


def _session_created_at(session: dict[str, object]) -> datetime | None:
    raw = session.get("time_created")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value > 10_000_000_000:
        return datetime.fromtimestamp(value / 1000, UTC)
    return datetime.fromtimestamp(value, UTC)


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise click.ClickException(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


_LOCAL_TOOL_RE = re.compile(r"^(read|bash|glob|todowrite|task|tokenscope|apply_patch|skill)$")


def _tool_group(tool_name: str) -> str:
    if "_" in tool_name:
        return tool_name.split("_", 1)[0]
    if "-" in tool_name:
        return tool_name.split("-", 1)[0]
    return tool_name


def _build_mcp_stats(tool_usage: list[object]) -> dict[str, object]:
    mcp_map: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "calls": 0.0})
    _accumulate_mcp(mcp_map, tool_usage)
    return _finalize_mcp_stats(mcp_map)


def _accumulate_mcp(mcp_map: dict[str, dict[str, float]], tool_usage: list[object]) -> None:
    for t in tool_usage:
        tool_name = getattr(t, "tool_name", None)
        output_tokens = getattr(t, "output_tokens", 0)
        call_count = getattr(t, "call_count", 0)
        if not isinstance(tool_name, str):
            continue
        if _LOCAL_TOOL_RE.match(tool_name):
            continue
        group = _tool_group(tool_name)
        mcp_map[group]["tokens"] += float(output_tokens)
        mcp_map[group]["calls"] += float(call_count)


def _finalize_mcp_stats(mcp_map: dict[str, dict[str, float]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total_tokens = sum(v["tokens"] for v in mcp_map.values())
    for name, data in mcp_map.items():
        calls = int(data["calls"])
        tokens = int(data["tokens"])
        rows.append(
            {
                "name": name,
                "tokens": tokens,
                "calls": calls,
                "tokens_per_call": round(tokens / max(calls, 1), 2),
                "percent": round((tokens / total_tokens * 100.0), 2) if total_tokens > 0 else 0.0,
            }
        )
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return {"rows": rows[:10], "total_tokens": int(total_tokens)}


def _build_component_stats(summary, attribution) -> dict[str, object]:
    component_map: dict[str, float] = defaultdict(float)
    _accumulate_components(component_map, summary, attribution)
    return _finalize_component_stats(component_map)


def _build_component_stats_from_attribution(attribution) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total = 0
    for tool in attribution.tool_usage:
        group = _tool_group(tool.tool_name)
        tokens = int(tool.output_tokens)
        total += tokens
        rows.append(
            {
                "component_type": "tool",
                "component_group": group,
                "component_name": tool.tool_name,
                "tokens": tokens,
                "estimated_session_tokens": tokens,
                "calls": int(tool.call_count),
            }
        )

    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    for row in rows:
        row["percent"] = round((int(row["tokens"]) / total * 100.0), 2) if total > 0 else 0.0
    return {"rows": rows[:15], "total_tokens": total}


def _accumulate_components(component_map: dict[str, float], summary, attribution) -> None:
    component_map["input"] += float(summary.input_tokens)
    component_map["output"] += float(summary.output_tokens)
    component_map["reasoning"] += float(summary.reasoning_tokens)
    component_map["cache_read"] += float(summary.cache_read_tokens)
    component_map["cache_write"] += float(summary.cache_write_tokens)
    component_map["system"] += float(attribution.totals.system_tokens)
    component_map["user"] += float(attribution.totals.user_tokens)
    component_map["assistant"] += float(attribution.totals.assistant_tokens)
    component_map["tool_output"] += float(attribution.totals.tool_output_tokens)


def _build_contributor_stats(attribution) -> dict[str, object]:
    cmap: dict[str, float] = defaultdict(float)
    _accumulate_contributors(cmap, attribution)
    return _finalize_contributor_stats(cmap)


def _accumulate_contributors(contributor_map: dict[str, float], attribution) -> None:
    for tool in attribution.tool_usage:
        contributor_map[tool.tool_name] += float(tool.output_tokens)
    if attribution.totals.system_tokens > 0:
        contributor_map["System"] += float(attribution.totals.system_tokens)


def _finalize_contributor_stats(contributor_map: dict[str, float]) -> dict[str, object]:
    total = sum(contributor_map.values())
    rows = []
    for name, tokens in contributor_map.items():
        t = int(tokens)
        rows.append(
            {
                "name": name,
                "tokens": t,
                "percent": round((tokens / total * 100.0), 2) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return {"rows": rows[:10], "total_tokens": int(total)}


def _finalize_component_stats(component_map: dict[str, float]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total = sum(component_map.values())
    for name, tokens in component_map.items():
        t = int(tokens)
        rows.append(
            {
                "group": name,
                "tokens": t,
                "percent": round((tokens / total * 100.0), 2) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return {"rows": rows, "total_tokens": int(total)}


def _finalize_component_stats_canonical(component_map: dict[str, float]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total = sum(component_map.values())
    for key, tokens in component_map.items():
        t = int(tokens)
        parts = key.split("|", 2)
        ctype = parts[0] if len(parts) > 0 else "component"
        cgroup = parts[1] if len(parts) > 1 else "unknown"
        cname = parts[2] if len(parts) > 2 else key
        rows.append(
            {
                "component_type": ctype,
                "component_group": cgroup,
                "component_name": cname,
                "tokens": t,
                "estimated_session_tokens": t,
                "calls": 0,
                "percent": round((tokens / total * 100.0), 2) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return {"rows": rows[:15], "total_tokens": int(total)}


def _extract_model_id_from_message(message: dict[str, object]) -> str:
    info = message.get("info")
    if isinstance(info, dict):
        model_id = info.get("modelID")
        if isinstance(model_id, str) and model_id:
            return model_id
        model = info.get("model")
        if isinstance(model, dict):
            nested = model.get("modelID")
            if isinstance(nested, str) and nested:
                return nested
    return "unknown"


def _build_model_costs_from_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    model_map: dict[str, float] = defaultdict(float)
    _accumulate_model_costs(model_map, messages)
    return _finalize_model_costs(model_map)


def _accumulate_model_costs(model_map: dict[str, float], messages: list[dict[str, object]]) -> None:
    for msg in messages:
        role = msg.get("role")
        if role != "assistant":
            continue
        model_id = _extract_model_id_from_message(msg)
        message_cost = summarize_telemetry(collect_telemetry_calls([msg])).total_cost
        model_map[model_id] += float(message_cost)


def _finalize_model_costs(model_map: dict[str, float]) -> list[dict[str, object]]:
    rows = [{"model": model, "cost": round(cost, 6)} for model, cost in model_map.items()]
    rows.sort(key=lambda x: float(x["cost"]), reverse=True)
    return rows[:10]
