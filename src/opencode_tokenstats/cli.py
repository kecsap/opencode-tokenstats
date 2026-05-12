from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from collections import defaultdict
from pathlib import Path
import re
import sys

import click

try:
    import tqdm as tqdm_mod

    TQDM_AVAILABLE = True
except Exception:
    tqdm_mod = None  # type: ignore[assignment]
    TQDM_AVAILABLE = False

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opencode_tokenstats.client import ApiClientError, OpencodeApiClient
    from opencode_tokenstats.activity_classifier import classify_session, CATEGORY_LABELS, extract_root_dir
    from opencode_tokenstats.canonical_metrics import build_canonical_metrics
    from opencode_tokenstats.compatibility import analyze_context_compatibility
    from opencode_tokenstats.local_session_service import LocalSessionService, LocalStorageError
    from opencode_tokenstats.renderer import print_period_report, print_session_report, print_status_report
    from opencode_tokenstats.report_schema import build_report_schema, report_to_markdown
    from opencode_tokenstats.session_service import SessionService
    from opencode_tokenstats.tokenization import TokenizerRegistry
    from opencode_tokenstats.pricing import load_model_aliases
else:
    from .activity_classifier import classify_session, CATEGORY_LABELS, extract_root_dir
    from .client import ApiClientError, OpencodeApiClient
    from .canonical_metrics import build_canonical_metrics
    from .compatibility import analyze_context_compatibility
    from .local_session_service import LocalSessionService, LocalStorageError
    from .renderer import print_period_report, print_session_report, print_status_report
    from .report_schema import build_report_schema, report_to_markdown
    from .session_service import SessionService
    from .tokenization import TokenizerRegistry
    from .pricing import load_model_aliases


class OrderedCommandsGroup(click.Group):
    _ORDER = [
        "daily",
        "weekly",
        "month",
        "range",
        "lifetime",
        "session",
        "status",
        "json",
        "health",
        "tokenizer-warmup",
    ]

    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = list(self.commands)
        rank = {name: idx for idx, name in enumerate(self._ORDER)}
        commands.sort(key=lambda name: (rank.get(name, len(self._ORDER)), name))
        return commands


@click.group(
    cls=OrderedCommandsGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "OpenCode TokenStats CLI. Local-first session analytics with canonical\n"
        "TokenScope-compatible metrics, Rich console panels, and JSON/Markdown outputs."
    ),
)
@click.option("--base-url", default="http://127.0.0.1:4096", show_default=True)
@click.option("--username", default=None)
@click.option("--password", default=None)
@click.option("--timeout", default=10.0, show_default=True, type=float)
@click.option("--retries", default=2, show_default=True, type=int)
@click.option("--mode", type=click.Choice(["local", "api"]), default="local", show_default=True)
@click.option("--db-path", default=None)
@click.option("--no-warmup", is_flag=True, help="Disable automatic tokenizer warmup")
@click.option("--model-alias-file", default=None, help="Path to models.conf alias file")
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
    no_warmup: bool,
    model_alias_file: str | None,
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
        "model_alias_file": model_alias_file,
        "no_warmup": no_warmup,
    }

    if not no_warmup and ctx.invoked_subcommand != "tokenizer-warmup":
        _run_default_warmup_silent()


@main.command(name="health", short_help="Health check + optional tokenizer/compat checks")
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
def health(
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


@main.command(name="tokenizer-warmup", short_help="Preload tokenizer caches")
@click.option(
    "--pair",
    "pairs",
    multiple=True,
    help="provider:model pair, repeatable (e.g. openai:gpt-5.3-codex)",
)
@click.option("--sample-text", default="warmup", show_default=True)
def tokenizer_warmup(pairs: tuple[str, ...], sample_text: str) -> None:
    """Warm tokenizer cache for selected models."""
    registry = TokenizerRegistry()
    parsed: list[tuple[str, str]] = []

    if pairs:
        for pair in pairs:
            if ":" not in pair:
                raise click.ClickException(f"Invalid --pair '{pair}', expected provider:model")
            provider, model = pair.split(":", 1)
            provider = provider.strip()
            model = model.strip()
            if not provider or not model:
                raise click.ClickException(f"Invalid --pair '{pair}', expected provider:model")
            parsed.append((provider, model))
    else:
        parsed = [
            ("local", "qwen3.6-27b"),
            ("openai", "gpt-5.3-codex"),
            ("anthropic", "claude-sonnet-4"),
        ]

    # Skip models with unavailable tokenizers (approx fallback)
    available_pairs = [
        (p, m) for p, m in parsed
        if registry.is_tokenizer_available(p, m)
    ]
    skipped = [
        (p, m) for p, m in parsed
        if not registry.is_tokenizer_available(p, m)
    ]
    if skipped:
        for p, m in skipped:
            resolved = registry.resolve_model(p, m)
            click.echo(f"Skipping {p}:{m} (tokenizer unavailable, would use {resolved.tokenizer.kind} fallback)")

    if not available_pairs:
        click.echo("No models with available tokenizers. Nothing to warmup.")
        return

    max_workers = min(len(available_pairs), max(os.cpu_count() or 1, 1))
    results = registry.warmup_parallel(available_pairs, sample_text=sample_text, max_workers=max_workers)
    warmed = sum(1 for r in results if r.status == "warmed")
    failed = sum(1 for r in results if r.status == "failed")

    click.echo(f"Tokenizer warmup: warmed={warmed} skipped={len(skipped)} failed={failed}")
    for r in results:
        click.echo(
            f"- {r.provider_id}:{r.model_id} kind={r.tokenizer_kind} value={r.tokenizer_value} status={r.status}"
        )
        if r.warning:
            click.echo(f"  warning: {r.warning}")


def _run_default_warmup_silent() -> None:
    try:
        registry = TokenizerRegistry()
        all_pairs = [
            ("local", "qwen3.6-27b"),
            ("openai", "gpt-5.3-codex"),
            ("anthropic", "claude-sonnet-4"),
        ]
        # Only warmup models with available tokenizers (skip approx fallback)
        available_pairs = [
            (p, m) for p, m in all_pairs
            if registry.is_tokenizer_available(p, m)
        ]
        if not available_pairs:
            # No tokenizers available; warmup would be a no-op.
            return
        with _SessionProgress(desc="Warming tokenizer cache") as prog:
            results = registry.warmup_parallel(available_pairs, sample_text="warmup")
            for idx, _ in enumerate(results, start=1):
                prog.update(idx, len(available_pairs))
    except Exception:
        # Best-effort optimization only; never block CLI commands.
        return


@main.command(short_help="Show one session summary")
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
        for t in canonical.tool_rows[:10]
    ]
    mcp_stats = {"rows": canonical.mcp_rows, "total_tokens": sum(r["tokens"] for r in canonical.mcp_rows)}
    core_stats = {"rows": canonical.core_rows, "total_tokens": sum(r["tokens"] for r in canonical.core_rows)}
    component_stats = {"rows": canonical.component_family_rows, "total_tokens": sum(r["tokens"] for r in canonical.component_family_rows)}
    model_costs = [
        {
            "model": canonical.model,
            "api_cost": round(canonical.actual_cost_usd, 6),
            "estimated_cost": round(canonical.estimated_cost_usd, 6),
            "cost": round(canonical.actual_cost_usd if canonical.actual_cost_usd > 0 else canonical.estimated_cost_usd, 6),
        }
    ]
    print_session_report(
        sid,
        canonical.api_calls,
        canonical.session_total_tokens,
        canonical.estimated_cost_usd,
        token_composition=canonical.token_composition,
        top_tools=top_tools,
        mcp_stats=mcp_stats,
        core_stats=core_stats,
        component_stats=component_stats,
        model_costs=model_costs,
    )


@main.command(short_help="Show source/session status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show quick status for current data source."""
    options = ctx.obj
    sessions = _list_sessions(options)
    print_status_report(str(options["mode"]), sessions)


@main.command(short_help="Aggregate last 24 hours")
@click.pass_context
def daily(ctx: click.Context) -> None:
    """Show last 1 day aggregate."""
    _print_period_report(ctx.obj, days=1, label="daily")


@main.command(short_help="Aggregate last 7 days")
@click.pass_context
def weekly(ctx: click.Context) -> None:
    """Show last 7 days aggregate."""
    _print_period_report(ctx.obj, days=7, label="weekly")


@main.command(name="month", short_help="Aggregate last 30 days or specific month")
@click.argument("month", required=False)
@click.pass_context
def month_cmd(ctx: click.Context, month: str | None) -> None:
    """Show last 30 days aggregate, or stats for a specific month (name or number)."""
    if month is None:
        _print_period_report(ctx.obj, days=30, label="month")
    else:
        start, end = _month_window(month)
        with _SessionProgress() as prog:
            report = _build_period_report(ctx.obj, start, end, progress_callback=prog.update)
        _print_report("month", report)


@main.command(short_help="Aggregate all available sessions")
@click.pass_context
def lifetime(ctx: click.Context) -> None:
    """Show all-time aggregate across all available sessions."""
    _print_lifetime_report(ctx.obj)


@main.command(short_help="Aggregate explicit date window (e.g. 2026-05-01..2026-05-07)")
@click.option("--from-date", required=True, help="YYYY-MM-DD (e.g. 2026-05-01)")
@click.option("--to-date", required=True, help="YYYY-MM-DD (e.g. 2026-05-07)")
@click.pass_context
def range(ctx: click.Context, from_date: str, to_date: str) -> None:
    """Show aggregate for explicit date range."""
    start = _parse_date(from_date)
    end = _parse_date(to_date) + timedelta(days=1)
    with _SessionProgress() as prog:
        report = _build_period_report(ctx.obj, start, end, progress_callback=prog.update)
    _print_report("range", report)


@main.command(name="json", short_help="Emit canonical report schema")
@click.option(
    "--period",
    type=click.Choice(["daily", "weekly", "month", "lifetime"]),
    default="daily",
    show_default=True,
)
@click.option("--format", "output_format", type=click.Choice(["json", "md"]), default="json", show_default=True)
@click.pass_context
def json_cmd(ctx: click.Context, period: str, output_format: str) -> None:
    """Emit aggregate report as JSON."""
    if period == "lifetime":
        start, end = _lifetime_window(ctx.obj)
    else:
        days = {"daily": 1, "weekly": 7, "month": 30}[period]
        end = datetime.now(UTC)
        start = end - timedelta(days=days)
    with _SessionProgress() as prog:
        session_metrics = _collect_period_session_metrics(
            ctx.obj, start, end, progress_callback=prog.update
        )
    payload = build_report_schema(
        period=period,
        mode=str(ctx.obj["mode"]),
        start=start,
        end=end,
        session_metrics=session_metrics,
        model_alias_file=ctx.obj.get("model_alias_file"),
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
        top = result.tool_schema_estimates[:10]
        for est in top:
            click.echo(
                f"Tool Estimate: {est.name} tokens={est.estimated_tokens} args={est.argument_count} complex={est.has_complex_args}"
            )


class _SessionProgress:
    """Context manager that shows a transient progress bar during session data collection."""

    def __init__(self, desc: str = "Gathering OpenCode session data") -> None:
        self._bar: object | None = None
        self._desc = desc

    def __enter__(self) -> "_SessionProgress":
        if TQDM_AVAILABLE and tqdm_mod is not None and sys.stderr.isatty():
            self._bar = tqdm_mod.tqdm(
                total=0,
                unit="sessions",
                unit_divisor=1,
                file=sys.stderr,
                leave=False,
                desc=self._desc,
                ascii="█░",
                bar_format="{desc}: |{bar:20}| {n_fmt}/{total_fmt}",
            )
        return self

    def __exit__(self, *args: object) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def update(self, current: int, total: int) -> None:
        if self._bar is not None:
            if self._bar.total == 0:
                self._bar.total = total
                self._bar.refresh()
            self._bar.update(1)


def _print_period_report(options: dict[str, object], *, days: int, label: str) -> None:
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    with _SessionProgress() as prog:
        report = _build_period_report(options, start, end, progress_callback=prog.update)
    _print_report(label, report)


def _print_lifetime_report(options: dict[str, object]) -> None:
    with _SessionProgress() as prog:
        start, end = _lifetime_window(options)
        report = _build_period_report(options, start, end, progress_callback=prog.update)
    _print_report("lifetime", report)


def _lifetime_window(options: dict[str, object]) -> tuple[datetime, datetime]:
    sessions = _list_sessions(options)
    created_times = [dt for dt in (_session_created_at(s) for s in sessions) if dt is not None]
    if created_times:
        start = min(created_times)
    else:
        start = datetime.fromtimestamp(0, UTC)
    end = datetime.now(UTC) + timedelta(seconds=1)
    return start, end


def _build_period_report(
    options: dict[str, object],
    start: datetime,
    end: datetime,
    *,
    progress_callback: callable | None = None,
) -> dict[str, object]:
    session_metrics = _collect_period_session_metrics(
        options, start, end, progress_callback=progress_callback
    )
    sessions = _list_sessions(options)
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    used = 0
    token_composition = {
        "input": 0,
        "cache_read": 0,
        "cache_write": 0,
        "output": 0,
        "reasoning": 0,
        "web_search_requests": 0,
    }
    tool_map: dict[str, dict[str, int]] = defaultdict(lambda: {"output_tokens": 0, "call_count": 0})
    mcp_map: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "calls": 0.0})
    component_map: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "calls": 0.0})
    core_map: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "calls": 0.0})
    aliases = load_model_aliases(options.get("model_alias_file"))
    model_map: dict[str, dict[str, float]] = defaultdict(lambda: {"api_cost": 0.0, "estimated_cost": 0.0})

    # Build directory lookup for root_dir extraction (from session.directory)
    session_dirs: dict[str, str] = {
        str(sess.get("id", "")): str(sess.get("directory", "")) for sess in sessions
    }
    # Activity aggregation maps
    activity_map: dict[str, dict[str, object]] = {}
    session_rows: list[dict[str, object]] = []

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
            if r["component_group"] == "opencode-core":
                core_map[r["component_name"]]["tokens"] += float(r["tokens"])
                core_map[r["component_name"]]["calls"] += int(r.get("calls", 0))
            else:
                component_map[f"{r['component_type']}|{r['component_group']}|{r['component_name']}"]["tokens"] += float(r["tokens"])
                component_map[f"{r['component_type']}|{r['component_group']}|{r['component_name']}"]["calls"] += int(r.get("calls", 0))
        model_key = aliases.get(canonical.model, canonical.model)
        model_map[model_key]["api_cost"] += float(canonical.actual_cost_usd)
        model_map[model_key]["estimated_cost"] += float(canonical.estimated_cost_usd)

        # Classify session and aggregate by activity
        category = classify_session(canonical)
        if category not in activity_map:
            activity_map[category] = {"tokens": 0, "calls": 0, "api_cost": 0.0, "estimated_cost": 0.0}
        activity_map[category]["tokens"] += canonical.session_total_tokens
        activity_map[category]["calls"] += canonical.api_calls
        activity_map[category]["api_cost"] += canonical.actual_cost_usd
        activity_map[category]["estimated_cost"] += canonical.estimated_cost_usd

        # Build per-session row for top_sessions
        raw_dir = session_dirs.get(canonical.session_id, "")
        root_dir = extract_root_dir(raw_dir)
        session_rows.append(
            {
                "root_dir": root_dir,
                "tokens": canonical.session_total_tokens,
                "api_cost": round(canonical.actual_cost_usd, 6),
                "estimated_cost": round(canonical.estimated_cost_usd, 6),
            }
        )

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
    )[:10]

    # Format by_activity rows sorted by cost desc
    by_activity = [
        {
            "category": cat,
            "label": CATEGORY_LABELS.get(cat, cat.title()),
            "tokens": data["tokens"],
            "calls": data["calls"],
            "api_cost": round(data["api_cost"], 6),
            "estimated_cost": round(data["estimated_cost"], 6),
        }
        for cat, data in activity_map.items()
    ]
    by_activity.sort(key=lambda x: x["api_cost"] if x["api_cost"] > 0 else x["estimated_cost"], reverse=True)

   # Aggregate sessions by root_dir, summing tokens and cost
    dir_map: dict[str, dict[str, object]] = {}
    for row in session_rows:
        rd = row["root_dir"]
        if rd not in dir_map:
            dir_map[rd] = {"root_dir": rd, "tokens": 0, "api_cost": 0.0, "estimated_cost": 0.0}
        dir_map[rd]["tokens"] += row["tokens"]
        dir_map[rd]["api_cost"] += row["api_cost"]
        dir_map[rd]["estimated_cost"] += row["estimated_cost"]
    top_sessions = [
        {
            "root_dir": rd,
            "tokens": data["tokens"],
            "api_cost": round(data["api_cost"], 6),
            "estimated_cost": round(data["estimated_cost"], 6),
        }
        for rd, data in dir_map.items()
    ]
    top_sessions.sort(key=lambda x: x["api_cost"] if x["api_cost"] > 0 else x["estimated_cost"], reverse=True)
    top_sessions = top_sessions[:10]

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
        "core_stats": _finalize_core_stats(core_map),
        "component_stats": _finalize_component_stats_canonical(
            component_map,
            core_tokens=sum(d["tokens"] for d in core_map.values()),
            core_calls=sum(d["calls"] for d in core_map.values()),
        ),
        "model_costs": _finalize_model_costs(model_map),
        "by_activity": by_activity,
        "top_sessions": top_sessions,
    }


def _collect_period_session_metrics(
    options: dict[str, object],
    start: datetime,
    end: datetime,
    *,
    progress_callback: callable | None = None,
) -> list[object]:
    sessions = _list_sessions(options)
    out = []
    total = len(sessions)

    # Filter sessions by date range first
    eligible_sessions = []
    for sess in sessions:
        created = _session_created_at(sess)
        sid = sess.get("id")
        if created is not None and created >= start and created < end and isinstance(sid, str) and sid:
            eligible_sessions.append((sid, sess))

    if not eligible_sessions:
        return out

    # Parallel fetch messages for eligible sessions
    def fetch_session(sid: str) -> tuple[str, list[dict[str, object]]]:
        messages = _get_messages(options, sid)
        return (sid, messages)

    max_workers = min(len(eligible_sessions), 8)  # Cap at 8 workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_session, sid): sid for sid, _ in eligible_sessions}

        results = {}
        for future in as_completed(futures):
            try:
                sid, messages = future.result()
                results[sid] = messages
            except Exception:
                pass

    # Build canonical metrics in parallel across CPU cores.
    worker_count = min(len(results), max(os.cpu_count() or 1, 1))
    if worker_count <= 1:
        for sid, messages in results.items():
            out.append(build_canonical_metrics(sid, messages))
            if progress_callback:
                progress_callback(len(out), len(eligible_sessions))
        return out

    start_method = "fork" if sys.platform.startswith("linux") else "spawn"
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context(start_method)) as executor:
        futures = {
            executor.submit(_build_session_metrics, sid, messages): sid
            for sid, messages in results.items()
        }
        done = 0
        for future in as_completed(futures):
            sid = futures[future]
            try:
                out.append(future.result())
            except Exception:
                # Best-effort: fall back to local compute if worker fails.
                messages = results.get(sid, [])
                out.append(build_canonical_metrics(sid, messages))
            done += 1
            if progress_callback:
                progress_callback(done, len(eligible_sessions))

    return out


def _build_session_metrics(session_id: str, messages: list[dict[str, object]]) -> object:
    return build_canonical_metrics(session_id, messages)


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


_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _month_window(month_arg: str) -> tuple[datetime, datetime]:
    """Resolve a month argument to a date range for the current year."""
    now = datetime.now(UTC)
    year = now.year

    lower = month_arg.lower().strip()

    # Try numeric (01-12)
    if lower.isdigit():
        month_num = int(lower)
        if month_num < 1 or month_num > 12:
            raise click.ClickException(f"Invalid month '{month_arg}', expected 01-12 or month name")
    else:
        # Try name lookup
        if lower not in _MONTHS:
            raise click.ClickException(
                f"Invalid month '{month_arg}', expected month name (jan-dec) or number (01-12)"
            )
        month_num = _MONTHS[lower]

    # Build date range: first day of month to first day of next month
    from datetime import datetime as dt
    start = dt(year, month_num, 1, tzinfo=UTC)
    if month_num == 12:
        end = dt(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = dt(year, month_num + 1, 1, tzinfo=UTC)

    return start, end


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
    family: dict[str, dict[str, object]] = {}
    total = 0
    for tool in attribution.tool_usage:
        group = _tool_group(tool.tool_name)
        tokens = int(tool.output_tokens)
        calls = int(tool.call_count)
        total += tokens
        if group not in family:
            family[group] = {
                "component_type": "tool",
                "component_group": group,
                "tokens": 0,
                "estimated_session_tokens": 0,
                "calls": 0,
            }
        family[group]["tokens"] += tokens
        family[group]["estimated_session_tokens"] += tokens
        family[group]["calls"] += calls

    rows = list(family.values())
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    for row in rows:
        row["percent"] = round((int(row["tokens"]) / total * 100.0), 2) if total > 0 else 0.0
    return {"rows": rows[:15], "total_tokens": total}


def _accumulate_components(component_map: dict[str, float], summary, attribution) -> None:
    component_map["input"] += float(summary.input_tokens)
    component_map["output"] += float(summary.output_tokens)
    component_map["reasoning"] += float(summary.reasoning_tokens)
    component_map["cache_read"] += float(summary.cache_read_tokens)
    component_map["system"] += float(attribution.totals.system_tokens)
    component_map["user"] += float(attribution.totals.user_tokens)
    component_map["assistant"] += float(attribution.totals.assistant_tokens)
    component_map["tool_output"] += float(attribution.totals.tool_output_tokens)


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


def _finalize_core_stats(core_map: dict[str, dict[str, float]]) -> dict[str, object]:
    normalized: dict[str, dict[str, float]] = {}
    for name, data in core_map.items():
        target = "general" if name == "invalid" else name
        if target not in normalized:
            normalized[target] = {"tokens": 0.0, "calls": 0.0}
        normalized[target]["tokens"] += float(data["tokens"])
        normalized[target]["calls"] += float(data["calls"])

    rows: list[dict[str, object]] = []
    for name, data in normalized.items():
        t = int(data["tokens"])
        c = int(data["calls"])
        rows.append(
            {
                "component_name": name,
                "tokens": t,
                "calls": c,
            }
        )
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return {"rows": rows, "total_tokens": int(sum(d["tokens"] for d in normalized.values()))}


def _finalize_component_stats_canonical(component_map: dict[str, dict[str, float]], *, core_tokens: float = 0.0, core_calls: float = 0.0) -> dict[str, object]:
    family: dict[str, dict[str, Any]] = {}
    type_sets: dict[str, set[str]] = {}
    for key, data in component_map.items():
        parts = key.split("|", 2)
        ctype = parts[0] if len(parts) > 0 else "component"
        cgroup = parts[1] if len(parts) > 1 else "unknown"
        if cgroup not in family:
            family[cgroup] = {"tokens": 0.0, "calls": 0.0}
        type_sets.setdefault(cgroup, set()).add(ctype)
        family[cgroup]["tokens"] += float(data["tokens"])
        family[cgroup]["calls"] += float(data["calls"])

    total = sum(v["tokens"] for v in family.values()) + float(core_tokens)
    rows: list[dict[str, object]] = []
    for cgroup, data in family.items():
        t = int(data["tokens"])
        c = int(data["calls"])
        types = type_sets.get(cgroup, set())
        rows.append(
            {
                "component_type": "mixed" if len(types) > 1 else (types.pop() if types else "unknown"),
                "component_group": cgroup,
                "tokens": t,
                "calls": c,
                "percent": round((data["tokens"] / total * 100.0), 2) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    if core_tokens > 0:
        t = int(core_tokens)
        c = int(core_calls)
        rows.append(
            {
                "component_type": "core",
                "component_group": "opencode-core",
                "tokens": t,
                "calls": c,
                "percent": round((core_tokens / total * 100.0), 2) if total > 0 else 0.0,
            }
        )
        rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return {"rows": rows[:15], "total_tokens": int(total)}


def _extract_model_id_from_message(message: dict[str, object]) -> str:
    info = message.get("info")
    if isinstance(info, dict):
        model_id = info.get("modelID")
        provider_id = info.get("providerID")
        if isinstance(model_id, str) and model_id:
            if isinstance(provider_id, str) and provider_id:
                return f"{provider_id}/{model_id}"
            return model_id
        model = info.get("model")
        if isinstance(model, dict):
            nested_model_id = model.get("modelID")
            nested_provider_id = model.get("providerID")
            if isinstance(nested_model_id, str) and nested_model_id:
                if isinstance(nested_provider_id, str) and nested_provider_id:
                    return f"{nested_provider_id}/{nested_model_id}"
                return nested_model_id
    return "unknown"


def _build_model_costs_from_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    aliases = load_model_aliases()
    model_map: dict[str, dict[str, float]] = defaultdict(lambda: {"api_cost": 0.0, "estimated_cost": 0.0})
    _accumulate_model_costs(model_map, messages, aliases)
    return _finalize_model_costs(model_map)


def _accumulate_model_costs(model_map: dict[str, dict[str, float]], messages: list[dict[str, object]], aliases: dict[str, str]) -> None:
    for msg in messages:
        role = msg.get("role")
        if role != "assistant":
            continue
        model_id = aliases.get(_extract_model_id_from_message(msg), _extract_model_id_from_message(msg))
        telemetry = summarize_telemetry(collect_telemetry_calls([msg]))
        model_map[model_id]["api_cost"] += float(telemetry.total_cost)
        # Estimated cost would need pricing lookup; for now use API cost as proxy for estimated
        model_map[model_id]["estimated_cost"] += float(telemetry.total_cost)


def _finalize_model_costs(model_map: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows = []
    for model, costs in model_map.items():
        api_cost = costs.get("api_cost", 0.0)
        estimated_cost = costs.get("estimated_cost", 0.0)
        primary_cost = api_cost if api_cost > 0 else estimated_cost
        rows.append(
            {
                "model": model,
                "api_cost": round(api_cost, 6),
                "estimated_cost": round(estimated_cost, 6),
                "cost": round(primary_cost, 6),
            }
        )
    rows.sort(key=lambda x: float(x["cost"]), reverse=True)
    return rows[:10]


if __name__ == "__main__":
    main()
