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
from urllib.parse import parse_qs, urlsplit

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
    from opencode_tokenstats.pricing import (
        load_model_aliases,
        load_pricing_ledger,
        merge_pricing_history,
        normalize_pricing_date,
        parse_official_openai_pricing,
        parse_official_openai_pricing_html,
        pricing_status_report,
        resolve_alias,
        write_pricing_ledger,
    )
    from opencode_tokenstats.trends import build_period_trends
    from opencode_tokenstats.telemetry import collect_telemetry_calls
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
    from .pricing import (
        load_model_aliases,
        load_pricing_ledger,
        merge_pricing_history,
        normalize_pricing_date,
        parse_official_openai_pricing,
        parse_official_openai_pricing_html,
        pricing_status_report,
        resolve_alias,
        write_pricing_ledger,
    )
    from .trends import build_period_trends
    from .telemetry import collect_telemetry_calls


_FORK_PERIOD_METRIC_INPUTS: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}


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
@click.option("-sf", "--session-filter", default=None, help="Comma-separated list of project root dir names to filter sessions by")
@click.option("-sm", "--model-filter", default=None, help="Comma-separated model IDs or aliases; use !name to exclude")
@click.option("--loc-scope", type=click.Choice(["code", "all"]), default="code", show_default=True, help="Git LOC scope")
@click.option("--loc-exclude", default=None, help="Comma-separated Git path patterns to exclude from LOC")
@click.option("-esl", "--export-session-list", default=None, help="Export selected session IDs to file (one per line)")
@click.option("-o", "--session-output-dir", default=None, help="Export selected session transcripts to a directory")
@click.option("--max-ext-tools", default=24, show_default=True, type=click.IntRange(1, None), help="Max external tools to include in External Tools panels")
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
    session_filter: str | None,
    model_filter: str | None,
    loc_scope: str,
    loc_exclude: str | None,
    export_session_list: str | None,
    session_output_dir: str | None,
    max_ext_tools: int,
) -> None:
    """OpenCode TokenStats CLI."""
    session_filter_set: set[str] | None = None
    if session_filter:
        session_filter_set = {v.strip() for v in session_filter.split(",") if v.strip()}

    model_filter_set: tuple[str, ...] | None = None
    if model_filter:
        selectors = tuple(v.strip() for v in model_filter.split(",") if v.strip())
        if any(selector == "!" for selector in selectors):
            raise click.ClickException("Model filter '!' must include a model name.")
        model_filter_set = selectors
    loc_exclude_set = tuple(v.strip() for v in (loc_exclude or "").split(",") if v.strip())

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
        "session_filter": session_filter_set,
        "model_filter": model_filter_set,
        "loc_scope": loc_scope,
        "loc_exclude": loc_exclude_set,
        "export_session_list": export_session_list,
        "session_output_dir": session_output_dir,
        "max_ext_tools": max_ext_tools,
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


@click.group(name="pricing", help="Inspect and maintain the tracked historical pricing ledger.")
def pricing_group() -> None:
    """Pricing ledger status, import, and refresh commands."""


main.add_command(pricing_group)


@pricing_group.command("status")
@click.option("--ledger", default=None, help="Path to pricing ledger JSON (default: bundled data/pricing-history.json)")
def pricing_status(ledger: str | None) -> None:
    """Report active/retired records, source metadata, coverage gaps, and Fast aliases."""
    try:
        payload = load_pricing_ledger(ledger)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"failed to load pricing ledger: {exc}") from exc
    report = pricing_status_report(payload)
    for item in report["records"]:
        period = item["effective_from"] + (" -> " + item["effective_to"] if item["effective_to"] else " -> open")
        click.echo(
            f"{item['provider']}/{item['model']} [{item['service_profile']}] {item['status']} "
            f"{period} source={item['source_url']} retrieved={item['retrieved_at']} confidence={item['confidence']}"
        )
    if report["fast_aliases"]:
        click.echo("fast aliases: " + ", ".join(report["fast_aliases"]))
    if report["coverage_gaps"]:
        click.echo("coverage gaps:")
        for gap in report["coverage_gaps"]:
            click.echo(f"  {gap}")


def _apply_pricing_write(merged: dict, changes: list[str], target: str, yes: bool) -> None:
    for line in changes:
        click.echo(f"  {line}")
    if yes:
        write_pricing_ledger(target, merged)
        click.echo(f"written: {target}")
    else:
        click.echo(f"preview only; pass --yes to write to {target}")


@pricing_group.command("import")
@click.argument("source", type=click.Path(exists=True, dir_okay=False))
@click.option("--ledger", default=None, help="Current ledger to merge into (default: bundled data/pricing-history.json)")
@click.option("--target", required=True, type=click.Path(dir_okay=False), help="Explicit ledger path to write")
@click.option("--yes", is_flag=True, help="Write the merged ledger to --target (preview only without it)")
def pricing_import(source: str, ledger: str | None, target: str, yes: bool) -> None:
    """Import reviewed dated records from a ledger-format JSON file."""
    try:
        proposed = load_pricing_ledger(source)["records"]
        current = load_pricing_ledger(ledger)
        merged, changes = merge_pricing_history(current, proposed)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"import validation failed; ledger unchanged: {exc}") from exc
    _apply_pricing_write(merged, changes, target, yes)


@pricing_group.command("refresh")
@click.option(
    "--source-url",
    default="https://platform.openai.com/docs/pricing",
    show_default=True,
    help="Official OpenAI pricing source URL (https on openai.com or *.openai.com only)",
)
@click.option("--effective-from", default=None, help="Effective date (YYYY-MM-DD or ISO datetime, UTC); default: today UTC")
@click.option("--ledger", default=None, help="Current ledger to merge into (default: bundled data/pricing-history.json)")
@click.option("--target", required=True, type=click.Path(dir_okay=False), help="Explicit ledger path to write")
@click.option("--yes", is_flag=True, help="Write the merged ledger to --target (preview only without it)")
@click.pass_context
def pricing_refresh(
    ctx: click.Context,
    source_url: str,
    effective_from: str | None,
    ledger: str | None,
    target: str,
    yes: bool,
) -> None:
    """Fetch official OpenAI pricing and propose dated Standard/Fast ledger records."""
    options = ctx.obj
    now = datetime.now(UTC)
    effective = normalize_pricing_date(effective_from or now.strftime("%Y-%m-%d"))
    retrieved = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    parts = urlsplit(source_url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not (host == "openai.com" or host.endswith(".openai.com")):
        raise click.ClickException(
            f"refresh source must be an official OpenAI https URL (openai.com or *.openai.com): {source_url}"
        )
    client = OpencodeApiClient(
        base_url=f"{parts.scheme}://{parts.netloc}",
        username=options["username"],
        password=options["password"],
        timeout=options["timeout"],
        retries=options["retries"],
    )
    try:
        page = client.get_text(parts.path or "/", params=parse_qs(parts.query))
    except ApiClientError as exc:
        raise click.ClickException(f"refresh fetch failed; ledger unchanged: {exc}") from exc
    finally:
        client.close()
    try:
        payload = parse_official_openai_pricing_html(page)
        records = parse_official_openai_pricing(
            payload,
            effective_from=effective,
            retrieved_at=retrieved,
            source_url=source_url,
        )
        current = load_pricing_ledger(ledger)
        merged, changes = merge_pricing_history(current, records)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"refresh validation failed; ledger unchanged: {exc}") from exc
    _apply_pricing_write(merged, changes, target, yes)


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
    messages = _filter_messages_by_model(messages, options)
    session_info = _find_session_info(sessions, sid) or _get_session_info(options, sid)
    canonical = build_canonical_metrics(sid, messages, session_info=session_info)
    if options.get("model_filter"):
        click.echo("WARNING: model filter active; tool/component attribution is approximate at assistant-message level.")
    top_tools_limit = _max_ext_tools(options)
    top_tools = [
        {
            "name": t["tool"],
            "output_tokens": t["tokens"],
            "call_count": t["calls"],
        }
        for t in canonical.tool_rows
        if _include_in_top_tools(t)
    ][:top_tools_limit]
    mcp_stats = {"rows": canonical.mcp_rows, "total_tokens": sum(r["tokens"] for r in canonical.mcp_rows)}
    core_stats = {"rows": canonical.core_rows, "total_tokens": sum(r["tokens"] for r in canonical.core_rows)}
    component_stats = {"rows": canonical.component_family_rows, "total_tokens": sum(r["tokens"] for r in canonical.component_family_rows)}
    model_costs = _finalize_model_costs(
        _accumulate_model_cost_rows(
            canonical.per_model_costs,
            load_model_aliases(options.get("model_alias_file")),
        )
    )
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
        pricing_coverage=canonical.pricing_coverage,
        pricing_warnings=[w for w in canonical.warnings if str(w).startswith("pricing:")],
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
@click.option("--from-date", required=True, help="YYYY-MM-DD (e.g. 2026-05-01), today, yesterday, or now")
@click.option("--to-date", required=True, help="YYYY-MM-DD (e.g. 2026-05-07), today, yesterday, or now")
@click.pass_context
def range(ctx: click.Context, from_date: str, to_date: str) -> None:
    """Show aggregate for explicit date range."""
    start = _parse_date(from_date)
    parsed_end = _parse_date(to_date)
    end = parsed_end if to_date.strip().lower() == "now" else parsed_end + timedelta(days=1)
    if end <= start:
        raise click.ClickException("Invalid date range: --to-date must be after --from-date")
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

    # Build directory lookup for root_dir extraction and session filtering
    sessions = _list_sessions(ctx.obj)
    session_dirs_map: dict[str, str] = {
        str(sess.get("id", "")): str(sess.get("directory", "")) for sess in sessions
    }
    session_lookup: dict[str, dict[str, object]] = {
        str(sess.get("id", "")): sess for sess in sessions if isinstance(sess.get("id"), str) and str(sess.get("id"))
    }

    # Apply session filter for JSON output
    session_filter = ctx.obj.get("session_filter")
    if session_filter:
        filtered_ids: set[str] = set()
        for sid, raw_dir in session_dirs_map.items():
            rd = extract_root_dir(raw_dir)
            if rd in session_filter:
                filtered_ids.add(sid)
        session_metrics = [c for c in session_metrics if c.session_id in filtered_ids]

    session_output_dir = ctx.obj.get("session_output_dir")
    if isinstance(session_output_dir, str) and session_output_dir.strip():
        session_ids = [str(c.session_id) for c in session_metrics if getattr(c, "session_id", None)]
        _export_session_transcripts(session_output_dir, session_ids, ctx.obj, session_lookup)

    payload = build_report_schema(
        period=period,
        mode=str(ctx.obj["mode"]),
        start=start,
        end=end,
        session_metrics=session_metrics,
        model_alias_file=ctx.obj.get("model_alias_file"),
        session_dirs=session_dirs_map,
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
            if total <= 0:
                self._bar.total = None
                self._bar.set_description("Finding in-period sessions")
                self._bar.n = 0
            else:
                self._bar.total = total
                self._bar.set_description(self._desc)
                self._bar.n = current
            self._bar.refresh()


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
    top_tools_limit = _max_ext_tools(options)
    session_metrics = _collect_period_session_metrics(
        options, start, end, progress_callback=progress_callback
    )
    sessions = _list_sessions(options)

    # Build directory lookup for root_dir extraction (from session.directory)
    session_dirs: dict[str, str] = {
        str(sess.get("id", "")): str(sess.get("directory", "")) for sess in sessions
    }
    session_lookup: dict[str, dict[str, object]] = {
        str(sess.get("id", "")): sess for sess in sessions if isinstance(sess.get("id"), str) and str(sess.get("id"))
    }

    # Apply session filter: keep only sessions whose root_dir matches the filter
    session_filter = options.get("session_filter")
    if session_filter:
        filtered_ids: set[str] = set()
        for sid, raw_dir in session_dirs.items():
            rd = extract_root_dir(raw_dir)
            if rd in session_filter:
                filtered_ids.add(sid)
        session_metrics = [c for c in session_metrics if c.session_id in filtered_ids]

    export_session_list = options.get("export_session_list")
    if isinstance(export_session_list, str) and export_session_list.strip():
        session_ids = [str(c.session_id) for c in session_metrics if getattr(c, "session_id", None)]
        _export_session_ids(export_session_list, session_ids)

    session_output_dir = options.get("session_output_dir")
    if isinstance(session_output_dir, str) and session_output_dir.strip():
        session_ids = [str(c.session_id) for c in session_metrics if getattr(c, "session_id", None)]
        _export_session_transcripts(session_output_dir, session_ids, options, session_lookup)

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
    model_map: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "api_cost": 0.0, "estimated_cost": 0.0, "tokens": 0.0, "input_tokens": 0.0,
            "output_tokens": 0.0, "reasoning_tokens": 0.0, "generated_tokens": 0.0,
            "priced_calls": 0, "future_fallback_calls": 0, "unpriced_calls": 0, "provenances": [],
        }
    )
    pricing_coverage = {"calls": 0, "priced_calls": 0, "future_fallback_calls": 0, "unpriced_calls": 0}
    warnings: list[str] = []

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
            if not _include_in_top_tools(t):
                continue
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
        for model_row in canonical.per_model_costs:
            model_key = resolve_alias(str(model_row["model"]), aliases)
            model_map[model_key]["api_cost"] += float(model_row["api_cost"])
            model_map[model_key]["estimated_cost"] += float(model_row["estimated_cost"])
            model_map[model_key]["tokens"] += int(model_row["tokens"])
            model_map[model_key]["input_tokens"] += int(model_row.get("input_tokens", 0))
            model_map[model_key]["output_tokens"] += int(model_row.get("output_tokens", 0))
            model_map[model_key]["reasoning_tokens"] += int(model_row.get("reasoning_tokens", 0))
            model_map[model_key]["generated_tokens"] += int(model_row.get("generated_tokens", 0))
            model_map[model_key]["priced_calls"] += int(model_row.get("priced_calls", 0))
            model_map[model_key]["future_fallback_calls"] += int(model_row.get("future_fallback_calls", 0))
            model_map[model_key]["unpriced_calls"] += int(model_row.get("unpriced_calls", 0))
            provenance = str(model_row.get("pricing_provenance", ""))
            if provenance and provenance not in model_map[model_key]["provenances"]:
                model_map[model_key]["provenances"].append(provenance)
        session_coverage = getattr(canonical, "pricing_coverage", None)
        if isinstance(session_coverage, dict):
            for key in pricing_coverage:
                pricing_coverage[key] += int(session_coverage.get(key, 0) or 0)
        for warning in canonical.warnings:
            if isinstance(warning, str) and warning and warning not in warnings:
                warnings.append(warning)

        # Attribute each assistant call to its preceding user turn. The fallback
        # preserves compatibility with CanonicalMetrics produced before activity_rows.
        activity_rows = canonical.activity_rows or [{
            "category": classify_session(canonical),
            "tokens": canonical.session_total_tokens,
            "input_tokens": canonical.input_tokens,
            "output_tokens": canonical.output_tokens,
            "reasoning_tokens": canonical.reasoning_tokens,
            "generated_tokens": canonical.output_tokens + canonical.reasoning_tokens,
            "calls": canonical.api_calls,
            "api_cost": canonical.actual_cost_usd,
            "estimated_cost": canonical.estimated_cost_usd,
        }]
        for activity in activity_rows:
            category = str(activity["category"])
            if category not in activity_map:
                activity_map[category] = {
                    "tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "generated_tokens": 0,
                    "calls": 0,
                    "api_cost": 0.0,
                    "estimated_cost": 0.0,
                }
            for field in ("tokens", "input_tokens", "output_tokens", "reasoning_tokens", "generated_tokens", "calls"):
                activity_map[category][field] += int(activity[field])
            for field in ("api_cost", "estimated_cost"):
                activity_map[category][field] += float(activity[field])

        # Build per-session row for top_sessions
        raw_dir = session_dirs.get(canonical.session_id, "")
        root_dir = extract_root_dir(raw_dir)
        session_rows.append(
            {
                "root_dir": root_dir,
                "tokens": canonical.session_total_tokens,
                "input_tokens": canonical.input_tokens,
                "output_tokens": canonical.output_tokens,
                "reasoning_tokens": canonical.reasoning_tokens,
                "generated_tokens": canonical.output_tokens + canonical.reasoning_tokens,
                "api_cost": round(canonical.actual_cost_usd, 6),
                "estimated_cost": round(canonical.estimated_cost_usd, 6),
                "pricing_coverage": getattr(canonical, "pricing_coverage", None) or {},
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
    )[:top_tools_limit]

    # Format by_activity rows sorted by cost desc
    by_activity = [
        {
            "category": cat,
            "label": CATEGORY_LABELS.get(cat, cat.title()),
            "tokens": data["tokens"],
            "input_percent": round(data["input_tokens"] / data["tokens"] * 100.0, 2) if data["tokens"] else 0.0,
            "output_percent": round(data["output_tokens"] / data["tokens"] * 100.0, 2) if data["tokens"] else 0.0,
            "reasoning_tokens": data["reasoning_tokens"],
            "reasoning_percent": round(data["reasoning_tokens"] / data["generated_tokens"] * 100.0, 2) if data["generated_tokens"] else 0.0,
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
            dir_map[rd] = {
                "root_dir": rd,
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "generated_tokens": 0,
                "api_cost": 0.0,
                "estimated_cost": 0.0,
            }
        dir_map[rd]["tokens"] += row["tokens"]
        dir_map[rd]["input_tokens"] += row["input_tokens"]
        dir_map[rd]["output_tokens"] += row["output_tokens"]
        dir_map[rd]["reasoning_tokens"] += row["reasoning_tokens"]
        dir_map[rd]["generated_tokens"] += row["generated_tokens"]
        dir_map[rd]["api_cost"] += row["api_cost"]
        dir_map[rd]["estimated_cost"] += row["estimated_cost"]
    top_sessions = [
        {
            "root_dir": rd,
            "tokens": data["tokens"],
            "input_percent": round(data["input_tokens"] / data["tokens"] * 100.0, 2) if data["tokens"] else 0.0,
            "output_percent": round(data["output_tokens"] / data["tokens"] * 100.0, 2) if data["tokens"] else 0.0,
            "reasoning_tokens": data["reasoning_tokens"],
            "reasoning_percent": round(data["reasoning_tokens"] / data["generated_tokens"] * 100.0, 2) if data["generated_tokens"] else 0.0,
            "api_cost": round(data["api_cost"], 6),
            "estimated_cost": round(data["estimated_cost"], 6),
        }
        for rd, data in dir_map.items()
    ]
    top_sessions.sort(key=lambda x: x["api_cost"] if x["api_cost"] > 0 else x["estimated_cost"], reverse=True)
    top_sessions = top_sessions[:10]

    trends = build_period_trends(
        start=start,
        end=end,
        session_metrics=session_metrics,
        session_dirs=session_dirs,
        loc_scope=str(options.get("loc_scope", "code")),
        loc_exclude=tuple(options.get("loc_exclude", ())),
    )

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
        "trends": trends,
        "pricing": {
            "calls": pricing_coverage["calls"],
            "priced_calls": pricing_coverage["priced_calls"],
            "future_fallback_calls": pricing_coverage["future_fallback_calls"],
            "unpriced_calls": pricing_coverage["unpriced_calls"],
            "coverage_percent": (
                round(pricing_coverage["priced_calls"] / pricing_coverage["calls"] * 100.0, 2)
                if pricing_coverage["calls"]
                else 0.0
            ),
        },
        "warnings": warnings,
        "model_filter_active": bool(options.get("model_filter")),
    }


def _collect_period_session_metrics(
    options: dict[str, object],
    start: datetime,
    end: datetime,
    *,
    progress_callback: callable | None = None,
) -> list[object]:
    if progress_callback:
        progress_callback(0, 0)
    sessions = _list_sessions(options)
    out = []
    model_aliases = load_model_aliases(options.get("model_alias_file"))
    results: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}

    if options["mode"] == "local":
        db_path = LocalSessionService.find_database_path(options.get("db_path"))
        session_by_id = {
            str(session["id"]): session
            for session in sessions
            if isinstance(session.get("id"), str) and session.get("id")
        }
        session_filter = options.get("session_filter")
        selected_ids: set[str] | None = None
        if session_filter:
            selected_ids = {
                sid
                for sid, session in session_by_id.items()
                if extract_root_dir(str(session.get("directory", ""))) in session_filter
            }
        period_messages = LocalSessionService(db_path=db_path).get_period_messages(
            int(start.timestamp() * 1000), int(end.timestamp() * 1000), session_ids=selected_ids
        )
        for sid, messages in period_messages.items():
            session_info = session_by_id.get(sid, {"id": sid})
            if session_filter and extract_root_dir(str(session_info.get("directory", ""))) not in session_filter:
                continue
            results[sid] = (
                session_info,
                _filter_messages_by_model(messages, options, aliases=model_aliases),
            )
        eligible_count = len(results)
    else:
        # API mode has no local part index; use the session time interval as a
        # candidate filter and retain exact part-time filtering below.
        eligible_sessions = []
        for sess in sessions:
            created = _session_created_at(sess)
            sid = sess.get("id")
            updated = _session_updated_at(sess)
            overlaps_period = (
                created is not None
                and created < end
                and (updated is None or updated >= start)
            )
            if overlaps_period and isinstance(sid, str) and sid:
                eligible_sessions.append((sid, sess))

        if not eligible_sessions:
            return out

        def fetch_session(sid: str, session_info: dict[str, object]) -> tuple[str, dict[str, object], list[dict[str, object]]]:
            messages = _get_messages(options, sid)
            messages = _filter_messages_to_period(messages, start, end)
            messages = _filter_messages_by_model(messages, options, aliases=model_aliases)
            return (sid, session_info, messages)

        max_workers = min(len(eligible_sessions), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_session, sid, sess): sid for sid, sess in eligible_sessions}
            for future in as_completed(futures):
                try:
                    sid, session_info, messages = future.result()
                    results[sid] = (session_info, messages)
                except Exception:
                    pass
        eligible_count = len(eligible_sessions)

    if not results:
        return out
    if progress_callback:
        progress_callback(0, eligible_count)

    # Build canonical metrics in parallel across CPU cores.
    worker_count = min(len(results), max(os.cpu_count() or 1, 1))
    if worker_count <= 1:
        for sid, (session_info, messages) in results.items():
            out.append(build_canonical_metrics(sid, messages, session_info=session_info))
            if progress_callback:
                progress_callback(len(out), eligible_count)
        return out

    start_method = "fork" if sys.platform.startswith("linux") else "spawn"
    global _FORK_PERIOD_METRIC_INPUTS
    try:
        if start_method == "fork":
            # Fork workers inherit this mapping copy-on-write. Submit IDs only so we
            # do not pickle hundreds of thousands of already-parsed SQLite parts.
            _FORK_PERIOD_METRIC_INPUTS = results
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context(start_method)) as executor:
            if start_method == "fork":
                futures = {executor.submit(_build_fork_period_session_metrics, sid): sid for sid in results}
            else:
                futures = {
                    executor.submit(_build_session_metrics, sid, messages, session_info): sid
                    for sid, (session_info, messages) in results.items()
                }
            done = 0
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    out.append(future.result())
                except Exception:
                    # Best-effort: fall back to local compute if worker fails.
                    session_info, messages = results.get(sid, ({}, []))
                    out.append(build_canonical_metrics(sid, messages, session_info=session_info))
                done += 1
                if progress_callback:
                    progress_callback(done, eligible_count)
    finally:
        _FORK_PERIOD_METRIC_INPUTS = {}

    return out


def _build_session_metrics(
    session_id: str,
    messages: list[dict[str, object]],
    session_info: dict[str, object] | None = None,
) -> object:
    return build_canonical_metrics(session_id, messages, session_info=session_info)


def _build_fork_period_session_metrics(session_id: str) -> object:
    """Build one metric from the parent process's fork-inherited input mapping."""
    session_info, messages = _FORK_PERIOD_METRIC_INPUTS[session_id]
    return build_canonical_metrics(session_id, messages, session_info=session_info)


def _filter_messages_by_model(
    messages: list[dict[str, object]],
    options: dict[str, object],
    *,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    selectors = options.get("model_filter")
    if not selectors:
        return messages
    positive = {str(value) for value in selectors if not str(value).startswith("!")}
    negative = {str(value)[1:] for value in selectors if str(value).startswith("!")}
    aliases = aliases if aliases is not None else load_model_aliases(options.get("model_alias_file"))

    def matches(message: dict[str, object]) -> bool:
        for call in collect_telemetry_calls([message]):
            provider = str(getattr(call, "provider_id", "") or "")
            model = str(getattr(call, "model_id", "") or "")
            raw = f"{provider}/{model}" if provider and model else model or provider
            names = {raw, model, resolve_alias(raw, aliases)} - {""}
            if (not positive or names & positive) and not names & negative:
                return True
        return False

    return [
        message
        for message in messages
        if message.get("role") != "assistant" or matches(message)
    ]


def _print_report(label: str, report: dict[str, object]) -> None:
    if report.get("model_filter_active"):
        click.echo("WARNING: model filter active; tool/component attribution is approximate at assistant-message level.")
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


def _get_session_info(options: dict[str, object], session_id: str) -> dict[str, object]:
    if options["mode"] == "local":
        db_path = LocalSessionService.find_database_path(options.get("db_path"))
        service = LocalSessionService(db_path=db_path)
        return service.get_session(session_id)

    with OpencodeApiClient(
        base_url=options["base_url"],
        username=options["username"],
        password=options["password"],
        timeout=options["timeout"],
        retries=options["retries"],
    ) as client:
        service = SessionService(client)
        data = service.get_session(session_id)
        return data if isinstance(data, dict) else {}


def _find_session_info(sessions: list[dict[str, object]], session_id: str) -> dict[str, object] | None:
    for session in sessions:
        if session.get("id") == session_id:
            return session
    return None


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


def _session_updated_at(session: dict[str, object]) -> datetime | None:
    """Return OpenCode's last session update time when available."""
    raw: object = session.get("time_updated")
    if raw is None:
        raw = session.get("updated_at")
    if raw is None:
        session_time = session.get("time")
        if isinstance(session_time, dict):
            raw = session_time.get("updated")
    if raw is None:
        data = session.get("data")
        if isinstance(data, dict):
            nested = data.get("time")
            if isinstance(nested, dict):
                raw = nested.get("updated")
            if raw is None:
                raw = data.get("time_updated")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value > 10_000_000_000:
        value /= 1000
    return datetime.fromtimestamp(value, UTC)


def _timestamp_from_record(record: dict[str, object]) -> datetime | None:
    """Read a record timestamp, accepting OpenCode seconds or milliseconds."""
    candidates = [
        record.get("_time_created"),
        record.get("time_created"),
        record.get("timestamp"),
        record.get("time"),
    ]
    info = record.get("info")
    if isinstance(info, dict):
        candidates.extend((info.get("time_created"), info.get("timestamp")))
        nested = info.get("time")
        if isinstance(nested, dict):
            candidates.append(nested.get("created"))
    for raw in candidates:
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 10_000_000_000:
            value /= 1000
        # Small serialized part `time` values are durations, not timestamps.
        if value >= 1_000_000_000:
            return datetime.fromtimestamp(value, UTC)
    return None


def _filter_messages_to_period(
    messages: list[dict[str, object]], start: datetime, end: datetime
) -> list[dict[str, object]]:
    """Keep only messages/parts with activity inside the requested period."""
    filtered: list[dict[str, object]] = []
    for message in messages:
        message_in_period = _timestamp_from_record(message)
        parts = message.get("parts")
        kept_parts: list[object] = []
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    timestamp = _timestamp_from_record(part)
                    if timestamp is not None and start <= timestamp < end:
                        kept_parts.append(part)
        if (message_in_period is not None and start <= message_in_period < end) or kept_parts:
            copy = dict(message)
            if isinstance(parts, list):
                copy["parts"] = kept_parts
            filtered.append(copy)
    return filtered


def _parse_date(value: str) -> datetime:
    keyword = value.strip().lower()
    if keyword == "now":
        return datetime.now(UTC)
    if keyword == "today":
        return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if keyword == "yesterday":
        return (datetime.now(UTC) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise click.ClickException(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


def _normalize_session_ids(session_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in session_ids:
        sid = raw.strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _export_session_ids(path_str: str, session_ids: list[str]) -> None:
    path = Path(path_str).expanduser()
    parent = path.parent
    if not parent.exists():
        raise click.ClickException(f"Export path parent directory does not exist: {parent}")
    ids = _normalize_session_ids(session_ids)
    content = "\n".join(ids)
    if content:
        content += "\n"
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"Failed to write export session list file '{path}': {exc}") from exc


def _json_dict(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _render_transcript_part(part: object) -> str:
    if isinstance(part, dict):
        raw = part.get("data")
        if isinstance(raw, str):
            data = _json_dict(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            data = part
    elif isinstance(part, str):
        data = _json_dict(part)
    else:
        return ""

    typ = str(data.get("type", "unknown"))

    if typ == "text":
        return str(data.get("text", ""))

    if typ == "tool":
        state = data.get("state", {})
        state_dict = state if isinstance(state, dict) else {}
        out = state_dict.get("output")
        err = state_dict.get("error")
        tool = str(data.get("tool", "tool"))

        if out:
            return f"[tool:{tool}]\n{out}"
        if err:
            return f"[tool:{tool} ERROR]\n{err}"
        return f"[tool:{tool}] {json.dumps(state_dict, ensure_ascii=False)}"

    if typ in {"reasoning", "step-start", "step-finish"}:
        return ""

    return json.dumps(data, ensure_ascii=False)


def _build_session_transcript_text(
    session_id: str,
    session: dict[str, object] | None,
    messages: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    lines.append(f"# session: {session_id}")
    lines.append(f"# title: {session.get('title', '') if isinstance(session, dict) else ''}")
    lines.append(f"# directory: {session.get('directory', '') if isinstance(session, dict) else ''}")
    created = _session_created_at(session) if isinstance(session, dict) else None
    lines.append(f"# created: {created.isoformat() if created is not None else ''}")
    lines.append("")

    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown"))
        lines.append("")
        lines.append("")
        lines.append(f"## {role} {idx}")

        parts = message.get("parts")
        if isinstance(parts, list):
            for part in parts:
                text = _render_transcript_part(part).strip()
                if text:
                    lines.append("")
                    lines.append(text)

        fallback = message.get("content") or message.get("text")
        if isinstance(fallback, str) and fallback.strip():
            lines.append("")
            lines.append(fallback.strip())

    return "\n".join(lines) + "\n"


def _export_session_transcripts(
    output_dir_str: str,
    session_ids: list[str],
    options: dict[str, object],
    session_lookup: dict[str, dict[str, object]],
) -> None:
    output_dir = Path(output_dir_str).expanduser()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise click.ClickException(f"Failed to create session output dir '{output_dir}': {exc}") from exc

    for session_id in _normalize_session_ids(session_ids):
        try:
            messages = _get_messages(options, session_id)
            transcript = _build_session_transcript_text(session_id, session_lookup.get(session_id), messages)
            (output_dir / f"{session_id}.txt").write_text(transcript, encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                f"Failed to write transcript for session '{session_id}' in '{output_dir}': {exc}"
            ) from exc
        except Exception as exc:
            raise click.ClickException(f"Failed to export transcript for session '{session_id}': {exc}") from exc


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
_TOP_TOOLS_LIMIT = 24


def _max_ext_tools(options: dict[str, object]) -> int:
    value = options.get("max_ext_tools", _TOP_TOOLS_LIMIT)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _TOP_TOOLS_LIMIT
    return parsed if parsed > 0 else _TOP_TOOLS_LIMIT


def _include_in_top_tools(row: dict[str, object]) -> bool:
    return not (bool(row.get("is_core")) or bool(row.get("is_skill")) or bool(row.get("is_subagent")))


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
    def _str(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    info = message.get("info") if isinstance(message.get("info"), dict) else {}
    data = message.get("data") if isinstance(message.get("data"), dict) else {}
    model = message.get("model") if isinstance(message.get("model"), dict) else {}
    info_model = info.get("model") if isinstance(info.get("model"), dict) else {}
    data_model = data.get("model") if isinstance(data.get("model"), dict) else {}

    provider_id = _str(
        info.get("providerID")
        or data.get("providerID")
        or message.get("providerID")
        or model.get("providerID")
        or info_model.get("providerID")
        or data_model.get("providerID")
    )
    model_id = _str(
        info.get("modelID")
        or data.get("modelID")
        or message.get("modelID")
        or model.get("modelID")
        or model.get("id")
        or info_model.get("modelID")
        or info_model.get("id")
        or data_model.get("modelID")
        or data_model.get("id")
    )
    if model_id:
        if provider_id:
            return f"{provider_id}/{model_id}"
        return model_id
    return "unknown"


def _build_model_costs_from_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    aliases = load_model_aliases()
    canonical = build_canonical_metrics("__model_costs__", messages)
    model_map = _accumulate_model_cost_rows(canonical.per_model_costs, aliases)
    return _finalize_model_costs(model_map)


def _accumulate_model_cost_rows(
    model_rows: list[dict[str, object]], aliases: dict[str, str]
) -> dict[str, dict[str, float]]:
    model_map: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "api_cost": 0.0, "estimated_cost": 0.0, "tokens": 0.0, "input_tokens": 0.0,
            "output_tokens": 0.0, "reasoning_tokens": 0.0, "generated_tokens": 0.0,
            "priced_calls": 0, "future_fallback_calls": 0, "unpriced_calls": 0, "provenances": [],
        }
    )
    for row in model_rows:
        model_id = resolve_alias(str(row.get("model", "unknown")), aliases)
        model_map[model_id]["api_cost"] += float(row.get("api_cost", 0.0))
        model_map[model_id]["estimated_cost"] += float(row.get("estimated_cost", 0.0))
        model_map[model_id]["tokens"] += float(row.get("tokens", 0.0))
        model_map[model_id]["input_tokens"] += float(row.get("input_tokens", 0.0))
        model_map[model_id]["output_tokens"] += float(row.get("output_tokens", 0.0))
        model_map[model_id]["reasoning_tokens"] += float(row.get("reasoning_tokens", 0.0))
        model_map[model_id]["generated_tokens"] += float(row.get("generated_tokens", 0.0))
        model_map[model_id]["priced_calls"] += int(row.get("priced_calls", 0))
        model_map[model_id]["future_fallback_calls"] += int(row.get("future_fallback_calls", 0))
        model_map[model_id]["unpriced_calls"] += int(row.get("unpriced_calls", 0))
        provenance = str(row.get("pricing_provenance", ""))
        if provenance and provenance not in model_map[model_id]["provenances"]:
            model_map[model_id]["provenances"].append(provenance)
    return model_map


def _finalize_model_costs(model_map: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for model, costs in model_map.items():
        api_cost = costs.get("api_cost", 0.0)
        estimated_cost = costs.get("estimated_cost", 0.0)
        if api_cost > 0:
            estimated_cost = 0.0
        primary_cost = api_cost if api_cost > 0 else estimated_cost
        tokens = int(costs.get("tokens", 0))
        input_tokens = int(costs.get("input_tokens", 0))
        output_tokens = int(costs.get("output_tokens", 0))
        reasoning_tokens = int(costs.get("reasoning_tokens", 0))
        generated_tokens = int(costs.get("generated_tokens", 0))
        priced_calls = int(costs.get("priced_calls", 0) or 0)
        future_fallback_calls = int(costs.get("future_fallback_calls", 0) or 0)
        unpriced_calls = int(costs.get("unpriced_calls", 0) or 0)
        if priced_calls + future_fallback_calls + unpriced_calls == 0:
            pricing_status = "unknown"
        elif priced_calls == 0:
            pricing_status = "unpriced"
        elif future_fallback_calls > 0:
            pricing_status = "future_fallback"
        elif unpriced_calls > 0:
            pricing_status = "unpriced"
        else:
            pricing_status = "active"
        rows.append(
            {
                "model": model,
                "tokens": tokens,
                "input_percent": round(input_tokens / tokens * 100.0, 2) if tokens else 0.0,
                "output_percent": round(output_tokens / tokens * 100.0, 2) if tokens else 0.0,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_percent": round(reasoning_tokens / generated_tokens * 100.0, 2) if generated_tokens else 0.0,
                "api_cost": round(api_cost, 6),
                "estimated_cost": round(estimated_cost, 6),
                "cost": round(primary_cost, 6),
                "priced_calls": priced_calls,
                "future_fallback_calls": future_fallback_calls,
                "unpriced_calls": unpriced_calls,
                "pricing_provenance": "; ".join(str(item) for item in costs.get("provenances", [])),
                "pricing_status": pricing_status,
            }
        )
    rows.sort(
        key=lambda x: (float(x["api_cost"]), float(x["estimated_cost"])),
        reverse=True,
    )
    return rows[:15]


if __name__ == "__main__":
    main()
