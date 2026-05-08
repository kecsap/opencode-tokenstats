from __future__ import annotations

import click

from .client import ApiClientError, OpencodeApiClient
from .compatibility import CompatMode, analyze_context_compatibility
from .local_session_service import LocalSessionService, LocalStorageError
from .session_service import SessionService
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
