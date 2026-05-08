from __future__ import annotations

import click

from .client import ApiClientError, OpencodeApiClient
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
@click.pass_context
def doctor(
    ctx: click.Context,
    check_tokenizer: bool,
    provider_id: str,
    model_id: str,
    sample_text: str,
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
