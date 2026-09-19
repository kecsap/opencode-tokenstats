from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from opencode_tokenstats.cli import main
from opencode_tokenstats.models_dev import collect_models_dev_history, collect_models_dev_snapshot, fetch_models_dev_catalog
from opencode_tokenstats.pricing import load_pricing_ledger


SHA = "0123456789abcdef0123456789abcdef01234567"


def catalog() -> dict:
    return {
        "providers": {
            "alpha": {"models": {
                "base": {"cost": {"input": 1, "output": 2, "cacheRead": 0.1, "tiers": [{"threshold": 200000, "input": 2, "output": 3}], "context_over_200k": {"input": 3, "output": 4}}},
                "child": {"extends": "base", "aliases": ["alias"], "cost": {"output": 4}},
                "free": {"cost": {"input": None, "output": None}},
            }},
        }
    }


def test_snapshot_resolves_inheritance_and_normalizes_rates() -> None:
    snapshot = collect_models_dev_snapshot(catalog(), source_url="https://models.dev/api.json", revision=SHA, retrieved_at="2026-09-19T00:00:00Z")
    child = next(record for record in snapshot["records"] if record["model"] == "child")
    assert child["rates"]["input"] == 1.0
    assert child["rates"]["output"] == 4.0
    assert child["aliases"] == ["alias", "alpha/child", "child"]
    assert child["source"]["revision"] == SHA
    assert child["rates"]["tiers"] == [{"threshold": 200000, "input": 2, "output": 3}]
    assert child["rates"]["contextOver200k"] == {"input": 3, "output": 4}
    assert len(snapshot["records"]) == 2


def test_openai_terra_catalog_parity_keeps_official_override() -> None:
    payload = catalog()
    payload["providers"]["openai"] = {"models": {"gpt-5.6-terra": {
        "cost": {
            "input": 2,
            "output": 0.2,
            "cacheRead": 2.5,
            "cacheWrite": 12,
            "tiers": [{"input": 4, "output": 18, "cache_read": 0.4, "cache_write": 5, "tier": {"type": "context", "size": 272000}}],
                "contextOver200k": {"input": 4, "output": 18, "cache_read": 0.4, "cache_write": 5},
        },
    }}}
    snapshot = collect_models_dev_snapshot(payload, source_url="https://models.dev/api.json", revision=SHA, retrieved_at="2026-09-19T00:00:00Z")
    terra = next(record for record in snapshot["records"] if record["provider"] == "openai")
    assert terra["rates"]["input"] == 2.0
    assert terra["rates"]["output"] == 0.2
    assert terra["rates"]["tiers"] == [{"input": 4, "output": 18, "cache_read": 0.4, "cache_write": 5, "threshold": 272000}]
    assert terra["rates"]["contextOver200k"] == {"input": 4, "output": 18, "cache_read": 0.4, "cache_write": 5}

    official = next(
        record for record in load_pricing_ledger().get("records", [])
        if record["model"] == "gpt-5.6-terra" and record.get("source", {}).get("kind") == "provider_official"
    )
    assert official["source"]["url"] == "https://openai.com/api/pricing/"


def test_fetch_requires_source_head_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        headers = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return catalog()

    monkeypatch.setattr("opencode_tokenstats.models_dev.httpx.get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="source HEAD SHA"):
        fetch_models_dev_catalog()


def test_fetch_uses_authoritative_source_head_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, payload: object) -> None:
            self.headers = {}
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self.payload

    responses = iter((Response(catalog()), Response([{"sha": SHA}])))
    monkeypatch.setattr("opencode_tokenstats.models_dev.httpx.get", lambda *args, **kwargs: next(responses))

    _, revision = fetch_models_dev_catalog()

    assert revision == SHA


def test_history_uses_pinned_revisions_and_effective_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fetch(url: str, **kwargs: object):
        calls.append(url)

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return catalog()

        return Response()

    monkeypatch.setattr("opencode_tokenstats.models_dev.httpx.get", fetch)
    records = collect_models_dev_history(
        [{"revision": SHA, "effective_from": "2026-09-01"}],
        source_url_template="https://example.test/{revision}.json",
    )
    assert calls == [f"https://example.test/{SHA}.json"]
    assert {record["effective_from"] for record in records} == {"2026-09-01T00:00:00Z"}
    assert {record["source_revision"] for record in records} == {SHA}


def test_snapshot_cli_preview_write_and_failure_preserves_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "pricing.json"
    target.write_text('{"sentinel": true}\n', encoding="utf-8")
    monkeypatch.setattr("opencode_tokenstats.cli.fetch_models_dev_catalog", lambda url: (catalog(), SHA))

    runner = CliRunner()
    preview = runner.invoke(main, ["--no-warmup", "pricing", "snapshot", "--target", str(target)])
    assert preview.exit_code == 0
    assert target.read_text(encoding="utf-8") == '{"sentinel": true}\n'

    written = runner.invoke(main, ["--no-warmup", "pricing", "snapshot", "--target", str(target), "--yes"])
    assert written.exit_code == 0
    assert len(load_pricing_ledger(target)["records"]) == 2
    before_failure = target.read_bytes()

    monkeypatch.setattr("opencode_tokenstats.cli.fetch_models_dev_catalog", lambda url: (_ for _ in ()).throw(ValueError("bad catalog")))
    failed = runner.invoke(main, ["--no-warmup", "pricing", "snapshot", "--target", str(target), "--yes"])
    assert failed.exit_code != 0
    assert target.read_bytes() == before_failure
