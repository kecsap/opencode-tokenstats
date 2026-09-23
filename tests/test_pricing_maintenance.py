from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from opencode_tokenstats import cli
from opencode_tokenstats.client import ApiClientError
from opencode_tokenstats.pricing import (
    ModelPricing,
    _parse_context_pricing,
    _parse_pricing_history,
    _select_pricing_rate,
    load_pricing_ledger,
    merge_pricing_history,
    normalize_pricing_date,
    parse_official_openai_pricing,
    parse_official_openai_pricing_html,
    pricing_status_report,
    write_pricing_ledger,
)


def _ledger(records: list[dict]) -> dict:
    return {"schema_version": 1, "unit": "USD per 1M tokens", "records": records}


def _record(
    model: str,
    profile: str,
    input_rate: float,
    output_rate: float,
    start: str = "2026-09-18T00:00:00Z",
    end: str | None = None,
) -> dict:
    return {
        "provider": "openai",
        "model": model,
        "aliases": [model, f"openai/{model}"],
        "service_profile": profile,
        "context": "short",
        "effective_from": start,
        "effective_to": end,
        "status": "active",
        "confidence": "observed",
        "source": {"url": "https://openai.com/api/pricing/", "retrieved_at": start},
        "rates": {"input": input_rate, "output": output_rate, "cacheRead": 0.0, "cacheWrite": 0.0},
    }


def _write_ledger_file(tmp_path: Path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_normalize_pricing_date() -> None:
    assert normalize_pricing_date("2026-10-02") == "2026-10-02T00:00:00Z"
    assert normalize_pricing_date("2026-10-02T03:04:05+02:00") == "2026-10-02T01:04:05Z"


def test_parse_official_openai_pricing_splits_standard_and_fast() -> None:
    payload = {
        "models": {
            "gpt-9": {"input": 1.0, "output": 2.0, "cacheRead": 0.5},
            "gpt-9-fast": {"input": 2.0, "output": 4.0},
        }
    }
    records = parse_official_openai_pricing(
        payload,
        effective_from="2026-09-19",
        retrieved_at="2026-09-19T12:00:00Z",
        source_url="https://openai.com/api/pricing/",
    )
    standard = next(r for r in records if r["model"] == "gpt-9")
    fast = next(r for r in records if r["model"] == "gpt-9-fast")
    assert standard["service_profile"] == "standard"
    assert fast["service_profile"] == "fast"
    assert standard["aliases"] == ["gpt-9", "openai/gpt-9"]
    assert standard["effective_from"] == "2026-09-19T00:00:00Z"
    assert standard["source"] == {
        "url": "https://openai.com/api/pricing/",
        "retrieved_at": "2026-09-19T12:00:00Z",
        "kind": "provider_official",
    }
    assert standard["rates"]["input"] == 1.0
    assert standard["rates"]["cacheRead"] == 0.5


def test_parse_official_openai_pricing_rejects_malformed_payload() -> None:
    kwargs = dict(effective_from="2026-09-19", retrieved_at="2026-09-19T00:00:00Z", source_url="u")
    with pytest.raises(ValueError):
        parse_official_openai_pricing([], **kwargs)
    with pytest.raises(ValueError):
        parse_official_openai_pricing({"models": {}}, **kwargs)
    with pytest.raises(ValueError):
        parse_official_openai_pricing({"models": {"m": {"input": 1.0}}}, **kwargs)
    with pytest.raises(ValueError):
        parse_official_openai_pricing({"models": {"m": "not-rates"}}, **kwargs)
    with pytest.raises(ValueError):
        parse_official_openai_pricing({"models": {"m": {"input": -1, "output": 1}}}, **kwargs)


def test_merge_supersedes_open_record_and_keeps_missing_models() -> None:
    current = _ledger([
        _record("gpt-9", "standard", 1.0, 2.0),
        _record("gpt-9-fast", "fast", 2.0, 4.0),
        _record("gpt-8", "standard", 0.5, 1.0),
    ])
    proposed = [_record("gpt-9", "standard", 3.0, 6.0, start="2026-10-01T00:00:00Z")]
    merged, changes = merge_pricing_history(current, proposed)

    by_key = {(r["model"], r["service_profile"]): r for r in merged["records"]}
    assert by_key[("gpt-9", "standard")]["effective_from"] == "2026-10-01T00:00:00Z"
    closed = [
        r for r in merged["records"]
        if r["model"] == "gpt-9" and r["service_profile"] == "standard" and r["effective_to"] is not None
    ]
    assert len(closed) == 1
    assert closed[0]["effective_to"] == "2026-10-01T00:00:00Z"
    assert by_key[("gpt-9-fast", "fast")]["effective_to"] is None
    assert by_key[("gpt-8", "standard")]["effective_to"] is None
    assert any(line.startswith("change ") for line in changes)
    _parse_pricing_history(merged)


def test_merge_equal_rates_leaves_ledger_unchanged() -> None:
    current = _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    proposed = [_record("gpt-9", "standard", 1.0, 2.0, start="2026-10-01T00:00:00Z")]
    merged, changes = merge_pricing_history(current, proposed)
    assert len(merged["records"]) == 1
    assert merged["records"][0]["effective_to"] is None
    assert changes == ["unchanged openai/gpt-9 [standard]"]


def test_merge_rejects_backdated_supersession() -> None:
    current = _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    proposed = [_record("gpt-9", "standard", 3.0, 6.0, start="2026-09-01T00:00:00Z")]
    with pytest.raises(ValueError):
        merge_pricing_history(current, proposed)


def test_merge_historical_models_dev_records_without_rewriting_open_record() -> None:
    current = _ledger([_record("gpt-9", "standard", 3.0, 6.0, start="2026-10-01T00:00:00Z")])
    proposed = _record("gpt-9", "standard", 1.0, 2.0, start="2026-09-01T00:00:00Z")
    proposed["source"] = {
        "url": "https://raw.githubusercontent.com/anomalyco/models.dev/revision/api.json",
        "retrieved_at": "2026-09-01T00:00:00Z",
        "kind": "models.dev",
        "revision": "0123456789abcdef0123456789abcdef01234567",
    }
    merged, _ = merge_pricing_history(current, [proposed])
    historical = next(record for record in merged["records"] if record["effective_from"] == "2026-09-01T00:00:00Z")
    assert historical["effective_to"] == "2026-10-01T00:00:00Z"
    assert merged["records"][0]["effective_to"] is None


def test_merge_multiple_historical_models_dev_revisions_for_new_model() -> None:
    def historical(input_rate: float, start: str, revision: str) -> dict:
        record = _record("new-model", "standard", input_rate, input_rate * 2, start=start)
        record["source"] = {
            "url": f"https://models.dev/{revision}.json",
            "retrieved_at": start,
            "kind": "models.dev",
            "revision": revision,
        }
        return record

    merged, _ = merge_pricing_history(
        _ledger([]),
        [
            historical(2.0, "2026-09-01T00:00:00Z", "a" * 40),
            historical(1.0, "2026-08-01T00:00:00Z", "b" * 40),
        ],
    )

    records = sorted(merged["records"], key=lambda record: record["effective_from"])
    assert records[0]["effective_to"] == "2026-09-01T00:00:00Z"
    assert records[1]["effective_to"] is None
    _parse_pricing_history(merged)


def test_write_pricing_ledger_is_atomic(tmp_path) -> None:
    target = tmp_path / "ledger.json"
    payload = _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    write_pricing_ledger(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_write_pricing_ledger_failure_leaves_target_unchanged(tmp_path) -> None:
    target = tmp_path / "ledger.json"
    good = _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    write_pricing_ledger(target, good)
    bad = _ledger([_record("gpt-9", "standard", -1.0, 2.0)])
    with pytest.raises(ValueError):
        write_pricing_ledger(target, bad)
    assert json.loads(target.read_text(encoding="utf-8")) == good


def test_pricing_status_report_bundled_ledger() -> None:
    payload = load_pricing_ledger()
    report = pricing_status_report(payload)
    by_key = {(r["provider"], r["model"], r["service_profile"]): r for r in report["records"]}
    assert by_key[("openai", "gpt-5.6-terra", "standard")]["status"] == "active"
    assert by_key[("openai", "gpt-5.6-terra", "standard")]["source_url"] == "https://developers.openai.com/api/docs/pricing"
    assert "gpt-5.6-terra-fast" in report["fast_aliases"]
    assert "gpt-5.6-luna-fast" in report["fast_aliases"]
    assert by_key[("openai", "gpt-5.6-luna", "standard")]["status"] == "active"


def test_bundled_official_rates_keep_catalog_history_and_terra_tiers() -> None:
    records = load_pricing_ledger()["records"]
    official = [
        row for row in records
        if row["provider"] == "openai" and row["source"].get("kind") == "provider_official"
    ]
    assert all(row["context"] == "short" and row["confidence"] for row in official)
    assert all(row["status"] in {"active", "retired"} for row in official)

    start = "2026-09-23T14:43:46Z"
    tiers = {
        "gpt-5.6-terra": ("standard", (2, 0.2, 2.5, 12), (4, 0.4, 5, 18)),
        "gpt-5.6-terra-fast": ("fast", (4, 0.4, 5, 24), (8, 0.8, 10, 36)),
        "gpt-5.6-luna-fast": ("fast", (0.4, 0.04, 0.5, 2.4), (0.8, 0.08, 1, 3.6)),
    }
    dimensions = ("input", "cacheRead", "cacheWrite", "output")
    for model, (profile, base, long) in tiers.items():
        dated = sorted(
            (row for row in official if row["model"] == model and row["service_profile"] == profile),
            key=lambda row: row["effective_from"],
        )
        assert [row["effective_from"] for row in dated] == ["2026-09-18T00:00:00Z", start]
        assert dated[0]["effective_to"] == start
        assert dated[1]["effective_to"] is None
        assert tuple(dated[1]["rates"][key] for key in dimensions) == base
        assert tuple(dated[1]["rates"]["contextOver200k"][key] for key in dimensions) == long

    embeddings = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13,
                  "text-embedding-ada-002": 0.10}
    for model, input_rate in embeddings.items():
        row = next(row for row in official if row["model"] == model)
        assert row["rates"] == {"input": input_rate, "cacheRead": 0,
                                "cacheWrite": 0, "output": 0}

    astra = next(row for row in records if row["provider"] == "openai"
                 and row["model"] == "gpt-6-astra" and row["source"].get("kind") == "models.dev")
    assert astra["effective_to"] is None


def test_cli_pricing_status_offline() -> None:
    result = CliRunner().invoke(cli.main, ["pricing", "status"])
    assert result.exit_code == 0, result.output
    assert "openai/gpt-5.6-terra [standard] active" in result.output
    assert "fast aliases:" in result.output


def test_cli_pricing_import_requires_explicit_approval(tmp_path) -> None:
    ledger_path = _write_ledger_file(
        tmp_path, "current.json", _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    )
    source_path = _write_ledger_file(
        tmp_path,
        "source.json",
        _ledger([_record("gpt-9", "standard", 3.0, 6.0, start="2026-10-01T00:00:00Z")]),
    )
    target = tmp_path / "target.json"

    result = CliRunner().invoke(
        cli.main, ["pricing", "import", source_path, "--ledger", ledger_path, "--target", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert "preview only" in result.output
    assert not target.exists()

    result = CliRunner().invoke(
        cli.main,
        ["pricing", "import", source_path, "--ledger", ledger_path, "--target", str(target), "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "written:" in result.output
    merged = json.loads(target.read_text(encoding="utf-8"))
    assert len(merged["records"]) == 2
    assert json.loads(Path(ledger_path).read_text(encoding="utf-8"))["records"][0]["effective_to"] is None


def test_cli_pricing_import_rejects_malformed_source(tmp_path) -> None:
    ledger_path = _write_ledger_file(
        tmp_path, "current.json", _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    )
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(_ledger([_record("gpt-9", "standard", -1.0, 2.0)])),
        encoding="utf-8",
    )
    target = tmp_path / "target.json"
    result = CliRunner().invoke(
        cli.main,
        ["pricing", "import", str(bad), "--ledger", ledger_path, "--target", str(target), "--yes"],
    )
    assert result.exit_code == 1
    assert "ledger unchanged" in result.output
    assert not target.exists()


def test_cli_pricing_backfill_preview_and_write(tmp_path, monkeypatch) -> None:
    ledger_path = _write_ledger_file(tmp_path, "current.json", _ledger([]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"revisions": [{"revision": "a" * 40, "effective_from": "2026-09-01"}]}),
        encoding="utf-8",
    )
    record = _record("new-model", "standard", 1.0, 2.0, start="2026-09-01T00:00:00Z")
    record["source"] = {
        "url": "https://models.dev/a.json",
        "kind": "models.dev",
        "revision": "a" * 40,
        "retrieved_at": record["effective_from"],
    }
    monkeypatch.setattr(cli, "collect_models_dev_history", lambda revisions, **kwargs: [record])
    target = tmp_path / "target.json"

    runner = CliRunner()
    preview = runner.invoke(
        cli.main,
        ["pricing", "backfill", str(manifest), "--ledger", ledger_path, "--target", str(target)],
    )
    assert preview.exit_code == 0, preview.output
    assert "preview only" in preview.output
    assert not target.exists()

    written = runner.invoke(
        cli.main,
        ["pricing", "backfill", str(manifest), "--ledger", ledger_path, "--target", str(target), "--yes"],
    )
    assert written.exit_code == 0, written.output
    assert len(json.loads(target.read_text(encoding="utf-8"))["records"]) == 1


def test_cli_pricing_backfill_manifest_failure_preserves_target(tmp_path) -> None:
    ledger_path = _write_ledger_file(tmp_path, "current.json", _ledger([]))
    manifest = tmp_path / "bad.json"
    manifest.write_text("{bad", encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_bytes(b"sentinel\n")

    result = CliRunner().invoke(
        cli.main,
        ["pricing", "backfill", str(manifest), "--ledger", ledger_path, "--target", str(target), "--yes"],
    )
    assert result.exit_code == 1
    assert "ledger unchanged" in result.output
    assert target.read_bytes() == b"sentinel\n"


def test_cli_pricing_backfill_fetch_failure_preserves_target(tmp_path, monkeypatch) -> None:
    ledger_path = _write_ledger_file(tmp_path, "current.json", _ledger([]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"revisions": [{"revision": "a" * 40, "effective_from": "2026-09-01"}]}), encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_bytes(b"sentinel\n")
    monkeypatch.setattr(cli, "collect_models_dev_history", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.HTTPError("boom")))

    result = CliRunner().invoke(
        cli.main,
        ["pricing", "backfill", str(manifest), "--ledger", ledger_path, "--target", str(target), "--yes"],
    )
    assert result.exit_code == 1
    assert "ledger unchanged" in result.output
    assert target.read_bytes() == b"sentinel\n"


_OFFICIAL_PRICING_HTML = """
<!doctype html>
<html>
<body>
<div id="content-switcher-latest-pricing" class="content-switcher-root" data-content-switcher-root data-content-switcher-id="latest-pricing" data-content-switcher-initial="standard" data-content-switcher-value="standard">
  <div class="content-switcher-selector"><button>Standard</button><button>Fast mode</button></div>
  <div class="content-switcher-panes">
    <div data-content-switcher-pane="true" data-value="standard">
      <table>
        <tr><th></th><th>Short context</th><th>Long context</th></tr>
        <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th></tr>
        <tr><td>gpt-9</td><td>$1.00</td><td>$0.10</td><td>$1.25</td><td>$5.00</td><td>$2.00</td><td>$0.20</td><td>$2.50</td><td>$10.00</td></tr>
        <tr><td>gpt-9-mini</td><td>$0.20</td><td>$0.02</td><td>$0.25</td><td>$1.00</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      </table>
    </div>
    <div data-content-switcher-pane="true" data-value="batch" hidden>
      <table>
        <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th></tr>
        <tr><td>gpt-9</td><td>$0.50</td><td>$0.05</td><td>$0.625</td><td>$2.50</td><td>$1.00</td><td>$0.10</td><td>$1.25</td><td>$5.00</td></tr>
      </table>
    </div>
    <div data-content-switcher-pane="true" data-value="fast" hidden>
      <table>
        <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th></tr>
        <tr><td>gpt-9</td><td>$2.00</td><td>$0.20</td><td>$2.50</td><td>$10.00</td><td>$4.00</td><td>$0.40</td><td>$5.00</td><td>$20.00</td></tr>
        <tr><td>gpt-9-cyber</td><td>$12.50</td><td>$1.25</td><td>$15.625</td><td>$75.00</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      </table>
      <table>
        <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th><th>Input</th><th>Cached input</th><th>Cache writes</th><th>Output</th></tr>
        <tr><td>gpt-9</td><td>$3.00</td><td>$0.30</td><td>$3.75</td><td>$15.00</td><td>$6.00</td><td>$0.60</td><td>$7.50</td><td>$30.00</td></tr>
      </table>
      <table>
        <tr><th>Model</th><th>Price per minute</th></tr>
        <tr><td>gpt-live-1</td><td>$0.05</td></tr>
      </table>
    </div>
  </div>
</div>
</body>
</html>
"""


def test_parse_official_openai_pricing_html() -> None:
    payload = parse_official_openai_pricing_html(_OFFICIAL_PRICING_HTML)
    models = payload["models"]
    assert set(models) == {"gpt-9", "gpt-9-mini", "gpt-9-fast", "gpt-9-cyber-fast"}
    assert models["gpt-9"] == {
        "input": 1.0,
        "cacheRead": 0.1,
        "cacheWrite": 1.25,
        "output": 5.0,
        "contextOver200k": {"input": 2.0, "cacheRead": 0.2, "cacheWrite": 2.5, "output": 10.0},
    }
    assert models["gpt-9-mini"] == {"input": 0.2, "cacheRead": 0.02, "cacheWrite": 0.25, "output": 1.0}
    assert "contextOver200k" not in models["gpt-9-mini"]
    # First fast table listing wins over the secondary fast table.
    assert models["gpt-9-fast"]["input"] == 2.0
    assert models["gpt-9-fast"]["contextOver200k"]["output"] == 20.0
    assert models["gpt-9-cyber-fast"]["cacheWrite"] == 15.625
    assert "contextOver200k" not in models["gpt-9-cyber-fast"]
    with pytest.raises(ValueError):
        parse_official_openai_pricing_html("<html><body>no pricing tables</body></html>")


def test_parse_specialized_tables_and_reject_invalid_rates() -> None:
    html = """
    <table><tr><th>Category</th><th>Model</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
    <tr><td>Standard</td><td>gpt-9-codex (1M context length)</td><td>$2</td><td>-</td><td>$4</td></tr>
    <tr><td>Fast</td><td>gpt-9-codex</td><td>$3</td><td>$0.3</td><td>$6</td></tr>
    <tr><td>Batch</td><td>gpt-9-batch</td><td>$1</td><td>-</td><td>$2</td></tr>
    <tr><td>Flex</td><td>gpt-9-flex</td><td>$1</td><td>-</td><td>$2</td></tr>
    <tr><td>Standard</td><td>bad-cache</td><td>$1</td><td>unknown</td><td>$2</td></tr>
    <tr><td>Standard</td><td>bad-input</td><td>-</td><td>-</td><td>$2</td></tr></table>
    <table><tr><th>Model</th><th>Audio input</th><th>Audio output</th></tr><tr><td>audio-model</td><td>$1</td><td>$2</td></tr></table>
    <div id="content-switcher-latest-pricing"><div data-value="standard">
    <table><tr><th>Model</th><th>Input</th><th>Training</th><th>Output</th></tr>
    <tr><td>gpt-9-finetune</td><td>$1</td><td>$3</td><td>$2</td></tr></table>
    </div></div>
    """
    models = parse_official_openai_pricing_html(html)["models"]
    assert models == {
        "gpt-9-codex": {"input": 2.0, "cacheRead": 0.0, "output": 4.0},
        "gpt-9-codex-fast": {"input": 3.0, "cacheRead": 0.3, "output": 6.0},
    }
    assert "gpt-9-flex" not in models
    assert "gpt-9-finetune" not in models


def test_standard_specialized_embedding_rates_are_input_only() -> None:
    html = """<div id="content-switcher-specialized-pricing"><div data-value="standard">
    <table><tr><th>Category</th><th>Model</th><th>Input</th>
    <th>Cached input</th><th>Output</th></tr>
    <tr><td>Codex</td><td>gpt-5.3-codex</td><td>$1.75</td><td>$0.175</td><td>$14</td></tr>
    <tr><td>Embedding</td><td>text-embedding-3-small</td><td>$0.02</td><td>-</td><td>-</td></tr>
    <tr><td>Embedding</td><td>text-embedding-3-large</td><td>$0.13</td><td>-</td><td>-</td></tr>
    <tr><td>Embedding</td><td>text-embedding-ada-002</td><td>$0.10</td><td>-</td><td>-</td></tr>
    <tr><td>Moderation</td><td>free-model</td><td>Free</td><td>-</td><td>-</td></tr>
    </table></div><div data-value="fast"><table>
    <tr><th>Category</th><th>Model</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
    <tr><td>Codex</td><td>gpt-5.3-codex</td><td>$3.50</td><td>$0.35</td><td>$28</td></tr>
    <tr><td>Embedding</td><td>not-fast</td><td>$0.01</td><td>-</td><td>-</td></tr>
    </table></div></div>"""
    models = parse_official_openai_pricing_html(html)["models"]
    assert models == {
        **{model: {"input": rate, "cacheRead": 0.0, "output": 0.0}
        for model, rate in {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13,
                            "text-embedding-ada-002": 0.10}.items()},
        "gpt-5.3-codex": {"input": 1.75, "cacheRead": 0.175, "output": 14.0},
        "gpt-5.3-codex-fast": {"input": 3.5, "cacheRead": 0.35, "output": 28.0},
    }


def test_specialized_duplicate_preserves_document_order_across_panes() -> None:
    html = """<div id="content-switcher-latest-pricing"><div data-value="standard">
    <table><tr><th>Model</th><th>Input</th><th>Output</th></tr>
    <tr><td>unrelated</td><td>$1</td><td>$2</td></tr></table>
    </div></div>
    <table><tr><th>Category</th><th>Model</th><th>Input</th><th>Output</th></tr>
    <tr><td>Standard</td><td>duplicate</td><td>$3</td><td>$4</td></tr></table>
    <div id="content-switcher-latest-pricing"><div data-value="standard">
    <table><tr><th>Model</th><th>Input</th><th>Output</th></tr>
    <tr><td>duplicate</td><td>$5</td><td>$6</td></tr></table>
    </div></div>"""
    assert parse_official_openai_pricing_html(html)["models"]["duplicate"] == {
        "input": 3.0, "output": 4.0,
    }


@pytest.mark.parametrize("input_rate,output_rate,cache_rate", [
    ("invalid", "$2", "-"), ("$1", "nan", "-"),
    ("-1", "$2", "-"), ("$1", "-2", "-"),
    ("$1", "$2", "-inf"), ("$1", "$2", "-1"),
])
def test_specialized_tables_skip_invalid_rates(input_rate, output_rate, cache_rate) -> None:
    html = f"""<table><tr><th>Category</th><th>Model</th><th>Input</th>
    <th>Cached input</th><th>Output</th></tr><tr><td>Standard</td>
    <td>invalid-model</td><td>{input_rate}</td><td>{cache_rate}</td>
    <td>{output_rate}</td></tr><tr><td>Standard</td><td>valid-model</td>
    <td>$1</td><td>-</td><td>$2</td></tr></table>"""
    models = parse_official_openai_pricing_html(html)["models"]
    assert models == {"valid-model": {"input": 1.0, "cacheRead": 0.0, "output": 2.0}}


def test_long_tier_presence_ignores_cache_dashes_and_keeps_explicit_threshold() -> None:
    html = """<div id="content-switcher-latest-pricing"><div data-value="standard"><table>
    <tr><th></th><th>Short context</th><th>Long context 1M</th></tr>
    <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Output</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
    <tr><td>qualified (1M context length)</td><td>$1</td><td>-</td><td>$2</td><td>$3</td><td>-</td><td>$4</td></tr>
    </table></div></div>"""
    rates = parse_official_openai_pricing_html(html)["models"]["qualified"]
    assert rates["contextOver200k"] == {
        "threshold": 1_000_000, "input": 3.0, "cacheRead": 0.0,
        "cacheWrite": 0.0, "output": 4.0,
    }
    tier = _parse_context_pricing(rates["contextOver200k"])
    pricing = ModelPricing(input=1, output=2, cache_read=0, cache_write=0, context_over_200k=tier)
    assert _select_pricing_rate(pricing, 1_000_000).input == 1.0
    assert _select_pricing_rate(pricing, 1_000_001).input == 3.0


def test_grouped_context_cache_columns_do_not_cross_contexts() -> None:
    html = """<div id="content-switcher-latest-pricing"><div data-value="standard"><table>
    <tr><th></th><th>Short context</th><th>Long context 1M</th></tr>
    <tr><th>Model</th><th>Input</th><th>Cached input</th><th>Output</th><th>Input</th><th>Cached input</th><th>Output</th></tr>
    <tr><td>grouped</td><td>$1</td><td>$0.1</td><td>$2</td><td>$3</td><td>$0.3</td><td>$4</td></tr>
    </table></div></div>"""
    assert parse_official_openai_pricing_html(html)["models"]["grouped"] == {
        "input": 1.0, "cacheRead": 0.1, "cacheWrite": 0.0, "output": 2.0,
        "contextOver200k": {
            "threshold": 1_000_000, "input": 3.0, "cacheRead": 0.3,
            "cacheWrite": 0.0, "output": 4.0,
        },
    }


class _StubPricingClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False

    def get_text(self, path: str, params: dict | None = None) -> str:
        self.requested_path = path
        return _OFFICIAL_PRICING_HTML

    def close(self) -> None:
        self.closed = True


def test_cli_pricing_refresh_preview_and_write(tmp_path, monkeypatch) -> None:
    created: dict[str, object] = {}

    class CapturingStub(_StubPricingClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            created.update(kwargs)

    monkeypatch.setattr(cli, "OpencodeApiClient", CapturingStub)
    ledger_path = _write_ledger_file(
        tmp_path, "current.json", _ledger([_record("gpt-9", "standard", 5.0, 5.0), _record("catalog-only", "standard", 0.5, 1.0)])
    )
    current = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    current["records"][0]["source"]["kind"] = "provider_official"
    current["records"][1]["source"] = {
        "url": "https://raw.githubusercontent.com/anomalyco/models.dev/revision/api.json",
        "retrieved_at": "2026-09-18T00:00:00Z",
        "kind": "models.dev",
        "revision": "0123456789abcdef0123456789abcdef01234567",
    }
    Path(ledger_path).write_text(json.dumps(current), encoding="utf-8")
    target = tmp_path / "target.json"
    args = ["pricing", "refresh", "--effective-from", "2026-10-02", "--ledger", ledger_path, "--target", str(target)]

    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    assert "change openai/gpt-9 [standard]" in result.output
    assert "add openai/gpt-9-mini [standard]" in result.output
    assert "add openai/gpt-9-fast [fast]" in result.output
    assert "add openai/gpt-9-cyber-fast [fast]" in result.output
    assert "preview only" in result.output
    assert not target.exists()
    assert created["base_url"] == "https://platform.openai.com"

    result = CliRunner().invoke(cli.main, args + ["--yes"])
    assert result.exit_code == 0, result.output
    merged = json.loads(target.read_text(encoding="utf-8"))
    assert any(
        r["model"] == "gpt-9"
        and r["service_profile"] == "standard"
        and r["rates"]["input"] == 1.0
        and r["effective_from"] == "2026-10-02T00:00:00Z"
        for r in merged["records"]
    )
    assert any(
        r["model"] == "gpt-9"
        and r["service_profile"] == "standard"
        and r["effective_to"] == "2026-10-02T00:00:00Z"
        for r in merged["records"]
    )
    assert any(
        r["model"] == "gpt-9-fast"
        and r["service_profile"] == "fast"
        and r["rates"]["input"] == 2.0
        for r in merged["records"]
    )
    refreshed = next(
        r for r in merged["records"]
        if r["model"] == "gpt-9" and r["effective_from"] == "2026-10-02T00:00:00Z"
    )
    assert refreshed["source"]["url"] == "https://platform.openai.com/docs/pricing"
    assert refreshed["source"]["retrieved_at"]
    assert refreshed["rates"]["contextOver200k"]["input"] == 2.0
    assert any(r["model"] == "catalog-only" and r["source"]["kind"] == "models.dev" for r in merged["records"])
    _parse_pricing_history(merged)


def test_cli_pricing_refresh_defaults_effective_from_to_retrieval_time(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(*cls.current, tzinfo=UTC)

    monkeypatch.setattr(cli, "datetime", FrozenDateTime)

    class DelayedClient(_StubPricingClient):
        def get_text(self, path: str, params: dict | None = None) -> str:
            FrozenDateTime.current = (2026, 10, 2, 3, 5, 6)
            return super().get_text(path, params)

    FrozenDateTime.current = (2026, 10, 2, 3, 4, 5)
    monkeypatch.setattr(cli, "OpencodeApiClient", DelayedClient)
    target = tmp_path / "target.json"
    result = CliRunner().invoke(
        cli.main, ["pricing", "refresh", "--ledger", _write_ledger_file(tmp_path, "empty.json", _ledger([])), "--target", str(target), "--yes"]
    )
    assert result.exit_code == 0, result.output
    records = json.loads(target.read_text(encoding="utf-8"))["records"]
    assert records
    assert all(r["effective_from"] == r["source"]["retrieved_at"] == "2026-10-02T03:05:06Z" for r in records)


def test_cli_pricing_refresh_explicit_effective_from_overrides_retrieval_time(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 2, 3, 4, 5, tzinfo=UTC)

    monkeypatch.setattr(cli, "datetime", FrozenDateTime)
    monkeypatch.setattr(cli, "OpencodeApiClient", _StubPricingClient)
    target = tmp_path / "target.json"
    ledger = _write_ledger_file(tmp_path, "empty.json", _ledger([]))
    result = CliRunner().invoke(
        cli.main, ["pricing", "refresh", "--effective-from", "2026-10-01", "--ledger", ledger, "--target", str(target), "--yes"]
    )
    assert result.exit_code == 0, result.output
    records = json.loads(target.read_text(encoding="utf-8"))["records"]
    assert all(r["effective_from"] == "2026-10-01T00:00:00Z" for r in records)
    assert all(r["source"]["retrieved_at"] == "2026-10-02T03:04:05Z" for r in records)


class _FailingPricingClient:
    def __init__(self, **kwargs: object) -> None:
        pass

    def get_text(self, path: str, params: dict | None = None) -> str:
        raise ApiClientError("boom")

    def close(self) -> None:
        pass


def test_cli_pricing_refresh_fetch_failure_leaves_ledger_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "OpencodeApiClient", _FailingPricingClient)
    ledger_path = _write_ledger_file(
        tmp_path, "current.json", _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    )
    target = tmp_path / "target.json"
    result = CliRunner().invoke(
        cli.main, ["pricing", "refresh", "--ledger", ledger_path, "--target", str(target), "--yes"]
    )
    assert result.exit_code == 1
    assert "ledger unchanged" in result.output
    assert not target.exists()


class _NoNetworkClient:
    def __init__(self, **kwargs: object) -> None:
        raise AssertionError("refresh source check must reject the URL before any request")


def test_cli_pricing_refresh_rejects_non_official_source_before_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "OpencodeApiClient", _NoNetworkClient)
    target = tmp_path / "target.json"
    for source_url in ("https://example.com/pricing", "http://openai.com/api/pricing/"):
        result = CliRunner().invoke(
            cli.main,
            ["pricing", "refresh", "--source-url", source_url, "--target", str(target), "--yes"],
        )
        assert result.exit_code == 1, result.output
        assert "official OpenAI" in result.output
        assert not target.exists()


class _EmptyPageClient:
    def __init__(self, **kwargs: object) -> None:
        pass

    def get_text(self, path: str, params: dict | None = None) -> str:
        return "<html><body>no pricing tables</body></html>"

    def close(self) -> None:
        pass


def test_cli_pricing_refresh_malformed_page_leaves_ledger_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "OpencodeApiClient", _EmptyPageClient)
    ledger_path = _write_ledger_file(
        tmp_path, "current.json", _ledger([_record("gpt-9", "standard", 1.0, 2.0)])
    )
    target = tmp_path / "target.json"
    result = CliRunner().invoke(
        cli.main, ["pricing", "refresh", "--ledger", ledger_path, "--target", str(target), "--yes"]
    )
    assert result.exit_code == 1
    assert "ledger unchanged" in result.output
    assert not target.exists()
