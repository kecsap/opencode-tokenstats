"""Collect priced models from models.dev."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

import httpx

from .pricing import _parse_pricing_history, normalize_pricing_date, write_pricing_ledger

DEFAULT_MODELS_DEV_URL = "https://models.dev/api.json"
DEFAULT_MODELS_DEV_REVISION_URL = "https://api.github.com/repos/anomalyco/models.dev/commits?per_page=1"
DEFAULT_MODELS_DEV_HISTORY_URL = "https://raw.githubusercontent.com/anomalyco/models.dev/{revision}/api.json"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _fetch_models_dev_revision(*, timeout: float) -> str:
    response = httpx.get(DEFAULT_MODELS_DEV_REVISION_URL, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError("models.dev source HEAD SHA response is invalid")
    revision = payload[0].get("sha")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision.strip()):
        raise ValueError("models.dev source HEAD SHA is required")
    return revision.strip().lower()


def fetch_models_dev_catalog(url: str = DEFAULT_MODELS_DEV_URL, *, timeout: float = 30.0) -> tuple[dict, str]:
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("models.dev catalog must be a JSON object")
    revision = (
        response.headers.get("x-revision")
        or response.headers.get("x-models-dev-revision")
        or payload.get("revision")
        or payload.get("source_revision")
    )
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision.strip()):
        revision = _fetch_models_dev_revision(timeout=timeout)
    return payload, revision.strip().lower()


def fetch_models_dev_revision_catalog(
    revision: str,
    *,
    source_url: str = DEFAULT_MODELS_DEV_HISTORY_URL,
    timeout: float = 30.0,
) -> tuple[dict, str]:
    """Fetch one catalog at an explicitly pinned models.dev Git revision."""
    revision = revision.strip().lower() if isinstance(revision, str) else ""
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError("models.dev historical revision must be a 40-character SHA")
    url = source_url.format(revision=revision)
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("models.dev historical catalog must be a JSON object")
    return payload, revision


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _tiers(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tier = deepcopy(item)
        metadata = tier.pop("tier", None)
        if "threshold" not in tier and isinstance(metadata, dict):
            tier["threshold"] = metadata.get("size", 200_000)
        result.append(tier)
    return result


def _entries(catalog: dict) -> list[tuple[str, str, dict]]:
    providers = catalog.get("providers", catalog)
    if not isinstance(providers, dict):
        raise ValueError("models.dev providers must be an object")
    result = []
    for provider, data in providers.items():
        if not isinstance(data, dict):
            continue
        models = data.get("models", data)
        if isinstance(models, dict):
            result.extend((str(provider), str(model), value) for model, value in models.items() if isinstance(value, dict))
    return result


def _resolve(provider: str, model: str, value: dict, all_models: dict[tuple[str, str], dict], seen: set[tuple[str, str]]) -> dict:
    key = (provider, model)
    if key in seen:
        raise ValueError(f"models.dev inheritance cycle at {provider}/{model}")
    parent_name = value.get("extends")
    if not parent_name:
        return deepcopy(value)
    parent_provider, parent_model = provider, str(parent_name)
    if "/" in parent_model:
        parent_provider, parent_model = parent_model.split("/", 1)
    parent = all_models.get((parent_provider, parent_model))
    if parent is None:
        raise ValueError(f"missing models.dev parent {parent_name!r}")
    result = _resolve(parent_provider, parent_model, parent, all_models, seen | {key})
    for field, child_value in value.items():
        if isinstance(result.get(field), dict) and isinstance(child_value, dict):
            merged = deepcopy(result[field])
            merged.update(deepcopy(child_value))
            result[field] = merged
        else:
            result[field] = deepcopy(child_value)
    return result


def collect_models_dev_snapshot(catalog: dict, *, source_url: str, revision: str, retrieved_at: str | None = None) -> dict:
    retrieved_at = retrieved_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    entries = _entries(catalog)
    all_models = {(provider, model): value for provider, model, value in entries}
    records = []
    for provider, model, raw in entries:
        value = _resolve(provider, model, raw, all_models, set())
        rates = value.get("cost", value.get("pricing", {}))
        if not isinstance(rates, dict):
            continue
        input_rate, output_rate = _number(rates.get("input")), _number(rates.get("output"))
        if input_rate is None or output_rate is None:
            continue
        tiers = _tiers(rates.get("tiers", value.get("tiers", [])))
        context_over_200k = rates.get("contextOver200k") or rates.get("context_over_200k")
        if context_over_200k is None and isinstance(value.get("contextOver200k"), dict):
            context_over_200k = value["contextOver200k"]
        if context_over_200k is None and isinstance(value.get("context_over_200k"), dict):
            context_over_200k = value["context_over_200k"]
        raw_aliases = value.get("aliases", [])
        if not isinstance(raw_aliases, list):
            raise ValueError(f"models.dev aliases must be a list for {provider}/{model}")
        aliases = sorted({model, f"{provider}/{model}", *(str(item) for item in raw_aliases)})
        record = {
            "provider": provider, "billing_channel": "direct_api", "model": model,
            "aliases": aliases, "service_profile": "standard", "profiles": ["standard"], "context": "short",
            "effective_from": retrieved_at, "effective_to": None, "status": "active", "confidence": "observed/inferred",
            "source_revision": revision, "observed_at": retrieved_at,
            "source": {"url": source_url, "retrieved_at": retrieved_at, "kind": "models.dev", "revision": revision},
            "rates": {"input": input_rate, "output": output_rate, "reasoning": _number(rates.get("reasoning")),
                       "cacheRead": _number(rates.get("cache_read", rates.get("cacheRead"))) or 0.0,
                       "cacheWrite": _number(rates.get("cache_write", rates.get("cacheWrite"))) or 0.0,
                       "tiers": tiers,
                       **({"contextOver200k": context_over_200k} if isinstance(context_over_200k, dict) else {})},
        }
        records.append(record)
    records.sort(key=lambda item: (item["provider"], item["model"]))
    snapshot = {"schema_version": 2, "unit": "USD per 1M tokens", "records": records}
    _parse_pricing_history(snapshot)
    return snapshot


def collect_models_dev_history(
    revisions: list[dict[str, Any]],
    *,
    source_url_template: str = DEFAULT_MODELS_DEV_HISTORY_URL,
    timeout: float = 30.0,
) -> list[dict]:
    """Fetch reviewed revisions and return deterministic dated pricing records."""
    if not isinstance(revisions, list) or not revisions:
        raise ValueError("models.dev historical revisions must be a non-empty list")
    reviewed: list[tuple[str, str, str]] = []
    for item in revisions:
        if not isinstance(item, dict):
            raise ValueError("models.dev historical revision entry must be an object")
        revision = item.get("revision")
        effective_from = item.get("effective_from")
        if not isinstance(revision, str) or not isinstance(effective_from, str):
            raise ValueError("models.dev historical entries require revision and effective_from")
        reviewed.append((normalize_pricing_date(effective_from), revision, str(item.get("source_url", source_url_template))))

    records: list[dict] = []
    for effective_from, revision, source_url in sorted(reviewed, key=lambda item: (item[0], item[1])):
        catalog, pinned_revision = fetch_models_dev_revision_catalog(
            revision, source_url=source_url, timeout=timeout
        )
        snapshot = collect_models_dev_snapshot(
            catalog,
            source_url=source_url.format(revision=pinned_revision),
            revision=pinned_revision,
            retrieved_at=effective_from,
        )
        records.extend(snapshot["records"])
    records.sort(key=lambda item: (item["effective_from"], item["provider"], item["model"]))
    return records


def write_models_dev_snapshot(path: str | Path, snapshot: dict) -> None:
    _parse_pricing_history(snapshot)
    write_pricing_ledger(path, snapshot)
