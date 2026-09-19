from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import fnmatch
import re
from datetime import datetime, timezone
from importlib.resources import files
from math import isfinite


def load_model_aliases(file_path: str | None = None) -> dict[str, str]:
    """Load model ID aliases from models.conf file.

    Format: one alias per line, e.g. 'alias_name = openai/gpt-5.4'
    Or grouped: 'alias_name = azure/gpt-5.4 openai/gpt-5.4'
    Wildcards: 'alias_name = myprovider/qwen*' matches myprovider/qwen3.6-27b, etc.

    Search locations (in order):
    1. Explicit file_path parameter
    2. OPTOKEN_MODEL_ALIAS_FILE environment variable
    3. Current working directory: models.conf (only if no explicit path or env var set)
    """
    candidates: list[Path] = []
    explicit_path = False

    if file_path:
        candidates.append(Path(file_path))
        explicit_path = True

    env_file = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE", "")
    if env_file:
        candidates.append(Path(env_file))
        explicit_path = True

    # Only check default models.conf if no explicit path or env var set
    if not explicit_path:
        candidates.append(Path.cwd() / "models.conf")

    aliases: dict[str, str] = {}
    wildcard_aliases: list[tuple[str, str]] = []  # List of (pattern, alias) for wildcard matching
    for conf_path in candidates:
        if conf_path.exists():
            try:
                for line in conf_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        # Strip @local prefix from alias name
                        alias_name = key
                        if alias_name.startswith("@local "):
                            alias_name = alias_name[len("@local "):]
                        for raw_id in val.split():
                            raw_id = raw_id.strip()
                            if raw_id:
                                if "*" in raw_id:
                                    wildcard_aliases.append((raw_id, alias_name))
                                else:
                                    aliases[raw_id] = alias_name
            except Exception:
                pass
            break
    # Prepend wildcard patterns with a marker so they can be distinguished
    for pattern, alias in wildcard_aliases:
        aliases[f"*{pattern}"] = alias
    return aliases


def resolve_alias(model_id: str, aliases: dict[str, str]) -> str:
    """Resolve a model ID to its alias, supporting wildcard patterns.

    Exact matches take precedence over wildcard matches.
    Wildcard patterns use * as a glob-like matcher (matches any suffix).
    """
    # Check exact match first
    if model_id in aliases:
        return aliases[model_id]

    # Check wildcard matches (glob semantics via fnmatch)
    for key, alias in aliases.items():
        if key.startswith("*"):
            pattern = key[1:]  # Remove the * marker
            if fnmatch.fnmatch(model_id, pattern):
                return alias

    return model_id


def load_local_model_patterns(file_path: str | None = None) -> list[str]:
    """Load local model wildcard patterns from models.conf file.

    Format: lines starting with '@local ' followed by space-separated patterns.
    Patterns support * wildcard, e.g.:
      @local myollama/* myllamacpp/* *qwen36*

    Search locations (in order):
    1. Explicit file_path parameter
    2. OPTOKEN_MODEL_ALIAS_FILE environment variable
    3. Current working directory: models.conf
    """
    candidates: list[Path] = []

    if file_path:
        candidates.append(Path(file_path))

    env_file = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE", "")
    if env_file:
        candidates.append(Path(env_file))

    candidates.append(Path.cwd() / "models.conf")

    patterns: list[str] = []
    for conf_path in candidates:
        if conf_path.exists():
            try:
                for line in conf_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("@local "):
                        for pat in line[len("@local "):].split():
                            pat = pat.strip()
                            if pat:
                                patterns.append(pat)
            except Exception:
                pass
            break
    return patterns


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input: float
    output: float
    cache_read: float
    cache_write: float = 0.0
    web_search: float = 0.0
    fast_multiplier: float = 1.0
    tiers: tuple["ContextPricing", ...] = field(default_factory=tuple)
    context_over_200k: "ContextPricing | None" = None


@dataclass(frozen=True, slots=True)
class ContextPricing:
    input: float
    output: float
    cache_read: float
    cache_write: float = 0.0
    threshold: int = 200_000


@dataclass(frozen=True, slots=True)
class PricingRecord:
    provider: str
    model: str
    service_profile: str
    context: str
    effective_from: str
    effective_to: str | None
    status: str
    confidence: str
    source_url: str
    retrieved_at: str
    pricing: ModelPricing
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PricingResolution:
    """Result of resolving the rate active for one telemetry call.

    status is one of: "active", "future_fallback", "flat_override", "unpriced".
    """

    pricing: ModelPricing | None
    status: str
    provenance: str = ""
    effective_from: str | None = None
    effective_to: str | None = None


class PricingLookup:
    def __init__(
        self,
        pricing_data: dict[str, ModelPricing],
        history: tuple[PricingRecord, ...] = (),
        flat_keys: "frozenset[str] | None" = None,
    ) -> None:
        self.pricing_data = pricing_data
        self.history = history
        if flat_keys is None:
            flat_keys = frozenset(pricing_data.keys())
        self.flat_keys = frozenset(key.lower() for key in flat_keys)

    @staticmethod
    def build_lookup_key(provider_id: str | None, model_id: str | None) -> str:
        provider = (provider_id or "").strip()
        model = (model_id or "").strip()
        if not provider:
            return model
        if not model:
            return provider

        provider_l = provider.lower()
        model_l = model.lower()
        if model_l.startswith(f"{provider_l}/"):
            return model
        return f"{provider}/{model}"

    def get_pricing(self, model_name: str) -> ModelPricing:
        found = self._find_pricing(model_name)
        if found is not None:
            return found
        return self.pricing_data.get(
            "default",
            ModelPricing(input=1, output=3, cache_read=0, cache_write=0, web_search=0, fast_multiplier=1),
        )

    def has_pricing(self, model_name: str) -> bool:
        return self._find_pricing(model_name) is not None

    def _find_pricing(self, model_name: str) -> ModelPricing | None:
        pricing, _ = self._find_pricing_key(model_name)
        return pricing

    def _find_pricing_key(self, model_name: str) -> tuple[ModelPricing | None, str | None]:
        raw_name = model_name.strip().lower()
        if not raw_name:
            return None, None

        for key in canonical_model_keys(raw_name):
            direct = self.pricing_data.get(key)
            if direct is not None:
                return direct, key

        exact = self.pricing_data.get(raw_name)
        if exact is not None:
            return exact, raw_name

        normalized = self._normalize_model_name(raw_name)
        normalized_exact = self.pricing_data.get(normalized)
        if normalized_exact is not None:
            return normalized_exact, normalized

        prefix, prefix_key = self._longest_prefix_match(raw_name)
        if prefix is not None:
            return prefix, prefix_key.lower() if prefix_key else None
        prefix, prefix_key = self._longest_prefix_match(normalized)
        if prefix is not None:
            return prefix, prefix_key.lower() if prefix_key else None
        return None, None

    def resolve_call_pricing(self, model_name: str, timestamp_ms: int | None = None) -> PricingResolution:
        """Resolve the rate active for one call at its timestamp.

        Historical records win for known models; calls before the first known
        record use the earliest later record ("future_fallback"); unknown,
        retired, or timestamp-less calls are unpriced unless an explicit
        flat-file override exists.
        """
        raw_name = (model_name or "").strip().lower()
        if not raw_name:
            return self._unpriced_resolution()

        if self.history:
            record, status = self._match_history_record(raw_name, timestamp_ms)
            if record is not None:
                return PricingResolution(
                    pricing=record.pricing,
                    status=status,
                    provenance=record.source_url,
                    effective_from=record.effective_from,
                    effective_to=record.effective_to,
                )

        pricing, key = self._find_pricing_key(raw_name)
        if pricing is not None and key is not None and key in self.flat_keys:
            return PricingResolution(pricing=pricing, status="flat_override", provenance=f"flat:{key}")
        return self._unpriced_resolution()

    @staticmethod
    def _unpriced_resolution() -> PricingResolution:
        return PricingResolution(pricing=None, status="unpriced", provenance="")

    def _match_history_record(self, raw_name: str, timestamp_ms: int | None) -> tuple[PricingRecord | None, str]:
        candidate_keys = set(canonical_model_keys(raw_name))
        matched: list[PricingRecord] = []
        for record in self.history:
            record_keys = {
                f"{record.provider.lower()}/{record.model.lower()}",
                record.model.lower(),
            }
            record_keys.update(alias.lower() for alias in record.aliases)
            if candidate_keys & record_keys:
                matched.append(record)
        if not matched:
            return None, "unpriced"

        moment = None
        if timestamp_ms is not None:
            try:
                moment = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                moment = None
        if moment is None:
            return None, "unpriced"

        active = [record for record in matched if _interval_contains(record, moment)]
        if active:
            return max(active, key=lambda record: record.effective_from), "active"

        later = [record for record in matched if _record_start(record) > moment]
        if later:
            return min(later, key=lambda record: _record_start(record)), "future_fallback"

        return None, "unpriced"

    def _longest_prefix_match(self, model_name: str) -> tuple[ModelPricing | None, str | None]:
        best_len = -1
        best: ModelPricing | None = None
        best_key: str | None = None

        for key, pricing in self.pricing_data.items():
            key_l = key.lower()
            if not model_name.startswith(key_l):
                continue
            suffix = model_name[len(key_l):]
            if suffix and not re.match(r"^[-.:/@_]", suffix):
                continue
            if len(key_l) > best_len:
                best_len = len(key_l)
                best = pricing
                best_key = key

        return best, best_key

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            return model_name.split("/")[-1].strip().lower()
        return model_name.strip().lower()


def estimate_session_cost_usd(
    pricing: ModelPricing,
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int = 0,
    web_search_requests: int = 0,
    context_tokens: int | None = None,
) -> float:
    rate = _select_pricing_rate(pricing, context_tokens)
    input_cost = (max(0, input_tokens) / 1_000_000) * rate.input
    output_cost = ((max(0, output_tokens) + max(0, reasoning_tokens)) / 1_000_000) * rate.output
    cache_read_cost = (max(0, cache_read_tokens) / 1_000_000) * rate.cache_read
    cache_write_cost = (max(0, cache_write_tokens) / 1_000_000) * rate.cache_write
    web_search_cost = max(0, web_search_requests) * pricing.web_search
    return input_cost + output_cost + cache_read_cost + cache_write_cost + web_search_cost


def _select_pricing_rate(pricing: ModelPricing, context_tokens: int | None) -> ModelPricing:
    if context_tokens is None:
        return pricing

    matched_tier: ContextPricing | None = None
    for tier in sorted(pricing.tiers, key=lambda item: item.threshold):
        threshold = tier.threshold or 0
        if context_tokens > threshold:
            matched_tier = tier

    if matched_tier is None:
        tier = pricing.context_over_200k
        if tier is not None:
            threshold = tier.threshold or 200_000
            if context_tokens > threshold:
                matched_tier = tier

    if matched_tier is not None:
        return ModelPricing(
            input=matched_tier.input,
            output=matched_tier.output,
            cache_read=matched_tier.cache_read,
            cache_write=matched_tier.cache_write,
            web_search=pricing.web_search,
            fast_multiplier=pricing.fast_multiplier,
            tiers=pricing.tiers,
            context_over_200k=pricing.context_over_200k,
        )
    return pricing


def canonical_model_keys(model: str) -> list[str]:
    raw = (model or "").strip().lower()
    if not raw:
        return []
    keys = [raw]
    if "/" in raw:
        provider, bare = raw.split("/", 1)
        keys.extend([bare, provider + "/" + bare])
    else:
        keys.extend([f"openai/{raw}", f"azure/{raw}"])
    out: list[str] = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def load_pricing_lookup() -> PricingLookup:
    candidates: list[Path] = []
    env = os.environ.get("OPENCODE_MODEL_PRICING_FILE")
    if env:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(env))))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "opencode-tokenscope" / "plugin" / "models.json")
    candidates.append(repo_root / "models.json")

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            data = _parse_flat_pricing(payload)
            data.setdefault("default", ModelPricing(input=1.0, output=3.0, cache_read=0.0, cache_write=0.0, web_search=0.0, fast_multiplier=1.0))
            return PricingLookup(data, flat_keys=frozenset(data.keys()))
        except Exception:
            continue

    try:
        payload = json.loads(files("opencode_tokenstats").joinpath("data/pricing-history.json").read_text(encoding="utf-8"))
        data, history = _parse_pricing_history(payload)
        data.setdefault("default", _default_pricing())
        return PricingLookup(data, history, flat_keys=frozenset())
    except Exception:
        return PricingLookup({"default": _default_pricing()}, flat_keys=frozenset())


def _default_pricing() -> ModelPricing:
    return ModelPricing(input=1.0, output=3.0, cache_read=0.0, cache_write=0.0, web_search=0.0, fast_multiplier=1.0)


def _record_start(record: PricingRecord) -> datetime:
    return datetime.fromisoformat(record.effective_from.replace("Z", "+00:00")).astimezone(timezone.utc)


def _record_end(record: PricingRecord) -> datetime | None:
    if not record.effective_to:
        return None
    return datetime.fromisoformat(record.effective_to.replace("Z", "+00:00")).astimezone(timezone.utc)


def _interval_contains(record: PricingRecord, moment: datetime) -> bool:
    start = _record_start(record)
    end = _record_end(record)
    return start <= moment and (end is None or moment < end)


def _parse_flat_pricing(payload: dict[object, object]) -> dict[str, ModelPricing]:
    data: dict[str, ModelPricing] = {}
    for key, val in payload.items():
        if not isinstance(key, str) or not isinstance(val, dict):
            continue
        tiers = _parse_context_tiers(val.get("tiers"))
        context_over_200k = _parse_context_pricing(val.get("contextOver200k") or val.get("context_over_200k"))
        data[key.lower()] = _model_pricing_from_rates(val, tiers=tiers, context_over_200k=context_over_200k)
    return data


def _model_pricing_from_rates(
    value: dict[object, object], *, tiers: tuple[ContextPricing, ...] = (), context_over_200k: ContextPricing | None = None
) -> ModelPricing:
    return ModelPricing(
        input=float(value.get("input", 0) or 0),
        output=float(value.get("output", 0) or 0),
        cache_read=float(value.get("cacheRead", value.get("cache_read", 0)) or 0),
        cache_write=float(value.get("cacheWrite", value.get("cache_write", 0)) or 0),
        web_search=float(value.get("webSearch", value.get("web_search", 0)) or 0),
        fast_multiplier=float(value.get("fastMultiplier", value.get("fast_multiplier", 1)) or 1),
        tiers=tiers,
        context_over_200k=context_over_200k,
    )


def _parse_pricing_history(payload: object) -> tuple[dict[str, ModelPricing], tuple[PricingRecord, ...]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("unit") != "USD per 1M tokens":
        raise ValueError("invalid pricing history header")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("pricing history records must be a list")

    records: list[PricingRecord] = []
    intervals: dict[tuple[str, str, str], list[tuple[datetime, datetime | None]]] = {}
    data: dict[str, ModelPricing] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("pricing history record must be an object")
        source = raw.get("source")
        rates = raw.get("rates")
        aliases = raw.get("aliases", [])
        if not isinstance(source, dict) or not isinstance(rates, dict) or not isinstance(aliases, list):
            raise ValueError("pricing history record has invalid fields")
        start = _parse_history_date(raw.get("effective_from"))
        end = _parse_history_date(raw.get("effective_to"), allow_none=True)
        if end is not None and end <= start:
            raise ValueError("pricing history interval is reversed")
        provider = _required_text(raw, "provider")
        model = _required_text(raw, "model")
        profile = _required_text(raw, "service_profile")
        interval_key = (provider.lower(), model.lower(), profile.lower())
        for old_start, old_end in intervals.setdefault(interval_key, []):
            if (end is None or old_start < end) and (old_end is None or start < old_end):
                raise ValueError("pricing history intervals overlap")
        intervals[interval_key].append((start, end))
        pricing = _model_pricing_from_rates(
            rates,
            tiers=_parse_context_tiers(rates.get("tiers"), strict=True),
            context_over_200k=_parse_context_pricing(
                rates.get("contextOver200k") or rates.get("context_over_200k"), strict=True
            ),
        )
        _validate_rates(pricing)
        retrieved_at = _parse_history_date(_required_text(source, "retrieved_at"))
        record = PricingRecord(
            provider=provider,
            model=model,
            service_profile=profile,
            context=_required_text(raw, "context"),
            effective_from=start.isoformat().replace("+00:00", "Z"),
            effective_to=end.isoformat().replace("+00:00", "Z") if end else None,
            status=_required_text(raw, "status"),
            confidence=_required_text(raw, "confidence"),
            source_url=_required_text(source, "url"),
            retrieved_at=retrieved_at.isoformat().replace("+00:00", "Z"),
            pricing=pricing,
            aliases=tuple(_required_alias(alias) for alias in aliases),
        )
        records.append(record)
        for alias in record.aliases:
            data[alias.lower()] = pricing
    return data, tuple(records)


def _required_text(value: dict[object, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"pricing history field {key} is required")
    return item.strip()


def _required_alias(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pricing history alias must be text")
    return value.strip()


def _parse_history_date(value: object, *, allow_none: bool = False) -> datetime | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ValueError("pricing history date must be text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("pricing history date is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("pricing history date must include timezone")
    return parsed.astimezone(timezone.utc)


def _validate_rates(pricing: ModelPricing) -> None:
    values = (pricing.input, pricing.output, pricing.cache_read, pricing.cache_write, pricing.web_search, pricing.fast_multiplier)
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("pricing history rates must be finite and non-negative")
    context_rates = list(pricing.tiers)
    if pricing.context_over_200k is not None:
        context_rates.append(pricing.context_over_200k)
    for tier in context_rates:
        values = (tier.input, tier.output, tier.cache_read, tier.cache_write)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("pricing history tier rates must be finite and non-negative")
        if tier.threshold <= 0:
            raise ValueError("pricing history tier threshold must be positive")


def _parse_context_pricing(value: object, *, strict: bool = False) -> ContextPricing | None:
    if not isinstance(value, dict):
        if strict and value is not None:
            raise ValueError("pricing history tier must be an object")
        return None
    raw_threshold = value.get("threshold", 200_000)
    if strict and (isinstance(raw_threshold, bool) or not isinstance(raw_threshold, int)):
        raise ValueError("pricing history tier threshold must be an integer")
    try:
        threshold = int(raw_threshold or 200_000)
    except (TypeError, ValueError) as exc:
        if strict:
            raise ValueError("pricing history tier threshold is invalid") from exc
        threshold = 200_000
    if strict and threshold <= 0:
        raise ValueError("pricing history tier threshold must be positive")
    return ContextPricing(
        input=float(value.get("input", 0) or 0),
        output=float(value.get("output", 0) or 0),
        cache_read=float(value.get("cacheRead", value.get("cache_read", 0)) or 0),
        cache_write=float(value.get("cacheWrite", value.get("cache_write", 0)) or 0),
        threshold=threshold,
    )


def _parse_context_tiers(value: object, *, strict: bool = False) -> tuple[ContextPricing, ...]:
    if not isinstance(value, list):
        if strict and value is not None:
            raise ValueError("pricing history tiers must be a list")
        return ()

    tiers: list[ContextPricing] = []
    for item in value:
        tier = _parse_context_pricing(item, strict=strict)
        if tier is not None:
            tiers.append(tier)

    tiers.sort(key=lambda item: item.threshold)
    if strict and any(left.threshold == right.threshold for left, right in zip(tiers, tiers[1:])):
        raise ValueError("pricing history tier thresholds must be unique")
    return tuple(tiers)


def default_ledger_path() -> Path:
    """Path of the tracked pricing ledger shipped with the package."""
    return Path(__file__).resolve().parent / "data" / "pricing-history.json"


def load_pricing_ledger(path: str | Path | None = None) -> dict:
    """Load a pricing history ledger file, validate it, and return its raw payload."""
    ledger = Path(path) if path is not None else default_ledger_path()
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    _parse_pricing_history(payload)
    return payload


def normalize_pricing_date(value: str) -> str:
    """Normalize a YYYY-MM-DD date or ISO datetime to a UTC '...Z' string."""
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text = f"{text}T00:00:00+00:00"
    parsed = _parse_history_date(text)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_official_openai_pricing(
    payload: object,
    *,
    effective_from: str,
    retrieved_at: str,
    source_url: str,
    provider: str = "openai",
) -> list[dict]:
    """Parse an official OpenAI pricing payload into dated ledger records.

    The payload is a JSON object mapping model IDs to rate objects, optionally
    wrapped under a "models" key. Model IDs ending in "-fast" produce a
    "fast" service profile record; every other model produces a "standard"
    record. Raises ValueError on malformed input.
    """
    if not isinstance(payload, dict):
        raise ValueError("official pricing payload must be a JSON object")
    models = payload.get("models", payload)
    if not isinstance(models, dict) or not models:
        raise ValueError("official pricing payload has no model entries")
    effective = _parse_history_date(normalize_pricing_date(effective_from))
    retrieved = _parse_history_date(normalize_pricing_date(retrieved_at))
    records: list[dict] = []
    for raw_model, rates in models.items():
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise ValueError("official pricing model names must be non-empty text")
        model = raw_model.strip()
        if not isinstance(rates, dict):
            raise ValueError(f"official pricing rates for {model} must be an object")
        for key in ("input", "output"):
            if key not in rates:
                raise ValueError(f"official pricing rates for {model} are missing {key}")
        pricing = _model_pricing_from_rates(rates)
        _validate_rates(pricing)
        profile = "fast" if model.lower().endswith("-fast") else "standard"
        rate_obj: dict[str, object] = {
            "input": pricing.input,
            "output": pricing.output,
            "cacheRead": pricing.cache_read,
            "cacheWrite": pricing.cache_write,
        }
        if pricing.web_search > 0:
            rate_obj["webSearch"] = pricing.web_search
        if pricing.fast_multiplier != 1.0:
            rate_obj["fastMultiplier"] = pricing.fast_multiplier
        if isinstance(rates.get("tiers"), list):
            rate_obj["tiers"] = rates["tiers"]
        context = rates.get("contextOver200k") or rates.get("context_over_200k")
        if isinstance(context, dict):
            rate_obj["contextOver200k"] = context
        records.append({
            "provider": provider,
            "model": model,
            "aliases": [model, f"{provider}/{model}"],
            "service_profile": profile,
            "context": "short",
            "effective_from": effective.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "effective_to": None,
            "status": "active",
            "confidence": "observed",
            "source": {
                "url": source_url,
                "retrieved_at": retrieved.isoformat(timespec="seconds").replace("+00:00", "Z"),
            },
            "rates": rate_obj,
        })
    return records


class _OfficialPricingHTMLParser(HTMLParser):
    """Collect per-1M-token price rows from the official OpenAI pricing page.

    Only tables inside the ``latest-pricing`` content-switcher panes with
    ``data-value`` of ``standard`` or ``fast`` are collected, so per-minute
    and modality tables in the same section are ignored.
    """

    def __init__(self) -> None:
        super().__init__()
        self._section_div_depth: int | None = None
        self._pane_value: str | None = None
        self._pane_div_depth: int | None = None
        self._div_depth = 0
        self._in_table = False
        self._in_row = False
        self._cell_parts: list[str] | None = None
        self._row: list[str] = []
        self.tables: dict[str, list[list[str]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs)
        if tag == "div":
            self._div_depth += 1
            if self._section_div_depth is None and attrs.get("id") == "content-switcher-latest-pricing":
                self._section_div_depth = self._div_depth
            elif (
                self._section_div_depth is not None
                and self._pane_div_depth is None
                and attrs.get("data-value") in ("standard", "fast")
            ):
                self._pane_value = attrs["data-value"]
                self._pane_div_depth = self._div_depth
        elif tag == "table" and self._pane_value is not None:
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._row = []
        elif tag in ("td", "th") and self._in_row:
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell_parts is not None:
            self._row.append("".join(self._cell_parts).strip())
            self._cell_parts = None
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._row and self._pane_value is not None:
                self.tables.setdefault(self._pane_value, []).append(self._row)
            self._row = []
        elif tag == "table" and self._in_table:
            self._in_table = False
        elif tag == "div":
            if self._pane_div_depth is not None and self._div_depth == self._pane_div_depth:
                self._pane_value = None
                self._pane_div_depth = None
            if self._section_div_depth is not None and self._div_depth == self._section_div_depth:
                self._section_div_depth = None
            self._div_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)


def _parse_price_cell(value: str) -> float | None:
    text = value.replace("$", "").replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_official_openai_pricing_html(html: str) -> dict:
    """Extract Standard/Fast model rates from the official OpenAI pricing page.

    Returns a ``{"models": {...}}`` payload in the shape accepted by
    ``parse_official_openai_pricing``: short-context rates plus an optional
    ``contextOver200k`` tier per model, and ``<model>-fast`` entries for the
    fast pane. The first table listing a model wins. Raises ValueError when
    the page has no per-1M-token model rate tables.
    """
    parser = _OfficialPricingHTMLParser()
    parser.feed(html)
    models: dict[str, dict[str, object]] = {}
    for pane in ("standard", "fast"):
        for row in parser.tables.get(pane, []):
            if len(row) != 9 or not row[0] or row[0] == "Model":
                continue
            model = row[0]
            key = f"{model}-fast" if pane == "fast" else model
            if key in models:
                continue
            rates: dict[str, object] = {}
            for field_name, value in zip(("input", "cacheRead", "cacheWrite", "output"), row[1:5]):
                price = _parse_price_cell(value)
                if price is None:
                    raise ValueError(f"official pricing page has no short-context {field_name} rate for {model}")
                rates[field_name] = price
            long_rates = [_parse_price_cell(value) for value in row[5:9]]
            if any(value is not None for value in long_rates):
                if any(value is None for value in long_rates):
                    raise ValueError(f"official pricing page has partial long-context rates for {model}")
                rates["contextOver200k"] = {
                    "input": long_rates[0],
                    "cacheRead": long_rates[1],
                    "cacheWrite": long_rates[2],
                    "output": long_rates[3],
                }
            models[key] = rates
    if not models:
        raise ValueError("official pricing page has no model rate tables")
    return {"models": models}


def _record_identity(record: dict[object, object]) -> tuple[str, str, str]:
    return (
        str(record.get("provider", "")).lower(),
        str(record.get("model", "")).lower(),
        str(record.get("service_profile", "")).lower(),
    )


def _pricing_from_record(record: dict[object, object]) -> ModelPricing:
    rates = record["rates"]
    return _model_pricing_from_rates(
        rates,
        tiers=_parse_context_tiers(rates.get("tiers"), strict=True),
        context_over_200k=_parse_context_pricing(
            rates.get("contextOver200k") or rates.get("context_over_200k"), strict=True
        ),
    )


def merge_pricing_history(current_payload: object, proposed_records: object) -> tuple[dict, list[str]]:
    """Merge proposed dated records into a current ledger payload.

    Returns (merged_payload, change_lines). A proposed record supersedes an open
    record with the same provider/model/service_profile only when their rates
    differ; records for models missing from the proposal are never retired.
    Raises ValueError when the merge would produce an invalid ledger.
    """
    if not isinstance(current_payload, dict):
        raise ValueError("current pricing ledger must be a JSON object")
    _parse_pricing_history(current_payload)
    if not isinstance(proposed_records, list):
        raise ValueError("proposed pricing records must be a list")
    _parse_pricing_history({
        "schema_version": 1,
        "unit": "USD per 1M tokens",
        "records": proposed_records,
    })

    current_records = [dict(record) for record in current_payload["records"]]
    open_by_key: dict[tuple[str, str, str], dict] = {}
    for record in current_records:
        if record.get("effective_to") is None:
            open_by_key[_record_identity(record)] = record

    changes: list[str] = []
    for record in proposed_records:
        record = dict(record)
        key = _record_identity(record)
        label = f"{key[0]}/{key[1]} [{key[2]}]"
        open_record = open_by_key.get(key)
        if open_record is None:
            current_records.append(record)
            changes.append(f"add {label}: effective from {record['effective_from']}")
            continue
        if _pricing_from_record(open_record) == _pricing_from_record(record):
            changes.append(f"unchanged {label}")
            continue
        start = _parse_history_date(record["effective_from"])
        if start <= _parse_history_date(open_record["effective_from"]):
            raise ValueError(f"proposed record for {label} must start after {open_record['effective_from']}")
        open_record["effective_to"] = start.isoformat(timespec="seconds").replace("+00:00", "Z")
        current_records.append(record)
        changes.append(
            f"change {label}: closes period from {open_record['effective_from']}, new period from {record['effective_from']}"
        )

    merged = dict(current_payload)
    merged["records"] = current_records
    _parse_pricing_history(merged)
    return merged, changes


def write_pricing_ledger(path: str | Path, payload: object) -> None:
    """Atomically write a validated pricing ledger payload to an explicit path."""
    _parse_pricing_history(payload)
    target = Path(path)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def pricing_status_report(payload: object) -> dict:
    """Summarize a ledger payload: per-record status, fast aliases, coverage gaps."""
    _parse_pricing_history(payload)
    _, records = _parse_pricing_history(payload)

    def base_model(model: str) -> str:
        model_l = model.lower()
        return model_l.removesuffix("-fast")

    grouped: dict[tuple[str, str], list[PricingRecord]] = {}
    for record in records:
        grouped.setdefault((record.provider.lower(), base_model(record.model)), []).append(record)

    entries: list[dict] = []
    fast_aliases: list[str] = []
    gaps: list[str] = []
    for (provider, model), group in sorted(grouped.items()):
        standard = [r for r in group if r.service_profile.lower() == "standard"]
        fast = [r for r in group if r.service_profile.lower() == "fast"]
        standard_active = any(r.effective_to is None for r in standard)
        fast_active = any(r.effective_to is None for r in fast)
        for record in sorted(group, key=lambda r: r.effective_from):
            entries.append({
                "provider": record.provider,
                "model": record.model,
                "service_profile": record.service_profile,
                "status": "active" if record.effective_to is None else "retired",
                "effective_from": record.effective_from,
                "effective_to": record.effective_to,
                "source_url": record.source_url,
                "retrieved_at": record.retrieved_at,
                "confidence": record.confidence,
                "aliases": list(record.aliases),
            })
        if not standard_active and not fast_active:
            gaps.append(f"{provider}/{model}: all records retired")
        if standard_active and not fast_active:
            gaps.append(f"{provider}/{model}: no active fast record")
        if fast_active and not standard_active:
            gaps.append(f"{provider}/{model}: no active standard record")
        fast_aliases.extend(r.model for r in fast if r.effective_to is None)
    return {"records": entries, "fast_aliases": sorted(fast_aliases), "coverage_gaps": gaps}
