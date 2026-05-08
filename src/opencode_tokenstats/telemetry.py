from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class TelemetrySchemaError(ValueError):
    """Raised when strict telemetry schema validation fails."""


@dataclass(frozen=True, slots=True)
class TelemetryCall:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    timestamp_ms: int | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True, slots=True)
class TelemetrySummary:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    total_cost: float = 0.0
    most_recent_call: TelemetryCall | None = None


@dataclass(frozen=True, slots=True)
class SessionTelemetryReport:
    session_path: str
    own: TelemetrySummary
    total_with_subagents: TelemetrySummary
    children: list["SessionTelemetryReport"] = field(default_factory=list)


class SessionReader(Protocol):
    def get_messages(self, path: str) -> list[dict[str, Any]]: ...

    def get_children(self, path: str) -> list[dict[str, Any]]: ...


def collect_telemetry_calls(messages: list[dict[str, Any]]) -> list[TelemetryCall]:
    calls: list[TelemetryCall] = []

    for message in messages:
        if message.get("role") != "assistant":
            continue

        parts = _parts_from_message(message)
        step_calls = _step_finish_calls(parts)
        if step_calls:
            calls.extend(step_calls)
            continue

        fallback_call = _fallback_message_call(message)
        if fallback_call is not None:
            calls.append(fallback_call)

    return calls


def summarize_telemetry(calls: list[TelemetryCall]) -> TelemetrySummary:
    if not calls:
        return TelemetrySummary()

    input_tokens = sum(c.input_tokens for c in calls)
    output_tokens = sum(c.output_tokens for c in calls)
    reasoning_tokens = sum(c.reasoning_tokens for c in calls)
    cache_read_tokens = sum(c.cache_read_tokens for c in calls)
    cache_write_tokens = sum(c.cache_write_tokens for c in calls)
    total_cost = sum(c.cost for c in calls)
    most_recent_call = _most_recent_call(calls)

    return TelemetrySummary(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        total_tokens=(
            input_tokens
            + output_tokens
            + reasoning_tokens
            + cache_read_tokens
            + cache_write_tokens
        ),
        api_calls=len(calls),
        total_cost=total_cost,
        most_recent_call=most_recent_call,
    )


def summarize_session_with_subagents(
    reader: SessionReader,
    session_path: str,
    *,
    source: str = "auto",
    strict_schema: bool = False,
) -> SessionTelemetryReport:
    messages = reader.get_messages(session_path)
    own_calls = collect_telemetry_calls(messages)
    own_summary = summarize_telemetry(own_calls)

    children_reports: list[SessionTelemetryReport] = []
    for child in reader.get_children(session_path):
        child_path = _child_path(child, source=source, strict_schema=strict_schema)
        if child_path is None:
            continue
        children_reports.append(
            summarize_session_with_subagents(
                reader,
                child_path,
                source=source,
                strict_schema=strict_schema,
            )
        )

    total_summary = own_summary
    for child in children_reports:
        total_summary = _merge_summaries(total_summary, child.total_with_subagents)

    return SessionTelemetryReport(
        session_path=session_path,
        own=own_summary,
        total_with_subagents=total_summary,
        children=children_reports,
    )


def _parts_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts")
    if isinstance(parts, list):
        return [p for p in parts if isinstance(p, dict)]
    return []


def _step_finish_calls(parts: list[dict[str, Any]]) -> list[TelemetryCall]:
    calls: list[TelemetryCall] = []
    for part in parts:
        if part.get("type") != "step-finish":
            continue
        tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
        cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
        calls.append(
            TelemetryCall(
                input_tokens=_safe_int(tokens.get("input")),
                output_tokens=_safe_int(tokens.get("output")),
                reasoning_tokens=_safe_int(tokens.get("reasoning")),
                cache_read_tokens=_safe_int(cache.get("read")),
                cache_write_tokens=_safe_int(cache.get("write")),
                cost=_safe_float(part.get("cost")),
                timestamp_ms=_safe_optional_int(part.get("timestamp"))
                or _safe_optional_int(part.get("time")),
            )
        )
    return calls


def _fallback_message_call(message: dict[str, Any]) -> TelemetryCall | None:
    info = message.get("info") if isinstance(message.get("info"), dict) else {}
    tokens = info.get("tokens") if isinstance(info.get("tokens"), dict) else {}
    if not tokens and isinstance(message.get("tokens"), dict):
        tokens = message.get("tokens")
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}

    cost_value = info.get("cost")
    if cost_value is None:
        cost_value = message.get("cost")

    has_tokens = bool(tokens)
    has_cost = cost_value is not None
    if not has_tokens and not has_cost:
        return None

    timestamp_ms = None
    time_info = info.get("time") if isinstance(info.get("time"), dict) else {}
    if isinstance(message.get("time"), dict):
        time_info = message.get("time")
    if time_info:
        timestamp_ms = _safe_optional_int(time_info.get("completed")) or _safe_optional_int(
            time_info.get("created")
        )

    return TelemetryCall(
        input_tokens=_safe_int(tokens.get("input")),
        output_tokens=_safe_int(tokens.get("output")),
        reasoning_tokens=_safe_int(tokens.get("reasoning")),
        cache_read_tokens=_safe_int(cache.get("read")),
        cache_write_tokens=_safe_int(cache.get("write")),
        cost=_safe_float(cost_value),
        timestamp_ms=timestamp_ms,
    )


def _child_path(
    child: dict[str, Any], *, source: str = "auto", strict_schema: bool = False
) -> str | None:
    key_order = _child_key_order_for_source(source)
    for key in key_order:
        value = child.get(key)
        if isinstance(value, str) and value:
            return value

    if strict_schema:
        raise TelemetrySchemaError(
            f"Invalid child schema for source='{source}': expected one of keys {key_order}, got keys {sorted(child.keys())}"
        )
    return None


def _child_key_order_for_source(source: str) -> tuple[str, ...]:
    if source == "local":
        return ("id", "session_id", "path", "sessionID")
    if source == "api":
        return ("path", "id", "sessionID", "session_id")
    return ("path", "id", "sessionID", "session_id")


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _most_recent_call(calls: list[TelemetryCall]) -> TelemetryCall | None:
    with_timestamp = [c for c in calls if c.timestamp_ms is not None]
    if with_timestamp:
        return max(with_timestamp, key=lambda c: int(c.timestamp_ms or 0))
    return calls[-1]


def _merge_summaries(a: TelemetrySummary, b: TelemetrySummary) -> TelemetrySummary:
    most_recent = _most_recent_call(
        [c for c in [a.most_recent_call, b.most_recent_call] if c is not None]
    )
    return TelemetrySummary(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        api_calls=a.api_calls + b.api_calls,
        total_cost=a.total_cost + b.total_cost,
        most_recent_call=most_recent,
    )
