from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from .content_attribution import collect_content_attribution
from .cost import build_default_pricing_lookup
from .activity_classifier import classify_turn, extract_assistant_activity, extract_root_dir, extract_user_text
from .telemetry import TelemetryCall, collect_telemetry_calls, summarize_telemetry
from .pricing import PricingLookup, PricingResolution, estimate_session_cost_usd, tier_applicability
from .pricing import load_local_model_patterns

import fnmatch


@dataclass(frozen=True, slots=True)
class CanonicalMetrics:
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    session_total_tokens: int
    api_calls: int
    actual_cost_usd: float
    estimated_cost_usd: float
    token_composition: dict[str, int]
    component_rows: list[dict[str, Any]]
    component_family_rows: list[dict[str, Any]]
    core_rows: list[dict[str, Any]]
    tool_rows: list[dict[str, Any]]
    mcp_rows: list[dict[str, Any]]
    per_model_costs: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    activity_rows: list[dict[str, Any]] = field(default_factory=list)
    trend_calls: list[TelemetryCall] = field(default_factory=list)
    pricing_coverage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ResolvedCall:
    call: TelemetryCall
    model_name: str
    resolution: PricingResolution
    context_tokens: int
    tier_status: str
    estimated_cost: float


def build_canonical_metrics(
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    session_info: dict[str, Any] | None = None,
    component_aliases: dict[str, str] | None = None,
) -> CanonicalMetrics:
    telemetry_calls = collect_telemetry_calls(messages)
    telemetry = summarize_telemetry(telemetry_calls)
    attribution = collect_content_attribution(messages)
    model = _detect_model(messages)
    warnings = list(attribution.warnings)
    warnings.extend(_collect_session_warnings(session_id, telemetry, session_info))

    pricing_lookup = build_default_pricing_lookup()
    is_local_model = _is_local_model(model)
    resolved_calls = _resolve_calls(
        telemetry_calls,
        fallback_model=model,
        pricing_lookup=pricing_lookup,
    )
    estimated_session_cost = _estimate_session_cost_per_call(
        resolved_calls,
    )
    activity_rows = _build_activity_rows(
        messages,
        resolved_calls=resolved_calls,
        include_actual_cost=not is_local_model,
        include_estimated_cost=True,
    )
    per_model_costs = _build_per_model_costs(resolved_calls)
    pricing_coverage = _build_pricing_coverage(per_model_costs)
    warnings.extend(_pricing_warnings(per_model_costs))
    tool_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    total_tool_tokens = sum(t.output_tokens for t in attribution.tool_usage)
    for t in attribution.tool_usage:
        percent = round((t.output_tokens / total_tool_tokens * 100.0), 2) if total_tool_tokens > 0 else 0.0
        is_core = t.tool_name in _CORE_OPENCODE_TOOLS
        ctype, group = _resolve_component_info(t.tool_name, t.is_skill, t.is_subagent)
        tool_rows.append(
            {
                "tool": t.tool_name,
                "tokens": int(t.output_tokens),
                "percent": percent,
                "calls": int(t.call_count),
                "is_skill": t.is_skill,
                "is_subagent": t.is_subagent,
                "is_core": is_core,
            }
        )
        component_rows.append(
            {
                "component_type": ctype,
                "component_group": group,
                "component_name": t.tool_name,
                "tokens": int(t.output_tokens),
                "estimated_session_tokens": int(t.output_tokens),
                "calls": int(t.call_count),
            }
        )

    skill_rows = _extract_available_skills(messages)
    subagent_rows = _extract_available_subagents(messages)
    for row in skill_rows:
        sname = row["name"]
        sctype, sgroup = _resolve_component_info(sname, True, False)
        component_rows.append(
            {
                "component_type": sctype,
                "component_group": sgroup,
                "component_name": sname,
                "tokens": row["tokens"],
                "estimated_session_tokens": row["tokens"],
                "calls": 0,
            }
        )
    for row in subagent_rows:
        sname = row["name"]
        sctype, sgroup = _resolve_component_info(sname, False, True)
        component_rows.append(
            {
                "component_type": sctype,
                "component_group": sgroup,
                "component_name": sname,
                "tokens": row["tokens"],
                "estimated_session_tokens": row["tokens"],
                "calls": 0,
            }
        )

    mcp_rows = _build_mcp_rows(tool_rows)
    component_family_rows = _build_component_family_rows(component_rows, component_aliases=component_aliases)

    token_composition = {
        "input": telemetry.input_tokens,
        "cache_read": telemetry.cache_read_tokens,
        "cache_write": telemetry.cache_write_tokens,
        "output": telemetry.output_tokens,
        "reasoning": telemetry.reasoning_tokens,
        "web_search_requests": telemetry.web_search_requests,
    }

    core_rows = _build_core_rows(component_rows)

    return CanonicalMetrics(
        session_id=session_id,
        model=model,
        input_tokens=telemetry.input_tokens,
        output_tokens=telemetry.output_tokens,
        reasoning_tokens=telemetry.reasoning_tokens,
        cache_read_tokens=telemetry.cache_read_tokens,
        session_total_tokens=telemetry.total_tokens,
        api_calls=telemetry.api_calls,
        actual_cost_usd=0.0 if is_local_model else telemetry.total_cost,
        estimated_cost_usd=estimated_session_cost,
        token_composition=token_composition,
        component_rows=component_rows,
        component_family_rows=component_family_rows,
        core_rows=core_rows,
        tool_rows=tool_rows,
        mcp_rows=mcp_rows,
        per_model_costs=per_model_costs,
        warnings=warnings,
        activity_rows=activity_rows,
        trend_calls=telemetry_calls,
        pricing_coverage=pricing_coverage,
    )


def _build_activity_rows(
    messages: list[dict[str, Any]],
    *,
    resolved_calls: list[_ResolvedCall],
    include_actual_cost: bool,
    include_estimated_cost: bool,
) -> list[dict[str, Any]]:
    """Attribute assistant telemetry to the preceding user turn.

    Messages arrive chronologically from both supported sources. We retain only
    numeric aggregates; prompt text and message objects are discarded immediately.
    """
    rows: list[dict[str, Any]] = []
    prompt = ""
    tools: set[str] = set()
    skills: set[str] = set()
    has_subagent = False
    calls: list[_ResolvedCall] = []
    resolved_call_iter = iter(resolved_calls)

    def flush() -> None:
        nonlocal tools, skills, has_subagent, calls
        if not calls:
            tools = set()
            skills = set()
            has_subagent = False
            return
        category = classify_turn(prompt, tools, has_subagent=has_subagent, skills=skills)
        input_tokens = sum(item.call.input_tokens for item in calls)
        output_tokens = sum(item.call.output_tokens for item in calls)
        reasoning_tokens = sum(item.call.reasoning_tokens for item in calls)
        cache_read_tokens = sum(item.call.cache_read_tokens for item in calls)
        cache_write_tokens = sum(item.call.cache_write_tokens for item in calls)
        estimated_cost = round(sum(item.estimated_cost for item in calls), 6) if include_estimated_cost else 0.0
        rows.append(
            {
                "category": category,
                "tokens": input_tokens + output_tokens + reasoning_tokens + cache_read_tokens + cache_write_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "generated_tokens": output_tokens + reasoning_tokens,
                "calls": len(calls),
                "api_cost": sum(item.call.cost for item in calls) if include_actual_cost else 0.0,
                "estimated_cost": estimated_cost,
                "tier_applicability": [
                    {
                        "status": item.tier_status,
                        "context_tokens": item.context_tokens if item.call.context_tokens_complete else None,
                        "context_token_source": "input_plus_cache" if item.call.context_tokens_complete else "incomplete",
                    }
                    for item in calls
                ],
            }
        )
        tools = set()
        skills = set()
        has_subagent = False
        calls = []

    for message in messages:
        if message.get("role") == "user":
            flush()
            prompt = extract_user_text(message)
            continue
        if message.get("role") != "assistant":
            continue
        message_tools, message_skills, message_has_subagent = extract_assistant_activity(message)
        tools.update(message_tools)
        skills.update(message_skills)
        has_subagent = has_subagent or message_has_subagent
        calls.extend(next(resolved_call_iter) for _ in collect_telemetry_calls([message]))
    flush()
    return rows


def _detect_model(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        detected = _detect_model_from_message(msg)
        if detected != "unknown":
            return detected
    return "unknown"


def _detect_model_from_message(message: dict[str, Any]) -> str:
    provider_id, model_id = _message_model_ref(message)
    if isinstance(model_id, str) and model_id:
        if isinstance(provider_id, str) and provider_id:
            return f"{provider_id}/{model_id}"
        return model_id
    return "unknown"


def _estimate_session_cost_per_call(
    calls: list[_ResolvedCall],
) -> float:
    return round(sum(item.estimated_cost for item in calls), 6)


def _resolve_calls(
    calls: list[TelemetryCall],
    *,
    fallback_model: str,
    pricing_lookup: PricingLookup,
) -> list[_ResolvedCall]:
    resolved: list[_ResolvedCall] = []
    for call in calls:
        model_name = PricingLookup.build_lookup_key(call.provider_id, call.model_id) or fallback_model
        if _is_local_model(model_name):
            resolution = pricing_lookup.resolve_local_call_pricing(model_name, call.timestamp_ms)
        else:
            resolution = pricing_lookup.resolve_call_pricing(model_name, call.timestamp_ms)
        context_tokens = call.input_tokens + call.cache_read_tokens + call.cache_write_tokens
        tier_status = (
            tier_applicability(resolution.pricing, context_tokens if call.context_tokens_complete else None)
            if resolution.pricing is not None
            else "unknown"
        )
        estimated_cost = 0.0
        if call.cost <= 0 and resolution.pricing is not None:
            estimated_cost = estimate_session_cost_usd(
                resolution.pricing,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                reasoning_tokens=call.reasoning_tokens,
                cache_read_tokens=call.cache_read_tokens,
                cache_write_tokens=call.cache_write_tokens,
                web_search_requests=call.web_search_requests,
                context_tokens=context_tokens,
            )
        resolved.append(_ResolvedCall(call, model_name, resolution, context_tokens, tier_status, estimated_cost))
    return resolved


def _build_pricing_coverage(per_model_costs: list[dict[str, Any]]) -> dict[str, Any]:
    total_calls = 0
    priced_calls = 0
    future_fallback_calls = 0
    default_fallback_calls = 0
    unpriced_calls = 0
    tier_applied_calls = 0
    base_rate_calls = 0
    tier_unknown_calls = 0
    for row in per_model_costs:
        total_calls += int(row.get("priced_calls", 0)) + int(row.get("unpriced_calls", 0))
        priced_calls += int(row.get("priced_calls", 0))
        future_fallback_calls += int(row.get("future_fallback_calls", 0))
        default_fallback_calls += int(row.get("default_fallback_calls", 0))
        unpriced_calls += int(row.get("unpriced_calls", 0))
        tier_applied_calls += int(row.get("tier_applied_calls", 0))
        base_rate_calls += int(row.get("base_rate_calls", 0))
        tier_unknown_calls += int(row.get("tier_unknown_calls", 0))
    return {
        "calls": total_calls,
        "priced_calls": priced_calls,
        "future_fallback_calls": future_fallback_calls,
        "default_fallback_calls": default_fallback_calls,
        "unpriced_calls": unpriced_calls,
        "coverage_percent": round(priced_calls / total_calls * 100.0, 2) if total_calls else 0.0,
        "tier_applied_calls": tier_applied_calls,
        "base_rate_calls": base_rate_calls,
        "tier_unknown_calls": tier_unknown_calls,
    }


def _pricing_warnings(per_model_costs: list[dict[str, Any]]) -> list[str]:
    return []


def _build_per_model_costs(
    calls: list[_ResolvedCall],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in calls:
        call = item.call
        if call.total_tokens == 0 and call.cost == 0:
            continue
        model_name = item.model_name
        resolution = item.resolution
        context_tokens = item.context_tokens
        api_cost = call.cost
        estimated_cost = item.estimated_cost
        row = grouped.get(model_name)
        if row is None:
            row = {
                "model": model_name,
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "generated_tokens": 0,
                "api_cost": 0.0,
                "estimated_cost": 0.0,
                "estimated_direct_cost": 0.0,
                "estimated_generic_cost": 0.0,
                "estimated_market_cost": 0.0,
                "estimated_future_market_cost": 0.0,
                "estimated_cloud_equivalent_cost": 0.0,
                "direct_estimate_calls": 0,
                "generic_estimate_calls": 0,
                "market_estimate_calls": 0,
                "future_market_estimate_calls": 0,
                "cloud_equivalent_estimate_calls": 0,
                "priced_calls": 0,
                "future_fallback_calls": 0,
                "default_fallback_calls": 0,
                "unpriced_calls": 0,
                "provenances": [],
                "pricing_channels": [],
                "pricing_revisions": [],
                "market_provider_counts": [],
                "market_rate_dates": [],
                "market_statuses": [],
                "cost_bases": [],
                "tier_applied_calls": 0,
                "base_rate_calls": 0,
                "tier_unknown_calls": 0,
                "context_tokens": 0,
                "context_token_sources": [],
            }
            grouped[model_name] = row
        row["tokens"] += call.input_tokens + call.output_tokens + call.reasoning_tokens + call.cache_read_tokens + call.cache_write_tokens
        row["input_tokens"] += call.input_tokens
        row["output_tokens"] += call.output_tokens
        row["reasoning_tokens"] += call.reasoning_tokens
        row["generated_tokens"] += call.output_tokens + call.reasoning_tokens
        row["api_cost"] += api_cost
        row["estimated_cost"] += estimated_cost
        basis = resolution.cost_basis.lower()
        channel = resolution.billing_channel.lower()
        if resolution.status == "default_fallback":
            amount_key, count_key = "estimated_generic_cost", "generic_estimate_calls"
        elif resolution.market_status == "future" or resolution.status == "future_fallback":
            amount_key, count_key = "estimated_future_market_cost", "future_market_estimate_calls"
        elif basis == "market" or channel == "market":
            amount_key, count_key = "estimated_market_cost", "market_estimate_calls"
        elif "cloud" in basis or "cloud" in channel:
            amount_key, count_key = "estimated_cloud_equivalent_cost", "cloud_equivalent_estimate_calls"
        else:
            amount_key, count_key = "estimated_direct_cost", "direct_estimate_calls"
        row[amount_key] += estimated_cost
        if estimated_cost > 0:
            row[count_key] += 1
        tier_status = item.tier_status
        row[f"{tier_status}_rate_calls" if tier_status == "base" else f"tier_{tier_status}_calls"] += 1
        if resolution.pricing is not None:
            if call.context_tokens_complete:
                row["context_tokens"] += context_tokens
                source = "input_plus_cache"
            else:
                source = "incomplete"
            if source not in row["context_token_sources"]:
                row["context_token_sources"].append(source)
        if resolution.pricing is not None:
            row["priced_calls"] += 1
            if resolution.status == "future_fallback":
                row["future_fallback_calls"] += 1
            if resolution.status == "default_fallback":
                row["default_fallback_calls"] += 1
            if resolution.provenance and resolution.provenance not in row["provenances"]:
                row["provenances"].append(resolution.provenance)
            if resolution.billing_channel and resolution.billing_channel not in row["pricing_channels"]:
                row["pricing_channels"].append(resolution.billing_channel)
            if resolution.source_revision and resolution.source_revision not in row["pricing_revisions"]:
                row["pricing_revisions"].append(resolution.source_revision)
            if resolution.provider_count and resolution.provider_count not in row["market_provider_counts"]:
                row["market_provider_counts"].append(resolution.provider_count)
            if resolution.rate_date and resolution.rate_date not in row["market_rate_dates"]:
                row["market_rate_dates"].append(resolution.rate_date)
            if resolution.market_status and resolution.market_status not in row["market_statuses"]:
                row["market_statuses"].append(resolution.market_status)
            if resolution.cost_basis and resolution.cost_basis not in row["cost_bases"]:
                row["cost_bases"].append(resolution.cost_basis)
        else:
            row["unpriced_calls"] += 1

    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        api_cost = float(row["api_cost"])
        estimated_cost = float(row["estimated_cost"])
        model_name = str(row["model"])
        primary_cost = api_cost if api_cost > 0 else estimated_cost
        tokens = int(row["tokens"])
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        reasoning_tokens = int(row["reasoning_tokens"])
        generated_tokens = int(row["generated_tokens"])
        rows.append(
            {
                "model": model_name,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_percent": round(input_tokens / tokens * 100.0, 2) if tokens else 0.0,
                "output_percent": round(output_tokens / tokens * 100.0, 2) if tokens else 0.0,
                "reasoning_tokens": reasoning_tokens,
                "generated_tokens": generated_tokens,
                "reasoning_percent": round(reasoning_tokens / generated_tokens * 100.0, 2) if generated_tokens else 0.0,
                "api_cost": round(api_cost, 6),
                "estimated_cost": round(estimated_cost, 6),
                "estimated_direct_cost": round(row["estimated_direct_cost"], 6),
                "estimated_generic_cost": round(row["estimated_generic_cost"], 6),
                "estimated_market_cost": round(row["estimated_market_cost"], 6),
                "estimated_future_market_cost": round(row["estimated_future_market_cost"], 6),
                "estimated_cloud_equivalent_cost": round(row["estimated_cloud_equivalent_cost"], 6),
                "direct_estimate_calls": int(row["direct_estimate_calls"]),
                "generic_estimate_calls": int(row["generic_estimate_calls"]),
                "market_estimate_calls": int(row["market_estimate_calls"]),
                "future_market_estimate_calls": int(row["future_market_estimate_calls"]),
                "cloud_equivalent_estimate_calls": int(row["cloud_equivalent_estimate_calls"]),
                "cost": round(primary_cost, 6),
                "priced_calls": int(row["priced_calls"]),
                "future_fallback_calls": int(row["future_fallback_calls"]),
                "default_fallback_calls": int(row["default_fallback_calls"]),
                "unpriced_calls": int(row["unpriced_calls"]),
                "pricing_provenance": "; ".join(str(item) for item in row["provenances"]),
                "pricing_channels": "; ".join(str(item) for item in row["pricing_channels"]),
                "pricing_revisions": "; ".join(str(item) for item in row["pricing_revisions"]),
                "market_provider_count": max(row["market_provider_counts"], default=0),
                "market_rate_date": "; ".join(str(item) for item in row["market_rate_dates"]),
                "market_status": "; ".join(str(item) for item in row["market_statuses"]),
                "cost_basis": "; ".join(str(item) for item in row["cost_bases"]),
                "tier_applied_calls": int(row["tier_applied_calls"]),
                "base_rate_calls": int(row["base_rate_calls"]),
                "tier_unknown_calls": int(row["tier_unknown_calls"]),
                "context_tokens": int(row["context_tokens"]),
                "context_token_source": "; ".join(str(item) for item in row["context_token_sources"]),
            }
        )
    rows.sort(key=lambda x: (float(x["api_cost"]), float(x["estimated_cost"])), reverse=True)
    return rows


def _message_model_ref(message: dict[str, Any]) -> tuple[str | None, str | None]:
    def _str(value: Any) -> str | None:
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
    return provider_id, model_id


def _is_local_model(model: str) -> bool:
    """Check if a model is a local model (no API cost).

    Uses wildcard patterns from models.conf @local directive.
    Patterns support * wildcard, e.g.: myollama/*, *qwen36*
    """
    patterns = load_local_model_patterns()
    if not patterns:
        return False
    for pattern in patterns:
        if fnmatch.fnmatch(model.lower(), pattern.lower()):
            return True
    return False


def _collect_session_warnings(
    session_id: str,
    telemetry: Any,
    session_info: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(session_info, dict):
        return []

    warnings: list[str] = []
    aggregate = _extract_session_aggregate(session_info)
    if aggregate is not None:
        token_mismatch = (
            aggregate["input"] != telemetry.input_tokens
            or aggregate["output"] != telemetry.output_tokens
            or aggregate["reasoning"] != telemetry.reasoning_tokens
            or aggregate["cache_read"] != telemetry.cache_read_tokens
            or aggregate["cache_write"] != telemetry.cache_write_tokens
        )
        cost_mismatch = abs(aggregate["cost"] - telemetry.total_cost) > 1e-9
        if token_mismatch or cost_mismatch:
            aggregate_total = (
                aggregate["input"]
                + aggregate["output"]
                + aggregate["reasoning"]
                + aggregate["cache_read"]
                + aggregate["cache_write"]
            )
            warnings.append(
                "session aggregate mismatch: "
                f"messages={telemetry.total_tokens} tokens/${telemetry.total_cost:.6f}, "
                f"session={aggregate_total} tokens/${aggregate['cost']:.6f}"
            )

    revert_message_id = _extract_revert_message_id(session_info)
    if revert_message_id:
        warnings.append(
            f"session has active revert at message {revert_message_id}; retained local content may include reverted history"
        )

    return warnings


def _extract_session_aggregate(session_info: dict[str, Any]) -> dict[str, float] | None:
    for source in _session_info_sources(session_info):
        tokens = source.get("tokens") if isinstance(source.get("tokens"), dict) else None
        if not tokens:
            continue
        cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
        return {
            "input": float(_safe_int(tokens.get("input"))),
            "output": float(_safe_int(tokens.get("output"))),
            "reasoning": float(_safe_int(tokens.get("reasoning"))),
            "cache_read": float(_safe_int(cache.get("read"))),
            "cache_write": float(_safe_int(cache.get("write"))),
            "cost": _safe_float(source.get("cost")),
        }
    return None


def _extract_revert_message_id(session_info: dict[str, Any]) -> str | None:
    for source in _session_info_sources(session_info):
        revert = source.get("revert") if isinstance(source.get("revert"), dict) else None
        if not revert:
            continue
        message_id = revert.get("messageID") or revert.get("messageId") or revert.get("message_id")
        if isinstance(message_id, str) and message_id.strip():
            return message_id.strip()
    return None


def _session_info_sources(session_info: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = [session_info]
    for key in ("data", "info", "session"):
        value = session_info.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _component_group(name: str) -> str:
    if "_" in name:
        head = name.split("_", 1)[0]
        return head or name
    if "-" in name:
        head = name.split("-", 1)[0]
        return head or name
    return name


def _resolve_component_info(
    tool_name: str, is_skill: bool, is_subagent: bool
) -> tuple[str, str]:
    if tool_name in _CORE_OPENCODE_TOOLS:
        return ("core", "opencode-core")
    if is_skill and tool_name in _CORE_OPENCODE_SKILLS:
        return ("core", "opencode-core")
    if is_skill:
        return ("skill", tool_name)
    if is_subagent and tool_name in _CORE_OPENCODE_SUBAGENTS:
        return ("core", "opencode-core")
    if is_subagent:
        return ("subagent", _component_group(tool_name))
    return ("tool", _component_group(tool_name))


_LOCAL_TOOL_RE = re.compile(r"^(read|bash|glob|todowrite|task|tokenscope|apply_patch|skill|quota_status)$")
_CORE_OPENCODE_TOOLS = {
    "read",
    "bash",
    "grep",
    "glob",
    "todowrite",
    "apply_patch",
    "apply",
    "webfetch",
    "invalid",
    "edit",
    "question",
    "compress",
    "write",
}
_CORE_OPENCODE_SKILLS = {"plan", "implement"}
_CORE_OPENCODE_SUBAGENTS = {"explore", "general"}


def _match_subagents_to_skills(
    component_rows: list[dict[str, Any]],
    component_aliases: dict[str, str] | None = None,
) -> tuple[set[str], set[int]]:
    """Match subagent rows to same-session skill names.

    A subagent matches a skill when the skill name is a case-insensitive
    prefix of the subagent name (no separator required). The longest
    matching skill wins, and the skill's effective component group (after
    explicit aliases) becomes the subagent's component group. Returns the
    set of matched raw skill names (protected from prefix collapsing) and
    the ids of the matched subagent rows. Explicitly aliased subagent rows
    are left for alias application.
    """
    alias_lookup = {str(name).lower(): str(label) for name, label in (component_aliases or {}).items()}
    skill_names: list[str] = []
    for row in component_rows:
        if row.get("component_type") != "skill":
            continue
        skill = str(row.get("component_name", row.get("component_group", "")))
        if skill:
            skill_names.append(skill)

    matched_skills: set[str] = set()
    matched_rows: set[int] = set()
    if not skill_names:
        return matched_skills, matched_rows

    for row in component_rows:
        if row.get("component_type") != "subagent":
            continue
        name = str(row.get("component_name", row.get("component_group", "")))
        if name.lower() in alias_lookup:
            continue
        lowered = name.lower()
        best: str | None = None
        for skill in skill_names:
            lowered_skill = skill.lower()
            if lowered.startswith(lowered_skill):
                if best is None or len(skill) > len(best):
                    best = skill
        if best is not None:
            row["component_group"] = alias_lookup.get(best.lower(), best)
            matched_skills.add(best)
            matched_rows.add(id(row))
    return matched_skills, matched_rows


def _session_root_dir(
    metrics: Any,
    session_dirs: dict[str, str] | None,
) -> str:
    raw_dir = str((session_dirs or {}).get(str(metrics.session_id), "") or "")
    return extract_root_dir(raw_dir) if raw_dir else (str(metrics.session_id) or "-")


def build_cross_session_skill_index(
    session_metrics: list[Any],
    session_dirs: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Collect skill component names from the selected sessions, keyed by root dir."""
    index: dict[str, list[str]] = {}
    for metrics in session_metrics:
        root_dir = _session_root_dir(metrics, session_dirs)
        names = index.setdefault(root_dir, [])
        for row in metrics.component_rows:
            if row.get("component_type") != "skill":
                continue
            name = str(row.get("component_name", row.get("component_group", "")))
            if name:
                names.append(name)
    return index


def apply_cross_session_skill_matching(
    session_metrics: list[Any],
    session_dirs: dict[str, str] | None = None,
    *,
    component_aliases: dict[str, str] | None = None,
) -> None:
    """Match subagent names to skill names across the selected sessions.

    Skill candidates come from skill component rows of the same report
    sessions (available and invoked), partitioned by root dir. A subagent
    matches a skill when its name starts with the skill name
    (case-insensitive, no separator required); the longest matching skill
    wins and the skill's effective component group (after explicit aliases)
    becomes the subagent's component group. Explicitly aliased names are
    left for alias application.
    """
    alias_lookup = {str(name).lower(): str(label) for name, label in (component_aliases or {}).items()}
    index = build_cross_session_skill_index(session_metrics, session_dirs)
    for metrics in session_metrics:
        root_dir = _session_root_dir(metrics, session_dirs)
        skill_names = index.get(root_dir)
        if not skill_names:
            continue
        for row in metrics.component_rows:
            if row.get("component_type") != "subagent":
                continue
            name = str(row.get("component_name", row.get("component_group", "")))
            if not name or name.lower() in alias_lookup:
                continue
            lowered = name.lower()
            best: str | None = None
            for skill in skill_names:
                lowered_skill = skill.lower()
                if lowered.startswith(lowered_skill) and (best is None or len(skill) > len(best)):
                    best = skill
            if best is not None:
                row["component_group"] = alias_lookup.get(best.lower(), best)


def _build_component_family_rows(
    component_rows: list[dict[str, Any]],
    *,
    component_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    alias_lookup: dict[str, str] = {}
    if component_aliases:
        alias_lookup = {str(name).lower(): str(label) for name, label in component_aliases.items()}
    aliased_names = frozenset(alias_lookup)
    matched_skill_names, matched_subagent_ids = _match_subagents_to_skills(
        component_rows, component_aliases=alias_lookup
    )
    _normalize_skill_component_groups(component_rows, matched_skill_names, aliased_names)
    # Explicit aliases override automatic grouping: reassign the group of
    # every row whose original name carries an alias.
    if alias_lookup:
        for row in component_rows:
            target = alias_lookup.get(str(row.get("component_name", "")).lower())
            if target is not None:
                row["component_group"] = target
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    type_sets: dict[tuple[str, str], set[str]] = {}
    for row in component_rows:
        ctype = row["component_type"]
        group = str(row["component_group"])
        name = str(row.get("component_name", ""))
        # Unmatched subagents keep their reduced label but aggregate only
        # with other unmatched subagents. A structured key can never equal
        # a tool/skill group label such as "subagent:make". Explicitly
        # aliased rows aggregate by their canonical group instead.
        if ctype == "subagent" and id(row) not in matched_subagent_ids and name.lower() not in aliased_names:
            key = ("subagent", group)
        else:
            key = ("group", group)
        if key not in grouped:
            grouped[key] = {
                "component_group": group,
                "tokens": 0,
                "estimated_session_tokens": 0,
                "calls": 0,
            }
        type_sets.setdefault(key, set()).add(ctype)
        g = grouped[key]
        g["tokens"] += int(row["tokens"])
        g["estimated_session_tokens"] += int(row["estimated_session_tokens"])
        g["calls"] += int(row["calls"])

    total_tokens = sum(v["tokens"] for v in grouped.values()) or 1
    out: list[dict[str, Any]] = []
    for key, g in grouped.items():
        group = g["component_group"]
        types = type_sets.get(key, set())
        out.append(
            {
                "component_type": "mixed" if len(types) > 1 else (types.pop() if types else "unknown"),
                "component_group": group,
                "tokens": g["tokens"],
                "estimated_session_tokens": g["estimated_session_tokens"],
                "calls": g["calls"],
                "percent": round((g["tokens"] / total_tokens * 100.0), 2),
            }
        )
    out.sort(key=lambda x: x["tokens"], reverse=True)
    return out


def _build_core_rows(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in component_rows:
        if row.get("component_group") != "opencode-core":
            continue
        name = str(row.get("component_name"))
        if name == "invalid":
            name = "general"
        if name not in grouped:
            grouped[name] = {
                "component_type": "core",
                "component_group": "opencode-core",
                "component_name": name,
                "tokens": 0,
                "estimated_session_tokens": 0,
                "calls": 0,
            }
        g = grouped[name]
        g["tokens"] += int(row.get("tokens", 0))
        g["estimated_session_tokens"] += int(row.get("estimated_session_tokens", 0))
        g["calls"] += int(row.get("calls", 0))
    rows = list(grouped.values())
    rows.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return rows


def _normalize_skill_component_groups(
    component_rows: list[dict[str, Any]],
    matched_skill_names: set[str] | None = None,
    aliased_names: frozenset[str] = frozenset(),
) -> None:
    if matched_skill_names is None:
        matched_skill_names = set()
    # Collect all prefixes from skills (hyphenated names)
    skill_prefix_counts: dict[str, int] = {}
    for row in component_rows:
        if row.get("component_type") != "skill":
            continue
        name = str(row.get("component_name", row.get("component_group", "")))
        if "-" not in name:
            continue
        prefix = name.split("-", 1)[0]
        if not prefix:
            continue
        skill_prefix_counts[prefix] = skill_prefix_counts.get(prefix, 0) + 1

    # Collect tool groups to check for overlapping prefixes
    tool_groups: set[str] = set()
    for row in component_rows:
        if row.get("component_type") == "tool":
            tool_groups.add(row.get("component_group", ""))

    for row in component_rows:
        if row.get("component_type") != "skill":
            continue
        name = str(row.get("component_name", row.get("component_group", "")))
        # Explicitly aliased skills keep their group for alias application.
        if name.lower() in aliased_names:
            continue
        if "-" not in name:
            continue
        prefix = name.split("-", 1)[0]
        # Matched skill groups keep their exact name (no prefix collapse).
        if name in matched_skill_names:
            row["component_group"] = name
            continue
        # Normalize to prefix if:
        # - multiple skills share the prefix, OR
        # - a tool already exists with the same prefix as group
        if prefix and (skill_prefix_counts.get(prefix, 0) > 1 or prefix in tool_groups):
            row["component_group"] = prefix
        else:
            row["component_group"] = name


def _build_mcp_rows(tool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in tool_rows:
        tool = str(row["tool"])
        if _LOCAL_TOOL_RE.match(tool):
            continue
        # Exclude skill/subagent/core calls from MCP Servers - they are not MCP tools
        if row.get("is_skill") or row.get("is_subagent") or row.get("is_core"):
            continue
        group = _component_group(tool)
        if group not in grouped:
            grouped[group] = {"tokens": 0.0, "calls": 0.0}
        grouped[group]["tokens"] += float(row["tokens"])
        grouped[group]["calls"] += float(row["calls"])

    total_tokens = sum(v["tokens"] for v in grouped.values())
    out: list[dict[str, Any]] = []
    for name, v in grouped.items():
        calls = int(v["calls"])
        tokens = int(v["tokens"])
        out.append(
            {
                "name": name,
                "tokens": tokens,
                "calls": calls,
                "tokens_per_call": round(tokens / max(calls, 1), 2),
                "percent": round((tokens / total_tokens * 100.0), 2) if total_tokens > 0 else 0.0,
            }
        )
    out.sort(key=lambda x: int(x["tokens"]), reverse=True)
    return out


def _extract_available_skills(messages: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    text = "\n".join(_collect_system_texts(messages))
    rows: list[dict[str, int | str]] = []
    pattern = re.compile(r"<skill>\s*<name>([^<]+)</name>\s*<description>(.*?)</description>", re.DOTALL)
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        desc = re.sub(r"\s+", " ", m.group(2)).strip()
        raw = f"{name}: {desc}"
        rows.append({"name": name, "tokens": _approx_tokens(raw)})
    return rows


def _extract_available_subagents(messages: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    text = "\n".join(_collect_system_texts(messages))
    rows: list[dict[str, int | str]] = []
    pattern = re.compile(r"-\s+([a-zA-Z0-9_-]+):\s+([^\n]+)")
    seen: set[str] = set()
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        desc = m.group(2).strip()
        if name in {"explore", "general", "svelte-file-editor"} or "agent" in desc.lower():
            if name in seen:
                continue
            seen.add(name)
            rows.append({"name": name, "tokens": _approx_tokens(f"{name}: {desc}")})
    return rows


def _collect_system_texts(messages: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for msg in messages:
        info = msg.get("info")
        if not isinstance(info, dict):
            continue
        system = info.get("system")
        if isinstance(system, str) and system.strip():
            out.append(system)
        elif isinstance(system, list):
            for item in system:
                if isinstance(item, str) and item.strip():
                    out.append(item)
    return out


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
