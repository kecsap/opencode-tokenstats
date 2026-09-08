from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from .content_attribution import collect_content_attribution
from .cost import build_default_pricing_lookup
from .telemetry import collect_telemetry_calls, summarize_telemetry
from .pricing import PricingLookup, estimate_session_cost_usd
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


def build_canonical_metrics(
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    session_info: dict[str, Any] | None = None,
) -> CanonicalMetrics:
    telemetry_calls = collect_telemetry_calls(messages)
    telemetry = summarize_telemetry(telemetry_calls)
    attribution = collect_content_attribution(messages)
    model = _detect_model(messages)
    warnings = list(attribution.warnings)
    warnings.extend(_collect_session_warnings(session_id, telemetry, session_info))

    pricing_lookup = build_default_pricing_lookup()
    estimated_session_cost = _estimate_session_cost_per_call(
        telemetry_calls,
        fallback_model=model,
        pricing_lookup=pricing_lookup,
    )
    per_model_costs = _build_per_model_costs(telemetry_calls, fallback_model=model, pricing_lookup=pricing_lookup)
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
    component_family_rows = _build_component_family_rows(component_rows)

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
        actual_cost_usd=0.0 if _is_local_model(model) else telemetry.total_cost,
        estimated_cost_usd=0.0 if (not _is_local_model(model) and telemetry.total_cost > 0) else estimated_session_cost,
        token_composition=token_composition,
        component_rows=component_rows,
        component_family_rows=component_family_rows,
        core_rows=core_rows,
        tool_rows=tool_rows,
        mcp_rows=mcp_rows,
        per_model_costs=per_model_costs,
        warnings=warnings,
    )


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
    calls: list[Any],
    *,
    fallback_model: str,
    pricing_lookup: PricingLookup,
) -> float:
    total = 0.0
    for call in calls:
        model_name = PricingLookup.build_lookup_key(call.provider_id, call.model_id)
        if not model_name:
            model_name = fallback_model
        pricing = pricing_lookup.get_pricing(model_name)
        context_tokens = call.input_tokens + call.cache_read_tokens + call.cache_write_tokens
        total += estimate_session_cost_usd(
            pricing,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            reasoning_tokens=call.reasoning_tokens,
            cache_read_tokens=call.cache_read_tokens,
            cache_write_tokens=call.cache_write_tokens,
            web_search_requests=call.web_search_requests,
            context_tokens=context_tokens,
        )
    return total


def _build_per_model_costs(
    calls: list[Any],
    *,
    fallback_model: str,
    pricing_lookup: PricingLookup,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for call in calls:
        if call.total_tokens == 0 and call.cost == 0:
            continue
        model_name = PricingLookup.build_lookup_key(call.provider_id, call.model_id)
        if not model_name:
            model_name = fallback_model
        pricing = pricing_lookup.get_pricing(model_name)
        context_tokens = call.input_tokens + call.cache_read_tokens + call.cache_write_tokens
        api_cost = call.cost
        estimated_cost = estimate_session_cost_usd(
            pricing,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            reasoning_tokens=call.reasoning_tokens,
            cache_read_tokens=call.cache_read_tokens,
            cache_write_tokens=call.cache_write_tokens,
            web_search_requests=call.web_search_requests,
            context_tokens=context_tokens,
        )
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
            }
            grouped[model_name] = row
        row["tokens"] += call.input_tokens + call.output_tokens + call.reasoning_tokens + call.cache_read_tokens + call.cache_write_tokens
        row["input_tokens"] += call.input_tokens
        row["output_tokens"] += call.output_tokens
        row["reasoning_tokens"] += call.reasoning_tokens
        row["generated_tokens"] += call.output_tokens + call.reasoning_tokens
        row["api_cost"] += api_cost
        row["estimated_cost"] += estimated_cost

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
                "estimated_cost": round(estimated_cost if api_cost <= 0 else 0.0, 6),
                "cost": round(primary_cost, 6),
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


def _build_component_family_rows(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _normalize_skill_component_groups(component_rows)
    grouped: dict[str, dict[str, Any]] = {}
    type_sets: dict[str, set[str]] = {}
    for row in component_rows:
        group = str(row["component_group"])
        if group not in grouped:
            grouped[group] = {
                "component_group": group,
                "tokens": 0,
                "estimated_session_tokens": 0,
                "calls": 0,
            }
        type_sets.setdefault(group, set()).add(row["component_type"])
        g = grouped[group]
        g["tokens"] += int(row["tokens"])
        g["estimated_session_tokens"] += int(row["estimated_session_tokens"])
        g["calls"] += int(row["calls"])

    total_tokens = sum(v["tokens"] for v in grouped.values()) or 1
    out: list[dict[str, Any]] = []
    for g in grouped.values():
        group = g["component_group"]
        types = type_sets.get(group, set())
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


def _normalize_skill_component_groups(component_rows: list[dict[str, Any]]) -> None:
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
        if "-" not in name:
            continue
        prefix = name.split("-", 1)[0]
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
