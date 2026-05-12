from __future__ import annotations

from dataclasses import dataclass
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


def build_canonical_metrics(session_id: str, messages: list[dict[str, Any]]) -> CanonicalMetrics:
    telemetry = summarize_telemetry(collect_telemetry_calls(messages))
    attribution = collect_content_attribution(messages)
    model = _detect_model(messages)

    pricing_lookup = build_default_pricing_lookup()
    estimated_session_cost = _estimate_session_cost_per_call(
        messages,
        fallback_model=model,
        pricing_lookup=pricing_lookup,
    )
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
                "estimated_session_tokens": row["tokens"] * telemetry.api_calls,
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
                "estimated_session_tokens": row["tokens"] * telemetry.api_calls,
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
    info = message.get("info") if isinstance(message.get("info"), dict) else {}
    model_id = info.get("modelID")
    provider_id = info.get("providerID")
    if isinstance(model_id, str) and model_id:
        if isinstance(provider_id, str) and provider_id:
            return f"{provider_id}/{model_id}"
        return model_id
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    nested_model_id = model.get("modelID")
    nested_provider_id = model.get("providerID")
    if isinstance(nested_model_id, str) and nested_model_id:
        if isinstance(nested_provider_id, str) and nested_provider_id:
            return f"{nested_provider_id}/{nested_model_id}"
        return nested_model_id
    return "unknown"


def _estimate_session_cost_per_call(
    messages: list[dict[str, Any]],
    *,
    fallback_model: str,
    pricing_lookup: PricingLookup,
) -> float:
    total = 0.0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        model_name = _detect_model_from_message(msg)
        if model_name == "unknown":
            model_name = fallback_model
        pricing = pricing_lookup.get_pricing(model_name)
        for call in collect_telemetry_calls([msg]):
            total += estimate_session_cost_usd(
                pricing,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                reasoning_tokens=call.reasoning_tokens,
                cache_read_tokens=call.cache_read_tokens,
                cache_write_tokens=call.cache_write_tokens,
                web_search_requests=call.web_search_requests,
            )
    return total


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
