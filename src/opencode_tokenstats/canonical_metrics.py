from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from .content_attribution import collect_content_attribution
from .cost import build_default_pricing_lookup, calculate_cost_summary
from .telemetry import collect_telemetry_calls, summarize_telemetry


@dataclass(frozen=True, slots=True)
class CanonicalMetrics:
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    session_total_tokens: int
    api_calls: int
    actual_cost_usd: float
    estimated_cost_usd: float
    token_composition: dict[str, int]
    component_rows: list[dict[str, Any]]
    contributor_rows: list[dict[str, Any]]
    tool_rows: list[dict[str, Any]]
    mcp_rows: list[dict[str, Any]]


def build_canonical_metrics(session_id: str, messages: list[dict[str, Any]]) -> CanonicalMetrics:
    telemetry = summarize_telemetry(collect_telemetry_calls(messages))
    attribution = collect_content_attribution(messages)
    model = _detect_model(messages)

    cost = calculate_cost_summary(
        telemetry,
        model_name=model,
        pricing_lookup=build_default_pricing_lookup(),
    )
    primary_cost = telemetry.total_cost if telemetry.total_cost > 0 else cost.estimated_session_cost

    tool_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    total_tool_tokens = sum(t.output_tokens for t in attribution.tool_usage)
    for t in attribution.tool_usage:
        percent = round((t.output_tokens / total_tool_tokens * 100.0), 2) if total_tool_tokens > 0 else 0.0
        group = _component_group(t.tool_name)
        tool_rows.append(
            {
                "tool": t.tool_name,
                "tokens": int(t.output_tokens),
                "percent": percent,
                "calls": int(t.call_count),
            }
        )
        component_rows.append(
            {
                "component_type": "tool",
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
        component_rows.append(
            {
                "component_type": "skill",
                "component_group": _component_group(row["name"]),
                "component_name": row["name"],
                "tokens": row["tokens"],
                "estimated_session_tokens": row["tokens"] * telemetry.api_calls,
                "calls": 0,
            }
        )
    for row in subagent_rows:
        component_rows.append(
            {
                "component_type": "subagent",
                "component_group": _component_group(row["name"]),
                "component_name": row["name"],
                "tokens": row["tokens"],
                "estimated_session_tokens": row["tokens"] * telemetry.api_calls,
                "calls": 0,
            }
        )

    contributor_rows = [{"name": row["component_name"], "tokens": row["tokens"]} for row in component_rows]
    if attribution.totals.system_tokens > 0:
        base = max(sum(r["tokens"] for r in component_rows) + attribution.totals.system_tokens, 1)
        contributor_rows.append(
            {
                "name": "System (inferred from API telemetry)",
                "tokens": int(attribution.totals.system_tokens),
                "percent": round((attribution.totals.system_tokens / base) * 100.0, 2),
            }
        )
    contrib_total = max(sum(int(r["tokens"]) for r in contributor_rows), 1)
    for row in contributor_rows:
        row["percent"] = round((int(row["tokens"]) / contrib_total) * 100.0, 2)
    contributor_rows.sort(key=lambda x: int(x["tokens"]), reverse=True)

    mcp_rows = _build_mcp_rows(tool_rows)

    token_composition = {
        "input": telemetry.input_tokens,
        "output": telemetry.output_tokens,
        "reasoning": telemetry.reasoning_tokens,
        "cache_read": telemetry.cache_read_tokens,
        "cache_write": telemetry.cache_write_tokens,
        "tool_output": attribution.totals.tool_output_tokens,
    }

    return CanonicalMetrics(
        session_id=session_id,
        model=model,
        input_tokens=telemetry.input_tokens,
        output_tokens=telemetry.output_tokens,
        reasoning_tokens=telemetry.reasoning_tokens,
        cache_read_tokens=telemetry.cache_read_tokens,
        cache_write_tokens=telemetry.cache_write_tokens,
        session_total_tokens=telemetry.total_tokens,
        api_calls=telemetry.api_calls,
        actual_cost_usd=telemetry.total_cost,
        estimated_cost_usd=primary_cost,
        token_composition=token_composition,
        component_rows=component_rows,
        contributor_rows=contributor_rows[:10],
        tool_rows=tool_rows,
        mcp_rows=mcp_rows,
    )


def _detect_model(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
        model_id = info.get("modelID")
        if isinstance(model_id, str) and model_id:
            return model_id
        model = info.get("model") if isinstance(info.get("model"), dict) else {}
        nested = model.get("modelID")
        if isinstance(nested, str) and nested:
            return nested
    return "unknown"


def _component_group(name: str) -> str:
    if "_" in name:
        head = name.split("_", 1)[0]
        return head or name
    if "-" in name:
        head = name.split("-", 1)[0]
        return head or name
    return name


_LOCAL_TOOL_RE = re.compile(r"^(read|bash|glob|todowrite|task|tokenscope|apply_patch|skill)$")


def _build_mcp_rows(tool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in tool_rows:
        tool = str(row["tool"])
        if _LOCAL_TOOL_RE.match(tool):
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
    return out[:10]


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
