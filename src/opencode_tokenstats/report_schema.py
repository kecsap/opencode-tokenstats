from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .canonical_metrics import CanonicalMetrics


@dataclass(frozen=True, slots=True)
class PeriodPoint:
    date: str
    sessions: int
    api_calls: int
    tokens: int
    api_cost: float


def build_report_schema(
    *,
    period: str,
    mode: str,
    start: datetime,
    end: datetime,
    session_metrics: list[CanonicalMetrics],
) -> dict[str, Any]:
    total_sessions = len(session_metrics)
    total_api_calls = sum(m.api_calls for m in session_metrics)
    total_tokens = sum(m.session_total_tokens for m in session_metrics)
    total_api_cost = round(sum(m.estimated_cost_usd for m in session_metrics), 6)

    token_totals = {
        "input": sum(m.input_tokens for m in session_metrics),
        "output": sum(m.output_tokens for m in session_metrics),
        "reasoning": sum(m.reasoning_tokens for m in session_metrics),
        "cache_read": sum(m.cache_read_tokens for m in session_metrics),
    }

    tools: dict[str, dict[str, int]] = {}
    contributors: dict[str, int] = {}
    models: dict[str, float] = {}
    components: dict[tuple[str, str, str], int] = {}
    skills: dict[str, int] = {}
    subagents: dict[str, int] = {}

    for m in session_metrics:
        models[m.model] = round(models.get(m.model, 0.0) + m.estimated_cost_usd, 6)
        for row in m.tool_rows:
            name = str(row["tool"])
            if name not in tools:
                tools[name] = {"tokens": 0, "calls": 0}
            tools[name]["tokens"] += int(row["tokens"])
            tools[name]["calls"] += int(row["calls"])
        for row in m.contributor_rows:
            name = str(row["name"])
            contributors[name] = contributors.get(name, 0) + int(row["tokens"])
        for row in m.component_rows:
            ctype = str(row["component_type"])
            cgroup = str(row["component_group"])
            cname = str(row["component_name"])
            key = (ctype, cgroup, cname)
            components[key] = components.get(key, 0) + int(row["estimated_session_tokens"])
            if ctype == "skill":
                skills[cname] = skills.get(cname, 0) + int(row["tokens"])
            if ctype == "subagent":
                subagents[cname] = subagents.get(cname, 0) + int(row["tokens"])

    tool_rows = [
        {"name": k, "tokens": v["tokens"], "calls": v["calls"]}
        for k, v in tools.items()
    ]
    tool_rows.sort(key=lambda x: (x["tokens"], x["calls"]), reverse=True)

    contributor_rows = [{"name": k, "tokens": v} for k, v in contributors.items()]
    contributor_rows.sort(key=lambda x: x["tokens"], reverse=True)

    model_rows = [{"model": k, "api_cost": round(v, 6)} for k, v in models.items()]
    model_rows.sort(key=lambda x: x["api_cost"], reverse=True)

    component_rows = [
        {
            "component_type": t,
            "component_group": g,
            "component_name": n,
            "estimated_session_tokens": v,
        }
        for (t, g, n), v in components.items()
    ]
    component_rows.sort(key=lambda x: x["estimated_session_tokens"], reverse=True)

    period_series = [
        {
            "date": start.date().isoformat(),
            "sessions": total_sessions,
            "api_calls": total_api_calls,
            "tokens": total_tokens,
            "api_cost": total_api_cost,
        }
    ]

    return {
        "overview": {
            "period": period,
            "mode": mode,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "sessions": total_sessions,
            "api_calls": total_api_calls,
            "tokens": total_tokens,
            "api_cost": total_api_cost,
        },
        "tokens": token_totals,
        "tools": tool_rows,
        "contributors": contributor_rows,
        "skills": [{"name": k, "tokens": v} for k, v in sorted(skills.items(), key=lambda x: x[1], reverse=True)],
        "subagents": [{"name": k, "tokens": v} for k, v in sorted(subagents.items(), key=lambda x: x[1], reverse=True)],
        "context_estimates": {
            "observed_tools_only": True,
            "components": component_rows,
        },
        "warnings": [],
        "period_series": period_series,
        "projects": [],
        "models": model_rows,
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    overview = report.get("overview", {})
    lines = [
        "## Overview",
        f"- Period: {overview.get('period')}",
        f"- Mode: {overview.get('mode')}",
        f"- Sessions: {overview.get('sessions')}",
        f"- API calls: {overview.get('api_calls')}",
        f"- Tokens: {overview.get('tokens')}",
        f"- API cost: {overview.get('api_cost')}",
        "",
        "## Top Tools",
    ]
    for tool in report.get("tools", [])[:10]:
        lines.append(f"- {tool['name']}: {tool['tokens']} tokens, {tool['calls']} calls")
    lines.append("")
    lines.append("## Top Models")
    for model in report.get("models", [])[:10]:
        lines.append(f"- {model['model']}: ${model['api_cost']}")
    return "\n".join(lines) + "\n"
