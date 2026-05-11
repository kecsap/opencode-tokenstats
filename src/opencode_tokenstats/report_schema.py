from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .activity_classifier import classify_session, CATEGORY_LABELS
from .canonical_metrics import CanonicalMetrics
from .pricing import load_model_aliases


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
    model_alias_file: str | None = None,
) -> dict[str, Any]:
    total_sessions = len(session_metrics)
    total_api_calls = sum(m.api_calls for m in session_metrics)
    total_tokens = sum(m.session_total_tokens for m in session_metrics)
    total_api_cost = round(
        sum(
            m.actual_cost_usd if m.actual_cost_usd > 0 else m.estimated_cost_usd
            for m in session_metrics
        ),
        6,
    )

    token_totals = {
        "input": sum(m.input_tokens for m in session_metrics),
        "output": sum(m.output_tokens for m in session_metrics),
        "reasoning": sum(m.reasoning_tokens for m in session_metrics),
        "cache_read": sum(m.cache_read_tokens for m in session_metrics),
    }

    tools: dict[str, dict[str, int]] = {}
    models: dict[str, dict[str, float]] = {}
    components: dict[tuple[str, str, str], int] = {}
    skills: dict[str, int] = {}
    subagents: dict[str, int] = {}

    aliases = load_model_aliases(model_alias_file)
    for m in session_metrics:
        model_key = aliases.get(m.model, m.model)
        if model_key not in models:
            models[model_key] = {"api_cost": 0.0, "estimated_cost": 0.0}
        models[model_key]["api_cost"] = round(models[model_key]["api_cost"] + m.actual_cost_usd, 6)
        models[model_key]["estimated_cost"] = round(models[model_key]["estimated_cost"] + m.estimated_cost_usd, 6)
        for row in m.tool_rows:
            name = str(row["tool"])
            if name not in tools:
                tools[name] = {"tokens": 0, "calls": 0}
            tools[name]["tokens"] += int(row["tokens"])
            tools[name]["calls"] += int(row["calls"])
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

    model_rows = []
    for k, costs in models.items():
        api_cost = costs["api_cost"]
        estimated_cost = costs["estimated_cost"]
        primary_cost = api_cost if api_cost > 0 else estimated_cost
        model_rows.append(
            {
                "model": k,
                "api_cost": round(api_cost, 6),
                "estimated_cost": round(estimated_cost, 6),
                "cost": round(primary_cost, 6),
            }
        )
    model_rows.sort(key=lambda x: x["cost"], reverse=True)

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

    # Build by_activity rows (session-level classification)
    activity_map: dict[str, dict[str, object]] = {}
    session_rows: list[dict[str, object]] = []
    for m in session_metrics:
        category = classify_session(m)
        if category not in activity_map:
            activity_map[category] = {"tokens": 0, "calls": 0, "api_cost": 0.0, "estimated_cost": 0.0}
        activity_map[category]["tokens"] += m.session_total_tokens
        activity_map[category]["calls"] += m.api_calls
        activity_map[category]["api_cost"] += m.actual_cost_usd
        activity_map[category]["estimated_cost"] += m.estimated_cost_usd
        session_rows.append(
            {
                "root_dir": m.session_id or "-",
                "tokens": m.session_total_tokens,
                "api_cost": round(m.actual_cost_usd, 6),
                "estimated_cost": round(m.estimated_cost_usd, 6),
            }
        )

    by_activity = [
        {
            "category": cat,
            "label": CATEGORY_LABELS.get(cat, cat.title()),
            "tokens": data["tokens"],
            "calls": data["calls"],
            "api_cost": round(data["api_cost"], 6),
            "estimated_cost": round(data["estimated_cost"], 6),
        }
        for cat, data in activity_map.items()
    ]
    by_activity.sort(key=lambda x: x["api_cost"] if x["api_cost"] > 0 else x["estimated_cost"], reverse=True)

    session_rows.sort(key=lambda x: x["api_cost"] if x["api_cost"] > 0 else x["estimated_cost"], reverse=True)
    top_sessions = session_rows[:10]

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
        "by_activity": by_activity,
        "top_sessions": top_sessions,
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
        lines.append(f"- {model['model']}: API=${model['api_cost']}, Est=${model['estimated_cost']}")
    return "\n".join(lines) + "\n"
