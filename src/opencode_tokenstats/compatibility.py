from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import json
import subprocess


CompatMode = Literal["strict_local", "strict_api", "tokenscope_compat"]


@dataclass(frozen=True, slots=True)
class ToolSchemaEstimate:
    name: str
    estimated_tokens: int
    argument_count: int
    has_complex_args: bool


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    mode: CompatMode
    observed_tools_only: bool
    tool_schema_estimates: list[ToolSchemaEstimate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def analyze_context_compatibility(
    messages: list[dict[str, Any]],
    *,
    mode: CompatMode,
    source: Literal["local", "api"],
) -> CompatibilityResult:
    observed_tools = _extract_observed_tools(messages)
    warnings: list[str] = []

    if mode in {"strict_local", "strict_api"}:
        if source == "api":
            warnings.append("observed tools only: API payload does not guarantee full enabled-tool inventory")
        return CompatibilityResult(
            mode=mode,
            observed_tools_only=True,
            tool_schema_estimates=[],
            warnings=warnings,
        )

    estimates = _estimate_tool_schemas(messages, observed_tools)
    warnings.append("tokenscope_compat mode: tool schema/context values are heuristic estimates")
    if source == "api":
        warnings.append("observed tools only: estimates are based on tools seen in message parts")

    return CompatibilityResult(
        mode=mode,
        observed_tools_only=True,
        tool_schema_estimates=estimates,
        warnings=warnings,
    )


def load_export_debug_session(session_id: str) -> dict[str, Any] | None:
    """Optional parity-debug helper, never required for normal operation."""
    try:
        completed = subprocess.run(
            ["opencode", "export", session_id],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    output = (completed.stdout or "").strip()
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def _extract_observed_tools(messages: list[dict[str, Any]]) -> list[str]:
    tools: set[str] = set()
    for msg in messages:
        for part in _parts(msg):
            if part.get("type") != "tool":
                continue
            name = part.get("tool")
            if isinstance(name, str) and name:
                tools.add(name)
    return sorted(tools)


def _estimate_tool_schemas(messages: list[dict[str, Any]], tool_names: list[str]) -> list[ToolSchemaEstimate]:
    out: list[ToolSchemaEstimate] = []
    for tool_name in tool_names:
        call_inputs: list[dict[str, Any]] = []
        for msg in messages:
            for part in _parts(msg):
                if part.get("type") != "tool" or part.get("tool") != tool_name:
                    continue
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                inp = state.get("input")
                if isinstance(inp, dict):
                    call_inputs.append(inp)

        arg_names: set[str] = set()
        complex_arg_names: set[str] = set()
        for inp in call_inputs:
            for key, value in inp.items():
                arg_names.add(key)
                if isinstance(value, (dict, list)):
                    complex_arg_names.add(key)

        arg_count = len(arg_names) if arg_names else 3
        complex_count = len(complex_arg_names) if arg_names else 1
        simple_count = max(0, arg_count - complex_count)
        has_complex = complex_count > 0

        estimated = 200 + (simple_count * 30) + (complex_count * 60) + (120 if has_complex else 80)
        out.append(
            ToolSchemaEstimate(
                name=tool_name,
                estimated_tokens=estimated,
                argument_count=arg_count,
                has_complex_args=has_complex,
            )
        )

    out.sort(key=lambda x: x.estimated_tokens, reverse=True)
    return out


def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts")
    if isinstance(parts, list):
        return [p for p in parts if isinstance(p, dict)]
    return []
