from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .tokenization import ResolvedModel, TokenCountResult, TokenizerRegistry


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class CategoryTotals:
    system_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0
    reasoning_tokens: int = 0
    tool_output_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.system_tokens
            + self.user_tokens
            + self.assistant_tokens
            + self.reasoning_tokens
            + self.tool_output_tokens
        )


@dataclass(frozen=True, slots=True)
class ToolUsageStat:
    tool_name: str
    output_tokens: int
    call_count: int
    is_skill: bool = False


@dataclass(frozen=True, slots=True)
class ContentAttribution:
    totals: CategoryTotals
    tool_usage: list[ToolUsageStat] = field(default_factory=list)
    observed_tools_only: bool = True
    tool_schema_context_estimate_tokens: int = 0
    approximate_tokenizer_used: bool = False
    warnings: list[str] = field(default_factory=list)


class ApproxTokenCounter:
    """Simple fallback counter until tokenizer registry is added."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


def collect_content_attribution(
    messages: list[dict[str, Any]], token_counter: TokenCounter | None = None
) -> ContentAttribution:
    counter = token_counter or ApproxTokenCounter()

    system_tokens = 0
    user_tokens = 0
    assistant_tokens = 0
    reasoning_tokens = 0

    tool_calls: dict[str, int] = {}
    tool_output_tokens: dict[str, int] = {}
    skill_tools: set[str] = set()

    for message in messages:
        system_tokens += _count_info_system(message, counter)

        role = message.get("role")
        parts = _parts_from_message(message)

        if role == "user":
            user_tokens += _count_text_parts(parts, counter)
        elif role == "assistant":
            assistant_tokens += _count_text_parts(parts, counter)

        reasoning_tokens += _count_reasoning_parts(parts, counter)
        _collect_tool_parts(parts, counter, tool_calls, tool_output_tokens, skill_tools)

    tool_stats = []
    for tool_name, call_count in tool_calls.items():
        is_skill = tool_name in skill_tools
        tool_stats.append(
            ToolUsageStat(
                tool_name=tool_name,
                output_tokens=tool_output_tokens.get(tool_name, 0),
                call_count=call_count,
                is_skill=is_skill,
            )
        )
    tool_stats.sort(key=lambda x: (x.output_tokens, x.call_count), reverse=True)

    total_tool_output_tokens = sum(tool_output_tokens.values())

    return ContentAttribution(
        totals=CategoryTotals(
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens,
            reasoning_tokens=reasoning_tokens,
            tool_output_tokens=total_tool_output_tokens,
        ),
        tool_usage=tool_stats,
        observed_tools_only=True,
        tool_schema_context_estimate_tokens=0,
    )


def collect_content_attribution_for_model(
    messages: list[dict[str, Any]],
    *,
    provider_id: str | None,
    model_id: str | None,
    tokenizer_registry: TokenizerRegistry | None = None,
) -> ContentAttribution:
    registry = tokenizer_registry or TokenizerRegistry()
    resolved_model = registry.resolve_model(provider_id, model_id)
    counter = _RegistryCounter(registry, resolved_model)
    result = collect_content_attribution(messages, token_counter=counter)
    return ContentAttribution(
        totals=result.totals,
        tool_usage=result.tool_usage,
        observed_tools_only=result.observed_tools_only,
        tool_schema_context_estimate_tokens=result.tool_schema_context_estimate_tokens,
        approximate_tokenizer_used=counter.approximate_used,
        warnings=counter.warnings,
    )


class _RegistryCounter:
    def __init__(self, registry: TokenizerRegistry, model: ResolvedModel) -> None:
        self._registry = registry
        self._model = model
        self.approximate_used = False
        self.warnings: list[str] = []

    def count(self, text: str) -> int:
        result: TokenCountResult = self._registry.count(text, self._model.tokenizer)
        if result.approximate:
            self.approximate_used = True
        if result.warning and result.warning not in self.warnings:
            self.warnings.append(result.warning)
        return result.tokens


def _parts_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts")
    if isinstance(parts, list):
        return [p for p in parts if isinstance(p, dict)]
    return []


def _count_info_system(message: dict[str, Any], counter: TokenCounter) -> int:
    info = message.get("info")
    if not isinstance(info, dict):
        return 0
    system = info.get("system")
    if isinstance(system, str):
        return counter.count(system)
    return 0


def _count_text_parts(parts: list[dict[str, Any]], counter: TokenCounter) -> int:
    total = 0
    for part in parts:
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str):
            total += counter.count(text)
    return total


def _count_reasoning_parts(parts: list[dict[str, Any]], counter: TokenCounter) -> int:
    total = 0
    for part in parts:
        if part.get("type") != "reasoning":
            continue
        text = part.get("text")
        if isinstance(text, str):
            total += counter.count(text)
    return total


def _collect_tool_parts(
    parts: list[dict[str, Any]],
    counter: TokenCounter,
    tool_calls: dict[str, int],
    tool_output_tokens: dict[str, int],
    skill_tools: set[str],
) -> None:
    for part in parts:
        if part.get("type") != "tool":
            continue

        tool_name = part.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            continue

        # Resolve skill tool calls to specific skill names
        if tool_name == "skill":
            tool_name = _resolve_skill_name(part)
            skill_tools.add(tool_name)

        tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1

        state = part.get("state")
        if not isinstance(state, dict):
            continue
        if state.get("status") != "completed":
            continue

        output = state.get("output")
        text = _tool_output_to_text(output)
        if not text:
            continue
        tool_output_tokens[tool_name] = tool_output_tokens.get(tool_name, 0) + counter.count(text)


def _resolve_skill_name(part: dict[str, Any]) -> str:
    state = part.get("state")
    if isinstance(state, dict):
        input_data = state.get("input")
        if isinstance(input_data, dict):
            skill_name = input_data.get("name")
            if isinstance(skill_name, str) and skill_name:
                return skill_name
            skill_name = input_data.get("skill")
            if isinstance(skill_name, str) and skill_name:
                return skill_name
            skill_name = input_data.get("skill_name")
            if isinstance(skill_name, str) and skill_name:
                return skill_name
    return "skill"


def _tool_output_to_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (int, float, bool)):
        return str(output)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            item_text = _tool_output_to_text(item)
            if item_text:
                parts.append(item_text)
        return "\n".join(parts)
    if isinstance(output, dict):
        parts = []
        for key, value in output.items():
            value_text = _tool_output_to_text(value)
            if value_text:
                parts.append(f"{key}: {value_text}")
        return "\n".join(parts)
    return ""
