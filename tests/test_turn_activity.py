from __future__ import annotations

from opencode_tokenstats.canonical_metrics import build_canonical_metrics


def _assistant(*, tool: str, tokens: dict[str, int], cost: float) -> dict:
    return {
        "role": "assistant",
        "info": {"providerID": "openai", "modelID": "gpt-5.3-codex"},
        "parts": [
            {"type": "tool", "tool": tool, "state": {"input": {}, "status": "completed"}},
            {"type": "step-finish", "tokens": tokens, "cost": cost},
        ],
    }


def test_build_canonical_metrics_attributes_each_turn_independently() -> None:
    messages = [
        {"role": "user", "parts": [{"type": "text", "text": "inspect the authentication flow"}]},
        _assistant(tool="read", tokens={"input": 10, "output": 2}, cost=0.01),
        {"role": "user", "parts": [{"type": "text", "text": "add a login setting"}]},
        _assistant(tool="edit", tokens={"input": 20, "output": 3, "reasoning": 1}, cost=0.02),
    ]

    metric = build_canonical_metrics("ses_test", messages)

    assert [row["category"] for row in metric.activity_rows] == ["exploration", "feature"]
    assert [row["tokens"] for row in metric.activity_rows] == [12, 24]
    assert sum(row["tokens"] for row in metric.activity_rows) == metric.session_total_tokens
    assert sum(row["calls"] for row in metric.activity_rows) == metric.api_calls


def test_activity_rows_keep_edit_turn_out_of_git_ops() -> None:
    messages = [
        {"role": "user", "parts": [{"type": "text", "text": "add this change and commit it later"}]},
        _assistant(tool="edit", tokens={"input": 10, "output": 2}, cost=0.01),
    ]

    metric = build_canonical_metrics("ses_test", messages)

    assert metric.activity_rows[0]["category"] == "feature"
