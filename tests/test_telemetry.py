from __future__ import annotations

import pytest

from opencode_tokenstats.telemetry import (
    TelemetrySchemaError,
    collect_telemetry_calls,
    summarize_session_with_subagents,
    summarize_telemetry,
)


def test_collect_telemetry_calls_multi_step_assistant_turn() -> None:
    messages = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 100,
                        "output": 10,
                        "reasoning": 5,
                        "cache": {"read": 0},
                    },
                    "cost": 0.12,
                    "timestamp": 1000,
                },
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 50,
                        "output": 7,
                        "reasoning": 3,
                        "cache": {"read": 2},
                    },
                    "cost": 0.03,
                    "timestamp": 2000,
                },
            ],
            "info": {
                "tokens": {
                    "input": 9999,
                    "output": 9999,
                    "reasoning": 9999,
                    "cache": {"read": 9999},
                },
                "cost": 99,
            },
        }
    ]

    calls = collect_telemetry_calls(messages)
    summary = summarize_telemetry(calls)

    assert len(calls) == 2
    assert summary.api_calls == 2
    assert summary.input_tokens == 150
    assert summary.output_tokens == 17
    assert summary.reasoning_tokens == 8
    assert summary.cache_read_tokens == 2
    assert summary.total_cost == 0.15
    assert summary.most_recent_call is not None
    assert summary.most_recent_call.timestamp_ms == 2000


def test_collect_telemetry_calls_message_level_cache_fallback() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "tokens": {
                    "input": 20,
                    "output": 30,
                    "reasoning": 10,
                    "cache": {"read": 80},
                },
                "cost": 0.2,
                "time": {"created": 1000, "completed": 1100},
            },
            "parts": [{"type": "text", "text": "done"}],
        }
    ]

    summary = summarize_telemetry(collect_telemetry_calls(messages))
    assert summary.api_calls == 1
    assert summary.input_tokens == 20
    assert summary.output_tokens == 30
    assert summary.reasoning_tokens == 10
    assert summary.cache_read_tokens == 80
    assert summary.total_cost == 0.2


def test_collect_telemetry_calls_zero_cost_subscription_turn() -> None:
    messages = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 40,
                        "output": 12,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                    "cost": 0,
                    "timestamp": 1234,
                }
            ],
        }
    ]

    summary = summarize_telemetry(collect_telemetry_calls(messages))
    assert summary.api_calls == 1
    assert summary.total_cost == 0.0
    assert summary.total_tokens == 52


class FakeSessionReader:
    def __init__(self) -> None:
        self.messages = {
            "root": [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "step-finish",
                            "tokens": {
                                "input": 10,
                                "output": 5,
                                "reasoning": 1,
                                "cache": {"read": 2},
                            },
                            "cost": 0.01,
                            "timestamp": 200,
                        }
                    ],
                }
            ],
            "child-1": [
                {
                    "role": "assistant",
                    "info": {
                        "tokens": {
                            "input": 7,
                            "output": 8,
                            "reasoning": 2,
                            "cache": {"read": 1},
                        },
                        "cost": 0.02,
                    },
                }
            ],
            "grandchild-1": [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "step-finish",
                            "tokens": {
                                "input": 3,
                                "output": 4,
                                "reasoning": 1,
"cache": {"read": 0},
                            },
                            "cost": 0.005,
                            "timestamp": 300,
                        }
                    ],
                }
            ],
        }
        self.children = {
            "root": [{"id": "child-1"}],
            "child-1": [{"path": "grandchild-1"}],
            "grandchild-1": [],
        }

    def get_messages(self, path: str):
        return self.messages[path]

    def get_children(self, path: str):
        return self.children[path]


def test_summarize_session_with_nested_subagents() -> None:
    report = summarize_session_with_subagents(FakeSessionReader(), "root")

    assert report.own.api_calls == 1
    assert len(report.children) == 1
    assert report.children[0].session_path == "child-1"
    assert len(report.children[0].children) == 1
    assert report.children[0].children[0].session_path == "grandchild-1"

    total = report.total_with_subagents
    assert total.api_calls == 3
    assert total.input_tokens == 20
    assert total.output_tokens == 17
    assert total.reasoning_tokens == 4
    assert total.cache_read_tokens == 3
    assert round(total.total_cost, 3) == 0.035


class BadChildReader(FakeSessionReader):
    def __init__(self) -> None:
        super().__init__()
        self.children["root"] = [{"child": "missing-key"}]


def test_strict_schema_raises_on_invalid_child_payload() -> None:
    with pytest.raises(TelemetrySchemaError):
        summarize_session_with_subagents(
            BadChildReader(),
            "root",
            source="api",
            strict_schema=True,
        )


def test_non_strict_schema_ignores_invalid_child_payload() -> None:
    report = summarize_session_with_subagents(
        BadChildReader(),
        "root",
        source="api",
        strict_schema=False,
    )
    assert report.own.api_calls == 1
    assert report.total_with_subagents.api_calls == 1
