from __future__ import annotations

from opencode_tokenstats.canonical_metrics import build_canonical_metrics


def test_build_canonical_metrics_basic_semantics() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {
                    "input": 100,
                    "output": 20,
                    "reasoning": 5,
                    "cache": {"read": 10, "write": 0},
                },
                "cost": 0.5,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_search",
                    "state": {"status": "completed", "output": "abc"},
                },
            ],
        }
    ]

    out = build_canonical_metrics("s1", messages)
    assert out.session_id == "s1"
    assert out.model == "gpt-5.3-codex"
    assert out.api_calls == 1
    assert out.input_tokens == 100
    assert len(out.tool_rows) == 1
    assert out.tool_rows[0]["tool"] == "lean-ctx_ctx_search"
    assert out.component_rows[0]["component_group"] == "lean-ctx"
    assert out.mcp_rows[0]["name"] == "lean-ctx"


def test_canonical_metrics_extracts_skill_and_subagent_components() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": """
<available_skills>
  <skill>
    <name>caveman</name>
    <description>Ultra-compressed communication mode.</description>
  </skill>
</available_skills>
- explore: Fast agent specialized for exploring codebases.
- general: General-purpose agent for researching complex questions.
""",
            },
            "parts": [{"type": "text", "text": "ok"}],
        }
    ]

    out = build_canonical_metrics("s2", messages)
    names = {(r["component_type"], r["component_name"]) for r in out.component_rows}
    assert ("skill", "caveman") in names
    assert ("subagent", "explore") in names
    assert ("subagent", "general") in names

    # skill/subagent estimated session burden = raw tokens * api_calls
    for row in out.component_rows:
        if row["component_type"] in {"skill", "subagent"}:
            assert row["estimated_session_tokens"] == row["tokens"] * out.api_calls


def test_model_includes_provider_prefix() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "providerID": "azure",
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
            },
            "parts": [{"type": "text", "text": "ok"}],
        }
    ]

    out = build_canonical_metrics("s3", messages)
    assert out.model == "azure/gpt-5.3-codex"


def test_local_model_has_zero_api_cost(tmp_path) -> None:
    from opencode_tokenstats.canonical_metrics import _is_local_model
    import os

    conf = tmp_path / "models.conf"
    conf.write_text("@local myollama/* myllamacpp/* *qwen36*\n")
    old_env = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE")
    os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = str(conf)
    try:
        # Local patterns
        assert _is_local_model("myollama/qwen3.6:35b-yarn")
        assert _is_local_model("myllamacpp/qwen3.6-27b1")
        assert _is_local_model("llamacpp_qwen36_gpu/qwen3.6-27b1")

        # Non-local patterns
        assert not _is_local_model("azure/gpt-5.4")
        assert not _is_local_model("openai/gpt-5.3-codex")
        assert not _is_local_model("anthropic/claude-sonnet-4")
    finally:
        if old_env is None:
            os.environ.pop("OPTOKEN_MODEL_ALIAS_FILE", None)
        else:
            os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = old_env


def test_local_model_cost_is_zero(tmp_path) -> None:
    import os

    conf = tmp_path / "models.conf"
    conf.write_text("@local myollama/*\n")
    old_env = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE")
    os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = str(conf)
    try:
        messages = [
            {
                "role": "assistant",
                "info": {
                    "providerID": "myollama",
                    "modelID": "qwen3.6:35b-yarn",
                    "tokens": {"input": 100, "output": 50, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                    "cost": 18.78,  # Cost in telemetry, but should be ignored for local models
                },
                "parts": [{"type": "text", "text": "ok"}],
            }
        ]

        out = build_canonical_metrics("s-local", messages)
        assert out.model == "myollama/qwen3.6:35b-yarn"
        assert out.actual_cost_usd == 0.0  # API cost should be 0 for local models
        assert out.estimated_cost_usd > 0  # Estimated cost should be calculated
    finally:
        if old_env is None:
            os.environ.pop("OPTOKEN_MODEL_ALIAS_FILE", None)
        else:
            os.environ["OPTOKEN_MODEL_ALIAS_FILE"] = old_env


def test_estimated_cost_uses_per_call_models(tmp_path) -> None:
    import json
    import os

    pricing_path = tmp_path / "models.json"
    pricing_path.write_text(
        json.dumps(
            {
                "openai/gpt-a": {
                    "input": 2.0,
                    "output": 8.0,
                    "cacheRead": 0.5,
                    "cacheWrite": 2.5,
                    "webSearch": 0.01,
                },
                "openai/gpt-b": {
                    "input": 1.0,
                    "output": 4.0,
                    "cacheRead": 0.0,
                    "cacheWrite": 1.0,
                    "webSearch": 0.02,
                },
                "default": {"input": 1.0, "output": 3.0, "cacheRead": 0.0},
            }
        )
    )

    old_pricing_env = os.environ.get("OPENCODE_MODEL_PRICING_FILE")
    os.environ["OPENCODE_MODEL_PRICING_FILE"] = str(pricing_path)
    try:
        messages = [
            {
                "role": "assistant",
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-a",
                    "tokens": {
                        "input": 1_000_000,
                        "output": 500_000,
                        "reasoning": 500_000,
                        "cache": {"read": 1_000_000, "write": 1_000_000},
                        "server_tool_use": {"web_search_requests": 1},
                    },
                    "cost": 0.0,
                },
                "parts": [{"type": "text", "text": "a"}],
            },
            {
                "role": "assistant",
                "info": {
                    "providerID": "openai",
                    "modelID": "gpt-b",
                    "tokens": {
                        "input": 1_000_000,
                        "output": 500_000,
                        "reasoning": 500_000,
                        "cache": {"read": 0, "write": 1_000_000},
                        "server_tool_use": {"web_search_requests": 2},
                    },
                    "cost": 0.0,
                },
                "parts": [{"type": "text", "text": "b"}],
            },
        ]
        out = build_canonical_metrics("s-mixed", messages)
        assert out.actual_cost_usd == 0.0
        assert out.estimated_cost_usd == 19.05
    finally:
        if old_pricing_env is None:
            os.environ.pop("OPENCODE_MODEL_PRICING_FILE", None)
        else:
            os.environ["OPENCODE_MODEL_PRICING_FILE"] = old_pricing_env
