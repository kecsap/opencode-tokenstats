from __future__ import annotations

from opencode_tokenstats.canonical_metrics import build_canonical_metrics, _build_component_family_rows
from opencode_tokenstats.content_attribution import collect_content_attribution


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
    # explore and general are core subagents, classified as "core" type
    names = {(r["component_type"], r["component_name"]) for r in out.component_rows}
    assert ("skill", "caveman") in names
    assert ("core", "explore") in names
    assert ("core", "general") in names

    # skill/subagent estimated session burden = raw tokens * api_calls
    for row in out.component_rows:
        if row["component_type"] in {"skill", "core"}:
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


def test_component_family_rows_aggregate_by_group() -> None:
    component_rows = [
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 100, "estimated_session_tokens": 100, "calls": 5},
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 200, "estimated_session_tokens": 200, "calls": 3},
        {"component_type": "tool", "component_group": "jcodemunch", "tokens": 50, "estimated_session_tokens": 50, "calls": 2},
        {"component_type": "skill", "component_group": "caveman", "tokens": 30, "estimated_session_tokens": 30, "calls": 0},
    ]
    family = _build_component_family_rows(component_rows)

    assert len(family) == 3
    assert family[0]["component_group"] == "lean-ctx"
    assert family[0]["tokens"] == 300
    assert family[0]["estimated_session_tokens"] == 300
    assert family[0]["calls"] == 8
    assert family[1]["component_group"] == "jcodemunch"
    assert family[1]["tokens"] == 50
    assert family[1]["calls"] == 2
    assert family[2]["component_group"] == "caveman"
    assert family[2]["component_type"] == "skill"


def test_component_family_rows_in_canonical_metrics() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {"type": "tool", "tool": "lean-ctx_ctx_read", "state": {"status": "completed", "output": "a"}},
                {"type": "tool", "tool": "lean-ctx_ctx_search", "state": {"status": "completed", "output": "b"}},
            ],
        }
    ]
    out = build_canonical_metrics("s-family", messages)

    lean_ctx_family = [r for r in out.component_family_rows if r["component_group"] == "lean-ctx"]
    assert len(lean_ctx_family) == 1
    assert lean_ctx_family[0]["calls"] == 2
    assert lean_ctx_family[0]["tokens"] > 0


def test_skill_call_attribution() -> None:
    """Test that skill tool calls are attributed to the correct skill component."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "caveman"},
                        "output": "skill loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "impeccable"},
                        "output": "skill loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-skill", messages)

    # Skill calls should be attributed with component_type "skill"
    skill_rows = [r for r in out.component_rows if r["component_type"] == "skill"]
    skill_names = {r["component_name"] for r in skill_rows}
    assert "caveman" in skill_names
    assert "impeccable" in skill_names

    # Tool calls should be attributed with component_type "tool"
    tool_rows = [r for r in out.component_rows if r["component_type"] == "tool"]
    assert any(r["component_name"] == "lean-ctx_ctx_read" for r in tool_rows)


def test_skill_call_grouping_in_family() -> None:
    """Test that skill calls are grouped with tools in the same family."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "svelte"},
                        "output": "svelte skill",
                    },
                },
                {
                    "type": "tool",
                    "tool": "svelte_list-sections",
                    "state": {"status": "completed", "output": "sections"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-svelte", messages)

    # Both skill and tool should be grouped under "svelte"
    svelte_family = [r for r in out.component_family_rows if r["component_group"] == "svelte"]
    assert len(svelte_family) == 1
    assert svelte_family[0]["calls"] == 2


def test_component_family_mixed_type_when_group_has_multiple_types() -> None:
    """Test that a group with multiple types shows 'mixed' instead of a single type."""
    component_rows = [
        {"component_type": "tool", "component_group": "svelte", "tokens": 100, "estimated_session_tokens": 100, "calls": 2},
        {"component_type": "skill", "component_group": "svelte", "tokens": 50, "estimated_session_tokens": 50, "calls": 1},
    ]
    family = _build_component_family_rows(component_rows)

    assert len(family) == 1
    assert family[0]["component_group"] == "svelte"
    assert family[0]["component_type"] == "mixed"
    assert family[0]["tokens"] == 150


def test_component_family_single_type_preserved() -> None:
    """Test that a group with a single type keeps that type."""
    component_rows = [
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 100, "estimated_session_tokens": 100, "calls": 3},
        {"component_type": "tool", "component_group": "lean-ctx", "tokens": 50, "estimated_session_tokens": 50, "calls": 2},
    ]
    family = _build_component_family_rows(component_rows)

    assert len(family) == 1
    assert family[0]["component_group"] == "lean-ctx"
    assert family[0]["component_type"] == "tool"
    assert family[0]["tokens"] == 150


def test_skill_family_preserves_single_hyphenated_skill_name() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "implement-slice"},
                        "output": "skill loaded",
                    },
                },
            ],
        }
    ]

    out = build_canonical_metrics("s-skill-hyphen-single", messages)

    family_names = {r["component_group"] for r in out.component_family_rows}
    assert "implement-slice" in family_names
    assert "implement" not in family_names


def test_skill_family_merges_multiple_hyphenated_skills_with_same_prefix() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "implement-slice"},
                        "output": "slice loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "implement-plan"},
                        "output": "plan loaded",
                    },
                },
            ],
        }
    ]

    out = build_canonical_metrics("s-skill-hyphen-multi", messages)

    implement_family = [r for r in out.component_family_rows if r["component_group"] == "implement"]
    assert len(implement_family) == 1
    assert implement_family[0]["calls"] == 2


def test_skill_calls_excluded_from_mcp_insights() -> None:
    """Test that skill calls are not included in MCP Insights."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "skill",
                    "state": {
                        "status": "completed",
                        "input": {"name": "caveman"},
                        "output": "skill loaded",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-mcp", messages)

    # MCP rows should only contain lean-ctx, not caveman skill
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "lean-ctx" in mcp_names
    assert "caveman" not in mcp_names


def test_subagent_call_attribution() -> None:
    """Test that task tool calls are attributed to the correct subagent component."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "task",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "explore", "prompt": "find files"},
                        "output": "subagent result",
                    },
                },
                {
                    "type": "tool",
                    "tool": "task",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "general", "prompt": "research"},
                        "output": "general result",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-subagent", messages)

    # explore and general are core subagents, attributed with component_type "core"
    core_rows = [r for r in out.component_rows if r["component_type"] == "core"]
    core_names = {r["component_name"] for r in core_rows}
    assert "explore" in core_names
    assert "general" in core_names

    # Tool calls should be attributed with component_type "tool"
    tool_rows = [r for r in out.component_rows if r["component_type"] == "tool"]
    assert any(r["component_name"] == "lean-ctx_ctx_read" for r in tool_rows)


def test_subagent_calls_excluded_from_mcp_insights() -> None:
    """Test that subagent calls are not included in MCP Insights."""
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": "sys",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {
                    "type": "tool",
                    "tool": "task",
                    "state": {
                        "status": "completed",
                        "input": {"subagent_type": "explore", "prompt": "find files"},
                        "output": "subagent result",
                    },
                },
                {
                    "type": "tool",
                    "tool": "lean-ctx_ctx_read",
                    "state": {"status": "completed", "output": "file content"},
                },
            ],
        }
    ]
    out = build_canonical_metrics("s-mcp-subagent", messages)

    # MCP rows should only contain lean-ctx, not explore subagent
    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "lean-ctx" in mcp_names
    assert "explore" not in mcp_names


def test_additional_core_components_classification() -> None:
    messages = [
        {
            "role": "assistant",
            "info": {
                "modelID": "gpt-5.3-codex",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "cost": 0.1,
                "system": """
<available_skills>
  <skill><name>plan</name><description>Planner skill.</description></skill>
  <skill><name>implement</name><description>Implementor skill.</description></skill>
</available_skills>
""",
            },
            "parts": [
                {"type": "text", "text": "ok"},
                {"type": "tool", "tool": "webfetch", "state": {"status": "completed", "output": "html"}},
                {"type": "tool", "tool": "invalid", "state": {"status": "completed", "output": "oops"}},
            ],
        }
    ]
    out = build_canonical_metrics("s-core-extra", messages)

    core_names = {r["component_name"] for r in out.core_rows}
    assert "webfetch" in core_names
    assert "general" in core_names
    assert "invalid" not in core_names
    assert "plan" in core_names
    assert "implement" in core_names

    family = [r for r in out.component_family_rows if r["component_group"] == "opencode-core"]
    assert len(family) == 1
    assert family[0]["component_type"] == "core"

    mcp_names = {r["name"] for r in out.mcp_rows}
    assert "webfetch" not in mcp_names
    assert "invalid" not in mcp_names
