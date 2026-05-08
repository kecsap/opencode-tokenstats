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
