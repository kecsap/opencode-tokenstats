from __future__ import annotations

from opencode_tokenstats.compatibility import analyze_context_compatibility


def _messages() -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool",
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"file": "a.txt", "limit": 10, "filters": ["x"]},
                    },
                },
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {"status": "completed", "input": {"command": "ls"}},
                },
            ],
        }
    ]


def test_strict_api_returns_observed_tools_only_with_warning() -> None:
    result = analyze_context_compatibility(_messages(), mode="strict_api", source="api")
    assert result.mode == "strict_api"
    assert result.observed_tools_only is True
    assert result.tool_schema_estimates == []
    assert any("observed tools only" in w for w in result.warnings)


def test_strict_local_returns_no_estimates_no_api_warning() -> None:
    result = analyze_context_compatibility(_messages(), mode="strict_local", source="local")
    assert result.mode == "strict_local"
    assert result.tool_schema_estimates == []
    assert not any("observed tools only" in w for w in result.warnings)


def test_tokenscope_compat_estimates_tools_from_observed_calls() -> None:
    result = analyze_context_compatibility(_messages(), mode="tokenscope_compat", source="api")
    assert result.mode == "tokenscope_compat"
    assert result.observed_tools_only is True
    assert len(result.tool_schema_estimates) == 2

    read_est = next(x for x in result.tool_schema_estimates if x.name == "read")
    bash_est = next(x for x in result.tool_schema_estimates if x.name == "bash")

    assert read_est.argument_count == 3
    assert read_est.has_complex_args is True
    assert read_est.estimated_tokens > bash_est.estimated_tokens
    assert any("heuristic" in w for w in result.warnings)
