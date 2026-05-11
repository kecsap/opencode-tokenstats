from __future__ import annotations

import pytest

from opencode_tokenstats.activity_classifier import (
    CATEGORY_LABELS,
    classify_session,
    extract_root_dir,
)
from opencode_tokenstats.canonical_metrics import CanonicalMetrics


def _make_canonical(tool_rows: list[dict] | None = None) -> CanonicalMetrics:
    return CanonicalMetrics(
        session_id="test-session",
        model="local/qwen3.6-27b",
        input_tokens=100,
        output_tokens=200,
        reasoning_tokens=0,
        cache_read_tokens=0,
        session_total_tokens=300,
        api_calls=5,
        actual_cost_usd=0.0,
        estimated_cost_usd=0.1,
        token_composition={"input": 100, "output": 200},
        component_rows=[],
        component_family_rows=[],
        core_rows=[],
        tool_rows=tool_rows or [],
        mcp_rows=[],
    )


class TestClassifySession:
    def test_edit_tools_yield_coding(self) -> None:
        canonical = _make_canonical(
            [{"tool": "edit", "tokens": 50, "calls": 3, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "coding"

    def test_write_tool_yield_coding(self) -> None:
        canonical = _make_canonical(
            [{"tool": "write", "tokens": 50, "calls": 1, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "coding"

    def test_apply_patch_tool_yield_coding(self) -> None:
        canonical = _make_canonical(
            [{"tool": "apply_patch", "tokens": 50, "calls": 2, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "coding"

    def test_bash_only_yield_build_deploy(self) -> None:
        canonical = _make_canonical(
            [{"tool": "bash", "tokens": 50, "calls": 3, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "build/deploy"

    def test_bash_with_edit_yield_coding(self) -> None:
        canonical = _make_canonical(
            [
                {"tool": "bash", "tokens": 30, "calls": 2, "is_skill": False, "is_subagent": False, "is_core": True},
                {"tool": "edit", "tokens": 50, "calls": 1, "is_skill": False, "is_subagent": False, "is_core": True},
            ]
        )
        assert classify_session(canonical) == "coding"

    def test_read_and_grep_yield_exploration(self) -> None:
        canonical = _make_canonical(
            [
                {"tool": "read", "tokens": 30, "calls": 2, "is_skill": False, "is_subagent": False, "is_core": True},
                {"tool": "grep", "tokens": 20, "calls": 1, "is_skill": False, "is_subagent": False, "is_core": True},
            ]
        )
        assert classify_session(canonical) == "exploration"

    def test_glob_only_yield_exploration(self) -> None:
        canonical = _make_canonical(
            [{"tool": "glob", "tokens": 20, "calls": 1, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "exploration"

    def test_read_only_yield_exploration(self) -> None:
        canonical = _make_canonical(
            [{"tool": "read", "tokens": 30, "calls": 5, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "exploration"

    def test_task_only_yield_planning(self) -> None:
        canonical = _make_canonical(
            [{"tool": "task", "tokens": 50, "calls": 2, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "planning"

    def test_no_tools_yield_conversation(self) -> None:
        canonical = _make_canonical([])
        assert classify_session(canonical) == "conversation"

    def test_subagent_yield_delegation(self) -> None:
        canonical = _make_canonical(
            [{"tool": "task", "tokens": 50, "calls": 1, "is_skill": False, "is_subagent": True, "is_core": True}]
        )
        assert classify_session(canonical) == "delegation"

    def test_planning_skill_yield_planning(self) -> None:
        canonical = _make_canonical(
            [{"tool": "plan", "tokens": 50, "calls": 1, "is_skill": True, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "planning"

    def test_implement_skill_yield_planning(self) -> None:
        canonical = _make_canonical(
            [{"tool": "implement", "tokens": 50, "calls": 1, "is_skill": True, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "planning"

    def test_skill_tool_yield_general(self) -> None:
        canonical = _make_canonical(
            [{"tool": "skill", "tokens": 50, "calls": 1, "is_skill": False, "is_subagent": False, "is_core": True}]
        )
        assert classify_session(canonical) == "general"

    def test_mcp_tools_yield_general(self) -> None:
        canonical = _make_canonical(
            [
                {
                    "tool": "mcp__server__tool",
                    "tokens": 50,
                    "calls": 1,
                    "is_skill": False,
                    "is_subagent": False,
                    "is_core": False,
                }
            ]
        )
        assert classify_session(canonical) == "general"


class TestExtractRootDir:
    def test_path_basename(self) -> None:
        assert extract_root_dir("/home/fafa/eju") == "eju"

    def test_deep_path_basename(self) -> None:
        assert extract_root_dir("/home/user/projects/my-project") == "my-project"

    def test_plain_title_preserved(self) -> None:
        assert extract_root_dir("my-project") == "my-project"

    def test_empty_string_fallback(self) -> None:
        assert extract_root_dir("") == "-"

    def test_dash_preserved(self) -> None:
        assert extract_root_dir("-") == "-"

    def test_session_id_style(self) -> None:
        assert extract_root_dir("/tmp/session-123") == "session-123"


class TestCategoryLabels:
    def test_all_categories_have_labels(self) -> None:
        expected = {
            "coding",
            "debugging",
            "feature",
            "refactoring",
            "testing",
            "exploration",
            "planning",
            "delegation",
            "git",
            "build/deploy",
            "conversation",
            "brainstorming",
            "general",
        }
        assert set(CATEGORY_LABELS.keys()) == expected

    def test_label_format(self) -> None:
        assert CATEGORY_LABELS["coding"] == "Coding"
        assert CATEGORY_LABELS["build/deploy"] == "Build/Deploy"
        assert CATEGORY_LABELS["conversation"] == "Conversation"
