from __future__ import annotations

from pathlib import Path

from .canonical_metrics import CanonicalMetrics


def extract_root_dir(raw_title: str) -> str:
    """Extract display root dir (last path segment) from session title.

    Deterministic, no filesystem probing.
    - Path-like: `/home/fafa/eju` -> `eju`
    - Plain title: `my-project` -> `my-project`
    - Empty/invalid: -> `-`
    """
    if not raw_title:
        return "-"
    name = Path(raw_title).name
    return name if name else "-"


# Tool sets for classification (mirrors codeburn classifier logic)
_EDIT_TOOLS = frozenset({"edit", "write", "apply_patch", "apply"})
_BASH_TOOLS = frozenset({"bash"})
_SEARCH_TOOLS = frozenset({"grep", "glob"})
_READ_TOOLS = frozenset({"read"})
_TASK_TOOLS = frozenset({"task"})
_SKILL_TOOL = frozenset({"skill"})
_PLANNING_SKILLS = frozenset({"plan", "implement"})

# Category labels (aligned with codeburn CATEGORY_LABELS)
CATEGORY_LABELS: dict[str, str] = {
    "coding": "Coding",
    "debugging": "Debugging",
    "feature": "Feature Dev",
    "refactoring": "Refactoring",
    "testing": "Testing",
    "exploration": "Exploration",
    "planning": "Planning",
    "delegation": "Delegation",
    "git": "Git Ops",
    "build/deploy": "Build/Deploy",
    "conversation": "Conversation",
    "brainstorming": "Brainstorming",
    "general": "General",
}


def classify_session(canonical: CanonicalMetrics) -> str:
    """Classify a session into a single activity category based on tools used.

    Mirrors codeburn's classifyTurn pipeline at session level:
    tool-pattern first → keyword refine → conversation fallback.

    Since we operate at session level (not turn level), keyword refinement
    is replaced with component-based detection (skills, subagents).
    """
    tools = {row["tool"] for row in canonical.tool_rows}

    if not tools:
        return "conversation"

    # Check for subagent delegation (task tool with subagent calls)
    has_subagent = any(row.get("is_subagent") for row in canonical.tool_rows)
    if has_subagent:
        return "delegation"

    # Check for planning skills (plan/implement)
    has_planning_skill = any(
        row.get("is_skill") and row["tool"] in _PLANNING_SKILLS
        for row in canonical.tool_rows
    )
    if has_planning_skill:
        return "planning"

    has_edits = bool(tools & _EDIT_TOOLS)
    has_bash = bool(tools & _BASH_TOOLS)
    has_search = bool(tools & _SEARCH_TOOLS)
    has_read = bool(tools & _READ_TOOLS)
    has_task = bool(tools & _TASK_TOOLS)
    has_skill = bool(tools & _SKILL_TOOL)

    # Edit tools present → coding (same as codeburn hasEdits branch)
    if has_edits:
        return "coding"

    # Bash without edits → build/deploy
    if has_bash and not has_edits:
        return "build/deploy"

    # Search / read-only → exploration
    if has_search or (has_read and not has_edits):
        return "exploration"

    # Task tools without edits → planning
    if has_task and not has_edits:
        return "planning"

    # Skill tool without specific planning skill → general (codeburn behavior)
    if has_skill:
        return "general"

    return "general"
