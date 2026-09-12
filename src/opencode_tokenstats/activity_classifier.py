from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
_WEB_TOOLS = frozenset({"webfetch", "websearch", "toolsearch"})
_LOCAL_TOOLS = (
    _EDIT_TOOLS
    | _BASH_TOOLS
    | _SEARCH_TOOLS
    | _READ_TOOLS
    | _TASK_TOOLS
    | _SKILL_TOOL
    | _WEB_TOOLS
    | frozenset({"todowrite", "question", "compress", "tokenscope", "quota_status", "invalid"})
)

_TEST_PATTERNS = re.compile(r"\b(test|pytest|vitest|jest|mocha|spec|coverage|npm\s+test|npx\s+vitest|npx\s+jest)\b", re.I)
_GIT_PATTERNS = re.compile(r"\bgit\s+(push|pull|commit|merge|rebase|checkout|branch|stash|log|diff|status|add|reset|cherry-pick|tag)\b", re.I)
_BUILD_PATTERNS = re.compile(r"\b(npm\s+run\s+build|npm\s+publish|pip\s+install|docker|deploy|make\s+build|npm\s+run\s+dev|npm\s+start|pm2|systemctl|brew|cargo\s+build)\b", re.I)
_INSTALL_PATTERNS = re.compile(r"\b(npm\s+install|pip\s+install|brew\s+install|apt\s+install|cargo\s+add)\b", re.I)
_DEBUG_KEYWORDS = re.compile(r"\b(fix|bug|error|broken|failing|crash|issue|debug|traceback|exception|stack\s*trace|not\s+working|wrong|unexpected|status\s+code|404|500|401|403)\b", re.I)
_FEATURE_KEYWORDS = re.compile(r"\b(add|create|implement|new|build|feature|introduce|set\s*up|scaffold|generate|make\s+(?:a|me|the)|write\s+(?:a|me|the))\b", re.I)
_REFACTOR_KEYWORDS = re.compile(r"\b(refactor|clean\s*up|rename|reorganize|simplify|extract|restructure|move|migrate|split)\b", re.I)
_BRAINSTORM_KEYWORDS = re.compile(r"\b(brainstorm|idea|what\s+if|explore|think\s+about|approach|strategy|design|consider|how\s+should|what\s+would|opinion|suggest|recommend)\b", re.I)
_RESEARCH_KEYWORDS = re.compile(r"\b(research|investigate|look\s+into|find\s+out|check|search|analyze|review|understand|explain|how\s+does|what\s+is|show\s+me|list|compare)\b", re.I)
_FILE_PATTERNS = re.compile(r"\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|sql|sh|go|rs|java|rb|php|css|html|md|csv|xml)\b", re.I)
_SCRIPT_PATTERNS = re.compile(r"\b(run\s+\S+\.\w+|execute|scrip?t|curl|api\s+\S+|endpoint|request\s+url|fetch\s+\S+|query|database|db\s+\S+)\b", re.I)
_URL_PATTERN = re.compile(r"https?://\S+", re.I)

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


def classify_session(canonical: "CanonicalMetrics") -> str:
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


def classify_turn(
    user_message: str,
    tools: set[str],
    *,
    has_subagent: bool = False,
    skills: set[str] | None = None,
) -> str:
    """Classify one prompt and its following assistant activity.

    Git is intentionally recognized only for bash-only OpenCode activity. External
    commits are neither observed nor relevant to this classifier.
    """
    skills = skills or set()
    if has_subagent:
        return "delegation"
    if skills & _PLANNING_SKILLS:
        return "planning"

    has_edits = bool(tools & _EDIT_TOOLS)
    has_bash = bool(tools & _BASH_TOOLS)
    has_reads = bool(tools & _READ_TOOLS)
    has_mcp = any(tool not in _LOCAL_TOOLS for tool in tools)
    has_search = bool(tools & (_SEARCH_TOOLS | _WEB_TOOLS)) or has_mcp
    has_task = bool(tools & _TASK_TOOLS)
    has_skill = bool(tools & _SKILL_TOOL) or bool(skills)

    if has_bash and not has_edits:
        if _TEST_PATTERNS.search(user_message):
            return "testing"
        if _GIT_PATTERNS.search(user_message):
            return "git"
        if _BUILD_PATTERNS.search(user_message) or _INSTALL_PATTERNS.search(user_message):
            return "build/deploy"

    if has_edits:
        return _refine_coding(user_message)
    if has_bash and has_reads:
        return _refine_exploration(user_message)
    if has_bash:
        return "coding"
    if has_search or has_reads:
        return _refine_exploration(user_message)
    if has_task:
        return "planning"
    if has_skill:
        return "general"
    return _classify_conversation(user_message)


def extract_user_text(message: dict[str, Any]) -> str:
    """Return the visible text from a user message without retaining it."""
    if message.get("role") != "user":
        return ""
    parts = message.get("parts")
    if not isinstance(parts, list):
        return ""
    return "\n".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
    )


def extract_assistant_activity(message: dict[str, Any]) -> tuple[set[str], set[str], bool]:
    """Extract classification-only tool signals from one assistant message."""
    tools: set[str] = set()
    skills: set[str] = set()
    has_subagent = False
    if message.get("role") != "assistant":
        return tools, skills, has_subagent
    parts = message.get("parts")
    if not isinstance(parts, list):
        return tools, skills, has_subagent
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "tool":
            continue
        tool = part.get("tool")
        if not isinstance(tool, str) or not tool:
            continue
        tool = tool.lower()
        tools.add(tool)
        state = part.get("state")
        inputs = state.get("input") if isinstance(state, dict) and isinstance(state.get("input"), dict) else {}
        if tool == "skill":
            name = inputs.get("name") or inputs.get("skill") or inputs.get("skill_name")
            if isinstance(name, str) and name.strip():
                skills.add(name.split("<", 1)[0].strip().lower())
        elif tool == "task":
            agent = inputs.get("subagent_type") or inputs.get("type")
            has_subagent = isinstance(agent, str) and bool(agent.strip())
    return tools, skills, has_subagent


def _refine_coding(user_message: str) -> str:
    return _first_matching_category(
        user_message,
        ((_REFACTOR_KEYWORDS, "refactoring"), (_FEATURE_KEYWORDS, "feature"), (_DEBUG_KEYWORDS, "debugging")),
    ) or "coding"


def _refine_exploration(user_message: str) -> str:
    if _RESEARCH_KEYWORDS.search(user_message):
        return "exploration"
    if _DEBUG_KEYWORDS.search(user_message):
        return "debugging"
    return "exploration"


def _classify_conversation(user_message: str) -> str:
    if _BRAINSTORM_KEYWORDS.search(user_message):
        return "brainstorming"
    if _RESEARCH_KEYWORDS.search(user_message):
        return "exploration"
    category = _first_matching_category(user_message, ((_FEATURE_KEYWORDS, "feature"), (_DEBUG_KEYWORDS, "debugging")))
    if category:
        return category
    if _FILE_PATTERNS.search(user_message) or _SCRIPT_PATTERNS.search(user_message):
        return "coding"
    if _URL_PATTERN.search(user_message):
        return "exploration"
    return "conversation"


def _first_matching_category(
    text: str, candidates: tuple[tuple[re.Pattern[str], str], ...]
) -> str | None:
    matches = ((match.start(), order, category) for order, (pattern, category) in enumerate(candidates) if (match := pattern.search(text)))
    best = min(matches, default=None)
    return best[2] if best else None
