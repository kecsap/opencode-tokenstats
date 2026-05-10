from __future__ import annotations

from opencode_tokenstats.cli import _finalize_component_stats_canonical


def test_component_stats_include_opencode_core_row() -> None:
    component_map = {
        "tool|lean-ctx|lean-ctx_ctx_read": 100.0,
        "tool|jcodemunch|jcodemunch_search_text": 50.0,
    }
    out = _finalize_component_stats_canonical(component_map, core_tokens=40.0)
    rows = out["rows"]
    core = [r for r in rows if r["component_type"] == "core" and r["component_group"] == "opencode-core"]
    assert len(core) == 1
    assert core[0]["tokens"] == 40
