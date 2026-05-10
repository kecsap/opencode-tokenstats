from __future__ import annotations

from opencode_tokenstats.cli import _finalize_component_stats_canonical, _finalize_core_stats


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


def test_core_stats_merge_invalid_into_general() -> None:
    out = _finalize_core_stats({"invalid": 13462.0, "general": 202.0, "read": 10.0})
    rows = out["rows"]
    names = {r["component_name"] for r in rows}

    assert "invalid" not in names
    assert "general" in names

    general_row = next(r for r in rows if r["component_name"] == "general")
    assert general_row["tokens"] == 13664
    assert out["total_tokens"] == 13674
