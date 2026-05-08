from __future__ import annotations

import re
from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _extract_metric_value(prom: str, name: str) -> float:
    m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.]+)$", prom, flags=re.MULTILINE)
    assert m is not None
    return float(m.group(1))


def test_prom_sample_contains_expected_primary_metrics() -> None:
    prom = _read("/home/kecsap/ai-tools/opencode-tokenstats/20260506_1525_ses_202e28e28ffeMhUAG1bsrZ7m7B.prom")
    assert _extract_metric_value(prom, "tokenscope_api_calls") == 172
    assert _extract_metric_value(prom, "tokenscope_input_tokens") == 1147492
    assert _extract_metric_value(prom, "tokenscope_cache_read_tokens") == 30306304
    assert _extract_metric_value(prom, "tokenscope_output_tokens") == 43683
    assert _extract_metric_value(prom, "tokenscope_reasoning_tokens") == 15960
    assert _extract_metric_value(prom, "tokenscope_session_total_tokens") == 31513439


def test_prom_sample_has_component_semantic_families() -> None:
    prom = _read("/home/kecsap/ai-tools/opencode-tokenstats/20260506_1525_ses_202e28e28ffeMhUAG1bsrZ7m7B.prom")
    assert "tokenscope_component_tokens" in prom
    assert "tokenscope_component_estimated_session_tokens" in prom
    assert "tokenscope_tool_tokens" in prom
    assert "tokenscope_tool_calls" in prom
    assert "tokenscope_contributor_tokens" in prom
