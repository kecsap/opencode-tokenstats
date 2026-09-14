from datetime import UTC, datetime
from pathlib import Path

from opencode_tokenstats.trends import build_period_trends


class Call:
    def __init__(self, timestamp_ms: int, tokens: int) -> None:
        self.timestamp_ms = timestamp_ms
        self._tokens = tokens

    def total_tokens(self) -> int:
        return self._tokens


class Metric:
    def __init__(self, session_id: str, calls: list[Call]) -> None:
        self.session_id = session_id
        self.trend_calls = calls


def test_trends_aggregate_repositories_and_keep_fixed_buckets(monkeypatch) -> None:
    roots = {"/a": "/repo-a", "/b": "/repo-b"}
    monkeypatch.setattr("opencode_tokenstats.trends._git_root", lambda directory: roots.get(directory))
    monkeypatch.setattr(
        "opencode_tokenstats.trends._git_churn",
        lambda root, _start, _end: [(datetime(2026, 1, 1, 1, tzinfo=UTC), 3, 2)] if root == "/repo-a" else [],
    )
    monkeypatch.setattr("opencode_tokenstats.trends._run_git", lambda *_args: "HEAD\n")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    result = build_period_trends(
        start=start,
        end=end,
        session_metrics=[Metric("one", [Call(1767229200000, 100)]), Metric("two", [Call(1767229200000, 50)])],
        session_dirs={"one": "/a", "two": "/b"},
    )
    assert result is not None
    assert result["bucket_count"] == 24
    assert len(result["points"]) == 24
    assert result["tokens"] == 150
    assert result["added_loc"] == 3
    assert result["deleted_loc"] == 2
    assert result["net_loc"] == 1
    assert result["tokens_per_loc"] == 30.0


def test_trends_require_timestamped_token_data(monkeypatch) -> None:
    monkeypatch.setattr("opencode_tokenstats.trends._git_root", lambda _directory: "/repo")
    monkeypatch.setattr("opencode_tokenstats.trends._git_churn", lambda *_args: [])
    monkeypatch.setattr("opencode_tokenstats.trends._run_git", lambda *_args: "HEAD\n")
    result = build_period_trends(
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        session_metrics=[Metric("one", [])],
        session_dirs={"one": str(Path("/repo"))},
    )
    assert result is None
