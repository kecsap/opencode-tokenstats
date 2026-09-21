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


def test_trends_count_each_session_once_per_active_bucket(monkeypatch) -> None:
    monkeypatch.setattr("opencode_tokenstats.trends._git_root", lambda _directory: "/repo")
    monkeypatch.setattr("opencode_tokenstats.trends._git_churn", lambda *_args: [])
    monkeypatch.setattr("opencode_tokenstats.trends._run_git", lambda *_args: "HEAD\n")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    session_calls = [
        Call(int(datetime(2026, 1, 1, 0, 30, tzinfo=UTC).timestamp() * 1000), 100),
        Call(int(datetime(2026, 1, 1, 0, 45, tzinfo=UTC).timestamp() * 1000), 25),
        Call(int(datetime(2026, 1, 1, 12, 30, tzinfo=UTC).timestamp() * 1000), 50),
    ]
    result = build_period_trends(
        start=start,
        end=end,
        session_metrics=[Metric("one", session_calls)],
        session_dirs={"one": "/repo"},
    )

    assert result is not None
    assert result["included_sessions"] == 1
    assert result["points"][0]["sessions"] == 1
    assert result["points"][12]["sessions"] == 1
    assert sum(point["sessions"] for point in result["points"]) == 2


def test_trends_selects_severe_outliers_then_orders_them_chronologically(monkeypatch) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    bucket_span_ms = int((end - start).total_seconds() * 1000 / 24)
    tokens = [1] * 7 + [20, 60, 10, 50, 30, 40]
    changes = [
        {
            "timestamp": datetime.fromtimestamp(
                start.timestamp() + (index + 0.5) * (end - start).total_seconds() / 24,
                UTC,
            ),
            "sha": str(index),
            "subject": "change",
            "added": 1,
            "deleted": 0,
            "files": [],
        }
        for index in range(len(tokens))
    ]
    monkeypatch.setattr("opencode_tokenstats.trends._git_root", lambda _directory: "/repo")
    monkeypatch.setattr("opencode_tokenstats.trends._git_churn", lambda *_args: [])
    monkeypatch.setattr("opencode_tokenstats.trends._git_changes", lambda *_args, **_kwargs: changes)
    monkeypatch.setattr("opencode_tokenstats.trends._run_git", lambda *_args: "HEAD\n")

    result = build_period_trends(
        start=start,
        end=end,
        session_metrics=[Metric("one", [Call(1_767_225_600_000 + index * bucket_span_ms, value) for index, value in enumerate(tokens)])],
        session_dirs={"one": "/repo"},
    )

    assert result is not None
    outliers = result["outliers"]
    assert len(outliers) == 5
    assert sorted(outlier["triggers"]["output_tokens_per_loc"]["ratio"] for outlier in outliers) == [20, 30, 40, 50, 60]
    assert [outlier["triggers"]["output_tokens_per_loc"]["ratio"] for outlier in outliers] == [20, 60, 50, 30, 40]
    dates = [(outlier["date"], outlier["end_date"]) for outlier in outliers]
    assert dates == sorted(dates)
