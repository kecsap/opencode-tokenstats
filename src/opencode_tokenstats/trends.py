from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
from statistics import median
from fnmatch import fnmatch
from typing import Any


MAX_BUCKETS = 48
DEFAULT_CODE_EXCLUDES = frozenset({
    "poetry.lock", "uv.lock", "Cargo.lock", "Gemfile.lock", "composer.lock",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "node_modules/**", "vendor/**", "third_party/**", "dist/**", "build/**",
    "target/**", "coverage/**", ".next/**", "*.min.js", "*.map",
    "*_generated.*", "*.generated.*", "*.pb.*",
})


def _loc_excluded(path: str, scope: str, patterns: tuple[str, ...]) -> bool:
    excludes = set(patterns)
    if scope == "code":
        excludes.update(DEFAULT_CODE_EXCLUDES)
    return any(fnmatch(path, pattern) or fnmatch(Path(path).name, pattern) for pattern in excludes)


def _call_tokens(call: Any) -> int:
    value = getattr(call, "total_tokens", 0)
    if callable(value):
        value = value()
    return int(value or 0)


def _call_datetime(call: Any) -> datetime | None:
    raw = getattr(call, "timestamp_ms", None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    # Accommodate legacy/test values in seconds while treating stored values
    # as milliseconds for normal OpenCode telemetry.
    if value < 10_000_000_000:
        value *= 1000
    try:
        return datetime.fromtimestamp(value / 1000, UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _bucket_count(start: datetime, end: datetime) -> int:
    days = max((end - start).total_seconds() / 86400, 1)
    if days <= 2:
        return 24
    if days <= 14:
        return max(1, round(days))
    return min(MAX_BUCKETS, max(1, round(days)))


def _bucket_index(when: datetime, start: datetime, end: datetime, count: int) -> int:
    fraction = (when - start).total_seconds() / max((end - start).total_seconds(), 1)
    return min(count - 1, max(0, int(fraction * count)))


def _run_git(directory: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", directory, *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _git_root(directory: str) -> str | None:
    raw = _run_git(directory, "rev-parse", "--show-toplevel")
    if not raw:
        return None
    root = raw.strip()
    return str(Path(root).resolve()) if root else None


def _git_changes(
    root: str,
    start: datetime,
    end: datetime,
    *,
    loc_scope: str = "code",
    loc_exclude: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    raw = _run_git(root, "log", "--numstat", "--format=commit%x09%H%x09%ct%x09%s", "HEAD", "--")
    if not raw:
        return []
    result: list[dict[str, Any]] = []
    commit_time: datetime | None = None
    commit: dict[str, Any] | None = None
    seen: set[str] = set()
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) >= 4 and fields[0] == "commit":
            commit_time = datetime.fromtimestamp(int(fields[2]), UTC)
            if fields[1] in seen:
                commit_time = None
                commit = None
            else:
                seen.add(fields[1])
                commit = {"sha": fields[1], "timestamp": commit_time, "subject": fields[3], "added": 0, "deleted": 0, "files": []}
                result.append(commit)
            continue
        if commit_time is None or commit is None or len(fields) < 3:
            continue
        if not (start <= commit_time < end) or fields[0] == "-" or fields[1] == "-":
            continue
        try:
            added, deleted = int(fields[0]), int(fields[1])
            if _loc_excluded(fields[2], loc_scope, loc_exclude):
                continue
            commit["added"] += added
            commit["deleted"] += deleted
            commit["files"].append({"path": fields[2], "added": added, "deleted": deleted, "changed": added + deleted})
        except ValueError:
            continue
    for commit in result:
        commit["files"].sort(key=lambda item: item["changed"], reverse=True)
    return result


def _git_churn(
    root: str,
    start: datetime,
    end: datetime,
    *,
    loc_scope: str = "code",
    loc_exclude: tuple[str, ...] = (),
) -> list[tuple[datetime, int, int]]:
    """Return file-level-compatible churn tuples for callers/tests."""
    changes = _git_changes(root, start, end, loc_scope=loc_scope, loc_exclude=loc_exclude)
    return [(change["timestamp"], change["added"], change["deleted"]) for change in changes]


def build_period_trends(
    *,
    start: datetime,
    end: datetime,
    session_metrics: list[Any],
    session_dirs: dict[str, str],
    loc_scope: str = "code",
    loc_exclude: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Build aggregate trends from sessions with Git and timestamped telemetry."""
    repo_sessions: dict[str, list[Any]] = defaultdict(list)
    for metric in session_metrics:
        directory = session_dirs.get(metric.session_id, "")
        root = _git_root(directory) if directory else None
        if root:
            repo_sessions[root].append(metric)
    if not repo_sessions:
        return None

    count = _bucket_count(start, end)
    span = (end - start) / count
    points = [
        {
            "date": (start + span * index).isoformat(),
            "tokens": 0,
            "input_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 0,
            "added_loc": 0,
            "deleted_loc": 0,
            "churn_loc": 0,
            "net_loc": 0,
            "tokens_per_loc": None,
            "sessions": 0,
        }
        for index in range(count)
    ]
    included_sessions = 0
    included_calls = 0
    repositories: list[str] = []
    bucket_changes: list[list[dict[str, Any]]] = [[] for _ in points]
    for root, metrics in repo_sessions.items():
        correlated_metrics = [
            metric
            for metric in metrics
            if any(getattr(call, "timestamp_ms", None) is not None for call in getattr(metric, "trend_calls", []))
        ]
        if not correlated_metrics:
            continue
        if loc_scope == "code" and not loc_exclude:
            churn = _git_churn(root, start, end)
        else:
            churn = _git_churn(root, start, end, loc_scope=loc_scope, loc_exclude=loc_exclude)
        changes = _git_changes(root, start, end, loc_scope=loc_scope, loc_exclude=loc_exclude)
        if churn or changes or _run_git(root, "rev-parse", "--verify", "HEAD"):
            repositories.append(root)
        if not changes:
            changes = [
                {"timestamp": when, "sha": "unknown", "subject": "Git changes", "added": added, "deleted": deleted, "files": []}
                for when, added, deleted in churn
            ]
        for change in changes:
            bucket = _bucket_index(change["timestamp"], start, end, count)
            bucket_changes[bucket].append({**change, "repository": root})
            point = points[bucket]
            added, deleted = change["added"], change["deleted"]
            point["added_loc"] += added
            point["deleted_loc"] += deleted
            point["churn_loc"] += added + deleted
            point["net_loc"] += added - deleted
        for metric in correlated_metrics:
            calls = getattr(metric, "trend_calls", [])
            timestamped = [call for call in calls if getattr(call, "timestamp_ms", None) is not None]
            if timestamped:
                included_sessions += 1
            active_buckets: set[int] = set()
            for call in timestamped:
                when = _call_datetime(call)
                if when is None:
                    continue
                if not (start <= when < end):
                    continue
                bucket = _bucket_index(when, start, end, count)
                if bucket not in active_buckets:
                    points[bucket]["sessions"] += 1
                    active_buckets.add(bucket)
                point = points[bucket]
                has_breakdown = any(
                    hasattr(call, attribute)
                    for attribute in ("input_tokens", "cache_read_tokens", "output_tokens", "reasoning_tokens")
                )
                input_tokens = int(getattr(call, "input_tokens", 0) or 0) if has_breakdown else 0
                cache_read_tokens = int(getattr(call, "cache_read_tokens", 0) or 0) if has_breakdown else 0
                output_tokens = int(getattr(call, "output_tokens", 0) or 0)
                if not has_breakdown:
                    output_tokens = _call_tokens(call)
                point["input_tokens"] += input_tokens
                point["cache_read_tokens"] += cache_read_tokens
                point["output_tokens"] += output_tokens
                point["tokens"] += input_tokens + cache_read_tokens + output_tokens
                included_calls += 1
    if not repositories or not included_calls:
        return None
    for point in points:
        if point["churn_loc"]:
            for key in ("input_tokens", "cache_read_tokens", "output_tokens"):
                point[f"{key}_per_loc"] = round(point[key] / point["churn_loc"], 2)
            point["tokens_per_loc"] = round(point["tokens"] / point["churn_loc"], 2)
    total_added = sum(p["added_loc"] for p in points)
    total_deleted = sum(p["deleted_loc"] for p in points)
    total_tokens = sum(p["tokens"] for p in points)
    ratio_keys = ("input_tokens_per_loc", "cache_read_tokens_per_loc", "output_tokens_per_loc")
    ratio_medians = {
        key: median([p[key] for p in points if p.get(key, 0) > 0])
        if any(p.get(key, 0) > 0 for p in points) else 0
        for key in ratio_keys
    }
    ratio_thresholds = {key: value * 2 for key, value in ratio_medians.items()}
    outliers: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        triggers = {
            key: {"ratio": point[key], "median": ratio_medians[key], "threshold": ratio_thresholds[key]}
            for key in ratio_keys
            if point.get(key, 0) > ratio_thresholds[key] and ratio_thresholds[key] > 0
        }
        if not triggers:
            continue
        selected_commits = sorted(
            bucket_changes[index],
            key=lambda item: item["added"] + item["deleted"],
            reverse=True,
        )[:3]
        selected_delta = sum(change["added"] + change["deleted"] for change in selected_commits)
        commits = []
        for change in selected_commits:
            commits.append({
                "repository": change["repository"],
                "sha": change["sha"],
                "subject": change["subject"],
                "added": change["added"],
                "deleted": change["deleted"],
                "delta_loc": change["added"] + change["deleted"],
                "files": change["files"][:5],
            })
        outliers.append({
            "date": point["date"],
            "end_date": points[index + 1]["date"] if index + 1 < len(points) else end.isoformat(),
            "added_loc": point["added_loc"],
            "deleted_loc": point["deleted_loc"],
            "churn_loc": point["churn_loc"],
            "triggers": triggers,
            "commits": commits,
            "other_commits_delta_loc": max(0, point["churn_loc"] - selected_delta),
        })
    outliers.sort(key=lambda item: max(trigger["ratio"] for trigger in item["triggers"].values()), reverse=True)
    selected_outliers = sorted(outliers[:5], key=lambda item: (item["date"], item["end_date"]))
    category_totals = {
        key: sum(p[key] for p in points)
        for key in ("input_tokens", "cache_read_tokens", "output_tokens")
    }
    return {
        "bucket_count": count,
        "points": points,
        "repositories": repositories,
        "included_sessions": included_sessions,
        "included_calls": included_calls,
        "added_loc": total_added,
        "deleted_loc": total_deleted,
        "churn_loc": total_added + total_deleted,
        "net_loc": total_added - total_deleted,
        "tokens": total_tokens,
        **category_totals,
        "category_tokens_per_loc": {
            key: round(value / (total_added + total_deleted), 2)
            if total_added + total_deleted else None
            for key, value in category_totals.items()
        },
        "tokens_per_loc": round(total_tokens / (total_added + total_deleted), 2)
        if total_added + total_deleted else None,
        "outlier_medians": ratio_medians,
        "outlier_thresholds": ratio_thresholds,
        "outliers": selected_outliers,
    }
