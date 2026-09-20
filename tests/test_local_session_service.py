from __future__ import annotations

import sqlite3
import json

import pytest

import opencode_tokenstats.local_session_service as local_session_service
from opencode_tokenstats.local_session_service import LocalSessionService, LocalStorageError


def test_list_sessions_from_sqlite(tmp_path) -> None:
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY,
            title TEXT,
            parent_id TEXT,
            time_created INTEGER,
            directory TEXT
        )
    """
    )
    conn.execute(
        "INSERT INTO session (id, title, parent_id, time_created, directory) VALUES (?, ?, ?, ?, ?)",
        ("s1", "Main", None, 100, "/home/user/project1"),
    )
    conn.execute(
        "INSERT INTO session (id, title, parent_id, time_created, directory) VALUES (?, ?, ?, ?, ?)",
        ("s2", "Child", "s1", 200, "/home/user/project2"),
    )
    conn.commit()
    conn.close()

    service = LocalSessionService(db_path=db)
    sessions = service.list_sessions()

    assert [s["id"] for s in sessions] == ["s2", "s1"]


def test_list_sessions_reads_time_updated_when_available(tmp_path) -> None:
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, title TEXT, parent_id TEXT, time_created INTEGER,
            time_updated INTEGER, directory TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?)",
        ("s1", "Active", None, 100, 200, "/repo"),
    )
    conn.commit()
    conn.close()

    sessions = LocalSessionService(db_path=db).list_sessions()

    assert sessions == [{
        "id": "s1", "title": "Active", "parent_id": None, "time_created": 100,
        "time_updated": 200, "directory": "/repo", "data": {},
    }]


def test_missing_db_raises(tmp_path) -> None:
    service = LocalSessionService(db_path=tmp_path / "missing.db")
    with pytest.raises(LocalStorageError):
        service.list_sessions()


def test_get_messages_from_sqlite(tmp_path) -> None:
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            data TEXT,
            time_created INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE part (
            id TEXT PRIMARY KEY,
            message_id TEXT,
            session_id TEXT,
            data TEXT,
            time_created INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO message (id, session_id, data, time_created) VALUES (?, ?, ?, ?)",
        ("m1", "s1", json.dumps({"role": "assistant"}), 1),
    )
    conn.execute(
        "INSERT INTO part (id, message_id, session_id, data, time_created) VALUES (?, ?, ?, ?, ?)",
        (
            "p1",
            "m1",
            "s1",
            json.dumps({"type": "tool", "tool": "read", "state": {"status": "completed"}}),
            1,
        ),
    )
    conn.commit()
    conn.close()

    service = LocalSessionService(db_path=db)
    messages = service.get_messages("s1")

    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    parts = messages[0]["parts"]
    assert isinstance(parts, list)
    assert parts[0]["tool"] == "read"


def test_get_period_messages_only_returns_in_window_parts_and_sessions(tmp_path) -> None:
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT, time_created INTEGER);
        """
    )
    rows = [
        ("m-before", "s-old", json.dumps({"role": "assistant"}), 10),
        ("m-active", "s-old", json.dumps({"role": "assistant"}), 110),
        ("m-message-only", "s-active", json.dumps({"role": "user"}), 120),
        ("m-other", "s-other", json.dumps({"role": "assistant"}), 120),
    ]
    conn.executemany("INSERT INTO message VALUES (?, ?, ?, ?)", rows)
    conn.executemany(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
        [
            ("p-old", "m-before", "s-old", json.dumps({"type": "tool"}), 20),
            ("p-active", "m-active", "s-old", json.dumps({"type": "step-finish"}), 125),
            ("p-outside", "m-active", "s-old", json.dumps({"type": "tool"}), 220),
            ("p-other", "m-other", "s-other", json.dumps({"type": "tool"}), 20),
        ],
    )
    conn.commit()
    conn.close()

    period = LocalSessionService(db_path=db).get_period_messages(100, 200)

    assert set(period) == {"s-old"}
    assert len(period["s-old"]) == 1
    assert period["s-old"][0]["_time_created"] == 110
    assert period["s-old"][0]["parts"] == [{"type": "step-finish", "_time_created": 125}]


def test_get_period_messages_bucketed_bounds_and_sorts_ids(monkeypatch) -> None:
    calls: list[set[str]] = []

    def fake_get_period_messages(self, _start, _end, *, session_ids=None):
        ids = set(session_ids or ())
        calls.append(ids)
        return {sid: [{"id": sid}] for sid in ids}

    monkeypatch.setattr(LocalSessionService, "get_period_messages", fake_get_period_messages)
    session_ids = {f"s{index:04d}" for index in range(1801)}
    result = LocalSessionService().get_period_messages_bucketed(
        0, 1, session_ids, workers=99
    )

    assert sorted(len(bucket) for bucket in calls) == [1, 900, 900]
    assert all(len(bucket) <= 900 for bucket in calls)
    assert list(result) == sorted(session_ids)


def test_get_period_messages_bucketed_live_equivalence_and_sequential_progress(tmp_path, monkeypatch) -> None:
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT, time_created INTEGER);
        """
    )
    session_ids = {f"s{index:04d}" for index in range(901)}
    conn.executemany(
        "INSERT INTO message VALUES (?, ?, ?, ?)",
        [(f"m{sid}", sid, json.dumps({"role": "assistant"}), 150) for sid in session_ids],
    )
    conn.executemany(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
        [(f"p{sid}", f"m{sid}", sid, json.dumps({"type": "tool"}), 150) for sid in session_ids],
    )
    conn.commit()
    conn.close()

    service = LocalSessionService(db_path=db)
    expected = service.get_period_messages(100, 200, session_ids=session_ids)
    calls: list[set[str]] = []
    original = LocalSessionService.get_period_messages

    def tracked(self, start, end, *, session_ids=None):
        ids = set(session_ids or ())
        calls.append(ids)
        return original(self, start, end, session_ids=session_ids)

    monkeypatch.setattr(LocalSessionService, "get_period_messages", tracked)
    for workers in (1, 2, 4, 8):
        progress: list[tuple[int, int]] = []
        result = service.get_period_messages_bucketed(
            100,
            200,
            session_ids,
            workers=workers,
            progress_callback=lambda current, total: progress.append((current, total)),
        )
        assert result == expected
        assert progress[0] == (0, 2)
        assert progress[-1] == (2, 2)
        assert all(len(bucket) <= 900 for bucket in calls)
        calls.clear()

    service.get_period_messages_bucketed(100, 200, session_ids, workers=1)
    assert calls == [set(sorted(session_ids)[:900]), set(sorted(session_ids)[900:])]

    seen_worker_counts: list[int] = []
    original_executor = local_session_service.ThreadPoolExecutor

    class RecordingExecutor(original_executor):
        def __init__(self, *args, **kwargs):
            seen_worker_counts.append(kwargs["max_workers"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(local_session_service, "ThreadPoolExecutor", RecordingExecutor)
    service.get_period_messages_bucketed(100, 200, session_ids, workers=99)
    assert seen_worker_counts and seen_worker_counts[0] <= 8
