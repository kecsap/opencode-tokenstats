from __future__ import annotations

import sqlite3
import json

import pytest

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
