from __future__ import annotations

import sqlite3

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
            time_created INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO session (id, title, parent_id, time_created) VALUES (?, ?, ?, ?)",
        ("s1", "Main", None, 100),
    )
    conn.execute(
        "INSERT INTO session (id, title, parent_id, time_created) VALUES (?, ?, ?, ?)",
        ("s2", "Child", "s1", 200),
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
