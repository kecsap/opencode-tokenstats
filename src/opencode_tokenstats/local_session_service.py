from __future__ import annotations

import os
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


class LocalStorageError(RuntimeError):
    """Raised when local OpenCode storage cannot be read."""


LOCAL_QUERY_BUCKET_SIZE = 900
MAX_LOCAL_QUERY_WORKERS = 8


@dataclass(slots=True)
class LocalSessionService:
    db_path: Path | None = None

    @staticmethod
    def find_database_path(custom_path: str | None = None) -> Path | None:
        candidates: list[Path] = []

        if custom_path:
            candidates.append(Path(os.path.expanduser(os.path.expandvars(custom_path))))

        env_path = os.environ.get("OPENCODE_DATABASE_FILE")
        if env_path:
            candidates.append(Path(os.path.expanduser(os.path.expandvars(env_path))))

        candidates.append(Path.home() / ".local" / "share" / "opencode" / "opencode.db")

        if os.name == "nt":
            appdata = os.environ.get("APPDATA")
            if appdata:
                candidates.append(Path(appdata) / "opencode" / "opencode.db")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def list_sessions(self) -> list[dict[str, object]]:
        path = self.db_path or self.find_database_path()
        if not path:
            raise LocalStorageError(
                "OpenCode local database not found. Set --db-path or OPENCODE_DATABASE_FILE."
            )
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT id, title, parent_id, time_created, time_updated, directory, data AS session_data
                    FROM session
                    ORDER BY time_created DESC
                    """
                ).fetchall()
            except sqlite3.Error:
                try:
                    rows = conn.execute(
                        """
                        SELECT id, title, parent_id, time_created, time_updated, directory
                        FROM session
                        ORDER BY time_created DESC
                        """
                    ).fetchall()
                except sqlite3.Error:
                    rows = conn.execute(
                        """
                        SELECT id, title, parent_id, time_created, directory
                        FROM session
                        ORDER BY time_created DESC
                        """
                    ).fetchall()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "parent_id": row["parent_id"],
                    "time_created": row["time_created"],
                    "time_updated": row["time_updated"] if "time_updated" in row.keys() else None,
                    "directory": row["directory"],
                    "data": _parse_json_dict(row["session_data"]) if "session_data" in row.keys() else {},
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            raise LocalStorageError(f"Failed to read OpenCode database at {path}: {exc}") from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_messages(self, session_id: str) -> list[dict[str, object]]:
        path = self.db_path or self.find_database_path()
        if not path:
            raise LocalStorageError(
                "OpenCode local database not found. Set --db-path or OPENCODE_DATABASE_FILE."
            )
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT m.id AS message_id, m.data AS message_data, p.data AS part_data,
                       p.time_created AS part_time_created
                FROM message m
                LEFT JOIN part p ON p.message_id = m.id
                WHERE m.session_id = ?
                ORDER BY m.time_created ASC, p.time_created ASC
                """,
                (session_id,),
            ).fetchall()

            by_message: dict[str, dict[str, object]] = {}
            for row in rows:
                message_id = row["message_id"]
                message = by_message.get(message_id)
                if message is None:
                    raw = _parse_json_dict(row["message_data"])
                    message = {
                        "role": raw.get("role", ""),
                        "info": raw,
                        "_time_created": row["message_time_created"] if "message_time_created" in row.keys() else None,
                        "parts": [],
                    }
                    by_message[message_id] = message

                part_raw = _parse_json_dict(row["part_data"])
                if part_raw:
                    # The serialized step-finish `time` is often a duration. Keep
                    # SQLite's authoritative part timestamp for period charts.
                    part_raw["_time_created"] = row["part_time_created"]
                    parts = message.get("parts")
                    if isinstance(parts, list):
                        parts.append(part_raw)

            return list(by_message.values())
        except sqlite3.Error as exc:
            raise LocalStorageError(f"Failed to read OpenCode database at {path}: {exc}") from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_period_messages(
        self,
        start_ms: int,
        end_ms: int,
        *,
        session_ids: set[str] | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        """Return only messages and parts active in a millisecond UTC window.

        A session is included only when it has an in-window part. Messages created in
        that window for those sessions are retained, matching the period filtering
        semantics without loading whole session histories.
        """
        path = self.db_path or self.find_database_path()
        if not path:
            raise LocalStorageError(
                "OpenCode local database not found. Set --db-path or OPENCODE_DATABASE_FILE."
            )
        if session_ids is not None and not session_ids:
            return {}

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            selected_ids = sorted(session_ids or ())
            # SQLite's default bound-variable limit is commonly 999. A session filter
            # normally narrows this far below that; otherwise retain correct results
            # and apply the filter in the caller.
            session_clause = ""
            session_params: tuple[str, ...] = ()
            if selected_ids and len(selected_ids) <= 900:
                session_clause = f" AND session_id IN ({','.join('?' for _ in selected_ids)})"
                session_params = tuple(selected_ids)
            rows = conn.execute(
                f"""
                WITH active_parts AS (
                    SELECT message_id, session_id, time_created, data
                    FROM part
                    WHERE time_created >= ? AND time_created < ?{session_clause}
                ), active_sessions AS (
                    SELECT DISTINCT session_id FROM active_parts
                ), selected_messages AS (
                    SELECT m.id, m.session_id, m.data, m.time_created
                    FROM message m
                    JOIN active_sessions s ON s.session_id = m.session_id
                    WHERE m.time_created >= ? AND m.time_created < ?
                    UNION
                    SELECT m.id, m.session_id, m.data, m.time_created
                    FROM message m
                    JOIN active_parts p ON p.message_id = m.id
                )
                SELECT m.id AS message_id, m.session_id, m.data AS message_data,
                       m.time_created AS message_time_created, p.data AS part_data,
                       p.time_created AS part_time_created
                FROM selected_messages m
                LEFT JOIN active_parts p ON p.message_id = m.id
                ORDER BY m.session_id, m.time_created ASC, p.time_created ASC
                """,
                (start_ms, end_ms, *session_params, start_ms, end_ms),
            ).fetchall()

            by_session: dict[str, dict[str, dict[str, object]]] = {}
            for row in rows:
                session_id = str(row["session_id"])
                messages = by_session.setdefault(session_id, {})
                message_id = str(row["message_id"])
                message = messages.get(message_id)
                if message is None:
                    raw = _parse_json_dict(row["message_data"])
                    message = {
                        "role": raw.get("role", ""),
                        "info": raw,
                        "_time_created": row["message_time_created"],
                        "parts": [],
                    }
                    messages[message_id] = message
                part_raw = _parse_json_dict(row["part_data"])
                if part_raw:
                    part_raw["_time_created"] = row["part_time_created"]
                    parts = message["parts"]
                    if isinstance(parts, list):
                        parts.append(part_raw)
            return {session_id: list(messages.values()) for session_id, messages in by_session.items()}
        except sqlite3.Error as exc:
            raise LocalStorageError(f"Failed to read OpenCode database at {path}: {exc}") from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_period_messages_bucketed(
        self,
        start_ms: int,
        end_ms: int,
        session_ids: set[str],
        *,
        workers: int,
        progress_callback: callable | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        """Read disjoint session-ID buckets without changing period semantics."""
        if not session_ids:
            return {}
        ids = sorted(session_ids)
        if len(ids) <= LOCAL_QUERY_BUCKET_SIZE:
            return self.get_period_messages(start_ms, end_ms, session_ids=session_ids)

        buckets = [
            set(ids[index : index + LOCAL_QUERY_BUCKET_SIZE])
            for index in range(0, len(ids), LOCAL_QUERY_BUCKET_SIZE)
        ]
        worker_count = min(max(workers, 1), MAX_LOCAL_QUERY_WORKERS, len(buckets))
        if progress_callback:
            progress_callback(0, len(buckets))
        results: dict[int, dict[str, list[dict[str, object]]]] = {}
        if worker_count == 1:
            for index, bucket in enumerate(buckets):
                results[index] = self.get_period_messages(
                    start_ms, end_ms, session_ids=bucket
                )
                if progress_callback:
                    progress_callback(index + 1, len(buckets))
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self.get_period_messages, start_ms, end_ms, session_ids=bucket): index
                    for index, bucket in enumerate(buckets)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    if progress_callback:
                        progress_callback(len(results), len(buckets))

        merged: dict[str, list[dict[str, object]]] = {}
        for index in range(len(buckets)):
            merged.update(results[index])
        return {session_id: merged[session_id] for session_id in sorted(merged)}

    def get_session(self, session_id: str) -> dict[str, object]:
        path = self.db_path or self.find_database_path()
        if not path:
            raise LocalStorageError(
                "OpenCode local database not found. Set --db-path or OPENCODE_DATABASE_FILE."
            )

        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """
                    SELECT id, title, parent_id, time_created, time_updated, directory, data AS session_data
                    FROM session
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            except sqlite3.Error:
                try:
                    row = conn.execute(
                        """
                        SELECT id, title, parent_id, time_created, time_updated, directory
                        FROM session
                        WHERE id = ?
                        LIMIT 1
                        """,
                        (session_id,),
                    ).fetchone()
                except sqlite3.Error:
                    row = conn.execute(
                        """
                        SELECT id, title, parent_id, time_created, directory
                        FROM session
                        WHERE id = ?
                        LIMIT 1
                        """,
                        (session_id,),
                    ).fetchone()

            if row is None:
                return {}

            return {
                "id": row["id"],
                "title": row["title"],
                "parent_id": row["parent_id"],
                "time_created": row["time_created"],
                "time_updated": row["time_updated"] if "time_updated" in row.keys() else None,
                "directory": row["directory"],
                "data": _parse_json_dict(row["session_data"]) if "session_data" in row.keys() else {},
            }
        except sqlite3.Error as exc:
            raise LocalStorageError(f"Failed to read OpenCode database at {path}: {exc}") from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


def _parse_json_dict(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}
