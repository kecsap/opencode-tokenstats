from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


class LocalStorageError(RuntimeError):
    """Raised when local OpenCode storage cannot be read."""


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
            rows = conn.execute(
                """
                SELECT id, title, parent_id, time_created
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
                SELECT m.id AS message_id, m.data AS message_data, p.data AS part_data
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
                        "parts": [],
                    }
                    by_message[message_id] = message

                part_raw = _parse_json_dict(row["part_data"])
                if part_raw:
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
