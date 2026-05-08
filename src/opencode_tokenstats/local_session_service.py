from __future__ import annotations

import os
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
