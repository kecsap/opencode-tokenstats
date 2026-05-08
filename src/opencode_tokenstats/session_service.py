from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import OpencodeApiClient


@dataclass(slots=True)
class SessionService:
    client: OpencodeApiClient

    def list_sessions(self) -> list[dict[str, Any]]:
        data = self.client.get("/session/list")
        return self._as_list(data)

    def get_session(self, path: str) -> dict[str, Any]:
        return self.client.post("/session/get", json={"path": path})

    def get_messages(self, path: str) -> list[dict[str, Any]]:
        data = self.client.post("/session/messages", json={"path": path})
        return self._as_list(data)

    def get_children(self, path: str) -> list[dict[str, Any]]:
        data = self.client.post("/session/children", json={"path": path})
        return self._as_list(data)

    @staticmethod
    def _as_list(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and "items" in value and isinstance(value["items"], list):
            return value["items"]
        return [value]
