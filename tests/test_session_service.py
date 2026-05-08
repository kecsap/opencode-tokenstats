from __future__ import annotations

from opencode_tokenstats.session_service import SessionService


class DummyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []

    def get(self, path: str):
        self.calls.append(("get", path, None))
        return {"items": [{"id": "s1"}]}

    def post(self, path: str, json=None):
        self.calls.append(("post", path, json))
        if path.endswith("/messages"):
            return None
        return {"id": "x"}


def test_session_service_shapes() -> None:
    service = SessionService(DummyClient())

    sessions = service.list_sessions()
    session = service.get_session("/tmp/s")
    messages = service.get_messages("/tmp/s")
    children = service.get_children("/tmp/s")

    assert sessions == [{"id": "s1"}]
    assert session == {"id": "x"}
    assert messages == []
    assert children == [{"id": "x"}]
