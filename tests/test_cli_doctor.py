from __future__ import annotations

from click.testing import CliRunner

from opencode_tokenstats import cli


class DummyService:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def list_sessions(self):
        return [{"id": "a"}, {"id": "b"}]


class DummyClient:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


class DummyLocalService:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    @staticmethod
    def find_database_path(_custom_path=None):
        return "/tmp/opencode.db"

    def list_sessions(self):
        return [{"id": "x"}]


def test_doctor_ok(monkeypatch) -> None:
    monkeypatch.setattr(cli, "OpencodeApiClient", DummyClient)
    monkeypatch.setattr(cli, "SessionService", DummyService)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--mode", "api", "doctor"])

    assert result.exit_code == 0
    assert "OpenCode API: OK" in result.output
    assert "list_sessions returned 2 entries" in result.output


def test_doctor_local_default(monkeypatch) -> None:
    monkeypatch.setattr(cli, "LocalSessionService", DummyLocalService)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["doctor"])

    assert result.exit_code == 0
    assert "OpenCode Local Storage: OK" in result.output
    assert "SQLite DB: /tmp/opencode.db" in result.output
    assert "list_sessions returned 1 entries" in result.output
