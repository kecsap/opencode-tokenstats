from __future__ import annotations

from click.testing import CliRunner

from opencode_tokenstats import cli


class DummyService:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def list_sessions(self):
        return [{"id": "a"}, {"id": "b"}]

    def get_messages(self, _session_id: str):
        return [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "tool": "read",
                        "state": {"status": "completed", "input": {"file": "x"}},
                    }
                ],
            }
        ]


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

    def get_messages(self, _session_id: str):
        return [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "tool": "bash",
                        "state": {"status": "completed", "input": {"command": "ls"}},
                    }
                ],
            }
        ]


class DummyRegistry:
    class Resolved:
        provider_id = "local"
        model_id = "qwen3.6-27b"

        class Tok:
            kind = "huggingface"
            value = "Qwen/Qwen3-32B"

        tokenizer = Tok()

    class Result:
        approximate = False
        warning = None

    def resolve_model(self, _provider_id: str, _model_id: str):
        return DummyRegistry.Resolved()

    def count(self, _text: str, _spec):
        return DummyRegistry.Result()

    def warmup(self, _pairs, sample_text="warmup"):
        class R:
            def __init__(self, provider_id, model_id, kind, value, status, warning=None):
                self.provider_id = provider_id
                self.model_id = model_id
                self.tokenizer_kind = kind
                self.tokenizer_value = value
                self.status = status
                self.warning = warning

        return [
            R("openai", "gpt-5.3-codex", "tiktoken", "gpt-4o", "warmed"),
            R("local", "qwen3.6-27b", "huggingface", "Qwen/Qwen3-32B", "approximate", "fallback"),
        ]


def test_health_ok(monkeypatch) -> None:
    monkeypatch.setattr(cli, "OpencodeApiClient", DummyClient)
    monkeypatch.setattr(cli, "SessionService", DummyService)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--mode", "api", "health"])

    assert result.exit_code == 0
    assert "OpenCode API: OK" in result.output
    assert "list_sessions returned 2 entries" in result.output


def test_health_local_default(monkeypatch) -> None:
    monkeypatch.setattr(cli, "LocalSessionService", DummyLocalService)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["health"])

    assert result.exit_code == 0
    assert "OpenCode Local Storage: OK" in result.output
    assert "SQLite DB: /tmp/opencode.db" in result.output
    assert "list_sessions returned 1 entries" in result.output


def test_health_tokenizer_check(monkeypatch) -> None:
    monkeypatch.setattr(cli, "LocalSessionService", DummyLocalService)
    monkeypatch.setattr(cli, "TokenizerRegistry", DummyRegistry)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["health", "--check-tokenizer"])

    assert result.exit_code == 0
    assert "Tokenizer Check: exact" in result.output
    assert "kind=huggingface" in result.output


def test_health_compatibility_check_local(monkeypatch) -> None:
    monkeypatch.setattr(cli, "LocalSessionService", DummyLocalService)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["health", "--compat-mode", "tokenscope_compat"])

    assert result.exit_code == 0
    assert "Compatibility Check:" in result.output
    assert "observed_tools_only=True" in result.output
    assert "Tool Estimate:" in result.output


def test_health_compatibility_check_api(monkeypatch) -> None:
    monkeypatch.setattr(cli, "OpencodeApiClient", DummyClient)
    monkeypatch.setattr(cli, "SessionService", DummyService)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["--mode", "api", "health", "--compat-mode", "strict_api", "--compat-source", "api"],
    )

    assert result.exit_code == 0
    assert "Compatibility Check:" in result.output
    assert "mode=strict_api" in result.output


def test_tokenizer_warmup_command(monkeypatch) -> None:
    monkeypatch.setattr(cli, "TokenizerRegistry", DummyRegistry)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["tokenizer-warmup", "--pair", "openai:gpt-5.3-codex"])
    assert result.exit_code == 0
    assert "Tokenizer warmup:" in result.output
    assert "status=warmed" in result.output


def test_auto_warmup_enabled_by_default(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_warmup():
        calls["n"] += 1

    monkeypatch.setattr(cli, "_run_default_warmup_silent", fake_warmup)
    monkeypatch.setattr(cli, "LocalSessionService", DummyLocalService)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["health"])
    assert result.exit_code == 0
    assert calls["n"] == 1


def test_auto_warmup_can_be_disabled(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_warmup():
        calls["n"] += 1

    monkeypatch.setattr(cli, "_run_default_warmup_silent", fake_warmup)
    monkeypatch.setattr(cli, "LocalSessionService", DummyLocalService)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--no-warmup", "health"])
    assert result.exit_code == 0
    assert calls["n"] == 0
