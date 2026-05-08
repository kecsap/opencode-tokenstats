from __future__ import annotations

import json

from opencode_tokenstats.tokenization import TokenizerRegistry, TokenizerSpec


def test_resolve_openai_model_map() -> None:
    registry = TokenizerRegistry()
    resolved = registry.resolve_model("openai", "gpt-5")
    assert resolved.tokenizer.kind == "tiktoken"
    assert resolved.tokenizer.value == "gpt-4o"


def test_resolve_provider_default_huggingface() -> None:
    registry = TokenizerRegistry()
    resolved = registry.resolve_model("anthropic", "unknown-model")
    assert resolved.tokenizer.kind == "huggingface"
    assert resolved.tokenizer.value == "Xenova/claude-tokenizer"


def test_count_approx_fallback_warning() -> None:
    registry = TokenizerRegistry()
    result = registry.count("abcd", TokenizerSpec(kind="approx", value=None))
    assert result.tokens == 1
    assert result.approximate is True
    assert result.warning is not None


def test_resolve_codex_aliases_to_openai_encodings() -> None:
    registry = TokenizerRegistry()
    resolved = registry.resolve_model("openai", "gpt-5.3-codex")
    assert resolved.tokenizer.kind == "tiktoken"
    assert resolved.tokenizer.value == "gpt-4o"

    resolved_mini = registry.resolve_model("openai", "gpt-5.1-codex-mini")
    assert resolved_mini.tokenizer.value == "gpt-4o-mini"


def test_huggingface_uses_local_tokenizer_cache(monkeypatch, tmp_path) -> None:
    hub_dir = tmp_path / "Xenova--claude-tokenizer"
    hub_dir.mkdir(parents=True)
    tok_file = hub_dir / "tokenizer.json"
    tok_file.write_text(json.dumps({"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None, "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1}, "fuse_unk": False}}), encoding="utf-8")

    monkeypatch.setenv("OPENCODE_TOKENIZER_CACHE_DIR", str(tmp_path))

    class FakeEncoded:
        def __init__(self):
            self.ids = [1, 2, 3]

    class FakeTokenizer:
        @staticmethod
        def from_file(_path: str):
            class T:
                def encode(self, _text: str):
                    return FakeEncoded()

            return T()

    class FakeTokenizersModule:
        Tokenizer = FakeTokenizer

    monkeypatch.setattr("opencode_tokenstats.tokenization.import_module", lambda name: FakeTokenizersModule if name == "tokenizers" else None)

    registry = TokenizerRegistry()
    result = registry.count("hello world", TokenizerSpec(kind="huggingface", value="Xenova/claude-tokenizer"))

    assert result.tokens == 3
    assert result.approximate is False


def test_resolve_qwen_model_to_huggingface_tokenizer() -> None:
    registry = TokenizerRegistry()
    resolved = registry.resolve_model("local", "qwen3.6-27b")
    assert resolved.tokenizer.kind == "huggingface"
    assert resolved.tokenizer.value == "Qwen/Qwen3-32B"


def test_huggingface_falls_back_to_transformers_tokenizer(monkeypatch) -> None:
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_hub: str, local_files_only: bool, trust_remote_code: bool):
            assert local_files_only is True
            assert trust_remote_code is True

            class T:
                def __call__(self, _text: str, add_special_tokens: bool = False):
                    assert add_special_tokens is False
                    return {"input_ids": [10, 11, 12, 13]}

            return T()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer

    def fake_import_module(name: str):
        if name == "transformers":
            return FakeTransformers
        if name == "tokenizers":
            class NoTokenizer:
                pass

            return NoTokenizer
        return None

    monkeypatch.setattr("opencode_tokenstats.tokenization.import_module", fake_import_module)

    registry = TokenizerRegistry()
    result = registry.count("hello", TokenizerSpec(kind="huggingface", value="Qwen/Qwen3-32B"))
    assert result.tokens == 4
    assert result.approximate is False
