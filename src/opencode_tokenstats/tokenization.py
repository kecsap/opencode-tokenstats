from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import os
from typing import Any


OPENAI_MODEL_MAP: dict[str, str] = {
    "gpt-5": "gpt-4o",
    "o4-mini": "gpt-4o",
    "o3": "gpt-4o",
    "o3-mini": "gpt-4o",
    "o1": "gpt-4o",
    "o1-pro": "gpt-4o",
    "gpt-4.1": "gpt-4o",
    "gpt-4.1-mini": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-5-codex": "gpt-4o",
    "gpt-5.1-codex": "gpt-4o",
    "gpt-5.1-codex-max": "gpt-4o",
    "gpt-5.1-codex-mini": "gpt-4o-mini",
    "gpt-5.2-codex": "gpt-4o",
    "gpt-5.3-codex": "gpt-4o",
    "gpt-5.3-codex-xhigh": "gpt-4o",
}

HUGGINGFACE_TOKENIZER_MODEL_MAP: dict[str, str] = {
    "claude-opus-4": "Xenova/claude-tokenizer",
    "claude-sonnet-4": "Xenova/claude-tokenizer",
    "claude-3.7-sonnet": "Xenova/claude-tokenizer",
    "claude-3.5-sonnet": "Xenova/claude-tokenizer",
    "claude-3.5-haiku": "Xenova/claude-tokenizer",
    "claude-3-opus": "Xenova/claude-tokenizer",
    "claude-3-sonnet": "Xenova/claude-tokenizer",
    "claude-3-haiku": "Xenova/claude-tokenizer",
    "llama-4": "Xenova/llama4-tokenizer",
    "llama-3.3": "unsloth/Llama-3.3-70B-Instruct",
    "llama-3.2": "Xenova/Llama-3.2-Tokenizer",
    "llama-3.1": "Xenova/Meta-Llama-3.1-Tokenizer",
    "llama-3": "Xenova/llama3-tokenizer-new",
    "llama-2": "Xenova/llama2-tokenizer",
    "code-llama": "Xenova/llama-code-tokenizer",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v2": "deepseek-ai/DeepSeek-V2",
    "mistral-large": "Xenova/mistral-tokenizer-v3",
    "mistral-small": "Xenova/mistral-tokenizer-v3",
    "mistral-nemo": "Xenova/Mistral-Nemo-Instruct-Tokenizer",
    "codestral": "Xenova/mistral-tokenizer-v3",
    "qwen3.6-27b": "Qwen/Qwen3-32B",
    "qwen3-coder-30b-a3b-instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}

PROVIDER_DEFAULT_HF: dict[str, str] = {
    "anthropic": "Xenova/claude-tokenizer",
    "meta": "Xenova/Meta-Llama-3.1-Tokenizer",
    "mistral": "Xenova/mistral-tokenizer-v3",
    "deepseek": "deepseek-ai/DeepSeek-V3",
}


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    kind: str  # tiktoken | huggingface | approx
    value: str | None = None


@dataclass(frozen=True, slots=True)
class TokenCountResult:
    tokens: int
    approximate: bool
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    provider_id: str
    model_id: str
    tokenizer: TokenizerSpec


@dataclass(frozen=True, slots=True)
class WarmupResult:
    provider_id: str
    model_id: str
    tokenizer_kind: str
    tokenizer_value: str | None
    status: str  # warmed | approximate | failed
    warning: str | None = None


class TokenizerRegistry:
    def __init__(self) -> None:
        self._tiktoken_encoders: dict[str, Any] = {}
        self._hf_tokenizers: dict[str, Any] = {}

    def resolve_model(self, provider_id: str | None, model_id: str | None) -> ResolvedModel:
        provider = _canonicalize(provider_id) or "anthropic"
        model = model_id or "claude-sonnet-4-20250514"

        openai_model = self._resolve_openai_model(provider, model)
        if openai_model is not None:
            return ResolvedModel(provider, model, TokenizerSpec("tiktoken", openai_model))

        hf_model = self._resolve_huggingface_model(provider, model)
        if hf_model is not None:
            return ResolvedModel(provider, model, TokenizerSpec("huggingface", hf_model))

        return ResolvedModel(provider, model, TokenizerSpec("approx", None))

    def is_tokenizer_available(self, provider_id: str | None, model_id: str | None) -> bool:
        """Check if a real tokenizer is available for the model (not approx fallback)."""
        resolved = self.resolve_model(provider_id, model_id)
        if resolved.tokenizer.kind == "approx":
            return False
        if resolved.tokenizer.kind == "tiktoken":
            # tiktoken is always available (bundled package)
            return True
        if resolved.tokenizer.kind == "huggingface":
            # Check if the local tokenizer file exists
            return self._resolve_local_hf_tokenizer_path(resolved.tokenizer.value or "") is not None
        return False

    def count(self, text: str, spec: TokenizerSpec) -> TokenCountResult:
        if not text.strip():
            return TokenCountResult(tokens=0, approximate=False)

        try:
            if spec.kind == "approx":
                return TokenCountResult(tokens=_approximate_count(text), approximate=True, warning="approximate tokenizer fallback")
            if spec.kind == "tiktoken":
                return self._count_tiktoken(text, spec.value or "cl100k_base")
            if spec.kind == "huggingface":
                return self._count_hf(text, spec.value or "Xenova/claude-tokenizer")
        except Exception as exc:
            return TokenCountResult(
                tokens=_approximate_count(text),
                approximate=True,
                warning=f"approximate tokenizer fallback: {exc}",
            )

        return TokenCountResult(tokens=_approximate_count(text), approximate=True, warning="approximate tokenizer fallback")

    def warmup(self, model_pairs: list[tuple[str, str]], sample_text: str = "warmup") -> list[WarmupResult]:
        results: list[WarmupResult] = []
        for provider_id, model_id in model_pairs:
            resolved = self.resolve_model(provider_id, model_id)
            if resolved.tokenizer.kind == "approx":
                results.append(
                    WarmupResult(
                        provider_id=resolved.provider_id,
                        model_id=resolved.model_id,
                        tokenizer_kind=resolved.tokenizer.kind,
                        tokenizer_value=resolved.tokenizer.value,
                        status="approximate",
                        warning="no exact tokenizer mapping, approximate mode",
                    )
                )
                continue

            try:
                count_result = self.count(sample_text, resolved.tokenizer)
                status = "approximate" if count_result.approximate else "warmed"
                results.append(
                    WarmupResult(
                        provider_id=resolved.provider_id,
                        model_id=resolved.model_id,
                        tokenizer_kind=resolved.tokenizer.kind,
                        tokenizer_value=resolved.tokenizer.value,
                        status=status,
                        warning=count_result.warning,
                    )
                )
            except Exception as exc:
                results.append(
                    WarmupResult(
                        provider_id=resolved.provider_id,
                        model_id=resolved.model_id,
                        tokenizer_kind=resolved.tokenizer.kind,
                        tokenizer_value=resolved.tokenizer.value,
                        status="failed",
                        warning=str(exc),
                    )
                )
        return results

    def warmup_parallel(
        self, model_pairs: list[tuple[str, str]], sample_text: str = "warmup", max_workers: int = 4
    ) -> list[WarmupResult]:
        """Warm multiple tokenizers in parallel using ThreadPoolExecutor."""
        results: list[WarmupResult] = []
        futures_map: dict[concurrent.futures.Future, tuple[str, str]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for provider_id, model_id in model_pairs:
                future = executor.submit(self._warmup_single, provider_id, model_id, sample_text)
                futures_map[future] = (provider_id, model_id)

            for future in as_completed(futures_map):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    provider_id, model_id = futures_map[future]
                    resolved = self.resolve_model(provider_id, model_id)
                    results.append(
                        WarmupResult(
                            provider_id=provider_id,
                            model_id=model_id,
                            tokenizer_kind=resolved.tokenizer.kind,
                            tokenizer_value=resolved.tokenizer.value,
                            status="failed",
                            warning=str(exc),
                        )
                    )

        return results

    def _warmup_single(self, provider_id: str, model_id: str, sample_text: str) -> WarmupResult:
        """Warm a single tokenizer model."""
        resolved = self.resolve_model(provider_id, model_id)
        if resolved.tokenizer.kind == "approx":
            return WarmupResult(
                provider_id=provider_id,
                model_id=model_id,
                tokenizer_kind=resolved.tokenizer.kind,
                tokenizer_value=resolved.tokenizer.value,
                status="approximate",
                warning="no exact tokenizer mapping, approximate mode",
            )

        try:
            count_result = self.count(sample_text, resolved.tokenizer)
            status = "approximate" if count_result.approximate else "warmed"
            return WarmupResult(
                provider_id=provider_id,
                model_id=model_id,
                tokenizer_kind=resolved.tokenizer.kind,
                tokenizer_value=resolved.tokenizer.value,
                status=status,
                warning=count_result.warning,
            )
        except Exception as exc:
            return WarmupResult(
                provider_id=provider_id,
                model_id=model_id,
                tokenizer_kind=resolved.tokenizer.kind,
                tokenizer_value=resolved.tokenizer.value,
                status="failed",
                warning=str(exc),
            )

    def _count_tiktoken(self, text: str, model: str) -> TokenCountResult:
        try:
            encoder = self._get_tiktoken_encoder(model)
            token_count = len(encoder.encode(text))
            return TokenCountResult(tokens=token_count, approximate=False)
        except Exception:
            return TokenCountResult(
                tokens=_approximate_count(text),
                approximate=True,
                warning=f"tiktoken unavailable for model '{model}', used approximate fallback",
            )

    def _count_hf(self, text: str, hub: str) -> TokenCountResult:
        try:
            tokenizer = self._get_hf_tokenizer(hub)
            if hasattr(tokenizer, "encode"):
                encoded = tokenizer.encode(text)
                ids = getattr(encoded, "ids", None)
                if isinstance(ids, list):
                    return TokenCountResult(tokens=len(ids), approximate=False)
                if isinstance(encoded, list):
                    return TokenCountResult(tokens=len(encoded), approximate=False)
            if callable(getattr(tokenizer, "__call__", None)):
                encoded = tokenizer(text, add_special_tokens=False)
                input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
                if isinstance(input_ids, list):
                    return TokenCountResult(tokens=len(input_ids), approximate=False)
        except Exception:
            pass
        return TokenCountResult(
            tokens=_approximate_count(text),
            approximate=True,
            warning=f"huggingface tokenizer unavailable for hub '{hub}', used approximate fallback",
        )

    def _get_tiktoken_encoder(self, model: str) -> Any:
        if model in self._tiktoken_encoders:
            return self._tiktoken_encoders[model]

        tiktoken = import_module("tiktoken")
        encoder: Any
        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
        self._tiktoken_encoders[model] = encoder
        return encoder

    def _get_hf_tokenizer(self, hub: str) -> Any:
        if hub in self._hf_tokenizers:
            return self._hf_tokenizers[hub]

        local_path = self._resolve_local_hf_tokenizer_path(hub)
        if local_path is not None:
            tokenizers = import_module("tokenizers")
            if hasattr(tokenizers, "Tokenizer"):
                tokenizer = tokenizers.Tokenizer.from_file(str(local_path))
                self._hf_tokenizers[hub] = tokenizer
                return tokenizer

        transformers_tokenizer = self._load_transformers_tokenizer(hub)
        if transformers_tokenizer is not None:
            self._hf_tokenizers[hub] = transformers_tokenizer
            return transformers_tokenizer

        raise RuntimeError(f"no local tokenizer assets found for hub '{hub}'")

    def _resolve_openai_model(self, provider: str, model_id: str) -> str | None:
        model_key = _canonicalize(model_id)
        if provider in {"openai", "opencode", "azure"}:
            if not model_key:
                return "cl100k_base"
            if model_key.startswith("gpt-5"):
                return OPENAI_MODEL_MAP.get(model_key, "gpt-4o")
            return OPENAI_MODEL_MAP.get(model_key, model_key)
        if model_key and model_key in OPENAI_MODEL_MAP:
            return OPENAI_MODEL_MAP[model_key]
        return None

    def _resolve_local_hf_tokenizer_path(self, hub: str) -> Path | None:
        candidates: list[Path] = []
        base_dir = os.environ.get("OPENCODE_TOKENIZER_CACHE_DIR")
        if base_dir:
            base = Path(os.path.expanduser(os.path.expandvars(base_dir)))
            safe = hub.replace("/", "--")
            candidates.extend(
                [
                    base / hub / "tokenizer.json",
                    base / safe / "tokenizer.json",
                    base / f"{safe}.json",
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        local_model_dirs = os.environ.get("OPENCODE_LOCAL_MODEL_DIRS")
        if local_model_dirs:
            safe = hub.replace("/", "--")
            parts = [p for p in local_model_dirs.split(os.pathsep) if p.strip()]
            for part in parts:
                base = Path(os.path.expanduser(os.path.expandvars(part.strip())))
                extra_candidates = [
                    base / hub / "tokenizer.json",
                    base / safe / "tokenizer.json",
                    base / _model_leaf(hub) / "tokenizer.json",
                ]
                for candidate in extra_candidates:
                    if candidate.exists():
                        return candidate
        return None

    def _resolve_huggingface_model(self, provider: str, model_id: str) -> str | None:
        model_key = _canonicalize(model_id)
        if model_key and model_key in HUGGINGFACE_TOKENIZER_MODEL_MAP:
            return HUGGINGFACE_TOKENIZER_MODEL_MAP[model_key]
        if provider in PROVIDER_DEFAULT_HF:
            return PROVIDER_DEFAULT_HF[provider]
        if model_key and model_key.startswith("claude"):
            return "Xenova/claude-tokenizer"
        if model_key and model_key.startswith("llama"):
            return HUGGINGFACE_TOKENIZER_MODEL_MAP.get(model_key, "Xenova/Meta-Llama-3.1-Tokenizer")
        if model_key and model_key.startswith("mistral"):
            return "Xenova/mistral-tokenizer-v3"
        if model_key and model_key.startswith("deepseek"):
            return "deepseek-ai/DeepSeek-V3"
        if model_key and model_key.startswith("qwen"):
            return os.environ.get("OPENCODE_QWEN_TOKENIZER_HUB", "Qwen/Qwen3-32B")
        return None

    def _load_transformers_tokenizer(self, hub: str) -> Any | None:
        try:
            transformers = import_module("transformers")
            auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
            if auto_tokenizer is None:
                return None
            return auto_tokenizer.from_pretrained(hub, local_files_only=True, trust_remote_code=True)
        except Exception:
            return None


def _approximate_count(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _canonicalize(value: str | None) -> str | None:
    if value is None:
        return None
    return value.split("/")[-1].strip().lower() if value.strip() else None


def _model_leaf(hub: str) -> str:
    return hub.split("/")[-1]
