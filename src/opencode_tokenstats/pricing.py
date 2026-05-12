from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import fnmatch


def load_model_aliases(file_path: str | None = None) -> dict[str, str]:
    """Load model ID aliases from models.conf file.

    Format: one alias per line, e.g. 'alias_name = openai/gpt-5.4'
    Or grouped: 'alias_name = azure/gpt-5.4 openai/gpt-5.4'
    Wildcards: 'alias_name = myprovider/qwen*' matches myprovider/qwen3.6-27b, etc.

    Search locations (in order):
    1. Explicit file_path parameter
    2. OPTOKEN_MODEL_ALIAS_FILE environment variable
    3. Current working directory: models.conf (only if no explicit path or env var set)
    """
    candidates: list[Path] = []
    explicit_path = False

    if file_path:
        candidates.append(Path(file_path))
        explicit_path = True

    env_file = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE", "")
    if env_file:
        candidates.append(Path(env_file))
        explicit_path = True

    # Only check default models.conf if no explicit path or env var set
    if not explicit_path:
        candidates.append(Path.cwd() / "models.conf")

    aliases: dict[str, str] = {}
    wildcard_aliases: list[tuple[str, str]] = []  # List of (pattern, alias) for wildcard matching
    for conf_path in candidates:
        if conf_path.exists():
            try:
                for line in conf_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        # Strip @local prefix from alias name
                        alias_name = key
                        if alias_name.startswith("@local "):
                            alias_name = alias_name[len("@local "):]
                        for raw_id in val.split():
                            raw_id = raw_id.strip()
                            if raw_id:
                                if "*" in raw_id:
                                    wildcard_aliases.append((raw_id, alias_name))
                                else:
                                    aliases[raw_id] = alias_name
            except Exception:
                pass
            break
    # Prepend wildcard patterns with a marker so they can be distinguished
    for pattern, alias in wildcard_aliases:
        aliases[f"*{pattern}"] = alias
    return aliases


def resolve_alias(model_id: str, aliases: dict[str, str]) -> str:
    """Resolve a model ID to its alias, supporting wildcard patterns.

    Exact matches take precedence over wildcard matches.
    Wildcard patterns use * as a glob-like matcher (matches any suffix).
    """
    # Check exact match first
    if model_id in aliases:
        return aliases[model_id]

    # Check wildcard matches (glob semantics via fnmatch)
    for key, alias in aliases.items():
        if key.startswith("*"):
            pattern = key[1:]  # Remove the * marker
            if fnmatch.fnmatch(model_id, pattern):
                return alias

    return model_id


def load_local_model_patterns(file_path: str | None = None) -> list[str]:
    """Load local model wildcard patterns from models.conf file.

    Format: lines starting with '@local ' followed by space-separated patterns.
    Patterns support * wildcard, e.g.:
      @local myollama/* myllamacpp/* *qwen36*

    Search locations (in order):
    1. Explicit file_path parameter
    2. OPTOKEN_MODEL_ALIAS_FILE environment variable
    3. Current working directory: models.conf
    """
    candidates: list[Path] = []

    if file_path:
        candidates.append(Path(file_path))

    env_file = os.environ.get("OPTOKEN_MODEL_ALIAS_FILE", "")
    if env_file:
        candidates.append(Path(env_file))

    candidates.append(Path.cwd() / "models.conf")

    patterns: list[str] = []
    for conf_path in candidates:
        if conf_path.exists():
            try:
                for line in conf_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("@local "):
                        for pat in line[len("@local "):].split():
                            pat = pat.strip()
                            if pat:
                                patterns.append(pat)
            except Exception:
                pass
            break
    return patterns


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input: float
    output: float
    cache_read: float
    cache_write: float = 0.0
    web_search: float = 0.0
    fast_multiplier: float = 1.0


class PricingLookup:
    def __init__(self, pricing_data: dict[str, ModelPricing]) -> None:
        self.pricing_data = pricing_data

    @staticmethod
    def build_lookup_key(provider_id: str | None, model_id: str | None) -> str:
        provider = (provider_id or "").strip()
        model = (model_id or "").strip()
        if not provider:
            return model
        if not model:
            return provider

        provider_l = provider.lower()
        model_l = model.lower()
        if model_l.startswith(f"{provider_l}/"):
            return model
        return f"{provider}/{model}"

    def get_pricing(self, model_name: str) -> ModelPricing:
        found = self._find_pricing(model_name)
        if found is not None:
            return found
        return self.pricing_data.get("default", ModelPricing(input=1, output=3, cache_read=0, cache_write=0, web_search=0, fast_multiplier=1))

    def has_pricing(self, model_name: str) -> bool:
        return self._find_pricing(model_name) is not None

    def _find_pricing(self, model_name: str) -> ModelPricing | None:
        raw_name = model_name.strip().lower()
        if not raw_name:
            return None

        for key in canonical_model_keys(raw_name):
            direct = self.pricing_data.get(key)
            if direct is not None:
                return direct

        exact = self.pricing_data.get(raw_name)
        if exact is not None:
            return exact

        normalized = self._normalize_model_name(raw_name)
        normalized_exact = self.pricing_data.get(normalized)
        if normalized_exact is not None:
            return normalized_exact

        return self._longest_prefix_match(raw_name) or self._longest_prefix_match(normalized)

    def _longest_prefix_match(self, model_name: str) -> ModelPricing | None:
        best_len = -1
        best: ModelPricing | None = None

        for key, pricing in self.pricing_data.items():
            key_l = key.lower()
            if model_name.startswith(key_l) and len(key_l) > best_len:
                best_len = len(key_l)
                best = pricing

        return best

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            return model_name.split("/")[-1].strip().lower()
        return model_name.strip().lower()


def estimate_session_cost_usd(
    pricing: ModelPricing,
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int = 0,
    web_search_requests: int = 0,
) -> float:
    input_cost = (max(0, input_tokens) / 1_000_000) * pricing.input
    output_cost = ((max(0, output_tokens) + max(0, reasoning_tokens)) / 1_000_000) * pricing.output
    cache_read_cost = (max(0, cache_read_tokens) / 1_000_000) * pricing.cache_read
    cache_write_cost = (max(0, cache_write_tokens) / 1_000_000) * pricing.cache_write
    web_search_cost = max(0, web_search_requests) * pricing.web_search
    return input_cost + output_cost + cache_read_cost + cache_write_cost + web_search_cost


def canonical_model_keys(model: str) -> list[str]:
    raw = (model or "").strip().lower()
    if not raw:
        return []
    keys = [raw]
    if "/" in raw:
        provider, bare = raw.split("/", 1)
        keys.extend([bare, provider + "/" + bare])
    else:
        keys.extend([f"openai/{raw}", f"azure/{raw}"])
    out: list[str] = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def load_pricing_lookup() -> PricingLookup:
    candidates: list[Path] = []
    env = os.environ.get("OPENCODE_MODEL_PRICING_FILE")
    if env:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(env))))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "opencode-tokenscope" / "plugin" / "models.json")
    candidates.append(repo_root / "models.json")

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            data: dict[str, ModelPricing] = {}
            for key, val in payload.items():
                if not isinstance(key, str) or not isinstance(val, dict):
                    continue
                data[key.lower()] = ModelPricing(
                    input=float(val.get("input", 0) or 0),
                    output=float(val.get("output", 0) or 0),
                    cache_read=float(val.get("cacheRead", val.get("cache_read", 0)) or 0),
                    cache_write=float(val.get("cacheWrite", val.get("cache_write", 0)) or 0),
                    web_search=float(val.get("webSearch", val.get("web_search", 0)) or 0),
                    fast_multiplier=float(val.get("fastMultiplier", val.get("fast_multiplier", 1)) or 1),
                )
            data.setdefault("default", ModelPricing(input=1.0, output=3.0, cache_read=0.0, cache_write=0.0, web_search=0.0, fast_multiplier=1.0))
            return PricingLookup(data)
        except Exception:
            continue

    return PricingLookup({"default": ModelPricing(input=1.0, output=3.0, cache_read=0.0, cache_write=0.0, web_search=0.0, fast_multiplier=1.0)})
