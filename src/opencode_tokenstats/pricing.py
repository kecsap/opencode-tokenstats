from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input: float
    output: float
    cache_write: float
    cache_read: float


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
        return self.pricing_data.get("default", ModelPricing(input=1, output=3, cache_write=0, cache_read=0))

    def has_pricing(self, model_name: str) -> bool:
        return self._find_pricing(model_name) is not None

    def _find_pricing(self, model_name: str) -> ModelPricing | None:
        raw_name = model_name.strip().lower()
        if not raw_name:
            return None

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
    cache_write_tokens: int,
) -> float:
    input_cost = (max(0, input_tokens) / 1_000_000) * pricing.input
    output_cost = ((max(0, output_tokens) + max(0, reasoning_tokens)) / 1_000_000) * pricing.output
    cache_read_cost = (max(0, cache_read_tokens) / 1_000_000) * pricing.cache_read
    cache_write_cost = (max(0, cache_write_tokens) / 1_000_000) * pricing.cache_write
    return input_cost + output_cost + cache_read_cost + cache_write_cost
