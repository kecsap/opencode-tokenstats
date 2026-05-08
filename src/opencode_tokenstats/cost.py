from __future__ import annotations

from dataclasses import dataclass

from .pricing import ModelPricing, PricingLookup, estimate_session_cost_usd
from .telemetry import TelemetrySummary


@dataclass(frozen=True, slots=True)
class CostSummary:
    api_session_cost: float
    estimated_session_cost: float
    estimated_input_cost: float
    estimated_output_cost: float
    estimated_cache_read_cost: float
    estimated_cache_write_cost: float
    is_subscription: bool


def calculate_cost_summary(
    telemetry: TelemetrySummary,
    *,
    model_name: str,
    pricing_lookup: PricingLookup,
) -> CostSummary:
    pricing = pricing_lookup.get_pricing(model_name)

    estimated_input_cost = (telemetry.input_tokens / 1_000_000) * pricing.input
    estimated_output_cost = (
        (telemetry.output_tokens + telemetry.reasoning_tokens) / 1_000_000
    ) * pricing.output
    estimated_cache_read_cost = (telemetry.cache_read_tokens / 1_000_000) * pricing.cache_read
    estimated_cache_write_cost = (telemetry.cache_write_tokens / 1_000_000) * pricing.cache_write
    estimated_session_cost = estimate_session_cost_usd(
        pricing,
        input_tokens=telemetry.input_tokens,
        output_tokens=telemetry.output_tokens,
        reasoning_tokens=telemetry.reasoning_tokens,
        cache_read_tokens=telemetry.cache_read_tokens,
        cache_write_tokens=telemetry.cache_write_tokens,
    )

    has_activity = telemetry.api_calls > 0 and (
        telemetry.input_tokens > 0
        or telemetry.output_tokens > 0
        or telemetry.reasoning_tokens > 0
        or telemetry.cache_read_tokens > 0
        or telemetry.cache_write_tokens > 0
    )
    is_subscription = has_activity and telemetry.total_cost == 0

    return CostSummary(
        api_session_cost=telemetry.total_cost,
        estimated_session_cost=estimated_session_cost,
        estimated_input_cost=estimated_input_cost,
        estimated_output_cost=estimated_output_cost,
        estimated_cache_read_cost=estimated_cache_read_cost,
        estimated_cache_write_cost=estimated_cache_write_cost,
        is_subscription=is_subscription,
    )


def build_default_pricing_lookup() -> PricingLookup:
    return PricingLookup(
        {
            "default": ModelPricing(input=1.0, output=3.0, cache_write=0.0, cache_read=0.0)
        }
    )
