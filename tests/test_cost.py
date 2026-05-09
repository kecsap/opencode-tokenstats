from __future__ import annotations

from opencode_tokenstats.cost import calculate_cost_summary
from opencode_tokenstats.pricing import ModelPricing, PricingLookup
from opencode_tokenstats.telemetry import TelemetrySummary


def test_calculate_cost_summary_components() -> None:
    telemetry = TelemetrySummary(
        input_tokens=1_000_000,
        output_tokens=500_000,
        reasoning_tokens=500_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        web_search_requests=2,
        api_calls=1,
        total_cost=2.0,
    )
    lookup = PricingLookup(
        {
            "gpt-5": ModelPricing(input=2.0, output=8.0, cache_read=0.5, cache_write=2.5, web_search=0.01),
            "default": ModelPricing(input=1.0, output=3.0, cache_read=0.0),
        }
    )

    summary = calculate_cost_summary(telemetry, model_name="gpt-5", pricing_lookup=lookup)

    assert summary.api_session_cost == 2.0
    assert summary.estimated_input_cost == 2.0
    assert summary.estimated_output_cost == 8.0
    assert summary.estimated_cache_read_cost == 0.5
    assert summary.estimated_cache_write_cost == 2.5
    assert summary.estimated_web_search_cost == 0.02
    assert summary.estimated_session_cost == 13.02
    assert summary.is_subscription is False


def test_calculate_cost_summary_subscription_detection() -> None:
    telemetry = TelemetrySummary(
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=0,
        cache_read_tokens=0,
        api_calls=1,
        total_cost=0.0,
    )
    lookup = PricingLookup(
        {
            "default": ModelPricing(input=1.0, output=3.0, cache_read=0.0),
        }
    )

    summary = calculate_cost_summary(telemetry, model_name="unknown", pricing_lookup=lookup)
    assert summary.is_subscription is True
