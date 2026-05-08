"""OpenCode TokenStats package."""

from .client import ApiClientError, OpencodeApiClient
from .canonical_metrics import CanonicalMetrics, build_canonical_metrics
from .compatibility import (
    CompatMode,
    CompatibilityResult,
    ToolSchemaEstimate,
    analyze_context_compatibility,
    load_export_debug_session,
)
from .content_attribution import (
    ApproxTokenCounter,
    CategoryTotals,
    ContentAttribution,
    ToolUsageStat,
    collect_content_attribution,
    collect_content_attribution_for_model,
)
from .pricing import ModelPricing, PricingLookup, estimate_session_cost_usd
from .cost import CostSummary, build_default_pricing_lookup, calculate_cost_summary
from .telemetry import (
    SessionTelemetryReport,
    TelemetrySchemaError,
    TelemetryCall,
    TelemetrySummary,
    collect_telemetry_calls,
    summarize_session_with_subagents,
    summarize_telemetry,
)
from .tokenization import ResolvedModel, TokenCountResult, TokenizerRegistry, TokenizerSpec

__all__ = [
    "OpencodeApiClient",
    "ApiClientError",
    "CanonicalMetrics",
    "build_canonical_metrics",
    "CompatMode",
    "ToolSchemaEstimate",
    "CompatibilityResult",
    "analyze_context_compatibility",
    "load_export_debug_session",
    "ApproxTokenCounter",
    "CategoryTotals",
    "ToolUsageStat",
    "ContentAttribution",
    "collect_content_attribution",
    "collect_content_attribution_for_model",
    "TokenizerSpec",
    "TokenCountResult",
    "ResolvedModel",
    "TokenizerRegistry",
    "ModelPricing",
    "PricingLookup",
    "estimate_session_cost_usd",
    "CostSummary",
    "calculate_cost_summary",
    "build_default_pricing_lookup",
    "TelemetryCall",
    "TelemetrySummary",
    "SessionTelemetryReport",
    "TelemetrySchemaError",
    "collect_telemetry_calls",
    "summarize_telemetry",
    "summarize_session_with_subagents",
]
