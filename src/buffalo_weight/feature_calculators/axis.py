from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_major_axis_length(ctx: FeatureContext) -> float:
    return 4 * ctx.moments_data["major_variance"] ** 0.5


def calculate_minor_axis_length(ctx: FeatureContext) -> float:
    return 4 * ctx.moments_data["minor_variance"] ** 0.5
