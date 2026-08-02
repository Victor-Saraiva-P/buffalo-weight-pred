from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext
from buffalo_weight.feature_calculators.crofton import crofton_perimeter


def calculate_convex_area(ctx: FeatureContext) -> float:
    return float(ctx.convex_mask.sum())


def calculate_convexity(ctx: FeatureContext) -> float:
    if not ctx.perimeter:
        return 0.0
    return crofton_perimeter(ctx.convex_mask) / ctx.perimeter
