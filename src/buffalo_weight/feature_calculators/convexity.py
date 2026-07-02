from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_convex_area(ctx: FeatureContext) -> float:
    _, hull_area, _ = ctx.hull_data
    return hull_area


def calculate_convexity(ctx: FeatureContext) -> float:
    _, _, hull_perimeter = ctx.hull_data
    perimeter = ctx.perimeter
    return hull_perimeter / perimeter if perimeter else 0
