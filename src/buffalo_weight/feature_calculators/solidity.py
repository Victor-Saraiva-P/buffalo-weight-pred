from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_solidity(ctx: FeatureContext) -> float:
    area = ctx.area
    hull_area = float(ctx.convex_mask.sum())
    return area / hull_area if hull_area else 0
