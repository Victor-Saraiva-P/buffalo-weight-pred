from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_solidity(ctx: FeatureContext) -> float:
    area = ctx.area
    _, hull_area, _ = ctx.hull_data
    return area / hull_area if hull_area else 0
