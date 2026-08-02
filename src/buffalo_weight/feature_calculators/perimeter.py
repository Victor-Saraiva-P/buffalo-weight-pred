from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_perimeter(ctx: FeatureContext) -> float:
    return ctx.perimeter
