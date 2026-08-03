from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext
from buffalo_weight.feature_calculators.crofton import crofton_perimeter


def calculate_convex_area(ctx: FeatureContext) -> float:
    """Count hull pixels.

    Example: ``calculate_convex_area(context)`` returns a pixel count.
    """
    return float(ctx.convex_mask.sum())


def calculate_convexity(ctx: FeatureContext) -> float:
    """Compare hull and mask perimeters; for example, ``calculate_convexity(context)``."""
    if not ctx.perimeter:
        return 0.0
    return crofton_perimeter(ctx.convex_mask) / ctx.perimeter
