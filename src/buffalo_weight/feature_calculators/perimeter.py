from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_perimeter(ctx: FeatureContext) -> float:
    """Return Crofton perimeter.

    Example: ``calculate_perimeter(context)`` measures its contour.
    """
    return ctx.perimeter
