from __future__ import annotations

import math
from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_equivalent_diameter(ctx: FeatureContext) -> float:
    area = ctx.area
    return math.sqrt(4 * area / math.pi) if area else 0
