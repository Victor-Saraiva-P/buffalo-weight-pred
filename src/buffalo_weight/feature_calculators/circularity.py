from __future__ import annotations

import math
from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_circularity(ctx: FeatureContext) -> float:
    area = ctx.area
    perimeter = ctx.perimeter
    return 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0
