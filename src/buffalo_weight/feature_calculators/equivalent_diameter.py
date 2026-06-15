from __future__ import annotations

import math


def calculate_equivalent_diameter(area: int) -> float:
    return math.sqrt(4 * area / math.pi) if area else 0
