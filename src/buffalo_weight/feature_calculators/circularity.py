from __future__ import annotations

import math


def calculate_circularity(area: int, perimeter: int) -> float:
    return 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0
