from __future__ import annotations

from buffalo_weight.feature_calculators.geometry import convex_hull, polygon_area


def calculate_solidity(pixels: list[tuple[int, int]], area: int) -> float:
    corners = []
    for x, y in pixels:
        corners.extend([(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)])
    hull_area = polygon_area(convex_hull(corners))
    return area / hull_area if hull_area else 0
