from __future__ import annotations

from buffalo_weight.feature_calculators.area import calculate_area
from buffalo_weight.feature_calculators.circularity import calculate_circularity
from buffalo_weight.feature_calculators.equivalent_diameter import (
    calculate_equivalent_diameter,
)
from buffalo_weight.feature_calculators.geometry import mask_pixels
from buffalo_weight.feature_calculators.hu_moments import calculate_hu_moments
from buffalo_weight.feature_calculators.perimeter import calculate_perimeter
from buffalo_weight.feature_calculators.solidity import calculate_solidity


def zero_features() -> dict[str, float]:
    return {
        "area": 0,
        "perimeter": 0,
        "solidity": 0,
        "circularity": 0,
        "equivalent_diameter": 0,
        "hu_moment_1": 0,
        "hu_moment_2": 0,
    }


def calculate_mask_features(mask: list[list[bool]]) -> dict[str, float]:
    pixels = mask_pixels(mask)
    area = calculate_area(pixels)
    if area == 0:
        return zero_features()

    perimeter = calculate_perimeter(mask, pixels)
    hu_moment_1, hu_moment_2 = calculate_hu_moments(pixels, area)

    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": calculate_solidity(pixels, area),
        "circularity": calculate_circularity(area, perimeter),
        "equivalent_diameter": calculate_equivalent_diameter(area),
        "hu_moment_1": hu_moment_1,
        "hu_moment_2": hu_moment_2,
    }
