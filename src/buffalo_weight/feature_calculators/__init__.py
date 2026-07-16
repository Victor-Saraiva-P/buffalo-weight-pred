from __future__ import annotations

from pathlib import Path

import numpy as np

from buffalo_weight.feature_calculators.area import calculate_area
from buffalo_weight.feature_calculators.allometry import calculate_allometric_features
from buffalo_weight.feature_calculators.axis import (
    calculate_major_axis_length,
    calculate_minor_axis_length,
)
from buffalo_weight.feature_calculators.bbox import (
    calculate_aspect_ratio,
    calculate_bbox_area,
    calculate_bbox_height,
    calculate_bbox_width,
    calculate_extent,
)
from buffalo_weight.feature_calculators.circularity import calculate_circularity
from buffalo_weight.feature_calculators.context import FeatureContext
from buffalo_weight.feature_calculators.convexity import (
    calculate_convex_area,
    calculate_convexity,
)
from buffalo_weight.feature_calculators.equivalent_diameter import (
    calculate_equivalent_diameter,
)
from buffalo_weight.feature_calculators.hu_moments import (
    calculate_hu_moment_1,
    calculate_hu_moment_2,
)
from buffalo_weight.feature_calculators.perimeter import calculate_perimeter
from buffalo_weight.feature_calculators.solidity import calculate_solidity


def zero_features() -> dict[str, float]:
    return {
        "area": 0,
        "perimeter": 0,
        "solidity": 0,
        "circularity": 0,
        "equivalent_diameter": 0,
        "bbox_width": 0,
        "bbox_height": 0,
        "bbox_area": 0,
        "aspect_ratio": 0,
        "extent": 0,
        "convex_area": 0,
        "convexity": 0,
        "major_axis_length": 0,
        "minor_axis_length": 0,
        "hu_moment_1": 0,
        "hu_moment_2": 0,
        "area_power_1_5": 0,
        "area_major_axis_product": 0,
        "middle_thickness": 0,
        "end_thickness_min": 0,
        "end_thickness_max": 0,
        "middle_to_end_ratio": 0,
        "centroid_x_offset": 0,
        "centroid_y_ratio": 0,
    }


def calculate_mask_features(mask: np.ndarray | Path | str) -> dict[str, float]:
    ctx = FeatureContext(mask)

    if ctx.area == 0:
        return zero_features()

    major_axis_length = calculate_major_axis_length(ctx)
    return {
        "area": calculate_area(ctx),
        "perimeter": calculate_perimeter(ctx),
        "solidity": calculate_solidity(ctx),
        "circularity": calculate_circularity(ctx),
        "equivalent_diameter": calculate_equivalent_diameter(ctx),
        "bbox_width": calculate_bbox_width(ctx),
        "bbox_height": calculate_bbox_height(ctx),
        "bbox_area": calculate_bbox_area(ctx),
        "aspect_ratio": calculate_aspect_ratio(ctx),
        "extent": calculate_extent(ctx),
        "convex_area": calculate_convex_area(ctx),
        "convexity": calculate_convexity(ctx),
        "major_axis_length": major_axis_length,
        "minor_axis_length": calculate_minor_axis_length(ctx),
        "hu_moment_1": calculate_hu_moment_1(ctx),
        "hu_moment_2": calculate_hu_moment_2(ctx),
        **calculate_allometric_features(ctx, major_axis_length),
    }
