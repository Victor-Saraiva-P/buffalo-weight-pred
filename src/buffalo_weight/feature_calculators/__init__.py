from __future__ import annotations

import math
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

APPROVED_FEATURES = (
    "area", "perimeter", "solidity", "circularity", "equivalent_diameter",
    "bbox_width", "bbox_height", "bbox_area", "aspect_ratio", "extent",
    "convex_area", "convexity", "major_axis_length", "minor_axis_length",
    "roundness", "feret_diameter", "hu_moment_1", "hu_moment_2",
    "area_power_1_5", "area_major_axis_product", "center_vertical_occupancy",
    "end_vertical_occupancy_min", "end_vertical_occupancy_max",
    "center_to_end_occupancy_ratio", "centroid_x_offset", "centroid_y_ratio",
)

LENGTH_FEATURES = {
    "perimeter", "equivalent_diameter", "bbox_width", "bbox_height",
    "major_axis_length", "minor_axis_length", "feret_diameter",
    "center_vertical_occupancy", "end_vertical_occupancy_min",
    "end_vertical_occupancy_max",
}
AREA_FEATURES = {"area", "bbox_area", "convex_area"}
VOLUME_FEATURES = {"area_power_1_5", "area_major_axis_product"}


def zero_features() -> dict[str, float]:
    return {name: 0.0 for name in APPROVED_FEATURES}


def calculate_mask_features(
    mask: np.ndarray | Path | str, canonical_long_side: int = 1024
) -> dict[str, float]:
    """Calculate the approved geometry; for example, ``calculate_mask_features(mask)``."""
    ctx = FeatureContext(mask)
    if ctx.area == 0:
        return zero_features()
    if canonical_long_side <= 0:
        raise ValueError(
            f"canonical_long_side was {canonical_long_side}; expected an integer greater than 0"
        )
    major_axis_length = calculate_major_axis_length(ctx)
    raw_features = _primary_features(ctx, major_axis_length)
    raw_features.update(_derived_features(ctx, major_axis_length))
    scale = canonical_long_side / max(ctx.mask.shape)
    return _scale_features(raw_features, scale)


def _primary_features(ctx: FeatureContext, major_axis_length: float) -> dict[str, float]:
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
    }


def _derived_features(ctx: FeatureContext, major_axis_length: float) -> dict[str, float]:
    return {
        "roundness": 4 * ctx.area / (math.pi * major_axis_length**2),
        "feret_diameter": _feret_diameter(ctx.hull_points),
        "hu_moment_1": calculate_hu_moment_1(ctx),
        "hu_moment_2": calculate_hu_moment_2(ctx),
        **calculate_allometric_features(ctx, major_axis_length),
    }


def _feret_diameter(hull_points: np.ndarray) -> float:
    if len(hull_points) < 2:
        return 0.0
    differences = hull_points[:, None, :] - hull_points[None, :, :]
    return float(np.sqrt((differences**2).sum(axis=2)).max())


def _scale_features(features: dict[str, float], scale: float) -> dict[str, float]:
    scaled = features.copy()
    for name in LENGTH_FEATURES:
        scaled[name] *= scale
    for name in AREA_FEATURES:
        scaled[name] *= scale**2
    for name in VOLUME_FEATURES:
        scaled[name] *= scale**3
    return scaled
