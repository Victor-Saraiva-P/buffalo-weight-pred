from __future__ import annotations

import numpy as np

from buffalo_weight.feature_calculators.context import FeatureContext


def _regional_occupancy(ctx: FeatureContext) -> tuple[float, float, float]:
    _, xs = ctx.nonzero_coords
    left, right = int(xs.min()), int(xs.max()) + 1
    widths = ctx.mask[:, left:right].sum(axis=0).astype(float)
    thirds = np.array_split(widths, 3)
    means = [float(region.mean()) if len(region) else 0.0 for region in thirds]
    return means[0], means[1], means[2]


def calculate_allometric_features(ctx: FeatureContext, major_axis_length: float) -> dict[str, float]:
    """Return volume and regional-shape proxies; for example, ``calculate_allometric_features(ctx, 10)``."""
    start, middle, end = _regional_occupancy(ctx)
    end_min, end_max = sorted((start, end))
    ys, xs = ctx.nonzero_coords
    bbox_width = float(xs.max() - xs.min() + 1)
    bbox_height = float(ys.max() - ys.min() + 1)
    return {
        "area_power_1_5": float(ctx.area**1.5),
        "area_major_axis_product": float(ctx.area * major_axis_length),
        "center_vertical_occupancy": middle,
        "end_vertical_occupancy_min": end_min,
        "end_vertical_occupancy_max": end_max,
        "center_to_end_occupancy_ratio": (
            middle / ((end_min + end_max) / 2) if end_min + end_max else 0.0
        ),
        "centroid_x_offset": abs(float(xs.mean() - xs.min() + 0.5) / bbox_width - 0.5),
        "centroid_y_ratio": float(ys.mean() - ys.min() + 0.5) / bbox_height,
    }
