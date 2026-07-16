from __future__ import annotations

import numpy as np

from buffalo_weight.feature_calculators.context import FeatureContext


def _resampled_widths(ctx: FeatureContext, sample_count: int = 9) -> np.ndarray:
    _, xs = ctx.nonzero_coords
    left, right = int(xs.min()), int(xs.max()) + 1
    widths = ctx.mask[:, left:right].sum(axis=0).astype(float)
    positions = np.linspace(0, max(len(widths) - 1, 0), sample_count)
    return np.interp(positions, np.arange(len(widths)), widths)


def calculate_allometric_features(ctx: FeatureContext, major_axis_length: float) -> dict[str, float]:
    """Return volume and regional-shape proxies; for example, ``calculate_allometric_features(ctx, 10)``."""
    sampled = _resampled_widths(ctx)
    start, middle, end = (float(region.mean()) for region in np.array_split(sampled, 3))
    end_min, end_max = sorted((start, end))
    ys, xs = ctx.nonzero_coords
    bbox_width = float(xs.max() - xs.min() + 1)
    bbox_height = float(ys.max() - ys.min() + 1)
    return {
        "area_power_1_5": float(ctx.area**1.5),
        "area_major_axis_product": float(ctx.area * major_axis_length),
        "middle_thickness": middle,
        "end_thickness_min": end_min,
        "end_thickness_max": end_max,
        "middle_to_end_ratio": middle / ((end_min + end_max) / 2) if end_min + end_max else 0.0,
        "centroid_x_offset": abs(float(xs.mean() - xs.min() + 0.5) / bbox_width - 0.5),
        "centroid_y_ratio": float(ys.mean() - ys.min() + 0.5) / bbox_height,
    }
