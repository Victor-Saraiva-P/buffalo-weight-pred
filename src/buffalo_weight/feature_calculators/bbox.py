from __future__ import annotations

from buffalo_weight.feature_calculators.context import FeatureContext


def calculate_bbox_width(ctx: FeatureContext) -> int:
    ys, xs = ctx.nonzero_coords
    return int(xs.max() - xs.min() + 1)


def calculate_bbox_height(ctx: FeatureContext) -> int:
    ys, xs = ctx.nonzero_coords
    return int(ys.max() - ys.min() + 1)


def calculate_bbox_area(ctx: FeatureContext) -> int:
    return calculate_bbox_width(ctx) * calculate_bbox_height(ctx)


def calculate_aspect_ratio(ctx: FeatureContext) -> float:
    width = calculate_bbox_width(ctx)
    height = calculate_bbox_height(ctx)
    return width / height if height else 0


def calculate_extent(ctx: FeatureContext) -> float:
    bbox_area = calculate_bbox_area(ctx)
    return ctx.area / bbox_area if bbox_area else 0
