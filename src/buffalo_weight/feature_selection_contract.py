"""Frozen feature-selection groups and public artifact schemas."""

from __future__ import annotations

from buffalo_weight.feature_evaluation import RemovalGroup

PERMUTATION_COUNT = 10
EVIDENCE_COLUMNS = [
    "experiment", "baseline", "target", "scope", "fold", "repetition",
    "permutation_seed", "n", "reference_mae_kg", "result_mae_kg",
    "delta_mae_kg", "effect",
]
REDUNDANCY_COLUMNS = [
    "feature_a", "feature_b", "structural_relation", "pearson", "spearman",
    "removal_group",
]
REMOVAL_GROUPS = (
    RemovalGroup("area_transformations", (
        "equivalent_diameter", "area_power_1_5", "area_major_axis_product",
    )),
    RemovalGroup("bounding_rectangle_relations", ("bbox_area", "aspect_ratio", "extent")),
    RemovalGroup("equivalent_ellipse_relation", ("roundness",)),
    RemovalGroup("vertical_occupancy_relation", ("center_to_end_occupancy_ratio",)),
    RemovalGroup("convex_hull_relations", ("solidity", "convexity")),
    RemovalGroup("area_contour_relation", ("circularity",)),
)
REDUNDANCY_FAMILIES = {
    "area_transformations": (
        "area", "equivalent_diameter", "area_power_1_5", "area_major_axis_product",
        "major_axis_length",
    ),
    "bounding_rectangle_relations": (
        "area", "bbox_width", "bbox_height", "bbox_area", "aspect_ratio", "extent",
    ),
    "equivalent_ellipse_relation": ("area", "major_axis_length", "minor_axis_length", "roundness"),
    "vertical_occupancy_relation": (
        "center_vertical_occupancy", "end_vertical_occupancy_min",
        "end_vertical_occupancy_max", "center_to_end_occupancy_ratio",
    ),
    "convex_hull_relations": ("area", "perimeter", "convex_area", "solidity", "convexity"),
    "area_contour_relation": ("area", "perimeter", "circularity"),
}
