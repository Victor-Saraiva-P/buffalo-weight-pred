"""Project-owned adapter for SciPy convex geometry operations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]


def mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    boundary = mask & ~binary_erosion(mask)
    ys, xs = np.nonzero(boundary)
    return np.column_stack((xs, ys)).astype(float)


def convex_hull_points(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return points
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return points


def convex_hull_equations(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    try:
        return np.asarray(ConvexHull(points).equations, dtype=float)
    except QhullError:
        return None
