"""Project-owned adapter for SciPy convex geometry operations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]


def mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Find digital boundary centers; for example, ``mask_boundary_points(mask)``."""
    boundary = mask & ~binary_erosion(mask)
    ys, xs = np.nonzero(boundary)
    return np.column_stack((xs, ys)).astype(float)


def convex_hull_points(points: np.ndarray) -> np.ndarray:
    """Select hull vertices; for example, ``convex_hull_points(boundary)``."""
    if len(points) < 3:
        return points
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return points


def convex_hull_equations(points: np.ndarray) -> np.ndarray | None:
    """Describe hull half-spaces; for example, ``convex_hull_equations(vertices)``."""
    if len(points) < 3:
        return None
    try:
        return np.asarray(ConvexHull(points).equations, dtype=float)
    except QhullError:
        return None
