from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from buffalo_weight.feature_calculators.crofton import crofton_perimeter
from buffalo_weight.feature_calculators.scipy_geometry import (
    convex_hull_equations,
    convex_hull_points,
    mask_boundary_points,
)


class FeatureContext:
    def __init__(self, mask: np.ndarray | Path | str) -> None:
        if isinstance(mask, (Path, str)):
            from PIL import Image

            loaded = np.asarray(Image.open(mask).convert("L"))
            self._mask = loaded > 0
        else:
            self._mask = np.asarray(mask, dtype=bool)

        if self._mask.ndim != 2:
            raise ValueError(f"mask shape was {self._mask.shape}; expected a two-dimensional array")

        self.area = int(self._mask.sum())

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @functools.cached_property
    def perimeter(self) -> float:
        return crofton_perimeter(self._mask)

    @functools.cached_property
    def hull_points(self) -> np.ndarray:
        return convex_hull_points(mask_boundary_points(self._mask))

    @functools.cached_property
    def convex_mask(self) -> np.ndarray:
        if self.area == 0:
            return np.zeros((0, 0), dtype=bool)
        ys, xs = self.nonzero_coords
        top, left = int(ys.min()), int(xs.min())
        bottom, right = int(ys.max()), int(xs.max())
        cropped = self._mask[top : bottom + 1, left : right + 1]
        return _rasterize_hull(self.hull_points, cropped, left, top)

    @functools.cached_property
    def nonzero_coords(self) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(self._mask)
        return ys, xs

    @functools.cached_property
    def moments_data(self) -> dict[str, float]:
        ys, xs = self.nonzero_coords
        x_values = xs.astype(float) + 0.5
        y_values = ys.astype(float) + 0.5
        dx = x_values - float(x_values.mean())
        dy = y_values - float(y_values.mean())
        normalized = _normalized_moments(dx, dy, float(self.area))
        return {**normalized, **_ellipse_variances(dx, dy, float(self.area))}


def _normalized_moments(
    dx: np.ndarray, dy: np.ndarray, pixel_count: float
) -> dict[str, float]:
    denominator = pixel_count**2
    return {
        "eta20": float((dx**2).sum() / denominator),
        "eta02": float((dy**2).sum() / denominator),
        "eta11": float((dx * dy).sum() / denominator),
    }


def _ellipse_variances(
    dx: np.ndarray, dy: np.ndarray, pixel_count: float
) -> dict[str, float]:
    mu20 = float((dx**2).sum() / pixel_count)
    mu02 = float((dy**2).sum() / pixel_count)
    mu11 = float((dx * dy).sum() / pixel_count)
    major, minor = _covariance_eigenvalues(mu20, mu02, mu11)
    return {"major_variance": major, "minor_variance": minor}


def _covariance_eigenvalues(
    mu20: float, mu02: float, mu11: float
) -> tuple[float, float]:
    trace = mu20 + mu02
    determinant = mu20 * mu02 - mu11**2
    discriminant = max(trace**2 / 4 - determinant, 0)
    major_variance = trace / 2 + discriminant**0.5
    minor_variance = max(trace / 2 - discriminant**0.5, 0)
    return major_variance, minor_variance


def _rasterize_hull(
    hull_points: np.ndarray, cropped_mask: np.ndarray, left: int, top: int
) -> np.ndarray:
    if len(hull_points) < 3:
        return cropped_mask.copy()
    equations = convex_hull_equations(hull_points)
    if equations is None:
        return cropped_mask.copy()
    return _pixels_inside_hull(cropped_mask.shape, equations, left, top)


def _pixels_inside_hull(
    shape: tuple[int, int], equations: np.ndarray, left: int, top: int
) -> np.ndarray:
    height, width = shape
    convex_mask: np.ndarray = np.zeros(shape, dtype=bool)
    x_values: np.ndarray = np.arange(left, left + width, dtype=float)
    for local_y in range(height):
        y_value = float(top + local_y)
        values = equations[:, 0, None] * x_values + equations[:, 1, None] * y_value
        convex_mask[local_y] = np.all(values + equations[:, 2, None] <= 1e-9, axis=0)
    return convex_mask
