from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from buffalo_weight.feature_calculators.geometry import (
    convex_hull,
    polygon_area,
    polygon_perimeter,
)


class FeatureContext:
    def __init__(self, mask: np.ndarray | Path | str) -> None:
        if isinstance(mask, (Path, str)):
            from PIL import Image

            self._mask = np.asarray(Image.open(mask).convert("L")) > 0
        else:
            self._mask = mask

        self.area = int(self._mask.sum())

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @functools.cached_property
    def perimeter(self) -> int:
        padded = np.pad(self._mask, 1, constant_values=False)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        return int(
            ((center & ~up).sum())
            + ((center & ~down).sum())
            + ((center & ~left).sum())
            + ((center & ~right).sum())
        )

    @functools.cached_property
    def hull_data(self) -> tuple[list[tuple[float, float]], float, float]:
        padded = np.pad(self._mask, 1, constant_values=False)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        boundary = center & (~up | ~down | ~left | ~right)
        boundary_y, boundary_x = np.nonzero(boundary)
        corners = []
        for x, y in zip(boundary_x.tolist(), boundary_y.tolist(), strict=True):
            corners.extend([(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)])
        hull = convex_hull(corners)
        hull_area = polygon_area(hull)
        hull_perimeter = polygon_perimeter(hull)
        return hull, hull_area, hull_perimeter

    @functools.cached_property
    def nonzero_coords(self) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(self._mask)
        return ys, xs

    @functools.cached_property
    def moments_data(self) -> dict[str, float]:
        ys, xs = self.nonzero_coords
        x_values = xs.astype(float) + 0.5
        y_values = ys.astype(float) + 0.5
        m00 = float(self.area)
        cx = float(x_values.sum() / m00)
        cy = float(y_values.sum() / m00)
        dx = x_values - cx
        dy = y_values - cy
        eta20 = float((dx**2).sum() / (m00**2))
        eta02 = float((dy**2).sum() / (m00**2))
        eta11 = float((dx * dy).sum() / (m00**2))
        mu20 = float((dx**2).sum() / m00)
        mu02 = float((dy**2).sum() / m00)
        mu11 = float((dx * dy).sum() / m00)
        trace = mu20 + mu02
        determinant = mu20 * mu02 - mu11**2
        discriminant = max(trace**2 / 4 - determinant, 0)
        major_variance = trace / 2 + discriminant**0.5
        minor_variance = max(trace / 2 - discriminant**0.5, 0)

        return {
            "eta20": eta20,
            "eta02": eta02,
            "eta11": eta11,
            "major_variance": major_variance,
            "minor_variance": minor_variance,
        }
