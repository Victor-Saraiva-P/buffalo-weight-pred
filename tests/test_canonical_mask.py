from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.canonical_mask import canonicalize_mask, principal_axis_angle


class CanonicalMaskTest(unittest.TestCase):
    def test_aligns_diagonal_silhouette_horizontally(self) -> None:
        mask = np.eye(9, dtype=bool)

        canonical = canonicalize_mask(mask, 32)

        ys, xs = np.nonzero(canonical)
        self.assertGreater(xs.max() - xs.min(), ys.max() - ys.min())

    def test_principal_axis_of_horizontal_mask_is_zero(self) -> None:
        mask = np.zeros((5, 9), dtype=bool)
        mask[2, 1:8] = True

        self.assertAlmostEqual(abs(principal_axis_angle(mask)), 0.0)


if __name__ == "__main__":
    unittest.main()
