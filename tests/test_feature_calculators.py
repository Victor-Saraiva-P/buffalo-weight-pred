from __future__ import annotations

import math
import unittest

import numpy as np

from buffalo_weight.feature_calculators import APPROVED_FEATURES, calculate_mask_features


class FeatureCalculatorTest(unittest.TestCase):
    def test_calculates_square_mask_features(self) -> None:
        features = calculate_mask_features(
            np.array([
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]),
            canonical_long_side=3,
        )

        self.assert_square_primary(features)
        self.assert_square_derived(features)

    def assert_square_primary(self, features: dict[str, float]) -> None:
        expected_axis = 4 * math.sqrt(2 / 3)
        self.assertEqual(features["area"], 9)
        self.assertAlmostEqual(features["perimeter"], 10.265992653082648)
        self.assertEqual(features["solidity"], 1)
        self.assertAlmostEqual(features["circularity"], 36 * math.pi / features["perimeter"] ** 2)
        self.assertAlmostEqual(features["equivalent_diameter"], math.sqrt(36 / math.pi))
        self.assertEqual(features["bbox_width"], 3)
        self.assertEqual(features["bbox_height"], 3)
        self.assertEqual(features["bbox_area"], 9)
        self.assertEqual(features["aspect_ratio"], 1)
        self.assertEqual(features["extent"], 1)
        self.assertEqual(features["convex_area"], 9)
        self.assertEqual(features["convexity"], 1)
        self.assertAlmostEqual(features["major_axis_length"], expected_axis)
        self.assertAlmostEqual(features["minor_axis_length"], expected_axis)

    def assert_square_derived(self, features: dict[str, float]) -> None:
        expected_axis = 4 * math.sqrt(2 / 3)
        self.assertAlmostEqual(features["roundness"], 27 / (8 * math.pi))
        self.assertAlmostEqual(features["feret_diameter"], 2 * math.sqrt(2))
        self.assertAlmostEqual(features["hu_moment_1"], 4 / 27)
        self.assertAlmostEqual(features["hu_moment_2"], 0)
        self.assertEqual(features["area_power_1_5"], 27)
        self.assertAlmostEqual(features["area_major_axis_product"], 9 * expected_axis)
        self.assertEqual(features["center_vertical_occupancy"], 3)
        self.assertEqual(features["end_vertical_occupancy_min"], 3)
        self.assertEqual(features["end_vertical_occupancy_max"], 3)
        self.assertEqual(features["center_to_end_occupancy_ratio"], 1)
        self.assertEqual(features["centroid_x_offset"], 0)
        self.assertEqual(features["centroid_y_ratio"], 0.5)

    def test_applies_canonical_scale_by_feature_dimension(self) -> None:
        mask = np.ones((3, 3), dtype=bool)

        original_scale = calculate_mask_features(mask, canonical_long_side=3)
        doubled_scale = calculate_mask_features(mask, canonical_long_side=6)

        self.assertAlmostEqual(doubled_scale["perimeter"], original_scale["perimeter"] * 2)
        self.assertEqual(doubled_scale["area"], original_scale["area"] * 4)
        self.assertEqual(doubled_scale["area_power_1_5"], original_scale["area_power_1_5"] * 8)
        self.assertEqual(doubled_scale["aspect_ratio"], original_scale["aspect_ratio"])
        self.assertEqual(doubled_scale["hu_moment_1"], original_scale["hu_moment_1"])

    def test_calculates_empty_mask_features(self) -> None:
        features = calculate_mask_features(
            np.array([
                [False, False],
                [False, False],
            ])
        )

        self.assertEqual(set(features), set(APPROVED_FEATURES))
        self.assertTrue(all(value == 0 for value in features.values()))

    def test_calculates_solidity_for_concave_mask(self) -> None:
        features = calculate_mask_features(
            np.array([
                [True, False],
                [True, True],
            ]),
            canonical_long_side=2,
        )

        self.assertEqual(features["area"], 3)
        self.assertEqual(features["solidity"], 1)


if __name__ == "__main__":
    unittest.main()
