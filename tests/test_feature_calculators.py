from __future__ import annotations

import math
import unittest

import numpy as np

from buffalo_weight.feature_calculators import calculate_mask_features


class FeatureCalculatorTest(unittest.TestCase):
    def test_calculates_square_mask_features(self) -> None:
        features = calculate_mask_features(
            np.array([
                [True, True],
                [True, True],
            ])
        )

        self.assertEqual(features["area"], 4)
        self.assertEqual(features["perimeter"], 8)
        self.assertEqual(features["solidity"], 1)
        self.assertAlmostEqual(features["circularity"], math.pi / 4)
        self.assertAlmostEqual(features["equivalent_diameter"], math.sqrt(16 / math.pi))
        self.assertEqual(features["bbox_width"], 2)
        self.assertEqual(features["bbox_height"], 2)
        self.assertEqual(features["bbox_area"], 4)
        self.assertEqual(features["aspect_ratio"], 1)
        self.assertEqual(features["extent"], 1)
        self.assertEqual(features["convex_area"], 4)
        self.assertEqual(features["convexity"], 1)
        self.assertEqual(features["major_axis_length"], 2)
        self.assertEqual(features["minor_axis_length"], 2)
        self.assertAlmostEqual(features["hu_moment_1"], 0.125)
        self.assertAlmostEqual(features["hu_moment_2"], 0)

    def test_calculates_empty_mask_features(self) -> None:
        features = calculate_mask_features(
            np.array([
                [False, False],
                [False, False],
            ])
        )

        self.assertEqual(
            features,
            {
                "area": 0,
                "perimeter": 0,
                "solidity": 0,
                "circularity": 0,
                "equivalent_diameter": 0,
                "bbox_width": 0,
                "bbox_height": 0,
                "bbox_area": 0,
                "aspect_ratio": 0,
                "extent": 0,
                "convex_area": 0,
                "convexity": 0,
                "major_axis_length": 0,
                "minor_axis_length": 0,
                "hu_moment_1": 0,
                "hu_moment_2": 0,
            },
        )

    def test_calculates_solidity_for_concave_mask(self) -> None:
        features = calculate_mask_features(
            np.array([
                [True, False],
                [True, True],
            ])
        )

        self.assertEqual(features["area"], 3)
        self.assertAlmostEqual(features["solidity"], 3 / 3.5)


if __name__ == "__main__":
    unittest.main()
