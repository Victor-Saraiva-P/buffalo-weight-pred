from __future__ import annotations

import math
import unittest

from buffalo_weight.feature_calculators import calculate_mask_features


class FeatureCalculatorTest(unittest.TestCase):
    def test_calculates_square_mask_features(self) -> None:
        features = calculate_mask_features(
            [
                [True, True],
                [True, True],
            ]
        )

        self.assertEqual(features["area"], 4)
        self.assertEqual(features["perimeter"], 8)
        self.assertEqual(features["solidity"], 1)
        self.assertAlmostEqual(features["circularity"], math.pi / 4)
        self.assertAlmostEqual(features["equivalent_diameter"], math.sqrt(16 / math.pi))
        self.assertAlmostEqual(features["hu_moment_1"], 0.125)
        self.assertAlmostEqual(features["hu_moment_2"], 0)

    def test_calculates_empty_mask_features(self) -> None:
        features = calculate_mask_features(
            [
                [False, False],
                [False, False],
            ]
        )

        self.assertEqual(
            features,
            {
                "area": 0,
                "perimeter": 0,
                "solidity": 0,
                "circularity": 0,
                "equivalent_diameter": 0,
                "hu_moment_1": 0,
                "hu_moment_2": 0,
            },
        )

    def test_calculates_solidity_for_concave_mask(self) -> None:
        features = calculate_mask_features(
            [
                [True, False],
                [True, True],
            ]
        )

        self.assertEqual(features["area"], 3)
        self.assertAlmostEqual(features["solidity"], 3 / 3.5)


if __name__ == "__main__":
    unittest.main()
