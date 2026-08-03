from __future__ import annotations

import unittest

from buffalo_weight.feature_evaluation import FeatureSample
from buffalo_weight.feature_redundancy import calculate_feature_redundancy


class FeatureRedundancyTest(unittest.TestCase):
    def test_reports_each_pair_and_structural_area_bijection(self) -> None:
        samples = redundancy_samples()
        rows = calculate_feature_redundancy(
            samples, (
                "area", "equivalent_diameter", "perimeter",
                "bbox_width", "bbox_height", "bbox_area",
            )
        )

        self.assertEqual(len(rows), 15)
        self.assertEqual(rows[0].structural_relation, "area_bijection")
        self.assertEqual(rows[0].removal_group, "area_transformations")
        self.assertAlmostEqual(rows[0].spearman or 0.0, 1.0)
        bbox_pair = next(row for row in rows if
                         (row.feature_a, row.feature_b) == ("bbox_width", "bbox_area"))
        self.assertIn("bbox_area_product", bbox_pair.structural_relation)

    def test_convexity_relation_does_not_claim_convex_area_dependency(self) -> None:
        samples = convexity_samples()
        rows = calculate_feature_redundancy(samples, ("perimeter", "convex_area", "convexity"))

        convex_area_pair = next(row for row in rows if row.feature_a == "convex_area")
        perimeter_pair = next(row for row in rows if row.feature_a == "perimeter"
                              and row.feature_b == "convexity")

        self.assertNotIn("convexity_ratio", convex_area_pair.structural_relation)
        self.assertEqual(perimeter_pair.structural_relation, "convexity_ratio")


def redundancy_samples() -> list[FeatureSample]:
    return [FeatureSample(str(index), 1, "B1", 80.0, {
        "area": float(value),
        "equivalent_diameter": float(value**0.5),
        "perimeter": float(10 - value),
        "bbox_width": float(value),
        "bbox_height": float(index + 2),
        "bbox_area": float(value * (index + 2)),
    }) for index, value in enumerate((1, 4, 9, 16))]


def convexity_samples() -> list[FeatureSample]:
    return [FeatureSample(str(index), 1, "B1", 80.0, {
        "perimeter": float(10 + index),
        "convex_area": float(20 + index * 2),
        "convexity": float(0.8 + index / 100),
    }) for index in range(4)]


if __name__ == "__main__":
    unittest.main()
