from __future__ import annotations

import unittest

from buffalo_weight.feature_selection_rules import (
    classify_mae_delta,
    conservative_removal_recommendation,
    permutation_seed,
)


class FeatureSelectionRulesTest(unittest.TestCase):
    def test_classifies_exact_practical_equivalence_boundaries_as_neutral(self) -> None:
        self.assertEqual(classify_mae_delta(-1.0), "neutral")
        self.assertEqual(classify_mae_delta(1.0), "neutral")
        self.assertEqual(classify_mae_delta(-1.000001), "improvement")
        self.assertEqual(classify_mae_delta(1.000001), "harm")

    def test_recommends_only_improvement_without_cross_baseline_harm(self) -> None:
        self.assertEqual(
            conservative_removal_recommendation({"random_forest": -1.2, "dense": 0.8}),
            "recommend_removal",
        )
        self.assertEqual(
            conservative_removal_recommendation({"random_forest": -1.2, "dense": 1.1}),
            "retain_harm_veto",
        )
        self.assertEqual(
            conservative_removal_recommendation({"random_forest": -1.0, "dense": 0.2}),
            "retain_double_neutral",
        )

    def test_permutation_seeds_are_deterministic_and_repetition_specific(self) -> None:
        first = permutation_seed(42, 3, "area", 0)
        self.assertEqual(first, permutation_seed(42, 3, "area", 0))
        self.assertNotEqual(first, permutation_seed(42, 3, "area", 1))
        self.assertNotEqual(first, permutation_seed(42, 4, "area", 0))


if __name__ == "__main__":
    unittest.main()
