from __future__ import annotations

import unittest

from buffalo_weight.tuning_types import (
    ALLOWED_APPROACHES,
    get_pre_registered_variations,
    validate_pre_registered_approach,
)


class TuningTypesTest(unittest.TestCase):
    def test_allowed_approaches_contains_all_four_classes(self) -> None:
        expected = {"random_forest", "dense_feature_network", "compact_cnn", "resnet18"}
        self.assertEqual(set(ALLOWED_APPROACHES), expected)

    def test_get_pre_registered_variations_returns_valid_recipes(self) -> None:
        for approach in ALLOWED_APPROACHES:
            with self.subTest(approach=approach):
                variations = get_pre_registered_variations(approach, budget=3)
                self.assertGreater(len(variations), 0)
                self.assertLessEqual(len(variations), 3)
                names = [v.name for v in variations]
                self.assertEqual(len(names), len(set(names)), "Variation names must be unique")

    def test_budget_zero_returns_empty_tuple(self) -> None:
        for approach in ALLOWED_APPROACHES:
            with self.subTest(approach=approach):
                variations = get_pre_registered_variations(approach, budget=0)
                self.assertEqual(variations, ())

    def test_unknown_approach_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "approach was 'invalid'"):
            validate_pre_registered_approach("invalid")


if __name__ == "__main__":
    unittest.main()
