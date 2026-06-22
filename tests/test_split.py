from __future__ import annotations

from collections import Counter
import unittest

from buffalo_weight.split import assign_folds, assign_weight_categories


class SplitTest(unittest.TestCase):
    def test_assigns_global_weight_categories_with_equal_counts(self) -> None:
        rows = [
            {"file_name": f"mask-{index:03d}", "weight": str(index)}
            for index in range(132, 0, -1)
        ]

        assign_weight_categories(rows)

        self.assertEqual(
            Counter(row["weight_category"] for row in rows),
            {"Q1": 33, "Q2": 33, "Q3": 33, "Q4": 33},
        )
        light_weights = [
            float(row["weight"]) for row in rows if row["weight_category"] == "Q1"
        ]
        heavy_weights = [
            float(row["weight"]) for row in rows if row["weight_category"] == "Q4"
        ]
        self.assertEqual(max(light_weights), 33)
        self.assertEqual(min(heavy_weights), 100)

    def test_assigns_reproducible_stratified_validation_folds(self) -> None:
        rows = [
            {"file_name": f"mask-{index:03d}", "weight": str(index)}
            for index in range(1, 133)
        ]
        assign_weight_categories(rows)

        assign_folds(rows, k=5, random_state=42)

        self.assertEqual({row["fold"] for row in rows}, {"1", "2", "3", "4", "5"})
        for fold in {"1", "2", "3", "4", "5"}:
            categories = Counter(
                row["weight_category"] for row in rows if row["fold"] == fold
            )
            self.assertEqual(set(categories), {"Q1", "Q2", "Q3", "Q4"})
            self.assertTrue(all(count in {6, 7} for count in categories.values()))


if __name__ == "__main__":
    unittest.main()
