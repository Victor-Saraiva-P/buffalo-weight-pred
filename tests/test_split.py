from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from buffalo_weight.split import assign_folds, assign_weight_categories, plot_weight_distribution


class SplitTest(unittest.TestCase):
    def test_assigns_global_weight_categories_with_equal_counts(self) -> None:
        rows = [
            {"file_name": f"mask-{index:03d}", "weight": str(index)}
            for index in range(132, 0, -1)
        ]

        assign_weight_categories(rows, category_count=4)

        self.assertEqual(
            Counter(row["weight_category"] for row in rows),
            {"B1": 33, "B2": 33, "B3": 33, "B4": 33},
        )
        light_weights = [
            float(row["weight"]) for row in rows if row["weight_category"] == "B1"
        ]
        heavy_weights = [
            float(row["weight"]) for row in rows if row["weight_category"] == "B4"
        ]
        self.assertEqual(max(light_weights), 33)
        self.assertEqual(min(heavy_weights), 100)

    def test_assigns_configurable_weight_category_count(self) -> None:
        rows = [
            {"file_name": f"mask-{index:03d}", "weight": str(index)}
            for index in range(1, 133)
        ]

        assign_weight_categories(rows, category_count=8)

        categories = Counter(row["weight_category"] for row in rows)
        self.assertEqual(set(categories), {f"B{index}" for index in range(1, 9)})
        self.assertTrue(all(count in {16, 17} for count in categories.values()))
        self.assertEqual(
            {row["weight_category_label"] for row in rows},
            {f"Faixa {index}" for index in range(1, 9)},
        )

    def test_assigns_reproducible_stratified_validation_folds(self) -> None:
        rows = [
            {"file_name": f"mask-{index:03d}", "weight": str(index)}
            for index in range(1, 133)
        ]
        assign_weight_categories(rows, category_count=4)

        assign_folds(rows, k=5, random_state=42)

        self.assertEqual({row["fold"] for row in rows}, {"1", "2", "3", "4", "5"})
        for fold in {"1", "2", "3", "4", "5"}:
            categories = Counter(
                row["weight_category"] for row in rows if row["fold"] == fold
            )
            self.assertEqual(set(categories), {"B1", "B2", "B3", "B4"})
            self.assertTrue(all(count in {6, 7} for count in categories.values()))

    def test_plots_configurable_weight_categories(self) -> None:
        rows = [
            {"file_name": f"mask-{index:03d}", "weight": str(index)}
            for index in range(1, 101)
        ]
        assign_weight_categories(rows, category_count=10)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight_categories.png"
            plot_weight_distribution(rows, path)

            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
