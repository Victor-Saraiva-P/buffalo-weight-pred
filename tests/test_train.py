from __future__ import annotations

import unittest

from buffalo_weight.train import evaluate_random_forest


class TrainTest(unittest.TestCase):
    def test_evaluates_random_forest_by_fold(self) -> None:
        rows = []
        labels = ["Leves", "Medio-Leves", "Medio-Pesados", "Pesados"]
        for index in range(20):
            category = f"Q{(index % 4) + 1}"
            rows.append(
                {
                    "file_name": f"mask-{index:03d}",
                    "weight": str(100 + index * 10),
                    "area": str(50 + index),
                    "perimeter": str(30 + index),
                    "weight_category": category,
                    "weight_category_label": labels[index % 4],
                    "fold": str((index % 5) + 1),
                }
            )

        metrics, predictions = evaluate_random_forest(
            rows,
            ["area", "perimeter"],
            n_estimators=5,
            random_state=42,
        )

        self.assertEqual(len(metrics), 5)
        self.assertEqual(len(predictions), 20)
        self.assertEqual({row["model"] for row in metrics}, {"random_forest"})
        self.assertEqual({row["fold"] for row in metrics}, {"1", "2", "3", "4", "5"})
        self.assertEqual(
            set(metrics[0]),
            {"model", "fold", "mae", "rmse", "r2", "n_train", "n_validation"},
        )
        self.assertEqual(
            set(predictions[0]),
            {
                "model",
                "fold",
                "file_name",
                "weight",
                "y_pred",
                "error",
                "abs_error",
                "weight_category",
                "weight_category_label",
            },
        )


if __name__ == "__main__":
    unittest.main()
