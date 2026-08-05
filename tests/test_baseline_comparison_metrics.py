from __future__ import annotations

import math
import unittest

from buffalo_weight.baseline_comparison_metrics import comparison_metric_rows
from buffalo_weight.baseline_comparison_types import ComparisonPrediction


class BaselineComparisonMetricsTest(unittest.TestCase):
    def test_recomputes_fold_and_pooled_metrics_with_signed_extreme_bias(self) -> None:
        predictions = [
            _prediction("a.png", "B1", 1, 10.0, 12.0),
            _prediction("b.png", "B1", 2, 20.0, 18.0),
            _prediction("c.png", "B10", 1, 30.0, 31.0),
            _prediction("d.png", "B10", 2, 40.0, 35.0),
        ]

        rows = comparison_metric_rows(predictions)

        pooled = next(row for row in rows if row.scope == "oof" and row.population == "all")
        self.assertEqual(pooled.n, 4)
        self.assertAlmostEqual(pooled.mae_kg, 2.5)
        self.assertAlmostEqual(pooled.rmse_kg or 0.0, 2.915475947)
        self.assertAlmostEqual(pooled.bias_kg, -1.0)
        self.assertAlmostEqual(pooled.r2 or 0.0, 0.932)
        b1 = next(row for row in rows if row.population == "B1")
        b10 = next(row for row in rows if row.population == "B10")
        self.assertEqual((b1.mae_kg, b1.bias_kg, b1.rmse_kg, b1.r2), (2.0, 0.0, None, None))
        self.assertEqual((b10.mae_kg, b10.bias_kg, b10.rmse_kg, b10.r2), (3.0, -2.0, None, None))

    def test_rejects_empty_and_non_finite_predictions_with_offending_values(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[\].*at least one finite"):
            comparison_metric_rows([])
        invalid = [_prediction("invalid.png", "B1", 1, 10.0, math.nan)]
        with self.assertRaisesRegex(ValueError, r"invalid\.png.*nan.*finite weights"):
            comparison_metric_rows(invalid)


def _prediction(
    file_name: str, category: str, fold: int, observed: float, predicted: float,
) -> ComparisonPrediction:
    return ComparisonPrediction(
        "random_forest_baseline", "random_forest", "candidate", file_name,
        category, fold, observed, predicted,
    )


if __name__ == "__main__":
    unittest.main()
