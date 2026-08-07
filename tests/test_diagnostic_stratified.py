from __future__ import annotations

import unittest

from buffalo_weight.diagnostic_stratified import (
    DiagnosticPrediction,
    StratifiedMetricRecord,
    compute_stratified_metrics,
)


class DiagnosticStratifiedTest(unittest.TestCase):
    def test_computes_stratified_metrics_with_resolution_threshold(self) -> None:
        predictions: list[DiagnosticPrediction] = []
        # Create 12 samples for ResA, 5 samples for ResB
        for i in range(12):
            predictions.append(DiagnosticPrediction(
                "random_forest_baseline", "baseline", f"a_{i}.png", "B1", "FarmA",
                "1024x768", 100.0, 100.0 + (i % 3),
            ))
        for i in range(5):
            predictions.append(DiagnosticPrediction(
                "random_forest_baseline", "baseline", f"b_{i}.png", "B10", "FarmB",
                "2048x1536", 400.0, 410.0,
            ))

        records = compute_stratified_metrics(predictions)

        # Check resolution metrics: 1024x768 has n=12 (included), 2048x1536 has n=5 (excluded)
        res_records = [r for r in records if r.stratum_type == "resolution"]
        self.assertEqual(len(res_records), 1)
        self.assertEqual(res_records[0].stratum_value, "1024x768")
        self.assertEqual(res_records[0].sample_count, 12)

        # Check category metrics
        b1_record = next(r for r in records if r.stratum_type == "weight_category" and r.stratum_value == "B1")
        self.assertEqual(b1_record.sample_count, 12)
        self.assertEqual(b1_record.evaluation_role, "baseline")

    def test_rejects_empty_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[\].*expected non-empty"):
            compute_stratified_metrics([])


if __name__ == "__main__":
    unittest.main()
