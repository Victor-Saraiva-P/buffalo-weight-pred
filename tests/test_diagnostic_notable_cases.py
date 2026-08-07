from __future__ import annotations

import unittest

from buffalo_weight.diagnostic_notable_cases import (
    NotableCaseRecord,
    identify_notable_cases,
)
from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


class DiagnosticNotableCasesTest(unittest.TestCase):
    def test_identifies_shared_hard_and_divergent_cases_preserving_metadata(self) -> None:
        # Create 20 samples across 4 baselines
        baselines = ["random_forest_baseline", "dense", "compact_cnn", "resnet18_pretrained_partial"]
        predictions: list[DiagnosticPrediction] = []

        # Sample 0 is hard in 3 baselines (large error in rf, dense, compact_cnn)
        for b in baselines:
            for i in range(20):
                obs = 100.0 + i * 10.0
                if i == 0 and b != "resnet18_pretrained_partial":
                    pred = obs + 100.0  # huge error (+100)
                elif i == 1 and b == "random_forest_baseline":
                    pred = obs + 50.0   # large error (+50)
                else:
                    pred = obs + 1.0    # small error (+1)
                predictions.append(DiagnosticPrediction(b, "baseline", f"sample_{i}.png", "B1", "FarmA", "1024x768", obs, pred))

        shared_hard, divergent = identify_notable_cases(predictions, baseline_names=baselines)

        # Sample 0 should be in shared hard cases
        sample_0_hard = [c for c in shared_hard if c.file_name == "sample_0.png"]
        self.assertEqual(len(sample_0_hard), 1)
        self.assertEqual(sample_0_hard[0].farm, "FarmA")
        self.assertEqual(sample_0_hard[0].resolution, "1024x768")
        self.assertIn("random_forest_baseline", sample_0_hard[0].predictions)

    def test_handles_decile_ties_consistently(self) -> None:
        baselines = ["random_forest_baseline", "dense", "compact_cnn", "resnet18_pretrained_partial"]
        predictions: list[DiagnosticPrediction] = []
        # Create 10 samples with identical errors to create ties at the boundary
        for b in baselines:
            for i in range(10):
                predictions.append(DiagnosticPrediction(b, "baseline", f"s_{i}.png", "B1", "F", "R", 100.0, 150.0 if i < 2 else 100.0))

        shared_hard, divergent = identify_notable_cases(predictions, baseline_names=baselines)
        self.assertTrue(len(shared_hard) >= 2)

    def test_rejects_insufficient_baselines(self) -> None:
        preds = [DiagnosticPrediction("m1", "baseline", "a.png", "B1", "F", "R", 100.0, 105.0)]
        with self.assertRaisesRegex(ValueError, r".*expected 4 baseline configurations"):
            identify_notable_cases(preds)


if __name__ == "__main__":
    unittest.main()
