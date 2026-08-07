from __future__ import annotations

import unittest

from buffalo_weight.diagnostic_coverage import DiagnosticCoverageSample
from buffalo_weight.diagnostic_descriptive_slice import (
    DescriptiveDiagnosticSlice,
    assert_no_prohibited_diagnostic_elements,
    build_descriptive_diagnostic_slice,
)
from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


class DiagnosticDescriptiveSliceTest(unittest.TestCase):
    def test_builds_complete_slice_and_passes_prohibition_audit(self) -> None:
        baselines = ["random_forest_baseline", "dense", "compact_cnn", "resnet18_pretrained_partial"]
        coverage_samples = [
            DiagnosticCoverageSample(f"s_{i}.png", "FarmA" if i < 10 else "FarmB", "B1", "1024x768", 100.0 + i * 10)
            for i in range(20)
        ]
        predictions: list[DiagnosticPrediction] = []
        for b in baselines:
            for s in coverage_samples:
                predictions.append(DiagnosticPrediction(
                    b, "baseline", s.file_name, s.weight_category, s.farm, s.resolution,
                    s.weight_kg, s.weight_kg + 2.0,
                ))

        slice_result = build_descriptive_diagnostic_slice(coverage_samples, predictions)

        self.assertIsInstance(slice_result, DescriptiveDiagnosticSlice)
        self.assertEqual(slice_result.coverage_summary.sample_count, 20)
        self.assertTrue(len(slice_result.stratified_metrics) > 0)
        self.assertTrue(len(slice_result.farm_comparisons) > 0)
        self.assertTrue(len(slice_result.residual_correlations) > 0)

        # Audit prohibition rules
        assert_no_prohibited_diagnostic_elements(slice_result)

    def test_prohibition_audit_catches_forbidden_terms(self) -> None:
        with self.assertRaisesRegex(ValueError, r"prohibited term 'p_value'"):
            assert_no_prohibited_diagnostic_elements({"summary": "contains p_value analysis"})


if __name__ == "__main__":
    unittest.main()
