from __future__ import annotations

import unittest
from dataclasses import dataclass

from buffalo_weight.diagnostic_coverage import (
    DiagnosticCoverageSample,
    DiagnosticCoverageSummary,
    compute_sample_coverage,
)


class DiagnosticCoverageTest(unittest.TestCase):
    def test_computes_counts_and_crosstabs_for_valid_samples(self) -> None:
        samples = [
            DiagnosticCoverageSample("a.png", "FarmA", "B1", "1024x768", 100.0),
            DiagnosticCoverageSample("b.png", "FarmA", "B1", "1024x768", 110.0),
            DiagnosticCoverageSample("c.png", "FarmB", "B10", "2048x1536", 400.0),
        ]
        summary = compute_sample_coverage(samples)

        self.assertEqual(summary.sample_count, 3)
        self.assertEqual(summary.category_counts, {"B1": 2, "B10": 1})
        self.assertEqual(summary.farm_counts, {"FarmA": 2, "FarmB": 1})
        self.assertEqual(summary.resolution_counts, {"1024x768": 2, "2048x1536": 1})
        self.assertIn({"weight_category": "B1", "farm": "FarmA", "n": 2}, summary.crosstab_category_farm)
        self.assertIn({"farm": "FarmA", "resolution": "1024x768", "n": 2}, summary.crosstab_farm_resolution)

    def test_rejects_empty_samples_with_descriptive_error(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[\].*expected non-empty"):
            compute_sample_coverage([])


if __name__ == "__main__":
    unittest.main()
