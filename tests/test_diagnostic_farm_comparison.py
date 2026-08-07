from __future__ import annotations

import unittest

from buffalo_weight.diagnostic_farm_comparison import (
    FarmComparisonRecord,
    REMAINING_CONFOUNDING_NOTE,
    compare_farms_under_approved_subsets,
)
from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


class DiagnosticFarmComparisonTest(unittest.TestCase):
    def test_compares_farms_in_full_sample_and_shared_range(self) -> None:
        predictions = [
            DiagnosticPrediction("rf", "baseline", "a.png", "B1", "FarmA", "1024x768", 80.0, 85.0), # Outside range (<92)
            DiagnosticPrediction("rf", "baseline", "b.png", "B2", "FarmA", "1024x768", 100.0, 105.0), # Inside range (100)
            DiagnosticPrediction("rf", "baseline", "c.png", "B3", "FarmB", "1024x768", 200.0, 202.0), # Inside range (200)
            DiagnosticPrediction("rf", "baseline", "d.png", "B10", "FarmB", "1024x768", 350.0, 340.0), # Outside range (>265)
        ]

        records = compare_farms_under_approved_subsets(predictions)

        full_records = [r for r in records if r.sample_scope == "full_sample"]
        shared_records = [r for r in records if r.sample_scope == "shared_range_92_265"]

        self.assertEqual(len(full_records), 2)  # FarmA, FarmB
        self.assertEqual(len(shared_records), 2)  # FarmA, FarmB

        farm_a_shared = next(r for r in shared_records if r.farm == "FarmA")
        self.assertEqual(farm_a_shared.sample_count, 1)
        self.assertAlmostEqual(farm_a_shared.mae_kg, 5.0)

        # Check confounding note present in all records
        for r in records:
            self.assertEqual(r.confounding_note, REMAINING_CONFOUNDING_NOTE)

    def test_rejects_empty_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[\].*expected non-empty"):
            compare_farms_under_approved_subsets([])


if __name__ == "__main__":
    unittest.main()
