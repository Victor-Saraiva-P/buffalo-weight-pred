from __future__ import annotations

import unittest

from buffalo_weight.diagnostic_residual_correlations import (
    ResidualCorrelationRecord,
    compute_residual_correlations,
)
from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


class DiagnosticResidualCorrelationsTest(unittest.TestCase):
    def test_computes_pearson_correlations_on_signed_residuals(self) -> None:
        preds = [
            DiagnosticPrediction("m1", "baseline", "a.png", "B1", "F", "R", 100.0, 110.0), # e1 = +10
            DiagnosticPrediction("m1", "baseline", "b.png", "B1", "F", "R", 200.0, 190.0), # e1 = -10
            DiagnosticPrediction("m2", "baseline", "a.png", "B1", "F", "R", 100.0, 105.0), # e2 = +5
            DiagnosticPrediction("m2", "baseline", "b.png", "B1", "F", "R", 200.0, 195.0), # e2 = -5
        ]

        records = compute_residual_correlations(preds)

        m1_m2 = next(r for r in records if r.configuration_1 == "m1" and r.configuration_2 == "m2")
        self.assertAlmostEqual(m1_m2.pearson_r, 1.0)

    def test_rejects_unpaired_or_empty_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[\].*expected non-empty"):
            compute_residual_correlations([])


if __name__ == "__main__":
    unittest.main()
