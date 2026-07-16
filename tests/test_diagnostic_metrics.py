from __future__ import annotations

import unittest

import numpy as np

from buffalo_weight.diagnostic_metrics import metric_summary, oracle_metrics, paired_bootstrap


class DiagnosticMetricsTest(unittest.TestCase):
    def test_metric_summary_reports_bias_and_tail(self) -> None:
        summary = metric_summary(np.asarray([100.0, 200.0]), np.asarray([110.0, 180.0]))

        self.assertEqual(float(summary["mae"]), 15.0)
        self.assertEqual(float(summary["bias"]), -5.0)

    def test_paired_bootstrap_detects_dominant_candidate(self) -> None:
        actual = np.arange(20, dtype=float)
        candidate = actual.copy()
        reference = actual + 10

        result = paired_bootstrap(actual, candidate, reference, repeats=500)

        self.assertEqual(float(result["probability_candidate_better"]), 1.0)

    def test_oracle_selects_smallest_error_per_animal(self) -> None:
        actual = np.asarray([100.0, 200.0])
        predictions = np.asarray([[90.0, 100.0], [200.0, 220.0]])

        result = oracle_metrics(actual, predictions)

        self.assertEqual(float(result["oracle_mae"]), 0.0)


if __name__ == "__main__":
    unittest.main()
