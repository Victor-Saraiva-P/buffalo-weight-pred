"""Tests for checking 100% baseline point reusability and freshness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.diagnostic_learning_freshness import (
    check_baseline_100_reusability,
    load_reused_fold_metrics,
)
from buffalo_weight.reproduction_config import InputsContract, ReportContract


class DiagnosticLearningFreshnessTest(unittest.TestCase):
    def test_reusability_returns_false_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = ReportContract(
                inputs=_dummy_inputs_contract(),
                artifacts_root=root,
            )
            reusable = check_baseline_100_reusability(contract, "random_forest_baseline")
            self.assertFalse(reusable)

    def test_reusability_returns_false_for_unknown_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = ReportContract(
                inputs=_dummy_inputs_contract(),
                artifacts_root=root,
            )
            reusable = check_baseline_100_reusability(contract, "unknown_config")
            self.assertFalse(reusable)

    def test_load_reused_fold_metrics_from_predictions_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = ReportContract(
                inputs=_dummy_inputs_contract(),
                artifacts_root=root,
            )
            base_dir = root / "baselines" / "random_forest_baseline"
            base_dir.mkdir(parents=True, exist_ok=True)

            rows = [
                {
                    "file_name": "s1.png",
                    "fold": "1",
                    "weight_category": "B1",
                    "observed_weight_kg": "100.0",
                    "predicted_weight_kg": "105.0",
                    "configuration": "random_forest_baseline",
                    "evaluation_role": "candidate",
                },
                {
                    "file_name": "s2.png",
                    "fold": "1",
                    "weight_category": "B1",
                    "observed_weight_kg": "200.0",
                    "predicted_weight_kg": "190.0",
                    "configuration": "random_forest_baseline",
                    "evaluation_role": "candidate",
                },
            ]
            fieldnames = list(rows[0].keys())
            write_csv_rows(rows, base_dir / "predictions.csv", fieldnames)

            mae, bias, n_eval = load_reused_fold_metrics(contract, "random_forest_baseline", fold=1)
            self.assertAlmostEqual(mae, 7.5)
            self.assertAlmostEqual(bias, -2.5)
            self.assertEqual(n_eval, 2)


def _dummy_inputs_contract() -> InputsContract:
    return InputsContract(Path("index.csv"), Path("masks"), 132, 1024, 10, 5, 42)


if __name__ == "__main__":
    unittest.main()
