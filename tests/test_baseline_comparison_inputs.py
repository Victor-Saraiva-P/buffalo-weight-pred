from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.baseline_comparison_inputs import load_comparison_predictions
from buffalo_weight.reproduction_config import load_report_contract
from tests.baseline_comparison_fixture import prepared_comparison_fixture


class BaselineComparisonInputsTest(unittest.TestCase):
    def test_normalizes_exactly_four_candidates_and_one_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            contract = load_report_contract(fixture.config_path)

            predictions = load_comparison_predictions(contract)

            self.assertEqual(len(predictions), 5 * 132)
            identities = {(row.configuration, row.approach, row.evaluation_role)
                          for row in predictions}
            self.assertEqual(identities, {
                ("random_forest_baseline", "random_forest", "candidate"),
                ("dense", "dense_feature_network", "candidate"),
                ("compact_cnn", "compact_cnn", "candidate"),
                ("resnet18_pretrained_partial", "resnet18", "candidate"),
                ("training_mean_reference", "training_mean", "reference"),
            })

    def test_rejects_manifest_whose_prediction_hash_is_not_integral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            manifest_path = (contract.artifacts_root / "baselines" / "dense" / "manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["outputs"]["predictions.csv"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "dense.*prediction.*integrity"):
                load_comparison_predictions(contract)


if __name__ == "__main__":
    unittest.main()
