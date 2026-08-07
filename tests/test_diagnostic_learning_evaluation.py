"""Tests for controlled learning curves evaluation logic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.diagnostic_learning_evaluation import evaluate_learning_curves
from buffalo_weight.reproduction_config import InputsContract, ReportContract
from tests.fake_compact_cnn import RecordingCompactCnnAdapter
from tests.fake_dense_baseline import FixedDenseBaselineRunner
from tests.fake_feature_evaluation import RecordingFeatureBaseline
from tests.fake_resnet_baseline import FixedResNetBaselineRunner
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_feature_confirmation_cli import _prepare_human_review, _run_confirmation


class DiagnosticLearningEvaluationTest(unittest.TestCase):
    def test_evaluate_learning_curves_produces_all_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CuratedInputsFixture(root, sample_count=132)
            contract_path, report_path = _prepare_human_review(fixture, ("area", "perimeter"))
            _run_confirmation(fixture, contract_path, report_path)

            from buffalo_weight.reproduction_config import load_report_contract
            contract = load_report_contract(fixture.config_path)

            rf_model = RecordingFeatureBaseline()
            dense_runner = FixedDenseBaselineRunner()
            compact_adapter = RecordingCompactCnnAdapter()
            resnet_runner = FixedResNetBaselineRunner()

            slice_data = evaluate_learning_curves(
                contract,
                random_forest_baseline=rf_model,
                dense_runner=dense_runner,
                compact_adapter=compact_adapter,
                resnet_runner=resnet_runner,
            )

            # 4 baselines * 5 folds * 3 fractions (0.50, 0.75, 1.00) = 60 point records
            self.assertEqual(len(slice_data.point_records), 60)
            # 4 baselines * 3 fractions = 12 summary records
            self.assertEqual(len(slice_data.summary_records), 12)

            configs = {p.configuration for p in slice_data.point_records}
            self.assertEqual(
                configs,
                {
                    "random_forest_baseline",
                    "dense_baseline",
                    "compact_cnn_baseline",
                    "resnet18_pretrained_partial",
                },
            )

            # Ensure tuned configuration is NOT present
            self.assertNotIn("tuning", configs)

            # Verify fractions present per config
            for config in configs:
                config_points = [p for p in slice_data.point_records if p.configuration == config]
                fractions = {p.fraction for p in config_points}
                self.assertEqual(fractions, {0.50, 0.75, 1.00})


def _dummy_inputs_contract() -> InputsContract:
    return InputsContract(Path("index.csv"), Path("masks"), 132, 1024, 10, 5, 42)


if __name__ == "__main__":
    unittest.main()
