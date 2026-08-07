from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.reproduction_config import load_report_contract
from buffalo_weight.tuning_evaluation import evaluate_tuning_variations
from buffalo_weight.tuning_types import get_pre_registered_variations
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_compact_cnn import RecordingCompactCnnAdapter
from tests.fake_dense_baseline import FixedCudaRuntimeProbe, RecordingDenseFeatureAdapter
from tests.fake_resnet_baseline import FixedResNetBaselineRunner
from tests.test_tuning_inputs import _setup_confirmed_approach


class TuningEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls._fixture = prepared_comparison_fixture(Path(cls._temp_dir.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_random_forest_tuning_evaluation(self) -> None:
        fixture = self._fixture
        _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 3)
        contract = load_report_contract(fixture.config_path)
        variations = get_pre_registered_variations("random_forest", 3)
        predictions, metrics = evaluate_tuning_variations(
            contract, "random_forest", "random_forest_baseline",
            ("area", "perimeter"), variations,
        )
        self.assertEqual(len(predictions), 3 * 132)
        self.assertGreater(len(metrics), 0)
        configurations = {p.configuration for p in predictions}
        self.assertEqual(configurations, {v.name for v in variations})

    def test_dense_tuning_evaluation(self) -> None:
        fixture = self._fixture
        _setup_confirmed_approach(fixture, "dense_feature_network", "dense", 2)
        contract = load_report_contract(fixture.config_path)
        variations = get_pre_registered_variations("dense_feature_network", 2)
        adapter = RecordingDenseFeatureAdapter()
        predictions, metrics = evaluate_tuning_variations(
            contract, "dense_feature_network", "dense",
            ("area", "perimeter"), variations, dense_adapter=adapter,
        )
        self.assertEqual(len(predictions), 2 * 132)
        self.assertGreater(len(metrics), 0)

    def test_compact_cnn_tuning_evaluation(self) -> None:
        fixture = self._fixture
        _setup_confirmed_approach(fixture, "compact_cnn", "compact_cnn", 2)
        contract = load_report_contract(fixture.config_path)
        variations = get_pre_registered_variations("compact_cnn", 2)
        adapter = RecordingCompactCnnAdapter()
        predictions, metrics = evaluate_tuning_variations(
            contract, "compact_cnn", "compact_cnn",
            None, variations, compact_cnn_adapter=adapter,
        )
        self.assertEqual(len(predictions), 2 * 132)
        self.assertGreater(len(metrics), 0)

    def test_resnet_tuning_evaluation(self) -> None:
        fixture = self._fixture
        _setup_confirmed_approach(fixture, "resnet18", "resnet18_pretrained_partial", 1)
        contract = load_report_contract(fixture.config_path)
        variations = get_pre_registered_variations("resnet18", 1)
        runner = FixedResNetBaselineRunner()
        predictions, metrics = evaluate_tuning_variations(
            contract, "resnet18", "resnet18_pretrained_partial",
            None, variations, resnet_runner=runner,
        )
        self.assertEqual(len(predictions), 1 * 132)
        self.assertGreater(len(metrics), 0)


if __name__ == "__main__":
    unittest.main()
