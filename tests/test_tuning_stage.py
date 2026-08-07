from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.reproduction_config import load_report_contract
from buffalo_weight.tuning_manifest import tuning_output_dir
from buffalo_weight.tuning_stage import run_tuning_stage
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_compact_cnn import RecordingCompactCnnAdapter
from tests.fake_dense_baseline import RecordingDenseFeatureAdapter
from tests.fake_resnet_baseline import FixedResNetBaselineRunner
from tests.fake_tuning_provenance import FixedTuningProvenance
from tests.test_tuning_inputs import _setup_confirmed_approach


class TuningStageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls._fixture = prepared_comparison_fixture(Path(cls._temp_dir.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_run_tuning_stage_rebuilds_and_reuses(self) -> None:
        fixture = self._fixture
        _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 3)
        contract = load_report_contract(fixture.config_path)
        provenance = FixedTuningProvenance()

        status1 = run_tuning_stage(contract, dry_run=False, provenance=provenance)
        self.assertEqual(status1, "rebuilt")
        self.assertTrue((tuning_output_dir(contract) / "manifest.json").is_file())

        status2 = run_tuning_stage(contract, dry_run=False, provenance=provenance)
        self.assertEqual(status2, "reusable")

    def test_run_tuning_stage_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 3)
            contract = load_report_contract(fixture.config_path)
            provenance = FixedTuningProvenance()

            status = run_tuning_stage(contract, dry_run=True, provenance=provenance)
            self.assertEqual(status, "absent")
            self.assertFalse((tuning_output_dir(contract) / "manifest.json").exists())

    def test_budget_zero_maintains_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 0)
            contract = load_report_contract(fixture.config_path)
            provenance = FixedTuningProvenance()

            status = run_tuning_stage(contract, dry_run=False, provenance=provenance)
            self.assertEqual(status, "released; baseline_maintained")

    def test_unconfirmed_gate_blocks_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            provenance = FixedTuningProvenance()
            with self.assertRaisesRegex(ValueError, "confirmed approach gate was blocked"):
                run_tuning_stage(contract, dry_run=False, provenance=provenance)


if __name__ == "__main__":
    unittest.main()
