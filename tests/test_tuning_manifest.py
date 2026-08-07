from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.reproduction_config import load_report_contract
from buffalo_weight.tuning_manifest import (
    build_tuning_manifest,
    tuning_output_dir,
    tuning_stage_status,
    validate_tuning_manifest,
)
from buffalo_weight.tuning_types import get_pre_registered_variations
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_tuning_provenance import FixedTuningProvenance
from tests.test_tuning_inputs import _setup_confirmed_approach


class TuningManifestTest(unittest.TestCase):
    def test_build_and_validate_tuning_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 3)
            contract = load_report_contract(fixture.config_path)
            output_dir = tuning_output_dir(contract)
            output_dir.mkdir(parents=True, exist_ok=True)
            variations = get_pre_registered_variations("random_forest", 3)
            (output_dir / "tuning_metrics.csv").write_text("header\n")
            (output_dir / "tuning_report.md").write_text("# Report\n")
            provenance = FixedTuningProvenance()
            manifest = build_tuning_manifest(
                output_dir, contract, "random_forest", "random_forest_baseline", 3,
                variations, provenance,
            )
            self.assertEqual(manifest["status"], "complete")
            validate_tuning_manifest(manifest, output_dir, contract, provenance)

    def test_manifest_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 3)
            contract = load_report_contract(fixture.config_path)
            output_dir = tuning_output_dir(contract)
            output_dir.mkdir(parents=True, exist_ok=True)
            variations = get_pre_registered_variations("random_forest", 3)
            (output_dir / "tuning_metrics.csv").write_text("header\n")
            (output_dir / "tuning_report.md").write_text("# Report\n")
            provenance = FixedTuningProvenance()
            manifest = build_tuning_manifest(
                output_dir, contract, "random_forest", "random_forest_baseline", 3,
                variations, provenance,
            )
            manifest["source_commit"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "tuning source commit"):
                validate_tuning_manifest(manifest, output_dir, contract, provenance)

    def test_absent_manifest_returns_absent_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            provenance = FixedTuningProvenance()
            status = tuning_stage_status(contract, provenance)
            self.assertEqual(status, "absent")


if __name__ == "__main__":
    unittest.main()
