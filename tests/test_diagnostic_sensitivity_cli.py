"""Tests for public CLI diagnostics-sensitivity subcommand."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from buffalo_weight.report_cli import main
from buffalo_weight.reproduction_config import load_report_contract
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_feature_evaluation import RecordingFeatureBaseline


class FakeCliMaskLoader:
    """Synthetic mask loader wired through the stage for CLI testing."""

    def load_mask(self, file_name: str) -> np.ndarray:
        """Return a centered 40×40 block in a 200×200 image.

        Example: ``loader.load_mask("img01")`` returns test mask.
        """
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[80:120, 80:120] = 1.0
        return mask


class DiagnosticSensitivityCliTest(unittest.TestCase):
    def test_cli_runs_diagnostics_sensitivity_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            stdout, stderr = io.StringIO(), io.StringIO()

            # Monkey-patch the stage to use our fake loader
            import buffalo_weight.diagnostic_sensitivity_stage as stage_mod
            original_rebuild = stage_mod._execute_stage_rebuild

            def patched_rebuild(contract, output_dir, publisher, mask_loader, rf_baseline):
                return original_rebuild(
                    contract, output_dir, publisher,
                    FakeCliMaskLoader(), rf_baseline,
                )

            stage_mod._execute_stage_rebuild = patched_rebuild
            try:
                result = main(
                    ["diagnostics-sensitivity", "--config", str(fixture.config_path)],
                    stdout=stdout, stderr=stderr,
                    random_forest_baseline=RecordingFeatureBaseline(),
                )
            finally:
                stage_mod._execute_stage_rebuild = original_rebuild

            self.assertEqual(result, 0, stderr.getvalue())
            self.assertIn("diagnostics_sensitivity: rebuilt", stdout.getvalue())

            contract = load_report_contract(fixture.config_path)
            out_dir = contract.artifacts_root / "diagnostics" / "sensitivity"
            self.assertTrue((out_dir / "manifest.json").is_file())
            self.assertTrue((out_dir / "sensitivity_perturbations.csv").is_file())
            self.assertTrue((out_dir / "morphology_eligibility.csv").is_file())
            self.assertTrue((out_dir / "sensitivity_report.md").is_file())

            # All 132 masks should appear in the eligibility CSV
            with (out_dir / "morphology_eligibility.csv").open(newline="", encoding="utf-8") as f:
                elig_rows = list(csv.DictReader(f))
            self.assertEqual(len(elig_rows), 132)

            # Perturbation CSV should have records for all configurations and perturbation kinds
            with (out_dir / "sensitivity_perturbations.csv").open(newline="", encoding="utf-8") as f:
                pert_rows = list(csv.DictReader(f))
            self.assertGreater(len(pert_rows), 0)

            # Verify delta convention: all non-rejected records have consistent deltas
            for row in pert_rows:
                if row["status"] == "eligible" and row["delta_kg"]:
                    delta = float(row["delta_kg"])
                    perturbed = float(row["perturbed_prediction_kg"])
                    original = float(row["original_prediction_kg"])
                    self.assertAlmostEqual(delta, perturbed - original, places=4)


if __name__ == "__main__":
    unittest.main()
