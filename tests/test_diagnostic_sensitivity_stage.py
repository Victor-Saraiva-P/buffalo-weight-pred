"""Tests for sensitivity diagnostic stage orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from buffalo_weight.diagnostic_sensitivity_evaluation import SensitivityMaskLoader
from buffalo_weight.diagnostic_sensitivity_stage import (
    diagnostic_sensitivity_output_dir,
    run_diagnostic_sensitivity_stage,
)
from buffalo_weight.reproduction_config import load_report_contract
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_feature_evaluation import RecordingFeatureBaseline


class FakeStageMaskLoader:
    """Synthetic mask loader for stage testing."""

    def load_mask(self, file_name: str) -> np.ndarray:
        """Return a centered 40×40 block in a 200×200 image.

        Example: ``loader.load_mask("img01")`` returns test mask.
        """
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[80:120, 80:120] = 1.0
        return mask


class DiagnosticSensitivityStageTest(unittest.TestCase):
    def test_dry_run_reports_reconstructible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            status = run_diagnostic_sensitivity_stage(contract, dry_run=True)
            self.assertEqual(status, "reconstructible")

    def test_stage_produces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            status = run_diagnostic_sensitivity_stage(
                contract, dry_run=False,
                mask_loader=FakeStageMaskLoader(),
                random_forest_baseline=RecordingFeatureBaseline(),
            )
            self.assertEqual(status, "rebuilt")
            out_dir = diagnostic_sensitivity_output_dir(contract)
            self.assertTrue((out_dir / "manifest.json").is_file())
            self.assertTrue((out_dir / "sensitivity_perturbations.csv").is_file())
            self.assertTrue((out_dir / "morphology_eligibility.csv").is_file())
            report_text = (out_dir / "sensitivity_report.md").read_text(encoding="utf-8")
            self.assertIn(
                "Deslocamentos de 5% que cortam o foreground são rejeitados da análise.",
                report_text,
            )
            self.assertNotIn("nunca cortam foreground", report_text)
            # Demo PNG should exist since the fake masks are eligible
            self.assertTrue((out_dir / "morphology_demo.png").is_file())
            # Verify manifest
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertGreater(manifest["total_perturbation_records"], 0)
            self.assertEqual(manifest["total_mask_count"], 132)

    def test_dry_run_after_rebuild_reports_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            run_diagnostic_sensitivity_stage(
                contract, dry_run=False,
                mask_loader=FakeStageMaskLoader(),
                random_forest_baseline=RecordingFeatureBaseline(),
            )
            status = run_diagnostic_sensitivity_stage(contract, dry_run=True)
            self.assertEqual(status, "reusable")


if __name__ == "__main__":
    unittest.main()
