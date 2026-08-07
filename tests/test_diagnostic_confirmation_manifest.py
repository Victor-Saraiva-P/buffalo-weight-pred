"""Unit tests for diagnostic confirmation manifest creation and validation.

Reference: GitHub Issue #27.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.diagnostic_confirmation_manifest import (
    build_confirmed_diagnostic_manifest,
    validate_canonical_tables,
    validate_confirmed_diagnostic_manifest,
)
from buffalo_weight.reproduction_config import load_report_contract


class TestDiagnosticConfirmationManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.artifacts_root = self.repo_dir / "generated" / "report"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.masks_dir = self.temp_dir / "masks"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.mask_index = self.temp_dir / "mask_index.csv"
        self.mask_index.write_text("file_name,farm,weight_kg,tag\n", encoding="utf-8")

        self.config_path = self.temp_dir / "report.yaml"
        self.config_content = (
            "inputs:\n"
            f"  mask_index_path: {self.mask_index}\n"
            f"  masks_dir: {self.masks_dir}\n"
            "  expected_mask_count: 5\n"
            "  canonical_long_side: 1024\n"
            "  weight_category_count: 5\n"
            "  fold_count: 5\n"
            "  fold_seed: 42\n"
            "artifacts:\n"
            f"  root: {self.artifacts_root}\n"
        )
        self.config_path.write_text(self.config_content, encoding="utf-8")
        self.contract = load_report_contract(self.config_path)

        self.package_dir = self.temp_dir / "package"
        self.package_dir.mkdir(parents=True, exist_ok=True)
        self._write_mock_canonical_tables(5)
        self.report_path = self.package_dir / "expanded_diagnostics_report.md"
        self.report_path.write_text(
            "# Relatório\n\nMAE OOF Pós-Seleção\n\n## Registro de revisão humana\n- Status: revisado\n",
            encoding="utf-8",
        )
        self.human_contract = {
            "human_decision": {
                "decision_url": "https://github.com/Victor-Saraiva-P/buffalo-weight-pred/issues/27",
                "reviewer": "Especialista",
                "reviewed_at": "2026-08-06",
            }
        }

    def _write_mock_canonical_tables(self, sample_count: int) -> None:
        # sample_coverage.csv
        cov_path = self.package_dir / "sample_coverage.csv"
        with cov_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stratum_type", "stratum_value", "sample_count"])
            w.writerow(["weight_category", "B1", str(sample_count)])

        # stratified_metrics.csv
        strat_path = self.package_dir / "stratified_metrics.csv"
        with strat_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["configuration", "evaluation_role", "stratum_type", "stratum_value", "sample_count", "mae_kg", "median_abs_error_kg", "bias_kg"])
            w.writerow(["random_forest_baseline", "baseline", "weight_category", "B1", str(sample_count), "12.5", "10.0", "-2.0"])

        # residual_correlations.csv
        corr_path = self.package_dir / "residual_correlations.csv"
        with corr_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["configuration_1", "configuration_2", "evaluation_role_1", "evaluation_role_2", "pearson_r"])
            w.writerow(["random_forest_baseline", "dense", "baseline", "baseline", "0.85"])

        # notable_cases.csv
        notable_path = self.package_dir / "notable_cases.csv"
        with notable_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "case_type", "observed_weight_kg", "weight_category", "farm", "resolution", "metric_value"])
            w.writerow(["mask_001.png", "shared_hard_case", "450.0", "B1", "Faco", "1024x768", "45.0"])

        # learning_curves_summary.csv
        lc_path = self.package_dir / "learning_curves_summary.csv"
        with lc_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["configuration", "fraction", "mean_n_train", "mean_mae_kg", "std_mae_kg", "mean_bias_kg", "reused_points_count"])
            w.writerow(["random_forest_baseline", "1.00", "105.0", "15.2", "1.2", "-1.0", "5"])

        # sensitivity_perturbations.csv
        sens_path = self.package_dir / "sensitivity_perturbations.csv"
        with sens_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["configuration", "evaluation_scope", "file_name", "perturbation", "status", "rejection_reason", "original_prediction_kg", "perturbed_prediction_kg", "delta_kg"])
            w.writerow(["random_forest_baseline", "all_eligible", "mask_001.png", "erosion_3x3", "eligible", "", "450.0", "448.0", "-2.0"])

    def test_validate_canonical_tables_success(self) -> None:
        validate_canonical_tables(self.package_dir, 5)

    def test_build_and_validate_manifest(self) -> None:
        manifest = build_confirmed_diagnostic_manifest(
            self.package_dir, self.contract, self.human_contract, commit="abc1234"
        )
        self.assertEqual(manifest["status"], "confirmed")
        self.assertEqual(manifest["repository_commit"], "abc1234")
        (self.package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        validate_confirmed_diagnostic_manifest(
            manifest, self.package_dir, self.human_contract, self.contract
        )

    def test_missing_table_raises(self) -> None:
        (self.package_dir / "sample_coverage.csv").unlink()
        with self.assertRaises(ValueError) as ctx:
            validate_canonical_tables(self.package_dir, 5)
        self.assertIn("canonical table sample_coverage.csv was missing", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
