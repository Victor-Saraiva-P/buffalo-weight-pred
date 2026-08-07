"""CLI tests for confirm-diagnostics command.

Reference: GitHub Issue #27.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.feature_confirmation_environment import FeatureConfirmationEnvironment
from buffalo_weight.hashing import sha256_file
from buffalo_weight.report_cli import main


class FakeCleanEnvironment(FeatureConfirmationEnvironment):
    def worktree_changes(self, repository_root: Path) -> list[str]:
        return []


class TestDiagnosticConfirmationCLI(unittest.TestCase):
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

        self._setup_mock_source_stages()

        self.review_dir = self.temp_dir / "review"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.review_dir / "expanded_diagnostics_report.md"
        self.report_path.write_text(
            "# Relatório\n\nMAE OOF Pós-Seleção\n\n## Registro de revisão humana\n- Status: revisado\n",
            encoding="utf-8",
        )
        self.contract_path = self.review_dir / "diagnostics_contract.json"
        self.human_contract = {
            "schema_version": 1,
            "status": "confirmed",
            "diagnostic_scope": "expanded",
            "source_report_sha256": sha256_file(self.report_path),
            "no_decision_reopening": True,
            "human_decision": {
                "decision_url": "https://github.com/Victor-Saraiva-P/buffalo-weight-pred/issues/27",
                "reviewer": "Especialista",
                "reviewed_at": "2026-08-06",
            },
        }
        self.contract_path.write_text(json.dumps(self.human_contract), encoding="utf-8")

    def _setup_mock_source_stages(self) -> None:
        diag_root = self.artifacts_root / "diagnostics"

        desc_dir = diag_root / "descriptive"
        desc_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(desc_dir / "sample_coverage.csv", ["stratum_type", "stratum_value", "sample_count"], [["weight_category", "B1", "5"]])
        self._write_csv(desc_dir / "stratified_metrics.csv", ["configuration", "evaluation_role", "stratum_type", "stratum_value", "sample_count", "mae_kg", "median_abs_error_kg", "bias_kg"], [["random_forest_baseline", "baseline", "weight_category", "B1", "5", "12.5", "10.0", "-2.0"]])
        self._write_csv(desc_dir / "residual_correlations.csv", ["configuration_1", "configuration_2", "evaluation_role_1", "evaluation_role_2", "pearson_r"], [["random_forest_baseline", "dense", "baseline", "baseline", "0.85"]])
        self._write_csv(desc_dir / "notable_cases.csv", ["file_name", "case_type", "observed_weight_kg", "weight_category", "farm", "resolution", "metric_value"], [["mask_001.png", "shared_hard_case", "450.0", "B1", "Faco", "1024x768", "45.0"]])
        (desc_dir / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

        learn_dir = diag_root / "learning_curves"
        learn_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(learn_dir / "learning_curves_summary.csv", ["configuration", "fraction", "mean_n_train", "mean_mae_kg", "std_mae_kg", "mean_bias_kg", "reused_points_count"], [["random_forest_baseline", "1.00", "105.0", "15.2", "1.2", "-1.0", "5"]])
        (learn_dir / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

        sens_dir = diag_root / "sensitivity"
        sens_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(sens_dir / "sensitivity_perturbations.csv", ["configuration", "evaluation_scope", "file_name", "perturbation", "status", "rejection_reason", "original_prediction_kg", "perturbed_prediction_kg", "delta_kg"], [["random_forest_baseline", "all_eligible", "mask_001.png", "erosion_3x3", "evaluated", "", "450.0", "448.0", "-2.0"]])
        (sens_dir / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    def _write_csv(self, path: Path, headers: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)

    def test_cli_confirm_diagnostics_dry_run(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "confirm-diagnostics",
                "--config", str(self.config_path),
                "--contract", str(self.contract_path),
                "--report", str(self.report_path),
                "--dry-run",
            ],
            stdout=stdout,
            stderr=stderr,
            feature_confirmation_environment=FakeCleanEnvironment(),
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("diagnostic_confirmation: released", stdout.getvalue())

    def test_cli_confirm_diagnostics_execution(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            [
                "confirm-diagnostics",
                "--config", str(self.config_path),
                "--contract", str(self.contract_path),
                "--report", str(self.report_path),
            ],
            stdout=stdout,
            stderr=stderr,
            feature_confirmation_environment=FakeCleanEnvironment(),
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("diagnostic_confirmation: confirmed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
