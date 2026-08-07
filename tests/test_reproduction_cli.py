from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.hashing import sha256_file
from buffalo_weight.report_cli import main
from buffalo_weight.report_inputs import clean_reconstructible_stage
from buffalo_weight.report_reproduction import (
    ReproductionDependencies,
    run_report_reproduction,
)
from buffalo_weight.reproduction_config import load_report_contract
from tests.fake_baseline_comparison import FixedBaselineComparisonProvenance
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_compact_cnn import FixedCompactCnnProvenance, RecordingCompactCnnAdapter
from tests.fake_dense_baseline import (
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
    FixedDenseBaselineRunner,
)
from tests.fake_feature_confirmation import FixedFeatureConfirmationEnvironment
from tests.fake_feature_evaluation import RecordingFeatureBaseline
from tests.fake_feature_selection import FixedFeatureEvidenceRunner
from tests.fake_report_provenance import FixedReportProvenance
from tests.fake_resnet_baseline import FixedResNetBaselineProvenance, FixedResNetBaselineRunner
from tests.fake_tuning_provenance import FixedTuningProvenance
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_approach_confirmation_cli import (
    _prepare_baseline_comparison as prepare_baseline_comparison,
    _prepare_human_review as prepare_approach_review,
    _run_confirmation as run_approach_confirmation,
)


class ReproductionCliTest(unittest.TestCase):
    def test_dry_run_presents_complete_graph_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory), sample_count=132)
            stdout, stderr = io.StringIO(), io.StringIO()
            code = main(
                ["reproduce", "--dry-run", "--config", str(fixture.config_path)],
                stdout=stdout,
                stderr=stderr,
                report_provenance=FixedReportProvenance(),
            )
            output = stdout.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("reproduction_plan:", output)
            self.assertIn("[stage] inputs: rebuild", output)
            self.assertIn("[gate] confirm-features: blocked", output)
            self.assertIn("[gate] confirm-approach: blocked", output)
            self.assertIn("[gate] confirm-diagnostics: blocked", output)
            self.assertFalse((fixture.root / "generated" / "report" / "inputs").exists())

    def test_unconfirmed_gate_halts_reproduction_with_actionable_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory), sample_count=132)
            stdout, stderr = io.StringIO(), io.StringIO()
            deps = _make_repro_deps()
            code = main(
                ["reproduce", "--config", str(fixture.config_path)],
                stdout=stdout,
                stderr=stderr,
                snapshot_publisher=deps.snapshot_publisher,
                report_provenance=deps.report_provenance,
                feature_evidence_runner=deps.feature_evidence_runner,
                feature_selection_provenance=deps.feature_selection_provenance,
                feature_confirmation_environment=deps.feature_confirmation_environment,
                random_forest_baseline=deps.random_forest_baseline,
                baseline_provenance=deps.baseline_provenance,
                dense_baseline_dependencies=deps.dense_baseline_dependencies,
                compact_cnn_adapter=deps.compact_cnn_adapter,
                compact_cnn_provenance=deps.compact_cnn_provenance,
                resnet_baseline_runner=deps.resnet_baseline_runner,
                resnet_baseline_provenance=deps.resnet_baseline_provenance,
                baseline_comparison_provenance=deps.baseline_comparison_provenance,
                tuning_provenance=deps.tuning_provenance,
            )
            output = stdout.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("inputs: rebuilt", output)
            self.assertIn("feature-selection: rebuilt", output)
            self.assertIn("blocked: confirm-features", output)
            self.assertIn("Action required:", output)
            self.assertIn("confirm-features", output)

    def test_full_reproduction_succeeds_when_all_gates_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_fully_confirmed_environment(Path(directory))
            stdout, stderr = io.StringIO(), io.StringIO()
            deps = _make_repro_deps()
            code = main(
                ["reproduce", "--config", str(fixture.config_path)],
                stdout=stdout,
                stderr=stderr,
                snapshot_publisher=deps.snapshot_publisher,
                report_provenance=deps.report_provenance,
                feature_evidence_runner=deps.feature_evidence_runner,
                feature_selection_provenance=deps.feature_selection_provenance,
                feature_confirmation_environment=deps.feature_confirmation_environment,
                random_forest_baseline=deps.random_forest_baseline,
                baseline_provenance=deps.baseline_provenance,
                dense_baseline_dependencies=deps.dense_baseline_dependencies,
                compact_cnn_adapter=deps.compact_cnn_adapter,
                compact_cnn_provenance=deps.compact_cnn_provenance,
                resnet_baseline_runner=deps.resnet_baseline_runner,
                resnet_baseline_provenance=deps.resnet_baseline_provenance,
                baseline_comparison_provenance=deps.baseline_comparison_provenance,
                tuning_provenance=deps.tuning_provenance,
            )
            output = stdout.getvalue()
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("reproduction: complete", output)
            self.assertIn("confirm-features: released", output)
            self.assertIn("confirm-approach: released", output)
            self.assertIn("confirm-diagnostics: released", output)

    def test_reexecution_resumes_and_rebuilds_only_affected_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_fully_confirmed_environment(Path(directory))
            deps = _make_repro_deps()
            contract = load_report_contract(fixture.config_path)

            stdout_first = io.StringIO()
            run_report_reproduction(contract, False, deps, stdout_first)
            self.assertIn("reproduction: complete", stdout_first.getvalue())

            # Corrupt feature selection manifest to force rebuild of feature-selection and downstream
            sel_manifest = fixture.root / "generated" / "report" / "feature_selection" / "manifest.json"
            sel_manifest.write_text(sel_manifest.read_text() + "tampered\n")

            stdout_second = io.StringIO()
            code_second = run_report_reproduction(contract, False, deps, stdout_second)
            output_second = stdout_second.getvalue()
            self.assertEqual(code_second, 0)
            self.assertIn("inputs: reusable", output_second)
            self.assertIn("feature-selection: rebuilt", output_second)
            self.assertIn("reproduction: complete", output_second)

    def test_stage_cleaning_does_not_reach_confirmed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _prepare_fully_confirmed_environment(Path(directory))
            contract = load_report_contract(fixture.config_path)
            confirmed_dir = contract.confirmed_feature_selection_dir
            self.assertTrue(confirmed_dir.exists())

            removed = clean_reconstructible_stage(contract, "inputs")
            self.assertIn("inputs", removed)
            self.assertIn("feature_selection", removed)
            self.assertTrue(confirmed_dir.exists())
            self.assertTrue(fixture.index_path.exists())


def _make_repro_deps() -> ReproductionDependencies:
    return ReproductionDependencies(
        report_provenance=FixedReportProvenance(),
        feature_evidence_runner=FixedFeatureEvidenceRunner(),
        feature_confirmation_environment=FixedFeatureConfirmationEnvironment(),
        random_forest_baseline=RecordingFeatureBaseline(),
        baseline_provenance=FixedBaselineProvenance(),
        dense_baseline_dependencies=DenseBaselineDependencies(
            FixedDenseBaselineRunner(), FixedDenseBaselineProvenance(), FixedCudaRuntimeProbe(),
        ),
        compact_cnn_adapter=RecordingCompactCnnAdapter(),
        compact_cnn_provenance=FixedCompactCnnProvenance(),
        resnet_baseline_runner=FixedResNetBaselineRunner(),
        resnet_baseline_provenance=FixedResNetBaselineProvenance(),
        baseline_comparison_provenance=FixedBaselineComparisonProvenance(),
        tuning_provenance=FixedTuningProvenance(),
    )


def _prepare_fully_confirmed_environment(root: Path) -> CuratedInputsFixture:
    fixture = prepare_baseline_comparison(root)

    # 1. Confirm approach
    app_contract, app_report = prepare_approach_review(
        fixture, "random_forest", "random_forest_baseline",
    )
    code_app, _, err_app = run_approach_confirmation(fixture, app_contract, app_report)
    if code_app != 0:
        raise AssertionError(f"approach confirmation failed: {err_app}")

    # 2. Run tuning stage to produce tuning source evidence
    _run_tuning_stage(fixture)

    # 3. Setup mock diagnostic source stages
    _setup_mock_diagnostic_source_stages(fixture)

    # 4. Confirm diagnostics
    _confirm_diagnostics_for_fixture(fixture)

    return fixture


def _run_tuning_stage(fixture: CuratedInputsFixture) -> None:
    deps = _make_repro_deps()
    code = main(
        ["tuning", "--config", str(fixture.config_path)],
        stdout=io.StringIO(), stderr=io.StringIO(),
        report_provenance=deps.report_provenance,
        random_forest_baseline=deps.random_forest_baseline,
        baseline_provenance=deps.baseline_provenance,
        dense_baseline_dependencies=deps.dense_baseline_dependencies,
        compact_cnn_adapter=deps.compact_cnn_adapter,
        resnet_baseline_runner=deps.resnet_baseline_runner,
        tuning_provenance=deps.tuning_provenance,
    )
    if code != 0:
        raise AssertionError(f"tuning failed with exit code {code}")


def _setup_mock_diagnostic_source_stages(fixture: CuratedInputsFixture) -> None:
    artifacts_root = fixture.root / "generated" / "report"
    diag_root = artifacts_root / "diagnostics"

    desc_dir = diag_root / "descriptive"
    desc_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = [["weight_category", f"B{i}", "27" if i <= 2 else "26"] for i in range(1, 6)]
    _write_csv(desc_dir / "sample_coverage.csv", ["stratum_type", "stratum_value", "sample_count"], coverage_rows)
    _write_csv(desc_dir / "stratified_metrics.csv", ["configuration", "evaluation_role", "stratum_type", "stratum_value", "sample_count", "mae_kg", "median_abs_error_kg", "bias_kg"], [["random_forest_baseline", "baseline", "weight_category", "B1", "5", "12.5", "10.0", "-2.0"]])
    _write_csv(desc_dir / "residual_correlations.csv", ["configuration_1", "configuration_2", "evaluation_role_1", "evaluation_role_2", "pearson_r"], [["random_forest_baseline", "dense", "baseline", "baseline", "0.85"]])
    _write_csv(desc_dir / "notable_cases.csv", ["file_name", "case_type", "observed_weight_kg", "weight_category", "farm", "resolution", "metric_value"], [["mask_001.png", "shared_hard_case", "450.0", "B1", "Faco", "1024x768", "45.0"]])
    (desc_dir / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    learn_dir = diag_root / "learning_curves"
    learn_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(learn_dir / "learning_curves_summary.csv", ["configuration", "fraction", "mean_n_train", "mean_mae_kg", "std_mae_kg", "mean_bias_kg", "reused_points_count"], [["random_forest_baseline", "1.00", "105.0", "15.2", "1.2", "-1.0", "5"]])
    (learn_dir / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    sens_dir = diag_root / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(sens_dir / "sensitivity_perturbations.csv", ["configuration", "evaluation_scope", "file_name", "perturbation", "status", "rejection_reason", "original_prediction_kg", "perturbed_prediction_kg", "delta_kg"], [["random_forest_baseline", "all_eligible", "mask_001.png", "erosion_3x3", "evaluated", "", "450.0", "448.0", "-2.0"]])
    (sens_dir / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _confirm_diagnostics_for_fixture(fixture: CuratedInputsFixture) -> None:
    review_dir = fixture.root / "review_diag"
    review_dir.mkdir(parents=True, exist_ok=True)
    report_path = review_dir / "expanded_diagnostics_report.md"
    report_path.write_text(
        "# Relatório\n\nMAE OOF Pós-Seleção\n\n## Registro de revisão humana\n- Status: revisado\n",
        encoding="utf-8",
    )
    contract_path = review_dir / "diagnostics_contract.json"
    contract_content = {
        "schema_version": 1,
        "status": "confirmed",
        "diagnostic_scope": "expanded",
        "source_report_sha256": sha256_file(report_path),
        "no_decision_reopening": True,
        "human_decision": {
            "decision_url": "https://github.com/Victor-Saraiva-P/buffalo-weight-pred/issues/27",
            "reviewer": "Especialista",
            "reviewed_at": "2026-08-06",
        },
    }
    contract_path.write_text(json.dumps(contract_content), encoding="utf-8")

    stdout, stderr = io.StringIO(), io.StringIO()
    code = main(
        [
            "confirm-diagnostics",
            "--config", str(fixture.config_path),
            "--contract", str(contract_path),
            "--report", str(report_path),
        ],
        stdout=stdout,
        stderr=stderr,
        feature_confirmation_environment=FixedFeatureConfirmationEnvironment(),
        report_provenance=FixedReportProvenance(),
    )
    if code != 0:
        raise AssertionError(f"confirm-diagnostics returned exit code {code}: {stderr.getvalue()}")


if __name__ == "__main__":
    unittest.main()
