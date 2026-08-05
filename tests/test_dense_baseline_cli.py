from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from buffalo_weight.dense_baseline_evaluation import (
    DenseBaselineRunner,
    ScientificDenseBaselineRunner,
)
from buffalo_weight.dense_baseline_provenance import (
    DenseBaselineProvenance,
    SystemDenseBaselineProvenance,
)
from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.dense_feature_adapter import DenseFeatureAdapter
from buffalo_weight.environment_contract import RuntimeProbe
from buffalo_weight.feature_baselines import DenseFeatureBaseline
from buffalo_weight.feature_selection_manifest import artifact_output_records
from buffalo_weight.report_cli import main
from buffalo_weight.report_provenance import ReportProvenance
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_compact_cnn import FixedCompactCnnProvenance, RecordingCompactCnnAdapter
from tests.fake_dense_baseline import (
    ChangedInputsProvenance,
    FailingDenseBaselineRunner,
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
    FixedDenseBaselineRunner,
    RecordingDenseFeatureAdapter,
    UnavailableDenseRuntimeProbe,
)
from tests.fake_feature_evaluation import RecordingFeatureBaseline
from tests.fake_report_provenance import FixedReportProvenance
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_feature_confirmation_cli import _prepare_human_review, _run_confirmation


class DenseBaselineCliTest(unittest.TestCase):
    def test_builds_oof_predictions_metrics_and_manifest_from_confirmed_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory), sample_count=100)
            contract_path, report_path = _prepare_human_review(
                fixture, ("area", "perimeter"),
            )
            self.assertEqual(_run_confirmation(fixture, contract_path, report_path)[0], 0)
            runner = FixedDenseBaselineRunner()
            runtime = FixedCudaRuntimeProbe()

            result, stdout, stderr = run_baselines(fixture, runner, runtime)

            self.assertEqual(result, 0, stderr)
            self.assertIn("dense: rebuilt", stdout)
            self.assertEqual(runtime.compute_checks, 1)
            self.assertEqual(runner.calls[0][1], ("area", "perimeter"))
            output_dir = fixture.root / "generated" / "report" / "baselines" / "dense"
            assert_public_tables(self, output_dir, fixture.sample_count)
            assert_dense_manifest(self, output_dir)

    def test_reuses_only_current_outputs_and_dry_run_never_probes_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, runner = prepared_fixture(Path(directory))
            built = run_baselines(fixture, runner, FixedCudaRuntimeProbe())
            self.assertEqual(built[0], 0, built[2])
            manifest = dense_output_dir(fixture) / "manifest.json"
            modified = manifest.stat().st_mtime_ns
            dry_runtime = FixedCudaRuntimeProbe()

            reused = run_baselines(fixture, runner, dry_runtime, "--dry-run")

            self.assertIn("dense: reusable", reused[1])
            self.assertEqual(dry_runtime.compute_checks, 0)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(manifest.stat().st_mtime_ns, modified)
            predictions = manifest.parent / "predictions.csv"
            predictions.write_text(predictions.read_text() + "tampered\n")
            obsolete = run_baselines(fixture, runner, dry_runtime, "--dry-run")
            self.assertIn("dense: obsolete", obsolete[1])
            self.assertEqual(dry_runtime.compute_checks, 0)

    def test_missing_cuda_fails_before_evaluation_and_artifact_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, runner = prepared_fixture(Path(directory))
            runtime = UnavailableDenseRuntimeProbe()

            result, _, stderr = run_baselines(fixture, runner, runtime)

            self.assertEqual(result, 1)
            self.assertIn("CUDA environment was", stderr)
            self.assertEqual(runtime.compute_checks, 1)
            self.assertEqual(runner.calls, [])
            output_dir = fixture.root / "generated" / "report" / "baselines" / "dense"
            self.assertFalse(output_dir.exists())

    def test_recipe_and_scientific_environment_changes_invalidate_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, runner = prepared_fixture(Path(directory))
            self.assertEqual(run_baselines(fixture, runner, FixedCudaRuntimeProbe())[0], 0)
            runtime = FixedCudaRuntimeProbe()

            changed_recipe = run_baselines(
                fixture, runner, runtime, "--dry-run",
                provenance=FixedDenseBaselineProvenance("6"),
            )
            changed_environment = run_baselines(
                fixture, runner, runtime, "--dry-run",
                provenance=FixedDenseBaselineProvenance(torch_version="2.14.0"),
            )

            self.assertIn("dense: obsolete", changed_recipe[1])
            self.assertIn("dense: obsolete", changed_environment[1])
            self.assertEqual(runtime.compute_checks, 0)
            self.assertEqual(len(runner.calls), 1)

    def test_source_commit_is_audit_only_and_does_not_invalidate_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, runner = prepared_fixture(Path(directory))
            built = run_baselines(fixture, runner, FixedCudaRuntimeProbe())
            self.assertEqual(built[0], 0, built[2])

            reused = run_baselines(
                fixture, runner, FixedCudaRuntimeProbe(), "--dry-run",
                provenance=FixedDenseBaselineProvenance(source_commit="3" * 40),
            )

            self.assertIn("dense: reusable", reused[1])
            self.assertEqual(len(runner.calls), 1)

    def test_semantically_invalid_prediction_order_rejects_current_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, runner = prepared_fixture(Path(directory))
            built = run_baselines(fixture, runner, FixedCudaRuntimeProbe())
            self.assertEqual(built[0], 0, built[2])
            output_dir = fixture.root / "generated" / "report" / "baselines" / "dense"
            prediction_path = output_dir / "predictions.csv"
            lines = prediction_path.read_text().splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            prediction_path.write_text("\n".join(lines) + "\n")
            _refresh_manifest_outputs(output_dir)

            result = run_baselines(
                fixture, runner, FixedCudaRuntimeProbe(), "--dry-run",
            )

            self.assertIn("dense: obsolete", result[1])

    def test_failed_retraining_leaves_no_stale_dense_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, runner = prepared_fixture(Path(directory))
            built = run_baselines(fixture, runner, FixedCudaRuntimeProbe())
            self.assertEqual(built[0], 0, built[2])
            output_dir = fixture.root / "generated" / "report" / "baselines" / "dense"
            predictions = output_dir / "predictions.csv"
            predictions.write_text(predictions.read_text() + "tampered\n")

            failed = run_baselines(
                fixture, FailingDenseBaselineRunner(), FixedCudaRuntimeProbe(),
            )

            self.assertEqual(failed[0], 1)
            self.assertIn("deliberate test failure", failed[2])
            self.assertFalse(output_dir.exists())

    def test_system_provenance_tracks_only_dense_scientific_dependencies(self) -> None:
        provenance = SystemDenseBaselineProvenance(FixedCudaRuntimeProbe())

        environment = provenance.scientific_environment()

        self.assertEqual(
            set(cast(dict[str, str], environment["direct_dependencies"])),
            {"numpy", "scikit-learn", "torch"},
        )
        self.assertRegex(provenance.dense_baseline_recipe_hash(), r"^[0-9a-f]{64}$")

    def test_confirmed_gate_accepts_identical_tables_from_a_later_input_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, runner = prepared_fixture(Path(directory))
            rebuilt = main(
                ["inputs", "--config", str(fixture.config_path)], stdout=io.StringIO(),
                stderr=io.StringIO(), report_provenance=ChangedInputsProvenance(),
            )
            self.assertEqual(rebuilt, 0)

            result, stdout, stderr = run_baselines(
                fixture, runner, FixedCudaRuntimeProbe(),
                inputs_provenance=ChangedInputsProvenance(),
            )

            self.assertEqual(result, 0, stderr)
            self.assertIn("dense: rebuilt", stdout)

    def test_real_orchestration_isolates_inner_selection_and_refits_external_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture, _ = prepared_fixture(Path(directory), sample_count=100)
            adapter = RecordingDenseFeatureAdapter()
            baseline = DenseFeatureBaseline(adapter=cast(DenseFeatureAdapter, adapter))
            runner = ScientificDenseBaselineRunner(baseline)

            result, _, stderr = run_baselines(fixture, runner, FixedCudaRuntimeProbe())

            self.assertEqual(result, 0, stderr)
            self.assertEqual(adapter.selection_sizes, [(64, 16)] * 5)
            self.assertEqual(adapter.refit_sizes, [80] * 5)
            self.assertTrue(all(recipe.inner_seed == 43 and recipe.training_seed == 44
                                for recipe in adapter.recipes))
            output_dir = fixture.root / "generated" / "report" / "baselines" / "dense"
            manifest = json.loads((output_dir / "manifest.json").read_text())
            fold_training = cast(list[dict[str, object]], manifest["fold_training"])
            assert_fold_training_counts(self, fold_training)


def run_baselines(
    fixture: CuratedInputsFixture, runner: DenseBaselineRunner,
    runtime: RuntimeProbe, *extra: str,
    provenance: DenseBaselineProvenance | None = None,
    inputs_provenance: ReportProvenance | None = None,
) -> tuple[int, str, str]:
    """Run the injected CLI; for example, tests avoid scientific CUDA training."""
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["baselines", "--config", str(fixture.config_path), *extra],
        stdout=stdout, stderr=stderr,
        dense_baseline_dependencies=DenseBaselineDependencies(
            runner, provenance or FixedDenseBaselineProvenance(), runtime,
        ),
        random_forest_baseline=RecordingFeatureBaseline(),
        baseline_provenance=FixedBaselineProvenance(),
        report_provenance=inputs_provenance or FixedReportProvenance(),
        compact_cnn_adapter=RecordingCompactCnnAdapter(),
        compact_cnn_provenance=FixedCompactCnnProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


def prepared_fixture(
    root: Path, sample_count: int = 100,
) -> tuple[CuratedInputsFixture, FixedDenseBaselineRunner]:
    """Build a released feature gate; for example, cache tests begin from valid inputs."""
    fixture = CuratedInputsFixture(root, sample_count)
    contract_path, report_path = _prepare_human_review(fixture, ("area", "perimeter"))
    confirmation = _run_confirmation(fixture, contract_path, report_path)
    if confirmation[0] != 0:
        raise AssertionError(f"feature confirmation returned {confirmation!r}; expected success")
    return fixture, FixedDenseBaselineRunner()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one public artifact; for example, tests inspect prediction rows."""
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def assert_public_tables(test: unittest.TestCase, output_dir: Path, sample_count: int) -> None:
    predictions = read_csv(output_dir / "predictions.csv")
    metrics = read_csv(output_dir / "fold_metrics.csv")
    test.assertEqual(len(predictions), sample_count)
    test.assertEqual(len({row["file_name"] for row in predictions}), sample_count)
    test.assertTrue(all(float(row["residual_kg"]) == int(row["fold"])
                        for row in predictions))
    test.assertEqual(len(metrics), 18)
    oof_all = next(row for row in metrics
                   if row["scope"] == "oof" and row["population"] == "all")
    test.assertEqual((float(oof_all["mae_kg"]), float(oof_all["bias_kg"])), (3.0, 3.0))


def assert_dense_manifest(test: unittest.TestCase, output_dir: Path) -> None:
    manifest = json.loads((output_dir / "manifest.json").read_text())
    test.assertEqual(manifest["status"], "complete")
    test.assertEqual(manifest["selected_features"], ["area", "perimeter"])
    test.assertEqual(manifest["execution"]["device"], "cuda")
    test.assertEqual(set(manifest["outputs"]), {"predictions.csv", "fold_metrics.csv"})
    recipe = manifest["recipe"]
    test.assertEqual(recipe["hidden_layers"], [64, 32])
    test.assertEqual(recipe["dropout"], 0.20)
    test.assertEqual((recipe["optimizer"], recipe["loss"]), ("adamw", "l1_standardized_target"))


def assert_fold_training_counts(
    test: unittest.TestCase, fold_training: list[dict[str, object]],
) -> None:
    expected = {
        "selection_count": 64, "stopping_count": 16,
        "retrain_count": 80, "held_out_count": 20,
    }
    for name, count in expected.items():
        test.assertEqual([row[name] for row in fold_training], [count] * 5)


def dense_output_dir(fixture: CuratedInputsFixture) -> Path:
    return cast(Path, fixture.root / "generated" / "report" / "baselines" / "dense")


def _refresh_manifest_outputs(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"] = artifact_output_records(
        output_dir, ("predictions.csv", "fold_metrics.csv"),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
