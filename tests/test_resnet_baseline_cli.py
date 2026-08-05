from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_cli import main
from tests.fake_resnet_baseline import (
    FixedResNetBaselineProvenance,
    FixedResNetBaselineRunner,
)
from tests.fake_report_provenance import FixedReportProvenance
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_feature_confirmation_cli import _prepare_human_review, _run_confirmation


PREDICTION_COLUMNS = [
    "model_config", "fold", "file_name", "weight_category", "weight_kg",
    "prediction_kg", "residual_kg", "absolute_error_kg",
]
METRIC_COLUMNS = [
    "model_config", "scope", "fold", "n", "mae_kg", "rmse_kg", "bias_kg", "r2",
]


class ResNetBaselineCliTest(unittest.TestCase):
    def test_builds_one_oof_prediction_per_mask_and_grouped_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            runner = FixedResNetBaselineRunner()

            result, stdout, stderr = run_baselines(fixture, runner)

            self.assertEqual(result, 0, stderr)
            self.assertIn("baselines: released", stdout)
            self.assertIn("resnet18_baseline: rebuilt", stdout)
            output_dir = baseline_output_dir(fixture)
            prediction_columns, predictions = read_csv(output_dir / "predictions.csv")
            metric_columns, metrics = read_csv(output_dir / "metrics.csv")
            self.assertEqual(prediction_columns, PREDICTION_COLUMNS)
            self.assertEqual(len(predictions), fixture.sample_count)
            self.assertEqual(len({row["file_name"] for row in predictions}), fixture.sample_count)
            self.assertEqual(predictions, sorted(predictions, key=lambda row: row["file_name"]))
            self.assertEqual(metric_columns, METRIC_COLUMNS)
            self.assertEqual({row["scope"] for row in metrics}, {"fold", "oof"})
            self.assertEqual(len(metrics), 6)
            self.assert_manifest_is_complete(output_dir, fixture.sample_count)

    def test_reuses_current_artifact_without_cuda_preflight_or_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            runner = FixedResNetBaselineRunner()
            self.assertEqual(run_baselines(fixture, runner)[0], 0)

            result, stdout, stderr = run_baselines(fixture, runner)

            self.assertEqual(result, 0, stderr)
            self.assertIn("resnet18_baseline: reusable", stdout)
            self.assertEqual(runner.preflight_count, 1)
            self.assertEqual(runner.evaluation_count, 1)

    def test_dry_run_reports_tampered_output_as_obsolete_without_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            runner = FixedResNetBaselineRunner()
            self.assertEqual(run_baselines(fixture, runner)[0], 0)
            predictions = baseline_output_dir(fixture) / "predictions.csv"
            predictions.write_text(f"{predictions.read_text()}tampered\n")

            result, stdout, stderr = run_baselines(fixture, runner, "--dry-run")

            self.assertEqual(result, 0, stderr)
            self.assertIn("resnet18_baseline: obsolete", stdout)
            self.assertEqual(runner.preflight_count, 1)
            self.assertEqual(runner.evaluation_count, 1)

    def test_recipe_change_invalidates_only_the_resnet_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            runner = FixedResNetBaselineRunner()
            self.assertEqual(run_baselines(fixture, runner)[0], 0)
            changed = FixedResNetBaselineProvenance("c" * 64)

            result, stdout, stderr = run_baselines(
                fixture, runner, "--dry-run", provenance=changed
            )

            self.assertEqual(result, 0, stderr)
            self.assertIn("resnet18_baseline: obsolete", stdout)
            self.assertTrue(baseline_output_dir(fixture).exists())
            self.assertEqual(runner.evaluation_count, 1)

    def assert_manifest_is_complete(self, output_dir: Path, row_count: int) -> None:
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["model_config"], "resnet18_pretrained_partial")
        self.assertEqual(manifest["outputs"]["predictions.csv"]["row_count"], row_count)
        self.assertEqual(set(manifest["outputs"]), {"metrics.csv", "predictions.csv"})
        latest_output = max((output_dir / name).stat().st_mtime_ns for name in manifest["outputs"])
        self.assertGreaterEqual(manifest_path.stat().st_mtime_ns, latest_output)


def prepared_fixture(root: Path) -> CuratedInputsFixture:
    fixture = CuratedInputsFixture(root)
    contract_path, report_path = _prepare_human_review(fixture, ("area", "perimeter"))
    result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
    if result != 0:
        raise AssertionError(f"feature confirmation returned {result}: {stderr}")
    return fixture


def run_baselines(
    fixture: CuratedInputsFixture, runner: FixedResNetBaselineRunner, *extra: str,
    provenance: FixedResNetBaselineProvenance | None = None,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["baselines", "--config", str(fixture.config_path), *extra],
        stdout=stdout, stderr=stderr, resnet_baseline_runner=runner,
        resnet_baseline_provenance=provenance or FixedResNetBaselineProvenance(),
        report_provenance=FixedReportProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


def baseline_output_dir(fixture: CuratedInputsFixture) -> Path:
    return fixture.root / "generated" / "report" / "baselines" / "resnet18_pretrained_partial"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


if __name__ == "__main__":
    unittest.main()
