from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_cli import main
from buffalo_weight.resnet_baseline_artifacts import METRIC_COLUMNS, PREDICTION_COLUMNS
from tests.fake_resnet_baseline import (
    FailingResNetBaselineRunner,
    FixedResNetBaselineProvenance,
    FixedResNetBaselineRunner,
)
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_feature_confirmation_cli import _prepare_human_review, _run_confirmation


class ResNetBaselineCliTest(unittest.TestCase):
    def test_builds_one_oof_prediction_per_mask_and_grouped_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            runner = FixedResNetBaselineRunner()

            result, stdout, stderr = run_baselines(fixture, runner)

            self.assertEqual(result, 0, stderr)
            self.assertIn("baselines: released", stdout)
            self.assertIn("resnet18_baseline: rebuilt", stdout)
            self.assertLess(
                stdout.index("resnet18_baseline: absent"),
                stdout.index("resnet18_baseline: rebuilt"),
            )
            output_dir = baseline_output_dir(fixture)
            prediction_columns, predictions = read_csv(output_dir / "predictions.csv")
            metric_columns, metrics = read_csv(output_dir / "fold_metrics.csv")
            self.assert_prediction_artifact(prediction_columns, predictions, fixture.sample_count)
            self.assert_metric_artifact(metric_columns, metrics)
            self.assert_manifest_is_complete(output_dir, fixture.sample_count)

    def assert_prediction_artifact(
        self, columns: list[str], rows: list[dict[str, str]], expected_count: int,
    ) -> None:
        self.assertEqual(columns, PREDICTION_COLUMNS)
        self.assertEqual(len(rows), expected_count)
        self.assertEqual(len({row["file_name"] for row in rows}), expected_count)
        self.assertEqual(rows, sorted(rows, key=lambda row: row["file_name"]))

    def assert_metric_artifact(
        self, columns: list[str], rows: list[dict[str, str]],
    ) -> None:
        self.assertEqual(columns, METRIC_COLUMNS)
        self.assertEqual({row["scope"] for row in rows}, {"fold", "oof"})
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["population"] for row in rows}, {"all", "B1", "B10"})
        category_rows = [row for row in rows if row["population"] != "all"]
        self.assertTrue(all(row["rmse_kg"] == row["r2"] == "" for row in category_rows))

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

    def test_nonofficial_execution_and_invalid_commit_are_obsolete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            runner = FixedResNetBaselineRunner()
            self.assertEqual(run_baselines(fixture, runner)[0], 0)
            manifest_path = baseline_output_dir(fixture) / "manifest.json"
            original = manifest_path.read_text()
            for field, value in (("execution", _nonofficial_execution()),
                                 ("source_commit", "g" * 40)):
                manifest = json.loads(original)
                manifest[field] = value
                manifest_path.write_text(json.dumps(manifest))
                self.assertIn("obsolete", run_baselines(fixture, runner, "--dry-run")[1])
                manifest_path.write_text(original)

    def test_failed_retraining_removes_obsolete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            runner = FixedResNetBaselineRunner()
            self.assertEqual(run_baselines(fixture, runner)[0], 0)
            predictions = baseline_output_dir(fixture) / "predictions.csv"
            predictions.write_text(f"{predictions.read_text()}tampered\n")

            result, _, stderr = run_baselines(fixture, FailingResNetBaselineRunner())

            self.assertEqual(result, 1)
            self.assertIn("injected ResNet training failure", stderr)
            self.assertFalse(baseline_output_dir(fixture).exists())

    def test_concurrent_baseline_execution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_fixture(Path(directory))
            lock_path = fixture.root / "generated" / "report" / "baselines" / ".train.lock"
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text(str(os.getpid()))

            result, _, stderr = run_baselines(fixture, FixedResNetBaselineRunner())

            lock_path.unlink()
            self.assertEqual(result, 1)
            self.assertIn("locked by", stderr)

    def assert_manifest_is_complete(self, output_dir: Path, row_count: int) -> None:
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["model_config"], "resnet18_pretrained_partial")
        self.assertEqual(manifest["outputs"]["predictions.csv"]["row_count"], row_count)
        self.assertEqual(set(manifest["outputs"]), {"fold_metrics.csv", "predictions.csv"})
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
    )
    return result, stdout.getvalue(), stderr.getvalue()


def baseline_output_dir(fixture: CuratedInputsFixture) -> Path:
    return fixture.root / "generated" / "report" / "baselines" / "resnet18_pretrained_partial"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def _nonofficial_execution() -> dict[str, object]:
    return {"device": "cpu", "deterministic": True, "official": False}


if __name__ == "__main__":
    unittest.main()
