from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.feature_evaluation import FeatureBaseline
from buffalo_weight.hashing import sha256_file
from buffalo_weight.report_cli import main
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_feature_evaluation import ConstantFeatureBaseline, RecordingFeatureBaseline
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_feature_confirmation_cli import _prepare_human_review, _run_confirmation


PREDICTION_COLUMNS = [
    "configuration", "evaluation_role", "file_name", "weight_category", "fold",
    "observed_weight_kg", "predicted_weight_kg", "residual_kg", "absolute_error_kg",
]
METRIC_COLUMNS = [
    "configuration", "evaluation_role", "fold", "n", "mae_kg", "rmse_kg", "bias_kg", "r2",
]
GROUPED_METRIC_COLUMNS = [
    "configuration", "evaluation_role", "population", "n", "mae_kg", "rmse_kg",
    "bias_kg", "r2",
]
MANIFEST_TAMPERING_CASES: tuple[tuple[str, object], ...] = (
    ("selected_features", ["perimeter", "area"]),
    ("fold_seed", 999),
    ("training_seed", 43),
    ("dependencies", {}),
    ("validations", []),
    ("unexpected_field", "not allowed"),
)


class RandomForestBaselineCliTest(unittest.TestCase):
    def test_execution_builds_isolated_oof_candidate_and_training_mean_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(directory)
            random_forest = RecordingFeatureBaseline()
            result, stdout, stderr = _run_baselines(fixture, random_forest)
            self.assertEqual(result, 0, stderr)
            self.assertIn("random_forest_baseline: rebuilt", stdout)
            self.assertIn("training_mean_reference: rebuilt", stdout)
            candidate = _read_predictions(fixture, "random_forest_baseline")
            reference = _read_predictions(fixture, "training_mean_reference")
            _assert_prediction_artifacts(self, fixture, candidate, reference)
            _assert_random_forest_partitions(self, random_forest)
            _assert_fold_training_means(self, reference)

    def test_outputs_have_deterministic_metric_schemas_hashes_and_signed_bias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(directory)
            result, _, stderr = _run_baselines(fixture, ConstantFeatureBaseline())
            self.assertEqual(result, 0, stderr)
            output_dir = _configuration_dir(fixture, "random_forest_baseline")
            _assert_metric_artifacts(self, output_dir)
            manifest = json.loads((output_dir / "manifest.json").read_text())
            _assert_manifest(self, output_dir, manifest)

    def test_reuse_and_recipe_invalidation_are_selective_by_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(directory)
            first_model = RecordingFeatureBaseline()
            provenance = FixedBaselineProvenance()
            self.assertEqual(_run_baselines(fixture, first_model, provenance)[0], 0)
            candidate_manifest = _manifest_path(fixture, "random_forest_baseline")
            reference_manifest = _manifest_path(fixture, "training_mean_reference")
            modified = (candidate_manifest.stat().st_mtime_ns,
                        reference_manifest.stat().st_mtime_ns)
            _assert_reuse(self, fixture, provenance, candidate_manifest, reference_manifest)
            changed_model = RecordingFeatureBaseline()
            changed = FixedBaselineProvenance(random_forest_hash="7" * 64)
            result, stdout, stderr = _run_baselines(fixture, changed_model, changed)
            self.assertEqual(result, 0, stderr)
            self.assertIn("random_forest_baseline: rebuilt", stdout)
            self.assertIn("training_mean_reference: reusable", stdout)
            self.assertEqual(len(changed_model.fit_calls), 5)
            self.assertNotEqual(candidate_manifest.stat().st_mtime_ns, modified[0])
            self.assertEqual(reference_manifest.stat().st_mtime_ns, modified[1])

    def test_tampered_output_is_obsolete_in_dry_run_and_rebuilt_on_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(directory)
            provenance = FixedBaselineProvenance()
            predictions = _build_then_tamper_predictions(self, fixture, provenance)
            result, stdout, stderr = _run_baselines(
                fixture, RecordingFeatureBaseline(), provenance, "--dry-run"
            )
            self.assertEqual(result, 0, stderr)
            self.assertIn("random_forest_baseline: obsolete", stdout)
            self.assertIn("training_mean_reference: reusable", stdout)
            rebuilt_model = RecordingFeatureBaseline()
            self.assertEqual(_run_baselines(fixture, rebuilt_model, provenance)[0], 0)
            self.assertEqual(len(rebuilt_model.fit_calls), 5)
            self.assertNotIn("tampered", predictions.read_text())

    def test_manifest_identity_tampering_invalidates_the_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _confirmed_fixture(directory)
            provenance = FixedBaselineProvenance()
            self.assertEqual(
                _run_baselines(fixture, RecordingFeatureBaseline(), provenance)[0], 0
            )
            manifest_path = _manifest_path(fixture, "random_forest_baseline")
            original = json.loads(manifest_path.read_text())
            _assert_manifest_tampering(self, fixture, provenance, manifest_path, original)


def _confirmed_fixture(directory: str) -> CuratedInputsFixture:
    fixture = CuratedInputsFixture(Path(directory))
    _confirm_features(fixture, ("area", "perimeter"))
    return fixture


def _build_then_tamper_predictions(
    test_case: unittest.TestCase, fixture: CuratedInputsFixture,
    provenance: FixedBaselineProvenance,
) -> Path:
    result = _run_baselines(fixture, RecordingFeatureBaseline(), provenance)[0]
    test_case.assertEqual(result, 0)
    predictions = _configuration_dir(fixture, "random_forest_baseline") / "predictions.csv"
    predictions.write_text(f"{predictions.read_text()}tampered\n")
    return predictions


def _assert_prediction_artifacts(
    test_case: unittest.TestCase, fixture: CuratedInputsFixture,
    candidate: list[dict[str, str]], reference: list[dict[str, str]],
) -> None:
    test_case.assertEqual(len(candidate), fixture.sample_count)
    test_case.assertEqual(len(reference), fixture.sample_count)
    names = [row["file_name"] for row in candidate]
    test_case.assertEqual(names, sorted(names))
    test_case.assertEqual({row["evaluation_role"] for row in candidate}, {"candidate"})
    test_case.assertEqual({row["evaluation_role"] for row in reference}, {"reference"})
    for row in [*candidate, *reference]:
        expected = float(row["predicted_weight_kg"]) - float(row["observed_weight_kg"])
        test_case.assertAlmostEqual(float(row["residual_kg"]), expected, places=5)


def _assert_random_forest_partitions(
    test_case: unittest.TestCase, random_forest: RecordingFeatureBaseline,
) -> None:
    isolated = all(
        set(call.training_ids).isdisjoint(call.prediction_ids)
        for call in random_forest.prediction_calls
    )
    test_case.assertTrue(isolated)
    test_case.assertEqual(
        {call.feature_names for call in random_forest.fit_calls},
        {("area", "perimeter")},
    )


def _assert_metric_artifacts(test_case: unittest.TestCase, output_dir: Path) -> None:
    fold_fields, fold_metrics = _read_csv(output_dir / "fold_metrics.csv")
    grouped_fields, grouped_metrics = _read_csv(output_dir / "grouped_metrics.csv")
    test_case.assertEqual(fold_fields, METRIC_COLUMNS)
    test_case.assertEqual(grouped_fields, GROUPED_METRIC_COLUMNS)
    test_case.assertEqual([row["fold"] for row in fold_metrics], ["1", "2", "3", "4", "5"])
    test_case.assertEqual([row["population"] for row in grouped_metrics], ["all", "B1", "B10"])
    test_case.assertEqual(grouped_metrics[0], _known_constant_prediction_metrics())


def _known_constant_prediction_metrics() -> dict[str, str]:
    return {
        "configuration": "random_forest_baseline", "evaluation_role": "candidate",
        "population": "all", "n": "50", "mae_kg": "56.580000",
        "rmse_kg": "68.822235", "bias_kg": "-53.500000", "r2": "-1.527144",
    }


def _assert_manifest(
    test_case: unittest.TestCase, output_dir: Path, manifest: dict[str, object],
) -> None:
    test_case.assertEqual(manifest["selected_features"], ["area", "perimeter"])
    test_case.assertEqual(manifest["training_seed"], 44)
    test_case.assertEqual(manifest["fold_seed"], 42)
    expected = {
        name: {"sha256": sha256_file(output_dir / name), "rows": rows, "columns": columns}
        for name, rows, columns in _expected_outputs()
    }
    test_case.assertEqual(manifest["outputs"], expected)


def _expected_outputs() -> tuple[tuple[str, int, list[str]], ...]:
    return (
        ("predictions.csv", 50, PREDICTION_COLUMNS),
        ("fold_metrics.csv", 5, METRIC_COLUMNS),
        ("grouped_metrics.csv", 3, GROUPED_METRIC_COLUMNS),
    )


def _assert_reuse(
    test_case: unittest.TestCase, fixture: CuratedInputsFixture,
    provenance: FixedBaselineProvenance, candidate_manifest: Path, reference_manifest: Path,
) -> None:
    modified = (candidate_manifest.stat().st_mtime_ns, reference_manifest.stat().st_mtime_ns)
    reused_model = RecordingFeatureBaseline()
    result, stdout, stderr = _run_baselines(fixture, reused_model, provenance)
    test_case.assertEqual(result, 0, stderr)
    test_case.assertIn("random_forest_baseline: reusable", stdout)
    test_case.assertIn("training_mean_reference: reusable", stdout)
    test_case.assertEqual(reused_model.fit_calls, [])
    test_case.assertEqual(candidate_manifest.stat().st_mtime_ns, modified[0])
    test_case.assertEqual(reference_manifest.stat().st_mtime_ns, modified[1])


def _assert_manifest_tampering(
    test_case: unittest.TestCase, fixture: CuratedInputsFixture,
    provenance: FixedBaselineProvenance, manifest_path: Path, original: dict[str, object],
) -> None:
    for field, value in MANIFEST_TAMPERING_CASES:
        with test_case.subTest(field=field):
            tampered = original.copy()
            tampered[field] = value
            manifest_path.write_text(json.dumps(tampered))
            _, stdout, _ = _run_baselines(
                fixture, RecordingFeatureBaseline(), provenance, "--dry-run"
            )
            test_case.assertIn("random_forest_baseline: obsolete", stdout)
            manifest_path.write_text(json.dumps(original))


def _confirm_features(fixture: CuratedInputsFixture, features: tuple[str, ...]) -> None:
    contract_path, report_path = _prepare_human_review(fixture, features)
    result, _, stderr = _run_confirmation(fixture, contract_path, report_path)
    if result != 0:
        raise AssertionError(f"feature confirmation returned {result}; expected success: {stderr}")


def _run_baselines(
    fixture: CuratedInputsFixture, random_forest: FeatureBaseline,
    provenance: FixedBaselineProvenance | None = None, *extra: str,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["baselines", "--config", str(fixture.config_path), *extra],
        stdout=stdout, stderr=stderr, random_forest_baseline=random_forest,
        baseline_provenance=provenance or FixedBaselineProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


def _read_predictions(
    fixture: CuratedInputsFixture, configuration: str,
) -> list[dict[str, str]]:
    return _read_csv(_configuration_dir(fixture, configuration) / "predictions.csv")[1]


def _configuration_dir(fixture: CuratedInputsFixture, configuration: str) -> Path:
    output_dir = fixture.root / "generated" / "report" / "baselines" / configuration
    return output_dir


def _manifest_path(fixture: CuratedInputsFixture, configuration: str) -> Path:
    manifest_path = _configuration_dir(fixture, configuration) / "manifest.json"
    return manifest_path


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def _assert_fold_training_means(
    test_case: unittest.TestCase, predictions: list[dict[str, str]],
) -> None:
    all_weights = [float(row["observed_weight_kg"]) for row in predictions]
    for row in predictions:
        fold = row["fold"]
        training_weights = [
            float(candidate["observed_weight_kg"])
            for candidate in predictions if candidate["fold"] != fold
        ]
        expected = sum(training_weights) / len(training_weights)
        test_case.assertAlmostEqual(float(row["predicted_weight_kg"]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
