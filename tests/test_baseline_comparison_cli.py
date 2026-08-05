from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.png_artifact import read_png_artifact_spec
from buffalo_weight.report_cli import main
from buffalo_weight.reproduction_config import load_report_contract
from buffalo_weight.baseline_comparison_stage import (
    BaselineComparisonUpstreamDependencies,
    run_baseline_comparison_stage,
)
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_baseline_comparison import FixedBaselineComparisonProvenance
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_compact_cnn import FixedCompactCnnProvenance
from tests.fake_dense_baseline import (
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
)
from tests.fake_report_provenance import FixedReportProvenance
from tests.fake_resnet_baseline import FixedResNetBaselineProvenance
from tests.report_inputs_fixture import CuratedInputsFixture


EXPECTED_METRIC_COLUMNS = [
    "configuration", "approach", "evaluation_role", "scope", "fold", "population",
    "n", "mae_kg", "rmse_kg", "bias_kg", "r2",
]


class BaselineComparisonCliTest(unittest.TestCase):
    def test_dry_run_reports_absent_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            output_dir = fixture.root / "generated/report/approach_selection"

            result, stdout, stderr = _run_comparison(fixture, "--dry-run")

            self.assertEqual(result, 0, stderr)
            self.assertIn("baseline_comparison: absent", stdout)
            self.assertFalse(output_dir.exists())

    def test_builds_provisional_evidence_package_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))

            result, stdout, stderr = _run_comparison(fixture)

            self.assertEqual(result, 0, stderr)
            self.assertIn("baseline_comparison: rebuilt", stdout)
            output_dir = fixture.root / "generated/report/approach_selection"
            columns, metrics = _read_csv(output_dir / "baseline_metrics.csv")
            self.assertEqual(columns, EXPECTED_METRIC_COLUMNS)
            self.assertEqual(len(metrics), 5 * (5 + 3))
            self.assertEqual(sum(row["evaluation_role"] == "candidate" for row in metrics), 32)
            self.assertTrue(all(row["rmse_kg"] == row["r2"] == "" for row in metrics
                                if row["population"] in {"B1", "B10"}))
            self._assert_review_artifacts(output_dir)
            reused = _run_comparison(fixture)
            self.assertEqual(reused[0], 0, reused[2])
            self.assertIn("baseline_comparison: reusable", reused[1])

    def test_rejects_an_obsolete_upstream_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            manifest_path = fixture.root / "generated/report/baselines/dense/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["outputs"]["predictions.csv"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))

            result, _, stderr = _run_comparison(fixture)

            self.assertEqual(result, 1)
            self.assertIn("dense", stderr)
            self.assertIn("expected all configurations reusable", stderr)

    def test_stage_api_rejects_manifest_corruption_outside_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            manifest_path = fixture.root / "generated/report/baselines/dense/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["outputs"]["fold_metrics.csv"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "dense.*all configurations reusable"):
                run_baseline_comparison_stage(
                    load_report_contract(fixture.config_path),
                    provenance=FixedBaselineComparisonProvenance(),
                    upstream_dependencies=_comparison_upstream_dependencies(),
                )

    def test_cleaning_baselines_removes_the_comparison_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            self.assertEqual(_run_comparison(fixture)[0], 0)
            output_dir = fixture.root / "generated/report/approach_selection"

            result = main(
                ["clean", "baselines", "--config", str(fixture.config_path)],
                stdout=io.StringIO(), stderr=io.StringIO(),
            )

            self.assertEqual(result, 0)
            self.assertFalse(output_dir.exists())

    def _assert_review_artifacts(self, output_dir: Path) -> None:
        decision = json.loads((output_dir / "selected_approach.json").read_text())
        manifest = json.loads((output_dir / "manifest.json").read_text())
        report = (output_dir / "approach_selection_report.md").read_text()
        self.assertIsNone(decision["human_decision"])
        self.assertEqual(manifest["status"], "provisional")
        self.assertIsNone(manifest["decision_url"])
        self.assertIn("MAE OOF Pós-Seleção", report)
        self.assertIn("Decisão humana: não preenchida", report)
        self.assertIn("não constitui uma decisão automática", report)
        for name in ("global_mae.png", "predicted_vs_observed.png", "residuals_vs_observed.png"):
            specification = read_png_artifact_spec(output_dir / name)
            self.assertEqual((specification.width_px, specification.height_px,
                              specification.dpi), (2400, 1800, 300))
        output_mtime = max((output_dir / name).stat().st_mtime_ns for name in manifest["outputs"])
        self.assertGreaterEqual((output_dir / "manifest.json").stat().st_mtime_ns, output_mtime)


def _run_comparison(
    fixture: CuratedInputsFixture, *extra: str,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(
        ["compare-baselines", "--config", str(fixture.config_path), *extra],
        stdout=stdout, stderr=stderr, report_provenance=FixedReportProvenance(),
        baseline_provenance=FixedBaselineProvenance(),
        dense_baseline_dependencies=DenseBaselineDependencies(
            provenance=FixedDenseBaselineProvenance(), runtime_probe=FixedCudaRuntimeProbe(),
        ),
        compact_cnn_provenance=FixedCompactCnnProvenance(),
        resnet_baseline_provenance=FixedResNetBaselineProvenance(),
        baseline_comparison_provenance=FixedBaselineComparisonProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


def _comparison_upstream_dependencies() -> BaselineComparisonUpstreamDependencies:
    return BaselineComparisonUpstreamDependencies(
        FixedReportProvenance(), FixedBaselineProvenance(),
        DenseBaselineDependencies(
            provenance=FixedDenseBaselineProvenance(), runtime_probe=FixedCudaRuntimeProbe(),
        ),
        FixedCompactCnnProvenance(), FixedResNetBaselineProvenance(),
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


if __name__ == "__main__":
    unittest.main()
