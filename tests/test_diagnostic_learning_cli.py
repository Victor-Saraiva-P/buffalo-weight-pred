"""Tests for public CLI diagnostics-learning subcommand."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.report_cli import main
from buffalo_weight.reproduction_config import load_report_contract
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_baseline_provenance import FixedBaselineProvenance
from tests.fake_compact_cnn import FixedCompactCnnProvenance, RecordingCompactCnnAdapter
from tests.fake_dense_baseline import (
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
    FixedDenseBaselineRunner,
)
from tests.fake_feature_evaluation import RecordingFeatureBaseline
from tests.fake_report_provenance import FixedReportProvenance
from tests.fake_resnet_baseline import FixedResNetBaselineProvenance, FixedResNetBaselineRunner


class DiagnosticLearningCliTest(unittest.TestCase):
    def test_cli_runs_diagnostics_learning_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            stdout, stderr = io.StringIO(), io.StringIO()

            result = main(
                ["diagnostics-learning", "--config", str(fixture.config_path)],
                stdout=stdout, stderr=stderr,
                random_forest_baseline=RecordingFeatureBaseline(),
                baseline_provenance=FixedBaselineProvenance(),
                report_provenance=FixedReportProvenance(),
                dense_baseline_dependencies=DenseBaselineDependencies(
                    FixedDenseBaselineRunner(), FixedDenseBaselineProvenance(), FixedCudaRuntimeProbe(),
                ),
                compact_cnn_adapter=RecordingCompactCnnAdapter(),
                compact_cnn_provenance=FixedCompactCnnProvenance(),
                resnet_baseline_runner=FixedResNetBaselineRunner(),
                resnet_baseline_provenance=FixedResNetBaselineProvenance(),
            )

            self.assertEqual(result, 0, stderr.getvalue())
            self.assertIn("diagnostics_learning: rebuilt", stdout.getvalue())

            contract = load_report_contract(fixture.config_path)
            out_dir = contract.artifacts_root / "diagnostics" / "learning_curves"
            self.assertTrue((out_dir / "manifest.json").is_file())
            self.assertTrue((out_dir / "learning_curves_points.csv").is_file())
            self.assertTrue((out_dir / "learning_curves_summary.csv").is_file())
            self.assertTrue((out_dir / "learning_curves_canonical.png").is_file())
            self.assertTrue((out_dir / "learning_curves_report.md").is_file())

            # Read points CSV to verify reusability and fraction fields
            with (out_dir / "learning_curves_points.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            # 4 baselines * 5 folds * 3 fractions (0.50, 0.75, 1.00) = 60 point records
            self.assertEqual(len(rows), 60)

            # Fractions 0.50 and 0.75 are ALWAYS retrained (4 baselines * 5 folds * 2 fractions = 40 points)
            sub_points = [r for r in rows if r["fraction"] in ("0.50", "0.75")]
            self.assertEqual(len(sub_points), 40)
            self.assertTrue(all(r["artifact_action"] == "retrained" for r in sub_points))

            # 100% points for matching reusable baselines are reused
            reused_100 = [r for r in rows if r["fraction"] == "1.00" and r["artifact_action"] == "reused"]
            self.assertGreaterEqual(len(reused_100), 10)


if __name__ == "__main__":
    unittest.main()
