"""Tests for controlled learning curves artifact writing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.diagnostic_learning_artifacts import write_learning_curves_artifacts
from buffalo_weight.diagnostic_learning_types import (
    LearningCurveSummaryRecord,
    LearningCurvesSlice,
    LearningPointRecord,
)


class DiagnosticLearningArtifactsTest(unittest.TestCase):
    def test_write_learning_curves_artifacts_creates_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "learning_curves"

            points = (
                LearningPointRecord("random_forest_baseline", 1, 0.50, 53, "oof", 26, 12.5, -0.4, "retrained"),
                LearningPointRecord("random_forest_baseline", 1, 0.75, 79, "oof", 26, 11.2, -0.2, "retrained"),
                LearningPointRecord("random_forest_baseline", 1, 1.00, 105, "oof", 26, 10.1, 0.0, "reused"),
            )
            summaries = (
                LearningCurveSummaryRecord("random_forest_baseline", 0.50, 53.0, 12.5, 0.0, -0.4, 0),
                LearningCurveSummaryRecord("random_forest_baseline", 0.75, 79.0, 11.2, 0.0, -0.2, 0),
                LearningCurveSummaryRecord("random_forest_baseline", 1.00, 105.0, 10.1, 0.0, 0.0, 1),
            )
            slice_data = LearningCurvesSlice(points, summaries)

            write_learning_curves_artifacts(output_dir, slice_data)

            self.assertTrue((output_dir / "learning_curves_points.csv").is_file())
            self.assertTrue((output_dir / "learning_curves_summary.csv").is_file())
            self.assertTrue((output_dir / "learning_curves_canonical.png").is_file())
            self.assertTrue((output_dir / "learning_curves_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
