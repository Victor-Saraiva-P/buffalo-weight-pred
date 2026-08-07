from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.baseline_comparison_metrics import comparison_metric_rows
from buffalo_weight.baseline_comparison_types import ComparisonPrediction
from buffalo_weight.tuning_artifacts import write_tuning_artifacts
from buffalo_weight.tuning_types import get_pre_registered_variations


class TuningArtifactsTest(unittest.TestCase):
    def test_write_tuning_artifacts_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            variations = get_pre_registered_variations("random_forest", 2)
            preds = [
                ComparisonPrediction("v1", "random_forest", "tuned", f"file_{i}.png", "B1" if i < 5 else "B10", 1, 100.0, 101.0)
                for i in range(10)
            ]
            metrics = comparison_metric_rows(preds)
            write_tuning_artifacts(
                output_dir, preds, metrics, "random_forest", "random_forest_baseline", variations,
            )
            self.assertTrue((output_dir / "tuning_metrics.csv").is_file())
            self.assertTrue((output_dir / "tuning_report.md").is_file())
            report_text = (output_dir / "tuning_report.md").read_text()
            self.assertIn("Relatório de Ajuste Fino de Configuração", report_text)
            self.assertIn("random_forest", report_text)


if __name__ == "__main__":
    unittest.main()
