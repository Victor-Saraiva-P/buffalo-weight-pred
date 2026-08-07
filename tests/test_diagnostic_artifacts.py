from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.diagnostic_artifacts import write_descriptive_diagnostic_artifacts
from buffalo_weight.diagnostic_coverage import DiagnosticCoverageSample
from buffalo_weight.diagnostic_descriptive_slice import build_descriptive_diagnostic_slice
from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


class DiagnosticArtifactsTest(unittest.TestCase):
    def test_writes_all_tidy_csv_files_and_report_markdown(self) -> None:
        baselines = ["random_forest_baseline", "dense", "compact_cnn", "resnet18_pretrained_partial"]
        coverage_samples = [
            DiagnosticCoverageSample(f"s_{i}.png", "FarmA" if i < 10 else "FarmB", "B1", "1024x768", 100.0 + i * 10)
            for i in range(20)
        ]
        predictions: list[DiagnosticPrediction] = []
        for b in baselines:
            for s in coverage_samples:
                predictions.append(DiagnosticPrediction(
                    b, "baseline", s.file_name, s.weight_category, s.farm, s.resolution,
                    s.weight_kg, s.weight_kg + 2.0,
                ))

        slice_result = build_descriptive_diagnostic_slice(coverage_samples, predictions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            write_descriptive_diagnostic_artifacts(output_dir, slice_result)

            self.assertTrue((output_dir / "sample_coverage.csv").is_file())
            self.assertTrue((output_dir / "stratified_metrics.csv").is_file())
            self.assertTrue((output_dir / "farm_comparison.csv").is_file())
            self.assertTrue((output_dir / "residual_correlations.csv").is_file())
            self.assertTrue((output_dir / "notable_cases.csv").is_file())
            self.assertTrue((output_dir / "descriptive_diagnostics_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
