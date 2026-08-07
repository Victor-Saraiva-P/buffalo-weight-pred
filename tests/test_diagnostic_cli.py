from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_cli import main
from buffalo_weight.reproduction_config import load_report_contract
from tests.baseline_comparison_fixture import prepared_comparison_fixture


class DiagnosticCliTest(unittest.TestCase):
    def test_cli_runs_diagnostics_descriptive_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            stdout, stderr = io.StringIO(), io.StringIO()

            result = main(
                ["diagnostics-descriptive", "--config", str(fixture.config_path)],
                stdout=stdout, stderr=stderr,
            )

            self.assertEqual(result, 0, stderr.getvalue())
            self.assertIn("diagnostics_descriptive: rebuilt", stdout.getvalue())

            contract = load_report_contract(fixture.config_path)
            out_dir = contract.artifacts_root / "diagnostics" / "descriptive"
            self.assertTrue((out_dir / "manifest.json").is_file())
            self.assertTrue((out_dir / "sample_coverage.csv").is_file())
            self.assertTrue((out_dir / "stratified_metrics.csv").is_file())
            self.assertTrue((out_dir / "farm_comparison.csv").is_file())
            self.assertTrue((out_dir / "residual_correlations.csv").is_file())
            self.assertTrue((out_dir / "notable_cases.csv").is_file())
            self.assertTrue((out_dir / "descriptive_diagnostics_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
