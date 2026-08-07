from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.report_cli import main
from buffalo_weight.reproduction_config import load_report_contract
from buffalo_weight.tuning_manifest import tuning_output_dir
from tests.baseline_comparison_fixture import prepared_comparison_fixture
from tests.fake_compact_cnn import FixedCompactCnnProvenance, RecordingCompactCnnAdapter
from tests.fake_dense_baseline import (
    FixedCudaRuntimeProbe,
    FixedDenseBaselineProvenance,
    RecordingDenseFeatureAdapter,
)
from tests.fake_resnet_baseline import FixedResNetBaselineProvenance, FixedResNetBaselineRunner
from tests.fake_tuning_provenance import FixedTuningProvenance
from tests.report_inputs_fixture import CuratedInputsFixture
from tests.test_tuning_inputs import _setup_confirmed_approach

APPROACH_CANDIDATES = (
    ("random_forest", "random_forest_baseline"),
    ("dense_feature_network", "dense"),
    ("compact_cnn", "compact_cnn"),
    ("resnet18", "resnet18_pretrained_partial"),
)


class TuningCliTest(unittest.TestCase):
    def test_cli_supports_all_four_approach_classes(self) -> None:
        for approach, baseline_config in APPROACH_CANDIDATES:
            with self.subTest(approach=approach):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = prepared_comparison_fixture(Path(directory))
                    _setup_confirmed_approach(fixture, approach, baseline_config, 2)
                    result, stdout, stderr = _run_tuning(fixture)
                    self.assertEqual(result, 0, stderr)
                    self.assertIn("tuning: rebuilt", stdout)
                    out_dir = tuning_output_dir(load_report_contract(fixture.config_path))
                    self.assertTrue((out_dir / "manifest.json").is_file())
                    self.assertTrue((out_dir / "tuning_metrics.csv").is_file())


    def test_budget_zero_maintains_baseline_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 0)
            result, stdout, stderr = _run_tuning(fixture)
            self.assertEqual(result, 0, stderr)
            self.assertIn("baseline_maintained", stdout)
            out_dir = tuning_output_dir(load_report_contract(fixture.config_path))
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "baseline_maintained")
            self.assertFalse((out_dir / "tuning_metrics.csv").exists())

    def test_posterior_tampering_invalidates_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 2)
            self.assertEqual(_run_tuning(fixture)[0], 0)
            out_dir = tuning_output_dir(load_report_contract(fixture.config_path))
            report = out_dir / "tuning_report.md"
            report.write_text(f"{report.read_text()}\ntampered\n")
            result, stdout, stderr = _run_tuning(fixture, dry_run=True)
            self.assertEqual(result, 0, stderr)
            self.assertIn("tuning: obsolete", stdout)

    def test_frozen_features_enforcement_for_feature_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = prepared_comparison_fixture(Path(directory))
            _setup_confirmed_approach(fixture, "random_forest", "random_forest_baseline", 2)
            contract_dir = fixture.root / "evidence" / "confirmed" / "feature_selection" / "v1"
            (contract_dir / "shared_feature_contract.json").unlink()
            result, stdout, stderr = _run_tuning(fixture)
            self.assertEqual(result, 1)
            self.assertIn("rejected:", stderr)


def _run_tuning(
    fixture: CuratedInputsFixture, dry_run: bool = False,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    dry_arg = ["--dry-run"] if dry_run else []
    dense_dep = DenseBaselineDependencies(
        adapter=RecordingDenseFeatureAdapter(),
        provenance=FixedDenseBaselineProvenance(),
        runtime_probe=FixedCudaRuntimeProbe(),
    )
    result = main(
        ["tuning", *dry_arg, "--config", str(fixture.config_path)],
        stdout=stdout, stderr=stderr,
        dense_baseline_dependencies=dense_dep,
        compact_cnn_adapter=RecordingCompactCnnAdapter(),
        compact_cnn_provenance=FixedCompactCnnProvenance(),
        resnet_baseline_runner=FixedResNetBaselineRunner(),
        resnet_baseline_provenance=FixedResNetBaselineProvenance(),
        tuning_provenance=FixedTuningProvenance(),
    )
    return result, stdout.getvalue(), stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
