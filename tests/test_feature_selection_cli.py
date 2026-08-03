from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from buffalo_weight.report_cli import main
from tests.fake_feature_selection import FixedFeatureEvidenceRunner
from tests.fake_report_provenance import FixedFeatureSelectionProvenance, FixedReportProvenance
from tests.report_inputs_fixture import CuratedInputsFixture


EVIDENCE_COLUMNS = [
    "experiment", "baseline", "target", "scope", "fold", "repetition",
    "permutation_seed", "n", "reference_mae_kg", "result_mae_kg",
    "delta_mae_kg", "effect",
]
REDUNDANCY_COLUMNS = [
    "feature_a", "feature_b", "structural_relation", "pearson", "spearman",
    "removal_group",
]


class FeatureSelectionCliTest(unittest.TestCase):
    def test_builds_complete_deterministic_provisional_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            provenance = FixedReportProvenance()
            self.assertEqual(run_main(fixture, "inputs", provenance=provenance), 0)
            runner = FixedFeatureEvidenceRunner()

            result, stdout, stderr = run_feature_selection(fixture, runner, provenance)

            self.assertEqual(result, 0, stderr)
            self.assertIn("feature_selection: rebuilt", stdout)
            output_dir = fixture.root / "generated" / "report" / "feature_selection"
            self.assert_canonical_tables(output_dir)
            self.assertEqual(runner.calls[0][0], 50)
            self.assertEqual(len(runner.calls[0][2]), 6)
            self.assert_package_is_provisional(output_dir)
            self.assert_figures_are_300_dpi(output_dir)

    def test_dry_run_reports_input_gate_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            provenance = FixedReportProvenance()
            blocked, stdout, _ = run_feature_selection(
                fixture, FixedFeatureEvidenceRunner(), provenance, "--dry-run"
            )
            self.assertEqual(blocked, 0)
            self.assertIn("feature_selection: blocked", stdout)
            self.assertFalse((fixture.root / "generated").exists())

    def test_dry_run_resume_and_output_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            provenance = FixedReportProvenance()
            self.assertEqual(run_main(fixture, "inputs", provenance=provenance), 0)
            runner = FixedFeatureEvidenceRunner()
            self.assertIn("feature_selection: absent", run_feature_selection(
                fixture, runner, provenance, "--dry-run"
            )[1])
            self.assertEqual(run_feature_selection(fixture, runner, provenance)[0], 0)
            self.assert_reuse_and_tamper(fixture, runner, provenance)

    def assert_reuse_and_tamper(
        self, fixture: CuratedInputsFixture, runner: FixedFeatureEvidenceRunner,
        provenance: FixedReportProvenance,
    ) -> None:
        manifest = fixture.root / "generated" / "report" / "feature_selection" / "manifest.json"
        modified = manifest.stat().st_mtime_ns
        self.assertIn("feature_selection: reusable", run_feature_selection(
            fixture, runner, provenance
        )[1])
        self.assertEqual(manifest.stat().st_mtime_ns, modified)
        evidence = manifest.parent / "feature_predictive_evidence.csv"
        evidence.write_text(evidence.read_text() + "tampered\n")
        self.assertIn("feature_selection: obsolete", run_feature_selection(
            fixture, runner, provenance, "--dry-run"
        )[1])

    def assert_canonical_tables(self, output_dir: Path) -> None:
        evidence = read_csv(output_dir / "feature_predictive_evidence.csv")
        redundancy = read_csv(output_dir / "feature_redundancy.csv")
        self.assertEqual(evidence[0], EVIDENCE_COLUMNS)
        self.assertEqual(len(evidence[1]), 3816)
        self.assertEqual(redundancy[0], REDUNDANCY_COLUMNS)
        self.assertEqual(len(redundancy[1]), 325)

    def assert_package_is_provisional(self, output_dir: Path) -> None:
        contract = json.loads((output_dir / "shared_feature_contract.json").read_text())
        manifest = json.loads((output_dir / "manifest.json").read_text())
        report = (output_dir / "feature_selection_report.md").read_text()
        self.assertEqual(contract["status"], "provisional")
        self.assertIsNone(contract["selected_features"])
        self.assertIsNone(contract["human_decision"])
        self.assertEqual(manifest["status"], "provisional")
        self.assertIn("recommend_removal", report)
        self.assertIn("retain_harm_veto", report)
        self.assertIn("retain_double_neutral", report)

    def assert_figures_are_300_dpi(self, output_dir: Path) -> None:
        names = ("redundancy_heatmap.png", "removal_heatmap.png", "permutation_effects.png")
        for name in names:
            with self.subTest(name=name), Image.open(output_dir / name) as figure:
                self.assertEqual(figure.size, (2400, 1800))
                self.assertAlmostEqual(figure.info["dpi"][0], 300.0, places=0)


def run_main(
    fixture: CuratedInputsFixture, command: str, provenance: FixedReportProvenance
) -> int:
    return main([command, "--config", str(fixture.config_path)], stdout=io.StringIO(),
                stderr=io.StringIO(), report_provenance=provenance)


def run_feature_selection(
    fixture: CuratedInputsFixture, runner: FixedFeatureEvidenceRunner,
    provenance: FixedReportProvenance, *extra: str,
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    result = main(["feature-selection", "--config", str(fixture.config_path), *extra],
                  stdout=stdout, stderr=stderr, report_provenance=provenance,
                  feature_evidence_runner=runner,
                  feature_selection_provenance=FixedFeatureSelectionProvenance())
    return result, stdout.getvalue(), stderr.getvalue()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


if __name__ == "__main__":
    unittest.main()
