from __future__ import annotations

import csv
import io
import json
import re
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
            self.assertEqual(runner.evaluation_count, 1)
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

    def test_manifest_metadata_tampering_invalidates_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            provenance = FixedReportProvenance()
            self.assertEqual(run_main(fixture, "inputs", provenance=provenance), 0)
            runner = FixedFeatureEvidenceRunner()
            self.assertEqual(run_feature_selection(fixture, runner, provenance)[0], 0)
            manifest_path = fixture.root / "generated" / "report" / "feature_selection" / "manifest.json"
            original = manifest_path.read_text()
            for field, value in manifest_tampering_cases():
                manifest = json.loads(original)
                manifest[field] = value
                manifest_path.write_text(json.dumps(manifest))
                plan = run_feature_selection(fixture, runner, provenance, "--dry-run")[1]
                self.assertIn("feature_selection: obsolete", plan, field)
                manifest_path.write_text(original)

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
        self.assertTrue(all(valid_redundancy_row(row) for row in redundancy[1]))
        self.assertEqual(evidence[1], sorted(evidence[1], key=evidence_sort_key))
        self.assertTrue(all(valid_evidence_row(row) for row in evidence[1]))

    def assert_package_is_provisional(self, output_dir: Path) -> None:
        contract = json.loads((output_dir / "shared_feature_contract.json").read_text())
        manifest = json.loads((output_dir / "manifest.json").read_text())
        report = (output_dir / "feature_selection_report.md").read_text()
        self.assertEqual(contract["status"], "provisional")
        self.assertIsNone(contract["selected_features"])
        self.assertIsNone(contract["human_decision"])
        self.assertEqual(manifest["status"], "provisional")
        self.assertIsNone(manifest["decision_url"])
        self.assertEqual(manifest["report_sha256"], contract["report_sha256"])
        self.assert_manifest_inputs(manifest)
        self.assertIn("recommend_removal", report)
        self.assertIn("retain_harm_veto", report)
        self.assertIn("retain_double_neutral", report)
        self.assertIn("## Desempenho isolado", report)
        self.assertIn("## Redundância estrutural e observada", report)
        self.assertIn("## Efeitos de permutação", report)
        self.assertEqual(report.count(" / `"), 325)

    def assert_manifest_inputs(self, manifest: dict[str, object]) -> None:
        inputs = manifest["inputs"]
        self.assertIsInstance(inputs, dict)
        assert isinstance(inputs, dict)
        self.assertEqual(set(inputs), {
            "manifest.json", "feature_index.csv", "canonical_split.csv",
        })
        self.assertTrue(all(set(record) == {"sha256", "row_count", "schema"}
                            for record in inputs.values() if isinstance(record, dict)))

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


def evidence_sort_key(row: dict[str, str]) -> tuple[object, ...]:
    scope_rank = 0 if row["scope"] == "fold" else 1
    return (row["experiment"], row["baseline"], row["target"], scope_rank,
            int(row["fold"] or 0), int(row["repetition"] or 0))


def valid_evidence_row(row: dict[str, str]) -> bool:
    numeric = re.compile(r"^-?\d+\.\d{6}$")
    allowed = row["experiment"] in {"isolated", "removal", "permutation"}
    allowed = allowed and row["baseline"] in {"random_forest", "dense"}
    allowed = allowed and row["scope"] in {"fold", "oof"}
    allowed = allowed and bool(numeric.fullmatch(row["result_mae_kg"]))
    return allowed and _valid_nullable_fields(row, numeric)


def _valid_nullable_fields(row: dict[str, str], numeric: re.Pattern[str]) -> bool:
    if row["experiment"] == "isolated":
        return row["reference_mae_kg"] == row["delta_mae_kg"] == row["effect"] == ""
    numeric_fields = (row["reference_mae_kg"], row["delta_mae_kg"])
    if not all(numeric.fullmatch(value) for value in numeric_fields):
        return False
    return row["effect"] in {"improvement", "neutral", "harm"}


def valid_redundancy_row(row: dict[str, str]) -> bool:
    numeric = re.compile(r"^-?\d+\.\d{6}$")
    groups = {"none", "area_transformations", "bounding_rectangle_relations",
              "equivalent_ellipse_relation", "vertical_occupancy_relation",
              "convex_hull_relations", "area_contour_relation"}
    relations = {"none", "area_bijection", "area_major_axis_product", "bbox_area_product",
                 "bbox_aspect_ratio", "bbox_extent", "ellipse_roundness",
                 "vertical_occupancy_ratio", "convex_solidity", "convexity_ratio",
                 "area_contour_circularity"}
    correlations = [row["pearson"], row["spearman"]]
    numeric_valid = all(value == "" or (numeric.fullmatch(value)
                        and -1.0 <= float(value) <= 1.0) for value in correlations)
    relation_parts = row["structural_relation"].split("|")
    return row["removal_group"] in groups and set(relation_parts) <= relations and numeric_valid


def manifest_tampering_cases() -> list[tuple[str, object]]:
    cases: list[tuple[str, object]] = [
        ("decision_url", "https://example.invalid/decision"), ("report_sha256", "0" * 64),
        ("source_commit", "0" * 40), ("command", "python main.py inputs"),
        ("revision", 2), ("validations", []),
    ]
    return cases


if __name__ == "__main__":
    unittest.main()
