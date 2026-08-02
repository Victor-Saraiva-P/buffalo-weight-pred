from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from buffalo_weight.report_cli import main
from buffalo_weight.snapshot_io import FilesystemSnapshotPublisher
from tests.fake_report_provenance import FixedReportProvenance
from tests.fake_snapshot import FailAfterSnapshotInstallOperations
from tests.report_inputs_fixture import CuratedInputsFixture


# This literal is an independent oracle from issue #14; importing production names
# would make the schema assertion pass when implementation and contract drift together.
APPROVED_FEATURES = [
    "area",
    "perimeter",
    "solidity",
    "circularity",
    "equivalent_diameter",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "aspect_ratio",
    "extent",
    "convex_area",
    "convexity",
    "major_axis_length",
    "minor_axis_length",
    "roundness",
    "feret_diameter",
    "hu_moment_1",
    "hu_moment_2",
    "area_power_1_5",
    "area_major_axis_product",
    "center_vertical_occupancy",
    "end_vertical_occupancy_min",
    "end_vertical_occupancy_max",
    "center_to_end_occupancy_ratio",
    "centroid_x_offset",
    "centroid_y_ratio",
]


class ReportInputsCliTest(unittest.TestCase):
    def test_dry_run_build_resume_and_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            dry_run = fixture.run("inputs", "--dry-run")
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
            self.assertIn("inputs: absent", dry_run.stdout)
            self.assertFalse(fixture.output_dir.exists())

    def test_build_resume_and_input_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            first_run = fixture.run("inputs")
            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            manifest_path = fixture.output_dir / "manifest.json"
            initial_manifest = manifest_path.read_bytes()
            first_mtime = manifest_path.stat().st_mtime_ns

            second_run = fixture.run("inputs")
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertIn("inputs: reusable", second_run.stdout)
            self.assertEqual(manifest_path.stat().st_mtime_ns, first_mtime)
            fixture.replace_weight(0, "81")
            obsolete = fixture.run("inputs", "--dry-run")
            self.assertEqual(obsolete.returncode, 0, obsolete.stderr)
            self.assertIn("inputs: obsolete", obsolete.stdout)
            rebuilt = fixture.run("inputs")
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            self.assertNotEqual(manifest_path.read_bytes(), initial_manifest)

    def test_output_change_invalidates_and_rebuilds_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            self.assertEqual(fixture.run("inputs").returncode, 0)
            feature_path = fixture.output_dir / "feature_index.csv"
            feature_path.write_text(feature_path.read_text() + "tampered\n")
            output_changed = fixture.run("inputs", "--dry-run")
            self.assertIn("inputs: obsolete", output_changed.stdout)
            self.assertEqual(fixture.run("inputs").returncode, 0)

    def test_outputs_approved_features_and_canonical_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))

            result = fixture.run("inputs")

            self.assertEqual(result.returncode, 0, result.stderr)
            feature_rows, feature_fields = read_csv(fixture.output_dir / "feature_index.csv")
            split_rows, split_fields = read_csv(fixture.output_dir / "canonical_split.csv")
            self.assertEqual(feature_fields, ["file_name", "farm", "weight_kg", *APPROVED_FEATURES])
            self.assertEqual(len(feature_rows), 50)
            self.assert_worked_feature_row(feature_rows[0])
            self.assertEqual(
                split_fields,
                ["file_name", "farm", "weight_kg", "weight_category", "fold"],
            )
            self.assert_canonical_counts(split_rows)
            self.assert_manifest_hashes(fixture.output_dir)

    def test_rejects_integrity_violations_with_offending_values(self) -> None:
        corruptions = [
            ("missing", self._remove_mask, "mask-000.png", "exactly one indexed PNG"),
            ("extra", self._add_extra_mask, "extra.png", "exactly the indexed PNG files"),
            ("invalid weight", self._invalidate_weight, "zero", "finite number greater than 0"),
            ("repeated name", self._repeat_name, "mask-000.png", "unique index names"),
            ("non-binary", self._make_non_binary, "128", "only pixel values 0 and 255"),
            ("duplicate pixels", self._duplicate_pixels, "mask-001.png", "pixel-wise unique masks"),
        ]
        for label, corrupt, offending, expected in corruptions:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = CuratedInputsFixture(Path(directory))
                corrupt(fixture)

                result = fixture.run("inputs")

                self.assertEqual(result.returncode, 1)
                self.assertIn(offending, result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertFalse((fixture.output_dir / "manifest.json").exists())

    def test_failed_rebuild_keeps_previous_snapshot_detectably_obsolete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            self.assertEqual(fixture.run("inputs").returncode, 0)
            manifest_path = fixture.output_dir / "manifest.json"
            manifest_before = manifest_path.read_bytes()
            self._make_non_binary(fixture)

            failed = fixture.run("inputs")

            self.assertEqual(failed.returncode, 1)
            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            plan = fixture.run("inputs", "--dry-run")
            self.assertIn("inputs: obsolete", plan.stdout)

    def test_publication_failure_keeps_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            self.assertEqual(fixture.run("inputs").returncode, 0)
            manifest_path = fixture.output_dir / "manifest.json"
            manifest_before = manifest_path.read_bytes()
            fixture.replace_weight(0, "81")
            operations = FailAfterSnapshotInstallOperations()
            publisher = FilesystemSnapshotPublisher(operations)
            stderr = io.StringIO()
            result = main(
                ["inputs", "--config", str(fixture.config_path)],
                stdout=io.StringIO(),
                stderr=stderr,
                snapshot_publisher=publisher,
            )
            self.assertEqual(result, 1)
            self.assertEqual(operations.replace_calls, 2)
            self.assertIn("post-install failure", stderr.getvalue())
            self.assertEqual(manifest_path.read_bytes(), manifest_before)

    def test_cli_injects_report_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            result = main(
                ["inputs", "--config", str(fixture.config_path)],
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                report_provenance=FixedReportProvenance(),
            )

            self.assertEqual(result, 0)
            manifest = json.loads((fixture.output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["recipe_sha256"], "1" * 64)
            self.assertEqual(manifest["dependencies"], {"fake-compute": "1.0"})
            self.assertEqual(manifest["source_commit"], "2" * 40)

    def test_clean_removes_only_reconstructible_stage_and_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            self.assertEqual(fixture.run("inputs").returncode, 0)
            descendant = fixture.root / "generated" / "report" / "baselines"
            evidence = self._write_cleaning_boundaries(fixture, descendant)

            result = fixture.run("clean", "inputs")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(fixture.output_dir.exists())
            self.assertFalse(descendant.exists())
            self.assertTrue(fixture.index_path.exists())
            self.assertTrue((evidence / "snapshot.txt").exists())

            rejected = fixture.run("clean", "../data")
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("../data", rejected.stderr)
            self.assertIn("expected one of", rejected.stderr)

    def assert_canonical_counts(self, rows: list[dict[str, str]]) -> None:
        categories = Counter(row["weight_category"] for row in rows)
        folds = Counter(row["fold"] for row in rows)
        self.assertEqual(categories, {f"B{index}": 5 for index in range(1, 11)})
        self.assertEqual(folds, {str(index): 10 for index in range(1, 6)})
        for fold in range(1, 6):
            within_fold = Counter(
                row["weight_category"] for row in rows if row["fold"] == str(fold)
            )
            self.assertEqual(within_fold, {f"B{index}": 1 for index in range(1, 11)})

    def assert_worked_feature_row(self, row: dict[str, str]) -> None:
        self.assertEqual(row["file_name"], "mask-000.png")
        self.assertEqual(row["area"], "180224.000000")
        self.assertEqual(row["perimeter"], "2699.180246")
        self.assertEqual(row["bbox_width"], "768.000000")
        self.assertEqual(row["bbox_area"], "589824.000000")
        self.assertEqual(row["extent"], "0.305556")
        self.assertEqual(row["center_vertical_occupancy"], "128.000000")
        self.assertEqual(row["center_to_end_occupancy_ratio"], "0.444444")

    def assert_manifest_hashes(self, output_dir: Path) -> None:
        manifest = json.loads((output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["row_count"], 50)
        for file_name in ("feature_index.csv", "canonical_split.csv"):
            digest = hashlib.sha256((output_dir / file_name).read_bytes()).hexdigest()
            self.assertEqual(manifest["outputs"][file_name]["sha256"], digest)
            self.assertEqual(manifest["outputs"][file_name]["rows"], 50)
            self.assertTrue(manifest["outputs"][file_name]["columns"])

    def _write_cleaning_boundaries(
        self, fixture: CuratedInputsFixture, descendant: Path
    ) -> Path:
        descendant.mkdir()
        (descendant / "cache.txt").write_text("reconstructible")
        evidence = fixture.root / "evidence" / "confirmed"
        evidence.mkdir(parents=True)
        (evidence / "snapshot.txt").write_text("reviewed")
        return evidence

    @staticmethod
    def _remove_mask(fixture: CuratedInputsFixture) -> None:
        missing_path = fixture.mask_path(0)
        if not missing_path.exists():
            raise AssertionError(f"fixture mask was {missing_path}; expected an existing PNG")
        missing_path.unlink()

    @staticmethod
    def _add_extra_mask(fixture: CuratedInputsFixture) -> None:
        Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8)).save(
            fixture.masks_dir / "extra.png"
        )

    @staticmethod
    def _invalidate_weight(fixture: CuratedInputsFixture) -> None:
        invalid_weight = "zero"
        fixture.replace_weight(0, invalid_weight)
        if fixture.index_rows()[0]["weight_kg"] != invalid_weight:
            raise AssertionError("fixture weight corruption was not persisted")

    @staticmethod
    def _repeat_name(fixture: CuratedInputsFixture) -> None:
        rows = fixture.index_rows()
        repeated_name = rows[0]["file_name"]
        fixture.replace_file_name(1, repeated_name)
        if fixture.index_rows()[1]["file_name"] != repeated_name:
            raise AssertionError("fixture name corruption was not persisted")

    @staticmethod
    def _make_non_binary(fixture: CuratedInputsFixture) -> None:
        invalid_pixels = np.asarray([[0, 128]], dtype=np.uint8)
        invalid_path = fixture.mask_path(0)
        Image.fromarray(invalid_pixels).save(invalid_path)

    @staticmethod
    def _duplicate_pixels(fixture: CuratedInputsFixture) -> None:
        original_bytes = fixture.mask_path(0).read_bytes()
        duplicate_path = fixture.mask_path(1)
        duplicate_path.write_bytes(original_bytes)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a public CSV; for example, ``read_csv(path)`` returns rows and columns."""
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), list(reader.fieldnames or [])


if __name__ == "__main__":
    unittest.main()
