from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.artifact_provenance import (
    TrainingEvidence,
    artifact_plan,
    expected_manifest,
    manifest_differences,
    prepare_artifacts,
    training_lock,
    write_manifest,
)
from buffalo_weight.models import ModelConfig


class ArtifactProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig("forest", "random_forest", {"n_estimators": 2, "random_state": 42})
        self.evidence = TrainingEvidence(
            split_rows=[{"file_name": "a", "weight": "10", "fold": "1", "weight_category": "B1"}],
            feature_rows=[{"file_name": "a", "weight": "10", "area": "2"}],
            feature_columns=["area"],
            masks_dir=None,
            device="auto",
        )

    def test_missing_manifest_requires_new_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model_dir = output_dir / self.config.name
            model_dir.mkdir()
            (model_dir / "old.csv").write_text("old")
            plan = artifact_plan(output_dir, self.config, self.evidence)
            prepare_artifacts(output_dir, [self.config], self.evidence)

        self.assertEqual(plan.status, "new")
        self.assertIn("manifest missing", plan.reasons)
        self.assertFalse(model_dir.exists())

    def test_manifest_and_output_hashes_make_artifact_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model_dir = output_dir / self.config.name
            model_dir.mkdir()
            (model_dir / "fold_metrics.csv").write_text("fold,mae\n1,2\n")
            (model_dir / "predictions.csv").write_text("file_name,y_pred\na,10\n")
            write_manifest(output_dir, self.config, self.evidence)

            plan = artifact_plan(output_dir, self.config, self.evidence)

        self.assertEqual(plan.status, "reuse")
        self.assertEqual(plan.reasons, ())

    def test_changed_output_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model_dir = output_dir / self.config.name
            model_dir.mkdir()
            (model_dir / "fold_metrics.csv").write_text("fold,mae\n1,2\n")
            (model_dir / "predictions.csv").write_text("file_name,y_pred\na,10\n")
            write_manifest(output_dir, self.config, self.evidence)
            (model_dir / "predictions.csv").write_text("fold,mae\n1,999\n")

            plan = artifact_plan(output_dir, self.config, self.evidence)

        self.assertEqual(plan.status, "stale")
        self.assertIn("predictions.csv changed", plan.reasons)

    def test_manifest_is_json_and_contains_configuration_identity(self) -> None:
        manifest = expected_manifest(self.config, self.evidence)

        self.assertEqual(manifest["model_config"], "forest")
        self.assertEqual(json.loads(json.dumps(manifest))["model"], "random_forest")

    def test_manifest_difference_reports_changed_configuration(self) -> None:
        expected = expected_manifest(self.config, self.evidence)
        actual = dict(expected)
        actual["params"] = {"n_estimators": 3}

        self.assertIn("params", manifest_differences(actual, expected, Path("missing")))

    def test_backend_change_does_not_make_artifact_stale(self) -> None:
        mask_config = ModelConfig(
            "cnn", "cnn_mask", {"epochs": 1, "batch_size": 1, "learning_rate": 0.1, "image_size": 8, "random_state": 1}
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model_dir = output_dir / mask_config.name
            model_dir.mkdir()
            (model_dir / "fold_metrics.csv").write_text("fold,mae\n1,2\n")
            (model_dir / "predictions.csv").write_text("file_name,y_pred\na,10\n")
            write_manifest(output_dir, mask_config, self.evidence)
            cpu_evidence = TrainingEvidence(
                self.evidence.split_rows,
                self.evidence.feature_rows,
                self.evidence.feature_columns,
                self.evidence.masks_dir,
                "cuda",
            )

            plan = artifact_plan(output_dir, mask_config, cpu_evidence)

        self.assertEqual(plan.status, "reuse")

    def test_split_metadata_change_makes_artifact_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model_dir = output_dir / self.config.name
            model_dir.mkdir()
            (model_dir / "fold_metrics.csv").write_text("fold,mae\n1,2\n")
            (model_dir / "predictions.csv").write_text("file_name,y_pred\na,10\n")
            write_manifest(output_dir, self.config, self.evidence)
            changed_split = [{**self.evidence.split_rows[0], "farm": "changed"}]
            evidence = TrainingEvidence(
                changed_split,
                self.evidence.feature_rows,
                self.evidence.feature_columns,
                None,
                "auto",
            )

            plan = artifact_plan(output_dir, self.config, evidence)

        self.assertEqual(plan.status, "stale")
        self.assertIn("input_hash", plan.reasons)

    def test_training_lock_rejects_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with training_lock(output_dir):
                with self.assertRaisesRegex(ValueError, "locked"):
                    with training_lock(output_dir):
                        pass

    def test_dry_run_does_not_delete_stale_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            model_dir = output_dir / self.config.name
            model_dir.mkdir()
            (model_dir / "provenance.json").write_text("{}")

            plans, pending = prepare_artifacts(output_dir, [self.config], self.evidence, dry_run=True)

            self.assertEqual(plans[0].status, "stale")
            self.assertEqual([config.name for config in pending], ["forest"])
            self.assertTrue(model_dir.exists())


if __name__ == "__main__":
    unittest.main()
