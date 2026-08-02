from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from buffalo_weight.artifact_provenance import TrainingEvidence
from buffalo_weight.models import ModelConfig
from buffalo_weight.pure_geometry_evaluation import PURE_GEOMETRY_FEATURES
from buffalo_weight.train_mask_geometry import train_mask_geometry_comparison


class FakeMaskGeometryEvaluator:
    feature_columns: list[str] = []
    models: list[str] = []

    def __call__(
        self,
        rows: list[dict[str, str]],
        feature_columns: list[str],
        model_configs: list[ModelConfig],
        masks_dir: Path,
        device: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        type(self).feature_columns = feature_columns
        type(self).models = [config.model for config in model_configs]
        return [], []


class FailingMaskGeometryEvaluator:
    def __call__(
        self,
        rows: list[dict[str, str]],
        feature_columns: list[str],
        model_configs: list[ModelConfig],
        masks_dir: Path,
        device: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        raise AssertionError("cached configurations must not be evaluated")


class FakeMaskGeometryReporter:
    def __call__(
        self,
        metrics: list[dict[str, str]],
        predictions: list[dict[str, str]],
        model_configs: list[ModelConfig],
        evidence: TrainingEvidence,
        output_dir: Path,
    ) -> list[dict[str, str]]:
        return [{"model": "cnn_mask_geometry", "mae_kg": "1", "r2": "0.9"}]


class TrainMaskGeometryTest(unittest.TestCase):
    def test_uses_only_pure_geometry_for_mask_and_fusion_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_config = self._write_shared_config(root)
            models_config = self._write_models_config(root)
            evaluator = FakeMaskGeometryEvaluator()

            comparison = train_mask_geometry_comparison(
                shared_config,
                models_config,
                root / "output",
                evaluator,
                FakeMaskGeometryReporter(),
                "cpu",
            )

        self.assertEqual(comparison[0]["mae_kg"], "1")
        self.assertEqual(evaluator.feature_columns, list(PURE_GEOMETRY_FEATURES))
        self.assertEqual(evaluator.models, ["cnn_mask", "cnn_mask_geometry"])

    def test_reuses_current_artifacts_without_evaluating_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_config = self._write_shared_config(root)
            models_config = self._write_models_config(root)
            with patch(
                "buffalo_weight.train_mask_geometry.prepare_artifacts",
                return_value=([], []),
            ):
                comparison = train_mask_geometry_comparison(
                    shared_config,
                    models_config,
                    root / "output",
                    FailingMaskGeometryEvaluator(),
                    FakeMaskGeometryReporter(),
                    "cpu",
                )

        self.assertEqual(comparison[0]["model"], "cnn_mask_geometry")

    def _write_shared_config(self, root: Path) -> Path:
        features_path = root / "features.csv"
        self._write_features(features_path)
        config_path = root / "shared.yaml"
        config_path.write_text(
            f"data:\n  masks_dir: {root / 'masks'}\nfeatures:\n  features_index_path: {features_path}\n"
        )
        return config_path

    def _write_models_config(self, root: Path) -> Path:
        path = root / "models.yaml"
        path.write_text(
            "model_configs:\n"
            "  mask_only:\n"
            "    model: cnn_mask\n"
            "    params: &params\n"
            "      epochs: 1\n      batch_size: 2\n      learning_rate: 0.001\n"
            "      image_size: 16\n      random_state: 42\n"
            "  fusion:\n    model: cnn_mask_geometry\n    params: *params\n"
        )
        return path

    def _write_features(self, path: Path) -> None:
        rows = []
        for index in range(1, 51):
            values = {feature: str(index) for feature in PURE_GEOMETRY_FEATURES}
            rows.append({"file_name": f"mask-{index}", "weight": str(index), **values})
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
