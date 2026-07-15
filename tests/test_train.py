from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from buffalo_weight.models import ModelConfig, ModelParam
from buffalo_weight.train_classical import train_classical
from buffalo_weight.train_cnn_mask import train_cnn_mask
from buffalo_weight.train import (
    evaluate_models,
    evaluate_random_forest,
    pending_model_configs,
    predict_fold_weights,
)


class FakeCnnMaskRegressor:
    fit_rows: list[dict[str, str]] = []
    early_stopping_rows: list[dict[str, str]] = []
    predicted_rows: list[dict[str, str]] = []

    def __init__(self, masks_dir: Path, params: dict[str, ModelParam]) -> None:
        self.masks_dir = masks_dir
        self.params = params

    def fit(
        self, rows: list[dict[str, str]], validation_rows: list[dict[str, str]] | None = None
    ) -> None:
        type(self).fit_rows = rows
        type(self).early_stopping_rows = validation_rows or []

    def predict(self, rows: list[dict[str, str]]) -> np.ndarray:
        type(self).predicted_rows = rows
        return np.zeros(len(rows), dtype=float)


class TrainTest(unittest.TestCase):
    def test_only_models_with_complete_csv_outputs_are_cached(self) -> None:
        configs = [
            ModelConfig("complete", "random_forest", {"n_estimators": 5, "random_state": 42}),
            ModelConfig("incomplete", "random_forest", {"n_estimators": 5, "random_state": 42}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            complete_dir = output_dir / "complete"
            complete_dir.mkdir()
            (complete_dir / "fold_metrics.csv").touch()
            (complete_dir / "predictions.csv").touch()
            incomplete_dir = output_dir / "incomplete"
            incomplete_dir.mkdir()
            (incomplete_dir / "fold_metrics.csv").touch()

            pending = pending_model_configs(output_dir, configs)

        self.assertEqual([config.name for config in pending], ["incomplete"])

    def test_cnn_uses_internal_validation_without_exposing_external_fold(self) -> None:
        train_rows = [
            {
                "file_name": f"train-{index}",
                "weight": str(100 + index),
                "weight_category": f"B{index % 4}",
            }
            for index in range(20)
        ]
        external_rows = [{"file_name": "external", "weight": "200", "weight_category": "B1"}]
        config = ModelConfig(
            "cnn_mask_baseline",
            "cnn_mask",
            {
                "epochs": 5,
                "batch_size": 4,
                "learning_rate": 0.001,
                "image_size": 64,
                "random_state": 42,
                "patience": 2,
                "validation_fraction": 0.25,
            },
        )

        with patch("buffalo_weight.train.CnnMaskRegressor", FakeCnnMaskRegressor):
            predictions = predict_fold_weights(train_rows, external_rows, [], config, Path("masks"))

        fitted_names = {row["file_name"] for row in FakeCnnMaskRegressor.fit_rows}
        stopping_names = {row["file_name"] for row in FakeCnnMaskRegressor.early_stopping_rows}
        self.assertEqual(len(FakeCnnMaskRegressor.early_stopping_rows), 5)
        self.assertEqual(fitted_names | stopping_names, {row["file_name"] for row in train_rows})
        self.assertFalse(fitted_names & stopping_names)
        self.assertEqual(FakeCnnMaskRegressor.predicted_rows, external_rows)
        self.assertNotIn("external", fitted_names | stopping_names)
        np.testing.assert_array_equal(predictions, np.zeros(1))

    def test_evaluates_random_forest_by_fold(self) -> None:
        rows = []
        labels = ["Leves", "Medio-Leves", "Medio-Pesados", "Pesados"]
        for index in range(20):
            category = f"Q{(index % 4) + 1}"
            rows.append(
                {
                    "file_name": f"mask-{index:03d}",
                    "weight": str(100 + index * 10),
                    "area": str(50 + index),
                    "perimeter": str(30 + index),
                    "weight_category": category,
                    "weight_category_label": labels[index % 4],
                    "fold": str((index % 5) + 1),
                }
            )

        metrics, predictions = evaluate_random_forest(
            rows,
            ["area", "perimeter"],
            n_estimators=5,
            random_state=42,
        )

        self.assertEqual(len(metrics), 5)
        self.assertEqual(len(predictions), 20)
        self.assertEqual({row["model_config"] for row in metrics}, {"random_forest"})
        self.assertEqual({row["model"] for row in metrics}, {"random_forest"})
        self.assertEqual({row["fold"] for row in metrics}, {"1", "2", "3", "4", "5"})
        self.assertEqual(
            set(metrics[0]),
            {"model_config", "model", "fold", "mae", "rmse", "r2", "n_train", "n_validation"},
        )
        self.assertEqual(
            set(predictions[0]),
            {
                "model",
                "model_config",
                "fold",
                "file_name",
                "weight",
                "y_pred",
                "error",
                "abs_error",
                "weight_category",
                "weight_category_label",
            },
        )

    def test_evaluates_multiple_models(self) -> None:
        rows = []
        labels = ["Leves", "Medio-Leves", "Medio-Pesados", "Pesados"]
        for index in range(20):
            category = f"Q{(index % 4) + 1}"
            rows.append(
                {
                    "file_name": f"mask-{index:03d}",
                    "weight": str(100 + index * 10),
                    "area": str(50 + index),
                    "perimeter": str(30 + index),
                    "weight_category": category,
                    "weight_category_label": labels[index % 4],
                    "fold": str((index % 5) + 1),
                }
            )

        metrics, predictions = evaluate_models(
            rows,
            ["area", "perimeter"],
            model_configs=[
                ModelConfig(
                    "random_forest_baseline",
                    "random_forest",
                    {"n_estimators": 5, "random_state": 42},
                ),
                ModelConfig(
                    "xgboost_baseline",
                    "xgboost",
                    {"n_estimators": 5, "random_state": 42},
                ),
            ],
        )

        self.assertEqual(len(metrics), 10)
        self.assertEqual(len(predictions), 40)
        self.assertEqual(
            {row["model_config"] for row in metrics},
            {"random_forest_baseline", "xgboost_baseline"},
        )
        self.assertEqual({row["model"] for row in metrics}, {"random_forest", "xgboost"})

    def test_train_classical_rejects_mask_prediction_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_classical only supports classical prediction models"):
            train_classical(Path("configs/shared.yaml"), Path("configs/cnn_mask_models.yaml"))

    def test_train_cnn_mask_rejects_classical_prediction_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_cnn_mask only supports mask prediction models"):
            train_cnn_mask(Path("configs/shared.yaml"), Path("configs/classical_models.yaml"))


if __name__ == "__main__":
    unittest.main()
