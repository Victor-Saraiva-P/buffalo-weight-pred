from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.artifact_provenance import TrainingEvidence
from buffalo_weight.mask_geometry_reports import write_mask_geometry_reports
from buffalo_weight.models import ModelConfig


class MaskGeometryReportsTest(unittest.TestCase):
    def test_reports_pooled_oof_metrics_for_both_input_contracts(self) -> None:
        configs = [self._config("mask_only", "cnn_mask"), self._config("fusion", "cnn_mask_geometry")]
        predictions = [
            *self._predictions("mask_only", "cnn_mask", (90.0, 220.0)),
            *self._predictions("fusion", "cnn_mask_geometry", (95.0, 205.0)),
        ]
        metrics = [self._metric(config, fold) for config in configs for fold in (1, 2)]
        evidence = TrainingEvidence([], [], ["area"], None, "cpu")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "mask_geometry"
            pure_dir = output_dir.parent / "pure_geometry"
            pure_dir.mkdir()
            (pure_dir / "model_comparison.csv").write_text(
                "model,mae_kg,r2,bias_kg,heavy_20pct_mae_kg,heavy_20pct_bias_kg\n"
                "random_forest,20,0.5,0,30,-10\n"
            )
            comparison = write_mask_geometry_reports(
                metrics, predictions, configs, evidence, output_dir
            )

            self.assertTrue((output_dir / "report.md").is_file())
            self.assertTrue((output_dir / "model_comparison.png").is_file())

        self.assertEqual(
            [row["model"] for row in comparison], ["fusion", "mask_only", "random_forest"]
        )
        self.assertEqual(float(comparison[0]["mae_kg"]), 5.0)
        self.assertEqual(comparison[0]["input"], "máscara + geometria pura")
        self.assertEqual(comparison[2]["input"], "geometria pura")

    def _config(self, name: str, model: str) -> ModelConfig:
        return ModelConfig(
            name,
            model,
            {
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": 0.1,
                "image_size": 8,
                "random_state": 1,
            },
        )

    def _metric(self, config: ModelConfig, fold: int) -> dict[str, str]:
        return {
            "model_config": config.name,
            "model": config.model,
            "fold": str(fold),
            "mae": "1",
            "rmse": "1",
            "r2": "",
            "n_train": "1",
            "n_validation": "1",
        }

    def _predictions(
        self, model_config: str, model: str, predicted: tuple[float, float]
    ) -> list[dict[str, str]]:
        return [
            self._prediction(model_config, model, 1, 100.0, predicted[0]),
            self._prediction(model_config, model, 2, 200.0, predicted[1]),
        ]

    def _prediction(
        self, model_config: str, model: str, fold: int, weight: float, predicted: float
    ) -> dict[str, str]:
        error = predicted - weight
        return {
            "model_config": model_config,
            "model": model,
            "fold": str(fold),
            "file_name": f"mask-{fold}",
            "weight": str(weight),
            "y_pred": str(predicted),
            "error": str(error),
            "abs_error": str(abs(error)),
            "weight_category": f"B{fold}",
            "weight_category_label": f"Faixa {fold}",
        }


if __name__ == "__main__":
    unittest.main()
