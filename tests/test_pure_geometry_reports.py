from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.pure_geometry_evaluation import PURE_GEOMETRY_FEATURES, NestedEvaluation
from buffalo_weight.pure_geometry_reports import (
    _geometry_target_matrix,
    _mean_importance_by_feature,
    _mean_training_mae,
    _residual_diagnostics,
    _strongest_weight_correlation,
    _strongest_feature_pair,
    feature_correlation_rows,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_residuals_vs_prediction,
    summarize_oof_predictions,
    write_pure_geometry_reports,
    write_scientific_report,
)


def source_rows() -> list[dict[str, str]]:
    return [
        {
            "file_name": f"mask-{index}",
            "weight": str(100 + 10 * index),
            **{feature: str(index * (column + 1) + column) for column, feature in enumerate(PURE_GEOMETRY_FEATURES)},
        }
        for index in range(1, 11)
    ]


def prediction_rows() -> list[dict[str, str]]:
    offsets = {"ridge": 20.0, "random_forest": 5.0, "xgboost": 10.0}
    return [
        {
            "fold": str((index - 1) % 5 + 1), "model": model, "file_name": f"mask-{index}",
            "weight_category": f"B{index}", "weight": str(100 + 10 * index),
            "prediction": str(100 + 10 * index + offset + index / 10),
            "residual": str(offset + index / 10), "absolute_error": str(offset + index / 10),
        }
        for model, offset in offsets.items()
        for index in range(1, 11)
    ]


def importance_rows() -> list[dict[str, str]]:
    return [
        {"fold": str(fold), "model": model, "feature": feature, "mae_increase_mean": str(index), "mae_increase_std": "0.1"}
        for model in ("ridge", "random_forest", "xgboost")
        for fold in range(1, 6)
        for index, feature in enumerate(PURE_GEOMETRY_FEATURES, start=1)
    ]


def evaluation_bundle() -> NestedEvaluation:
    metrics = [
        {"fold": "1", "model": model, "mae": "20", "r2": "0", "train_mae": "5"}
        for model in ("ridge", "random_forest", "xgboost")
    ]
    tuning = [{"outer_fold": "1", "model": "ridge", "params": "{'alpha': 1}", "mae": "20"}]
    return NestedEvaluation(metrics, prediction_rows(), tuning, importance_rows())


class PureGeometryReportsTest(unittest.TestCase):
    def test_summarizes_pooled_oof_metrics_in_mae_order(self) -> None:
        comparison = summarize_oof_predictions(prediction_rows())

        self.assertEqual(comparison[0]["model"], "random_forest")
        self.assertLess(float(comparison[0]["mae_kg"]), float(comparison[-1]["mae_kg"]))

    def test_correlation_helpers_include_target_and_both_methods(self) -> None:
        rows = source_rows()
        correlations = feature_correlation_rows(rows)

        self.assertEqual(_geometry_target_matrix(rows).shape, (10, len(PURE_GEOMETRY_FEATURES) + 1))
        self.assertEqual({row["method"] for row in correlations}, {"pearson", "spearman"})
        self.assertIn(_strongest_weight_correlation(correlations)[0], PURE_GEOMETRY_FEATURES)

    def test_importance_and_residual_helpers_aggregate_outer_predictions(self) -> None:
        values = _mean_importance_by_feature(importance_rows(), "random_forest")
        diagnostics = _residual_diagnostics(prediction_rows(), "random_forest")

        self.assertEqual(set(values), set(PURE_GEOMETRY_FEATURES))
        self.assertAlmostEqual(diagnostics["pred_spearman"], 1.0)
        self.assertEqual(_strongest_feature_pair(feature_correlation_rows(source_rows()))[2], 1.0)
        self.assertEqual(_mean_training_mae([{"model": "random_forest", "train_mae": "5"}], "random_forest"), 5.0)
        with self.assertRaisesRegex(ValueError, "0 times.*expected at least one"):
            _mean_training_mae([], "missing")

    def test_writes_all_required_csvs_plots_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            comparison = write_pure_geometry_reports(evaluation_bundle(), source_rows(), output_dir)

            self.assertEqual(comparison[0]["model"], "random_forest")
            for name in ("model_comparison.csv", "feature_importance.png", "residuals_vs_prediction.png", "correlation_matrix.png", "report.md"):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_individual_plot_and_report_functions_create_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            correlations = feature_correlation_rows(source_rows())
            comparison = summarize_oof_predictions(prediction_rows())
            plot_feature_importance(importance_rows(), output / "importance.png")
            plot_residuals_vs_prediction(prediction_rows()[:10], output / "residuals.png")
            plot_correlation_matrix(source_rows(), output / "correlation.png")
            write_scientific_report(evaluation_bundle(), comparison, correlations, output / "report.md")

            self.assertTrue(all(path.is_file() for path in output.iterdir()))
            self.assertIn("seleção de hiperparâmetros", (output / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
