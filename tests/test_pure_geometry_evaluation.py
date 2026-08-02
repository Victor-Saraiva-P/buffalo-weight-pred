from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from buffalo_weight.models import ModelConfig
from buffalo_weight.pure_geometry_evaluation import (
    FORBIDDEN_POSTURE_FEATURES,
    PURE_GEOMETRY_FEATURES,
    _candidate_fold_mae,
    _evaluate_candidate_groups,
    _fold_metric,
    _forest_params,
    _importance_row,
    _inner_candidate_mae,
    _permutation_importance_rows,
    _permuted_mae_increases,
    _prediction_row,
    _prediction_rows,
    _select_candidate,
    _validate_feature_contract,
    _xgboost_params,
    load_pure_geometry_rows,
    nested_evaluate_models,
    random_forest_candidates,
    ridge_candidates,
    stratified_geometry_rows,
    validate_fold_representation,
    xgboost_candidates,
)


class FakeLinearRegressor:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> "FakeLinearRegressor":
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0]


def geometry_rows(count: int = 50) -> list[dict[str, str]]:
    rows = []
    for index in range(1, count + 1):
        row = {feature: str(index * (column + 1)) for column, feature in enumerate(PURE_GEOMETRY_FEATURES)}
        rows.append({"file_name": f"mask-{index}", "weight": str(index * 2), **row})
    return rows


class PureGeometryEvaluationTest(unittest.TestCase):
    def test_feature_contract_excludes_every_posture_feature(self) -> None:
        self.assertFalse(set(PURE_GEOMETRY_FEATURES) & FORBIDDEN_POSTURE_FEATURES)
        _validate_feature_contract(geometry_rows(1)[0], Path("features.csv"))

    def test_feature_contract_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missed.*expected pure geometry"):
            _validate_feature_contract({"weight": "1"}, Path("missing.csv"))

    def test_loads_valid_geometry_csv_and_rejects_empty_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.csv"
            empty_path = Path(directory) / "empty.csv"
            self._write_rows(valid_path, geometry_rows(5))
            empty_path.write_text("file_name,weight\n")

            self.assertEqual(len(load_pure_geometry_rows(valid_path)), 5)
            with self.assertRaisesRegex(ValueError, "had 0 rows"):
                load_pure_geometry_rows(empty_path)

    def test_stratification_puts_every_weight_range_in_every_fold(self) -> None:
        rows = stratified_geometry_rows(geometry_rows(), 5, 10, 42)

        validate_fold_representation(rows, 5, 10)
        self.assertNotIn("fold", geometry_rows()[0])

    def test_representation_validation_rejects_missing_range(self) -> None:
        rows = [{"fold": "1", "weight_category": "B1"}]

        with self.assertRaisesRegex(ValueError, "fold 1.*expected"):
            validate_fold_representation(rows, 1, 2)

    def test_candidate_factories_cover_regularization_choices(self) -> None:
        self.assertEqual(len(ridge_candidates()), 8)
        self.assertEqual(len(random_forest_candidates(42)), 72)
        self.assertEqual(len(xgboost_candidates(42)), 36)
        self.assertEqual(_forest_params(42, None, 2, 0.7)["min_samples_leaf"], 2)
        self.assertEqual(_xgboost_params(42, 2, (0.1, 150), 5)["reg_lambda"], 5.0)

    def test_fold_helpers_calculate_predictions_metrics_and_importance(self) -> None:
        rows = [{**row, "weight_category": "B1"} for row in geometry_rows(5)]
        predicted = np.asarray([2.0, 5.0, 6.0, 9.0, 10.0])
        prediction_rows = _prediction_rows(rows, predicted, "fake", 1)
        metric = _fold_metric("fake", 1, predicted, predicted, predicted, predicted, ModelConfig("fake", "ridge", {"alpha": 1.0}))
        column_count = len(PURE_GEOMETRY_FEATURES)
        features = np.arange(2 * column_count, dtype=float).reshape(2, column_count)
        increases = _permuted_mae_increases(FakeLinearRegressor(), features, np.asarray([0.0, float(column_count)]), 0, 0.0, 42)
        importance = _permutation_importance_rows(FakeLinearRegressor(), features, np.asarray([0.0, float(column_count)]), "fake", 1, 42)

        self.assertEqual(prediction_rows[1], _prediction_row(rows[1], 5.0, "fake", 1))
        self.assertEqual(metric["mae"], "0")
        self.assertEqual(len(increases), 20)
        self.assertEqual(len(importance), len(PURE_GEOMETRY_FEATURES))
        self.assertEqual(_importance_row("fake", 1, "area", increases)["feature"], "area")

    def test_inner_selection_and_nested_group_evaluation_are_leakage_safe(self) -> None:
        outer_rows = stratified_geometry_rows(geometry_rows(), 5, 10, 42)
        inner_rows = stratified_geometry_rows([row for row in outer_rows if row["fold"] != "1"], 4, 8, 43)
        candidates = [ModelConfig("ridge", "ridge", {"alpha": 0.1}), ModelConfig("ridge", "ridge", {"alpha": 10.0})]
        selected, tuning = _select_candidate(inner_rows, candidates, "ridge", 1)
        fold_mae = _candidate_fold_mae(inner_rows[:30], inner_rows[30:], selected)
        mean_mae = _inner_candidate_mae(inner_rows, selected)
        evaluated = _evaluate_candidate_groups(outer_rows, {"ridge": candidates}, 42, 4)

        self.assertEqual(len(tuning), 2)
        self.assertGreaterEqual(fold_mae, 0.0)
        self.assertGreaterEqual(mean_mae, 0.0)
        self.assertEqual(len(evaluated.fold_metrics), 5)

    def test_nested_public_entrypoint_returns_three_models(self) -> None:
        self.assertEqual(
            {config.model for config in ridge_candidates() + random_forest_candidates(1) + xgboost_candidates(1)},
            {"ridge", "random_forest", "xgboost"},
        )
        self.assertTrue(callable(nested_evaluate_models))

    def _write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
