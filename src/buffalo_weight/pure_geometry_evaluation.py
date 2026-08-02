from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from buffalo_weight.models import ModelConfig, build_model
from buffalo_weight.split import assign_folds, assign_weight_categories, parse_weight, read_rows
from buffalo_weight.train import format_metric, rows_to_arrays


PURE_GEOMETRY_FEATURES = [
    "area",
    "perimeter",
    "solidity",
    "circularity",
    "equivalent_diameter",
    "convex_area",
    "convexity",
    "hu_moment_1",
    "hu_moment_2",
    "area_power_1_5",
]
FORBIDDEN_POSTURE_FEATURES = {
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "aspect_ratio",
    "extent",
    "major_axis_length",
    "minor_axis_length",
    "middle_thickness",
    "end_thickness_min",
    "end_thickness_max",
    "middle_to_end_ratio",
    "centroid_x_offset",
    "centroid_y_ratio",
}


@dataclass(frozen=True)
class NestedEvaluation:
    fold_metrics: list[dict[str, str]]
    predictions: list[dict[str, str]]
    tuning_results: list[dict[str, str]]
    importance_rows: list[dict[str, str]]


def load_pure_geometry_rows(features_path: Path) -> list[dict[str, str]]:
    """Load and validate the geometry-only training table.

    Example: ``load_pure_geometry_rows(Path("generated/features.csv"))``.
    """
    rows = read_rows(features_path)
    if not rows:
        raise ValueError(f"feature index {features_path} had 0 rows; expected at least 5 rows")
    _validate_feature_contract(rows[0], features_path)
    rows_to_arrays(rows, PURE_GEOMETRY_FEATURES)
    return rows


def _validate_feature_contract(first_row: dict[str, str], features_path: Path) -> None:
    missing = sorted(set(PURE_GEOMETRY_FEATURES) - set(first_row))
    if missing:
        raise ValueError(f"feature index {features_path} missed {missing!r}; expected pure geometry columns")
    selected_forbidden = sorted(set(PURE_GEOMETRY_FEATURES) & FORBIDDEN_POSTURE_FEATURES)
    if selected_forbidden:
        raise ValueError(f"selected features included {selected_forbidden!r}; expected no posture features")


def stratified_geometry_rows(
    rows: list[dict[str, str]], k: int, category_count: int, random_state: int
) -> list[dict[str, str]]:
    """Create representative weight folds without modifying source rows.

    Example: ``stratified_geometry_rows(rows, 5, 10, 42)``.
    """
    split_rows = [row.copy() for row in rows]
    assign_weight_categories(split_rows, category_count)
    assign_folds(split_rows, k, random_state)
    validate_fold_representation(split_rows, k, category_count)
    return split_rows


def validate_fold_representation(
    rows: list[dict[str, str]], k: int, category_count: int
) -> None:
    """Require every weight range in every validation fold.

    Example: ``validate_fold_representation(rows, 5, 10)``.
    """
    expected = {f"B{index}" for index in range(1, category_count + 1)}
    for fold in range(1, k + 1):
        observed = {row["weight_category"] for row in rows if int(row["fold"]) == fold}
        if observed != expected:
            raise ValueError(f"fold {fold} had categories {sorted(observed)!r}; expected {sorted(expected)!r}")


def ridge_candidates() -> list[ModelConfig]:
    """Return Ridge baselines whose alpha is selected inside each outer fold.

    Example: ``ridge_candidates()[0].model == "ridge"``.
    """
    alphas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    return [ModelConfig("ridge", "ridge", {"alpha": alpha}) for alpha in alphas]


def random_forest_candidates(random_state: int) -> list[ModelConfig]:
    """Return regularized Random Forest candidates for nested selection.

    Example: ``random_forest_candidates(42)[0].model == "random_forest"``.
    """
    candidates = []
    choices = product(("identity", "log"), (None, 4, 8), (1, 2, 4, 6), ("sqrt", 0.7, 1.0))
    for target_transform, max_depth, min_leaf, max_features in choices:
        params = _forest_params(random_state, max_depth, min_leaf, max_features)
        params["target_transform"] = target_transform
        candidates.append(ModelConfig("random_forest", "random_forest", params))
    return candidates


def _forest_params(
    random_state: int, max_depth: int | None, min_leaf: int, max_features: float | str
) -> dict[str, bool | float | int | str]:
    params: dict[str, bool | float | int | str] = {
        "n_estimators": 600,
        "random_state": random_state,
        "n_jobs": -1,
        "min_samples_leaf": min_leaf,
        "max_features": max_features,
    }
    if max_depth is not None:
        params["max_depth"] = max_depth
    return params


def xgboost_candidates(random_state: int) -> list[ModelConfig]:
    """Return conservative XGBoost candidates for the 132-row dataset.

    Example: ``xgboost_candidates(42)[0].model == "xgboost"``.
    """
    candidates = []
    schedules = ((0.02, 600), (0.05, 300), (0.1, 150))
    choices = product(("identity", "log"), (1, 2, 3), schedules, (1, 5))
    for target_transform, depth, min_step, min_child_weight in choices:
        params = _xgboost_params(random_state, depth, min_step, min_child_weight)
        params["target_transform"] = target_transform
        candidates.append(ModelConfig("xgboost", "xgboost", params))
    return candidates


def _xgboost_params(
    random_state: int, depth: int, learning_step: tuple[float, int], min_child_weight: int
) -> dict[str, bool | float | int | str]:
    learning_rate, n_estimators = learning_step
    return {
        "n_estimators": n_estimators,
        "random_state": random_state,
        "learning_rate": learning_rate,
        "max_depth": depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": float(min_child_weight),
        "reg_alpha": 0.1,
        "n_jobs": -1,
    }


def nested_evaluate_models(
    rows: list[dict[str, str]], random_state: int, inner_k: int = 4
) -> NestedEvaluation:
    """Evaluate Ridge, RF, and XGBoost with tuning isolated inside outer folds.

    Example: ``nested_evaluate_models(stratified_rows, 42)``.
    """
    candidates_by_model = {
        "ridge": ridge_candidates(),
        "random_forest": random_forest_candidates(random_state),
        "xgboost": xgboost_candidates(random_state),
    }
    return _evaluate_candidate_groups(rows, candidates_by_model, random_state, inner_k)


def _evaluate_candidate_groups(
    rows: list[dict[str, str]], candidates_by_model: dict[str, list[ModelConfig]], seed: int, inner_k: int
) -> NestedEvaluation:
    metrics: list[dict[str, str]] = []
    predictions: list[dict[str, str]] = []
    tuning: list[dict[str, str]] = []
    importances: list[dict[str, str]] = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        fold_outputs = _evaluate_outer_fold(rows, candidates_by_model, fold, seed, inner_k)
        metrics.extend(fold_outputs.fold_metrics)
        predictions.extend(fold_outputs.predictions)
        tuning.extend(fold_outputs.tuning_results)
        importances.extend(fold_outputs.importance_rows)
    return NestedEvaluation(metrics, predictions, tuning, importances)


def _evaluate_outer_fold(
    rows: list[dict[str, str]], candidates_by_model: dict[str, list[ModelConfig]], fold: int, seed: int, inner_k: int
) -> NestedEvaluation:
    train_rows = [row for row in rows if int(row["fold"]) != fold]
    validation_rows = [row for row in rows if int(row["fold"]) == fold]
    outputs = NestedEvaluation([], [], [], [])
    for model_name, candidates in candidates_by_model.items():
        result = _fit_outer_model(train_rows, validation_rows, candidates, model_name, fold, seed, inner_k)
        outputs.fold_metrics.extend(result.fold_metrics)
        outputs.predictions.extend(result.predictions)
        outputs.tuning_results.extend(result.tuning_results)
        outputs.importance_rows.extend(result.importance_rows)
    return outputs


def _fit_outer_model(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], candidates: list[ModelConfig],
    model_name: str, fold: int, seed: int, inner_k: int,
) -> NestedEvaluation:
    inner_rows = stratified_geometry_rows(train_rows, inner_k, min(8, len(train_rows) // inner_k), seed + fold)
    selected, tuning_rows = _select_candidate(inner_rows, candidates, model_name, fold)
    model = build_model(selected)
    x_train, y_train = rows_to_arrays(train_rows, PURE_GEOMETRY_FEATURES)
    x_validation, y_validation = rows_to_arrays(validation_rows, PURE_GEOMETRY_FEATURES)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    validation_pred = model.predict(x_validation)
    metric = _fold_metric(model_name, fold, y_train, train_pred, y_validation, validation_pred, selected)
    predictions = _prediction_rows(validation_rows, validation_pred, model_name, fold)
    importance = _permutation_importance_rows(model, x_validation, y_validation, model_name, fold, seed)
    return NestedEvaluation([metric], predictions, tuning_rows, importance)


def _select_candidate(
    rows: list[dict[str, str]], candidates: list[ModelConfig], model_name: str, outer_fold: int
) -> tuple[ModelConfig, list[dict[str, str]]]:
    scored = [(candidate, _inner_candidate_mae(rows, candidate)) for candidate in candidates]
    tuning_rows = [
        {"outer_fold": str(outer_fold), "model": model_name, "params": repr(candidate.params), "mae": format_metric(mae)}
        for candidate, mae in scored
    ]
    return min(scored, key=lambda item: item[1])[0], tuning_rows


def _inner_candidate_mae(rows: list[dict[str, str]], candidate: ModelConfig) -> float:
    errors = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        validation_rows = [row for row in rows if int(row["fold"]) == fold]
        errors.append(_candidate_fold_mae(train_rows, validation_rows, candidate))
    return float(np.mean(errors))


def _candidate_fold_mae(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], candidate: ModelConfig
) -> float:
    x_train, y_train = rows_to_arrays(train_rows, PURE_GEOMETRY_FEATURES)
    x_validation, y_validation = rows_to_arrays(validation_rows, PURE_GEOMETRY_FEATURES)
    model = build_model(candidate)
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_validation, model.predict(x_validation)))


def _fold_metric(
    model_name: str, fold: int, y_train: np.ndarray, train_pred: np.ndarray,
    y_validation: np.ndarray, validation_pred: np.ndarray, selected: ModelConfig,
) -> dict[str, str]:
    return {
        "fold": str(fold), "model": model_name,
        "train_mae": format_metric(mean_absolute_error(y_train, train_pred)),
        "mae": format_metric(mean_absolute_error(y_validation, validation_pred)),
        "r2": format_metric(r2_score(y_validation, validation_pred)),
        "n_train": str(len(y_train)), "n_validation": str(len(y_validation)),
        "selected_params": repr(selected.params),
    }


def _prediction_rows(
    rows: list[dict[str, str]], predicted: np.ndarray, model_name: str, fold: int
) -> list[dict[str, str]]:
    return [
        _prediction_row(row, float(y_pred), model_name, fold)
        for row, y_pred in zip(rows, predicted, strict=True)
    ]


def _prediction_row(
    row: dict[str, str], predicted: float, model_name: str, fold: int
) -> dict[str, str]:
    actual = parse_weight(row["weight"], row.get("file_name", ""))
    return {
        "fold": str(fold), "model": model_name, "file_name": row["file_name"],
        "weight_category": row["weight_category"], "weight": format_metric(actual),
        "prediction": format_metric(predicted), "residual": format_metric(predicted - actual),
        "absolute_error": format_metric(abs(predicted - actual)),
    }


def _permutation_importance_rows(
    model: object, x_validation: np.ndarray, y_validation: np.ndarray,
    model_name: str, fold: int, seed: int,
) -> list[dict[str, str]]:
    baseline = mean_absolute_error(y_validation, model.predict(x_validation))  # type: ignore[attr-defined]
    rows = []
    for feature_index, feature in enumerate(PURE_GEOMETRY_FEATURES):
        increases = _permuted_mae_increases(model, x_validation, y_validation, feature_index, baseline, seed + fold)
        rows.append(_importance_row(model_name, fold, feature, increases))
    return rows


def _permuted_mae_increases(
    model: object, x_values: np.ndarray, y_values: np.ndarray, feature_index: int, baseline: float, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed + feature_index * 1009)
    increases = []
    for _ in range(20):
        permuted = x_values.copy()
        permuted[:, feature_index] = generator.permutation(permuted[:, feature_index])
        prediction = model.predict(permuted)  # type: ignore[attr-defined]
        increases.append(mean_absolute_error(y_values, prediction) - baseline)
    return np.asarray(increases, dtype=float)


def _importance_row(
    model_name: str, fold: int, feature: str, increases: np.ndarray
) -> dict[str, str]:
    return {
        "fold": str(fold), "model": model_name, "feature": feature,
        "mae_increase_mean": format_metric(float(np.mean(increases))),
        "mae_increase_std": format_metric(float(np.std(increases))),
    }
