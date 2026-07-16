from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from buffalo_weight.models import ModelConfig, build_model
from buffalo_weight.split import assign_folds, assign_weight_categories, parse_weight
from buffalo_weight.train import format_metric, rows_to_arrays


NON_NEGATIVE_FEATURES = {
    "area", "perimeter", "solidity", "circularity", "equivalent_diameter",
    "bbox_width", "bbox_height", "bbox_area", "aspect_ratio", "extent",
    "convex_area", "convexity", "major_axis_length", "minor_axis_length",
    "area_power_1_5", "area_major_axis_product", "middle_thickness",
    "end_thickness_min", "end_thickness_max", "middle_to_end_ratio",
    "centroid_x_offset", "centroid_y_ratio",
}


def parse_feature_value(row: dict[str, str], feature: str) -> float:
    raw_value = row.get(feature, "")
    try:
        value = float(raw_value.replace(",", "."))
    except ValueError as error:
        file_name = row.get("file_name", "")
        message = f"Invalid feature {feature} for {file_name}: {raw_value!r}; expected a finite number"
        raise ValueError(message) from error
    if math.isfinite(value):
        return value
    file_name = row.get("file_name", "")
    raise ValueError(f"Invalid feature {feature} for {file_name}: {raw_value!r}; expected a finite number")


def validate_feature_values(rows: list[dict[str, str]], features: list[str]) -> None:
    if not rows:
        raise ValueError("feature index is empty; run make features before make analyze-features")
    missing = sorted(feature for feature in features if feature not in rows[0])
    if missing:
        raise ValueError(f"feature index missing columns: {', '.join(missing)}")
    for row in rows:
        validate_feature_row(row, features)


def validate_feature_row(row: dict[str, str], features: list[str]) -> None:
    for feature in features:
        value = parse_feature_value(row, feature)
        if feature in NON_NEGATIVE_FEATURES and value < 0:
            file_name = row.get("file_name", "")
            raise ValueError(f"Invalid feature {feature} for {file_name}: {value}; expected a non-negative number")


def feature_matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return np.asarray([[parse_feature_value(row, feature) for feature in features] for row in rows], dtype=float)


def metric_values(y_validation: np.ndarray, y_pred: np.ndarray) -> dict[str, str]:
    r2 = r2_score(y_validation, y_pred) if len(y_validation) > 1 else math.nan
    return {
        "mae": format_metric(mean_absolute_error(y_validation, y_pred)),
        "rmse": format_metric(mean_squared_error(y_validation, y_pred) ** 0.5),
        "r2": format_metric(r2),
    }


def mean_prediction_metrics(train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]]) -> dict[str, str]:
    y_train = np.asarray([parse_weight(row["weight"], row.get("file_name", "")) for row in train_rows], dtype=float)
    y_validation = np.asarray([parse_weight(row["weight"], row.get("file_name", "")) for row in validation_rows], dtype=float)
    return metric_values(y_validation, np.repeat(float(np.mean(y_train)), len(validation_rows)))


def model_prediction_metrics(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], feature_columns: list[str], model_config: ModelConfig
) -> dict[str, str]:
    x_train, y_train = rows_to_arrays(train_rows, feature_columns)
    x_validation, y_validation = rows_to_arrays(validation_rows, feature_columns)
    model = build_model(model_config)
    model.fit(x_train, y_train)
    return metric_values(y_validation, model.predict(x_validation))


def permuted_prediction_metrics(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], features: list[str], feature: str,
    model_config: ModelConfig, seed: int,
) -> dict[str, str]:
    x_train, y_train = rows_to_arrays(train_rows, features)
    x_validation, y_validation = rows_to_arrays(validation_rows, features)
    x_validation[:, features.index(feature)] = np.random.default_rng(seed).permutation(x_validation[:, features.index(feature)])
    model = build_model(model_config)
    model.fit(x_train, y_train)
    return metric_values(y_validation, model.predict(x_validation))


def metric_row(
    model_config: ModelConfig, scenario: str, feature: str, seed: int, fold: int,
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], metrics: dict[str, str],
) -> dict[str, str]:
    return {
        "model_config": model_config.name, "model": model_config.model, "scenario": scenario,
        "feature": feature, "split_random_state": str(seed), "fold": str(fold),
        "n_train": str(len(train_rows)), "n_validation": str(len(validation_rows)), **metrics,
    }


def permutation_seed(split_seed: int, fold: int, feature: str) -> int:
    feature_hash = sum((index + 1) * ord(char) for index, char in enumerate(feature))
    return split_seed * 1009 + fold * 9176 + feature_hash


def evaluate_feature_scenarios(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], features: list[str],
    feature: str, model_config: ModelConfig, seed: int, fold: int,
) -> list[dict[str, str]]:
    rows = [scenario_row(train_rows, validation_rows, [feature], model_config, "single_feature", feature, seed, fold)]
    without_columns = [column for column in features if column != feature]
    if without_columns:
        rows.append(scenario_row(train_rows, validation_rows, without_columns, model_config, "without_feature", feature, seed, fold))
    metrics = permuted_prediction_metrics(train_rows, validation_rows, features, feature, model_config, permutation_seed(seed, fold, feature))
    return [*rows, metric_row(model_config, "permuted_feature", feature, seed, fold, train_rows, validation_rows, metrics)]


def scenario_row(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], columns: list[str],
    model_config: ModelConfig, scenario: str, feature: str, seed: int, fold: int,
) -> dict[str, str]:
    metrics = model_prediction_metrics(train_rows, validation_rows, columns, model_config)
    return metric_row(model_config, scenario, feature, seed, fold, train_rows, validation_rows, metrics)


def evaluate_fold_scenarios(
    split_rows: list[dict[str, str]], features: list[str], model_config: ModelConfig, seed: int, fold: int
) -> list[dict[str, str]]:
    train_rows = [row for row in split_rows if int(row["fold"]) != fold]
    validation_rows = [row for row in split_rows if int(row["fold"]) == fold]
    rows = base_scenario_rows(train_rows, validation_rows, features, model_config, seed, fold)
    for feature in features:
        rows.extend(evaluate_feature_scenarios(train_rows, validation_rows, features, feature, model_config, seed, fold))
    return rows


def base_scenario_rows(
    train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], features: list[str],
    model_config: ModelConfig, seed: int, fold: int,
) -> list[dict[str, str]]:
    rows = [metric_row(model_config, "mean_baseline", "", seed, fold, train_rows, validation_rows, mean_prediction_metrics(train_rows, validation_rows))]
    rows.append(scenario_row(train_rows, validation_rows, features, model_config, "all_features", "", seed, fold))
    if "area" in features:
        rows.append(scenario_row(train_rows, validation_rows, ["area"], model_config, "area_baseline", "area", seed, fold))
    return rows


def evaluate_feature_analysis(
    rows: list[dict[str, str]], features: list[str], model_configs: list[ModelConfig],
    k: int, weight_category_count: int, seeds: list[int],
) -> list[dict[str, str]]:
    metric_rows = []
    for seed in seeds:
        split_rows = split_feature_rows(rows, weight_category_count, k, seed)
        for model_config in model_configs:
            for fold in sorted({int(row["fold"]) for row in split_rows}):
                metric_rows.extend(evaluate_fold_scenarios(split_rows, features, model_config, seed, fold))
    return metric_rows


def split_feature_rows(rows: list[dict[str, str]], weight_category_count: int, k: int, seed: int) -> list[dict[str, str]]:
    split_rows = [row.copy() for row in rows]
    assign_weight_categories(split_rows, weight_category_count)
    assign_folds(split_rows, k, seed)
    return split_rows
