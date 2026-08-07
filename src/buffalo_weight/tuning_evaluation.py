"""Evaluation orchestration for pre-registered tuning variations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from buffalo_weight.baseline_comparison_metrics import comparison_metric_rows
from buffalo_weight.baseline_comparison_types import ComparisonMetric, ComparisonPrediction
from buffalo_weight.compact_cnn_adapter import CompactCnnAdapter
from buffalo_weight.compact_cnn_evaluation import evaluate_compact_cnn, load_compact_cnn_samples
from buffalo_weight.compact_cnn_types import CompactCnnRecipe, CompactCnnTrainingAdapter
from buffalo_weight.dense_baseline_stage import DenseBaselineDependencies
from buffalo_weight.dense_feature_adapter import DenseFeatureAdapter, DenseTrainingRecipe
from buffalo_weight.feature_baselines import DenseFeatureBaseline
from buffalo_weight.feature_evaluation import PredictionPartition, TrainingPartition
from buffalo_weight.input_schema import SPLIT_COLUMNS
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_artifacts import load_resnet_samples
from buffalo_weight.resnet_baseline_evaluation import ResNetBaselineEvaluator, ResNetOofPrediction
from buffalo_weight.resnet_baseline_stage import ResNetBaselineRunner, ScientificResNetBaselineRunner
from buffalo_weight.tuning_types import TuningVariation


def evaluate_tuning_variations(
    contract: ReportContract, approach: str, baseline_config: str,
    frozen_features: tuple[str, ...] | None,
    variations: tuple[TuningVariation, ...],
    dense_adapter: DenseFeatureAdapter | None = None,
    compact_cnn_adapter: CompactCnnTrainingAdapter | None = None,
    resnet_runner: ResNetBaselineRunner | None = None,
) -> tuple[list[ComparisonPrediction], list[ComparisonMetric]]:
    """Evaluate pre-registered variations for the confirmed approach.

    Example: ``evaluate_tuning_variations(contract, "random_forest", ...)`` returns predictions and metrics.
    """
    predictions: list[ComparisonPrediction] = []
    for variation in variations:
        variation_predictions = _evaluate_single_variation(
            contract, variation, frozen_features, dense_adapter,
            compact_cnn_adapter, resnet_runner,
        )
        predictions.extend(variation_predictions)
    metrics = _compute_metrics(predictions, variations)
    return predictions, metrics


def _evaluate_single_variation(
    contract: ReportContract, variation: TuningVariation,
    frozen_features: tuple[str, ...] | None,
    dense_adapter: DenseFeatureAdapter | None,
    compact_cnn_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
) -> list[ComparisonPrediction]:
    if variation.approach == "random_forest":
        return _evaluate_rf_variation(contract, variation, frozen_features)
    if variation.approach == "dense_feature_network":
        return _evaluate_dense_variation(contract, variation, frozen_features, dense_adapter)
    if variation.approach == "compact_cnn":
        return _evaluate_compact_cnn_variation(contract, variation, compact_cnn_adapter)
    if variation.approach == "resnet18":
        return _evaluate_resnet_variation(contract, variation, resnet_runner)
    raise ValueError(f"approach was {variation.approach!r}; expected an allowed approach")


def _evaluate_rf_variation(
    contract: ReportContract, variation: TuningVariation,
    frozen_features: tuple[str, ...] | None,
) -> list[ComparisonPrediction]:
    if frozen_features is None or not frozen_features:
        raise ValueError(f"frozen features were {frozen_features!r}; expected non-empty feature tuple")
    features_path = contract.inputs_output_dir / "feature_index.csv"
    split_path = contract.inputs_output_dir / "canonical_split.csv"
    feature_index = _load_feature_index(features_path)
    split_rows = _load_csv_rows(split_path, SPLIT_COLUMNS)
    recipe_dict = _extract_dict_recipe(variation.recipe)
    predictions: list[ComparisonPrediction] = []
    for fold in sorted({int(row["fold"]) for row in split_rows}):
        predictions.extend(_evaluate_rf_fold(
            variation, recipe_dict, frozen_features, split_rows, feature_index, fold,
        ))
    return predictions


def _evaluate_rf_fold(
    variation: TuningVariation, recipe_dict: dict[str, object],
    frozen_features: tuple[str, ...], split_rows: list[dict[str, str]],
    feature_index: dict[str, dict[str, float]], fold: int,
) -> list[ComparisonPrediction]:
    train_split = [r for r in split_rows if int(r["fold"]) != fold]
    val_split = [r for r in split_rows if int(r["fold"]) == fold]
    train_partition = _build_training_partition(train_split, feature_index, frozen_features)
    val_partition = _build_prediction_partition(val_split, feature_index, frozen_features)
    regressor = RandomForestRegressor(**recipe_dict)
    regressor.fit(train_partition.values, train_partition.targets_kg)
    preds = regressor.predict(val_partition.values)
    return [
        ComparisonPrediction(
            variation.name, variation.approach, "tuned", row["file_name"],
            row["weight_category"], fold, float(row["weight_kg"]), float(pred),
        )
        for row, pred in zip(val_split, preds)
    ]


def _evaluate_dense_variation(
    contract: ReportContract, variation: TuningVariation,
    frozen_features: tuple[str, ...] | None, adapter: DenseFeatureAdapter | None,
) -> list[ComparisonPrediction]:
    if frozen_features is None or not frozen_features:
        raise ValueError(f"frozen features were {frozen_features!r}; expected non-empty feature tuple")
    features_path = contract.inputs_output_dir / "feature_index.csv"
    split_path = contract.inputs_output_dir / "canonical_split.csv"
    feature_index = _load_feature_index(features_path)
    split_rows = _load_csv_rows(split_path, SPLIT_COLUMNS)
    recipe = _extract_dense_recipe(variation.recipe)
    resolved_adapter = adapter or DenseFeatureAdapter()
    baseline_evaluator = DenseFeatureBaseline(recipe=recipe, adapter=resolved_adapter)
    predictions: list[ComparisonPrediction] = []
    for fold in sorted({int(row["fold"]) for row in split_rows}):
        train_split = [r for r in split_rows if int(r["fold"]) != fold]
        val_split = [r for r in split_rows if int(r["fold"]) == fold]
        train_part = _build_training_partition(train_split, feature_index, frozen_features)
        val_part = _build_prediction_partition(val_split, feature_index, frozen_features)
        predictor = baseline_evaluator.fit(train_part, frozen_features)
        preds = predictor.predict(val_part)
        for row, pred in zip(val_split, preds):
            predictions.append(ComparisonPrediction(
                variation.name, variation.approach, "tuned", row["file_name"],
                row["weight_category"], fold, float(row["weight_kg"]), float(pred),
            ))
    return predictions


def _evaluate_compact_cnn_variation(
    contract: ReportContract, variation: TuningVariation,
    adapter: CompactCnnTrainingAdapter | None,
) -> list[ComparisonPrediction]:
    recipe = _extract_compact_recipe(variation.recipe)
    resolved_adapter = adapter or CompactCnnAdapter()
    split_path = contract.inputs_output_dir / "canonical_split.csv"
    samples = load_compact_cnn_samples(split_path, contract.inputs.masks_dir)
    evaluation = evaluate_compact_cnn(samples, resolved_adapter, recipe)
    return [
        ComparisonPrediction(
            variation.name, variation.approach, "tuned", row.file_name,
            row.weight_category, row.fold, row.observed_weight_kg, row.predicted_weight_kg,
        )
        for row in evaluation.predictions
    ]


def _evaluate_resnet_variation(
    contract: ReportContract, variation: TuningVariation,
    runner: ResNetBaselineRunner | None,
) -> list[ComparisonPrediction]:
    recipe = _extract_resnet_recipe(variation.recipe)
    split_path = contract.inputs_output_dir / "canonical_split.csv"
    samples = load_resnet_samples(split_path, contract.inputs.masks_dir)
    if runner is not None:
        oof_preds = runner.evaluate(samples)
    else:
        from buffalo_weight.resnet_baseline_adapter import ResNet18BaselineAdapter
        adapter = ResNet18BaselineAdapter(recipe=recipe)
        evaluator = ResNetBaselineEvaluator(adapter, recipe.inner_seed)
        oof_preds = evaluator.evaluate(samples)
    return [
        ComparisonPrediction(
            variation.name, variation.approach, "tuned", pred.file_name,
            pred.weight_category, pred.fold, pred.weight_kg, pred.prediction_kg,
        )
        for pred in oof_preds
    ]


def _compute_metrics(
    predictions: list[ComparisonPrediction], variations: tuple[TuningVariation, ...],
) -> list[ComparisonMetric]:
    metrics: list[ComparisonMetric] = []
    for variation in variations:
        var_preds = [p for p in predictions if p.configuration == variation.name]
        metrics.extend(comparison_metric_rows(var_preds))
    return metrics


def _load_feature_index(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    return {
        row["file_name"]: {
            k: float(v) for k, v in row.items() if k not in ("file_name", "farm", "weight_kg")
        }
        for row in rows
    }


def _load_csv_rows(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if fields != expected_columns:
        raise ValueError(f"columns were {fields}; expected {expected_columns}")
    return rows


def _build_training_partition(
    split_rows: list[dict[str, str]], feature_index: dict[str, dict[str, float]],
    frozen_features: tuple[str, ...],
) -> TrainingPartition:
    sample_ids = tuple(r["file_name"] for r in split_rows)
    targets = np.asarray([float(r["weight_kg"]) for r in split_rows], dtype=np.float64)
    strata = tuple(r["weight_category"] for r in split_rows)
    matrix = np.asarray([[feature_index[f_name][feat] for feat in frozen_features]
                         for f_name in sample_ids], dtype=np.float64)
    return TrainingPartition(matrix, targets, sample_ids, strata)


def _build_prediction_partition(
    split_rows: list[dict[str, str]], feature_index: dict[str, dict[str, float]],
    frozen_features: tuple[str, ...],
) -> PredictionPartition:
    sample_ids = tuple(r["file_name"] for r in split_rows)
    matrix = np.asarray([[feature_index[f_name][feat] for feat in frozen_features]
                         for f_name in sample_ids], dtype=np.float64)
    return PredictionPartition(matrix, sample_ids)


def _extract_dict_recipe(recipe: object) -> dict[str, object]:
    if not isinstance(recipe, dict):
        raise ValueError(f"recipe was {recipe!r}; expected a dictionary")
    return recipe


def _extract_dense_recipe(recipe: object) -> DenseTrainingRecipe:
    if not isinstance(recipe, DenseTrainingRecipe):
        raise ValueError(f"recipe was {recipe!r}; expected a DenseTrainingRecipe")
    return recipe


def _extract_compact_recipe(recipe: object) -> CompactCnnRecipe:
    if not isinstance(recipe, CompactCnnRecipe):
        raise ValueError(f"recipe was {recipe!r}; expected a CompactCnnRecipe")
    return recipe


def _extract_resnet_recipe(recipe: object) -> object:
    return recipe
