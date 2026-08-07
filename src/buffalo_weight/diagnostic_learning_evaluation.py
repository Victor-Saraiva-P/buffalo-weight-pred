"""Evaluation logic for controlled learning curves across four baselines.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.baseline_provenance import BaselineProvenance
from buffalo_weight.compact_cnn_evaluation import (
    CompactCnnSample,
    fit_compact_target_scale,
    load_compact_cnn_samples,
    load_letterboxed_mask,
)
from buffalo_weight.compact_cnn_provenance import CompactCnnProvenance
from buffalo_weight.compact_cnn_types import (
    COMPACT_CNN_RECIPE,
    CompactCnnTrainingAdapter,
    MaskBatch,
)
from buffalo_weight.diagnostic_learning_freshness import (
    check_baseline_100_reusability,
    load_reused_fold_metrics,
)
from buffalo_weight.diagnostic_learning_subsets import generate_nested_subsets
from buffalo_weight.diagnostic_learning_types import (
    LEARNING_CURVE_CONFIGURATIONS,
    LearningCurveSummaryRecord,
    LearningCurvesSlice,
    LearningPointRecord,
)
from buffalo_weight.feature_baselines import DenseFeatureBaseline, RandomForestBaseline
from buffalo_weight.feature_confirmation_manifest import validate_frozen_feature_contract
from buffalo_weight.feature_evaluation import (
    FeatureBaseline,
    FeatureSample,
    PredictionPartition,
    TrainingPartition,
)
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet_baseline_evaluation import (
    ResNetSample,
)
from buffalo_weight.resnet_baseline_provenance import ResNetBaselineProvenance
from buffalo_weight.resnet_baseline_stage import ResNetBaselineRunner


class DenseBaselineRunnerSeam(Protocol):
    """Evaluation seam protocol for DenseBaselineRunner."""

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...]
    ) -> object:
        ...


def evaluate_learning_curves(
    contract: ReportContract,
    random_forest_baseline: FeatureBaseline | None = None,
    dense_runner: DenseBaselineRunnerSeam | None = None,
    compact_adapter: CompactCnnTrainingAdapter | None = None,
    resnet_runner: ResNetBaselineRunner | None = None,
    baseline_provenance: BaselineProvenance | None = None,
    compact_provenance: CompactCnnProvenance | None = None,
    resnet_provenance: ResNetBaselineProvenance | None = None,
) -> LearningCurvesSlice:
    """Evaluate controlled learning curves across the four baselines.

    Example: ``evaluate_learning_curves(contract)`` returns a LearningCurvesSlice.
    """
    feature_names = validate_frozen_feature_contract(contract)
    feature_samples = load_feature_samples(contract.inputs_output_dir, feature_names)
    split_path = contract.inputs_output_dir / "canonical_split.csv"
    cnn_samples = load_compact_cnn_samples(split_path, contract.inputs.masks_dir)

    points = _evaluate_all_configurations(
        contract, feature_names, feature_samples, cnn_samples,
        random_forest_baseline, dense_runner, compact_adapter, resnet_runner,
        baseline_provenance, compact_provenance, resnet_provenance,
    )

    ordered_points = tuple(sorted(points, key=lambda p: (p.configuration, p.fraction, p.fold)))
    summaries = _build_summary_records(ordered_points)
    return LearningCurvesSlice(ordered_points, summaries)


def _evaluate_all_configurations(
    contract: ReportContract,
    feature_names: tuple[str, ...],
    feature_samples: list[FeatureSample],
    cnn_samples: list[CompactCnnSample],
    rf_baseline: FeatureBaseline | None,
    dense_runner: DenseBaselineRunnerSeam | None,
    compact_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
    baseline_prov: BaselineProvenance | None,
    compact_prov: CompactCnnProvenance | None,
    resnet_prov: ResNetBaselineProvenance | None,
) -> list[LearningPointRecord]:
    points: list[LearningPointRecord] = []
    for config in LEARNING_CURVE_CONFIGURATIONS:
        points.extend(
            _evaluate_config_folds(
                contract, config, feature_names, feature_samples, cnn_samples,
                rf_baseline, dense_runner, compact_adapter, resnet_runner,
                baseline_prov, compact_prov, resnet_prov,
            )
        )
    return points


def _evaluate_config_folds(
    contract: ReportContract,
    config: str,
    feature_names: tuple[str, ...],
    feature_samples: list[FeatureSample],
    cnn_samples: list[CompactCnnSample],
    rf_baseline: FeatureBaseline | None,
    dense_runner: DenseBaselineRunnerSeam | None,
    compact_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
    baseline_prov: BaselineProvenance | None,
    compact_prov: CompactCnnProvenance | None,
    resnet_prov: ResNetBaselineProvenance | None,
) -> list[LearningPointRecord]:
    points: list[LearningPointRecord] = []
    is_reusable = check_baseline_100_reusability(
        contract, config, baseline_prov, compact_prov, resnet_prov
    )
    for fold in range(1, 6):
        points.extend(
            _evaluate_fold_fractions(
                contract, config, fold, is_reusable, feature_names, feature_samples,
                cnn_samples, rf_baseline, dense_runner, compact_adapter, resnet_runner,
            )
        )
    return points


def _evaluate_fold_fractions(
    contract: ReportContract,
    config: str,
    fold: int,
    is_100_reusable: bool,
    feature_names: tuple[str, ...],
    feature_samples: list[FeatureSample],
    cnn_samples: list[CompactCnnSample],
    rf_baseline: FeatureBaseline | None,
    dense_runner: DenseBaselineRunnerSeam | None,
    compact_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
) -> list[LearningPointRecord]:
    points: list[LearningPointRecord] = []
    feat_subs = generate_nested_subsets(feature_samples, outer_fold=fold, seed=45)
    cnn_subs = generate_nested_subsets(cnn_samples, outer_fold=fold, seed=45)

    for fraction in (0.50, 0.75, 1.00):
        if fraction == 1.00 and is_100_reusable:
            points.append(_reused_point(contract, config, fold, len(feat_subs[1.00])))
        else:
            rec = _evaluate_single_point(
                config, fold, fraction, feature_names, feat_subs[fraction],
                feature_samples, cnn_subs[fraction], cnn_samples,
                rf_baseline, dense_runner, compact_adapter, resnet_runner,
            )
            points.append(rec)
    return points


def _reused_point(
    contract: ReportContract, config: str, fold: int, n_train: int,
) -> LearningPointRecord:
    mae, bias, n_eval = load_reused_fold_metrics(contract, config, fold)
    return LearningPointRecord(config, fold, 1.00, n_train, "oof", n_eval, mae, bias, "reused")


def _evaluate_single_point(
    config: str,
    fold: int,
    fraction: float,
    feature_names: tuple[str, ...],
    train_feature_sub: list[FeatureSample],
    all_feature_samples: list[FeatureSample],
    train_cnn_sub: list[CompactCnnSample],
    all_cnn_samples: list[CompactCnnSample],
    rf_baseline: FeatureBaseline | None,
    dense_runner: DenseBaselineRunnerSeam | None,
    compact_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
) -> LearningPointRecord:
    held_feature = [s for s in all_feature_samples if s.fold == fold]
    held_cnn = [s for s in all_cnn_samples if s.fold == fold]

    mae, bias = _dispatch_fit_predict(
        config, feature_names, train_feature_sub, held_feature,
        train_cnn_sub, held_cnn, rf_baseline, dense_runner, compact_adapter, resnet_runner,
    )

    return LearningPointRecord(
        config, fold, fraction, len(train_feature_sub), "oof", len(held_feature), mae, bias, "retrained",
    )


def _dispatch_fit_predict(
    config: str,
    feature_names: tuple[str, ...],
    train_feat: list[FeatureSample],
    held_feat: list[FeatureSample],
    train_cnn: list[CompactCnnSample],
    held_cnn: list[CompactCnnSample],
    rf_baseline: FeatureBaseline | None,
    dense_runner: DenseBaselineRunnerSeam | None,
    compact_adapter: CompactCnnTrainingAdapter | None,
    resnet_runner: ResNetBaselineRunner | None,
) -> tuple[float, float]:
    if config == "random_forest_baseline":
        return _fit_predict_rf(train_feat, held_feat, feature_names, rf_baseline)
    if config == "dense_baseline":
        return _fit_predict_dense(train_feat, held_feat, feature_names, dense_runner)
    if config == "compact_cnn_baseline":
        return _fit_predict_compact_cnn(train_cnn, held_cnn, compact_adapter)
    if config == "resnet18_pretrained_partial":
        return _fit_predict_resnet(train_cnn, held_cnn, resnet_runner)
    raise ValueError(
        f"unknown baseline configuration {config!r}; expected one of {LEARNING_CURVE_CONFIGURATIONS!r}"
    )


def _fit_predict_rf(
    train_samples: list[FeatureSample],
    held_out_samples: list[FeatureSample],
    feature_names: tuple[str, ...],
    model: FeatureBaseline | None,
) -> tuple[float, float]:
    model_impl = model or RandomForestBaseline()
    train_p = _to_training_partition(train_samples, feature_names)
    predictor = model_impl.fit(train_p, feature_names)
    held_p = PredictionPartition(_feature_matrix(held_out_samples, feature_names), tuple(s.file_name for s in held_out_samples))
    preds = predictor.predict(held_p)
    obs = np.asarray([s.weight_kg for s in held_out_samples], dtype=np.float64)
    return _calculate_mae_and_bias(preds, obs)


def _fit_predict_dense(
    train_samples: list[FeatureSample],
    held_out_samples: list[FeatureSample],
    feature_names: tuple[str, ...],
    runner: DenseBaselineRunnerSeam | None,
) -> tuple[float, float]:
    if runner is not None:
        eval_res = runner.evaluate([*train_samples, *held_out_samples], feature_names)
        held_ids = {s.file_name for s in held_out_samples}
        predictions = getattr(eval_res, "predictions", ())
        preds_list = [p for p in predictions if getattr(p, "file_name", None) in held_ids]
        preds = np.asarray([getattr(p, "predicted_weight_kg") for p in preds_list], dtype=np.float64)
        obs = np.asarray([getattr(p, "observed_weight_kg") for p in preds_list], dtype=np.float64)
        return _calculate_mae_and_bias(preds, obs)

    dense_model = DenseFeatureBaseline()
    train_p = _to_training_partition(train_samples, feature_names)
    predictor = dense_model.fit(train_p, feature_names)
    held_p = PredictionPartition(_feature_matrix(held_out_samples, feature_names), tuple(s.file_name for s in held_out_samples))
    preds = predictor.predict(held_p)
    obs = np.asarray([s.weight_kg for s in held_out_samples], dtype=np.float64)
    return _calculate_mae_and_bias(preds, obs)


def _fit_predict_compact_cnn(
    train_samples: list[CompactCnnSample],
    held_out_samples: list[CompactCnnSample],
    adapter: CompactCnnTrainingAdapter | None,
) -> tuple[float, float]:
    if adapter is None:
        raise ValueError(
            f"compact_cnn_adapter was unavailable (received {adapter!r}); GPU training requires adapter"
        )

    recipe = COMPACT_CNN_RECIPE
    sel, stop = _inner_split(train_samples, recipe.inner_seed)
    sel_b, stop_b = _to_mask_batch(sel), _to_mask_batch(stop)
    scale = fit_compact_target_scale(sel_b.targets_kg)
    epochs = adapter.select_epoch_count(sel_b, stop_b, scale, recipe)

    refit_b = _to_mask_batch(train_samples)
    predictor = adapter.fit_epochs(refit_b, fit_compact_target_scale(refit_b.targets_kg), epochs, recipe)

    held_b = _to_mask_batch(held_out_samples)
    preds = predictor.predict_kg(held_b)
    obs = np.asarray([s.weight_kg for s in held_out_samples], dtype=np.float64)
    return _calculate_mae_and_bias(preds, obs)


def _fit_predict_resnet(
    train_samples: list[CompactCnnSample],
    held_out_samples: list[CompactCnnSample],
    runner: ResNetBaselineRunner | None,
) -> tuple[float, float]:
    if runner is None:
        raise ValueError(
            f"resnet_runner was unavailable (received {runner!r}); GPU training requires runner"
        )

    resnet_train = tuple(ResNetSample(s.file_name, s.mask_path, s.weight_category, s.fold, s.weight_kg) for s in train_samples)
    resnet_held = tuple(ResNetSample(s.file_name, s.mask_path, s.weight_category, s.fold, s.weight_kg) for s in held_out_samples)
    predictions = runner.evaluate(resnet_train + resnet_held)
    held_ids = {s.file_name for s in held_out_samples}
    held_preds = [p for p in predictions if p.file_name in held_ids]
    preds = np.asarray([p.prediction_kg for p in held_preds], dtype=np.float64)
    obs = np.asarray([p.weight_kg for p in held_preds], dtype=np.float64)
    return _calculate_mae_and_bias(preds, obs)


def _calculate_mae_and_bias(predicted: NDArray[np.float64], observed: NDArray[np.float64]) -> tuple[float, float]:
    diff = predicted - observed
    return float(np.mean(np.abs(diff))), float(np.mean(diff))


def _to_training_partition(samples: list[FeatureSample], feature_names: tuple[str, ...]) -> TrainingPartition:
    return TrainingPartition(
        _feature_matrix(samples, feature_names),
        np.asarray([s.weight_kg for s in samples], dtype=np.float64),
        tuple(s.file_name for s in samples),
        tuple(s.weight_category for s in samples),
    )


def _feature_matrix(samples: list[FeatureSample], feature_names: tuple[str, ...]) -> NDArray[np.float64]:
    values = [[s.feature_values[name] for name in feature_names] for s in samples]
    return np.asarray(values, dtype=np.float64)


def _inner_split(samples: list[CompactCnnSample], seed: int) -> tuple[list[CompactCnnSample], list[CompactCnnSample]]:
    from sklearn.model_selection import StratifiedShuffleSplit
    strata = [s.weight_category for s in samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    sel_idx, stop_idx = next(splitter.split(np.zeros(len(samples)), strata))
    return [samples[int(i)] for i in sel_idx], [samples[int(i)] for i in stop_idx]


def _to_mask_batch(samples: list[CompactCnnSample]) -> MaskBatch:
    pixels = np.stack([load_letterboxed_mask(s.mask_path) for s in samples])
    targets = np.asarray([s.weight_kg for s in samples], dtype=np.float64)
    return MaskBatch(pixels, targets, tuple(s.file_name for s in samples), tuple(s.weight_category for s in samples))


def _build_summary_records(points: tuple[LearningPointRecord, ...]) -> tuple[LearningCurveSummaryRecord, ...]:
    summaries: list[LearningCurveSummaryRecord] = []
    configs = sorted({p.configuration for p in points})

    for config in configs:
        for frac in (0.50, 0.75, 1.00):
            group = [p for p in points if p.configuration == config and abs(p.fraction - frac) < 1e-4]
            if group:
                summaries.append(_summarize_point_group(config, frac, group))
    return tuple(summaries)


def _summarize_point_group(config: str, frac: float, group: list[LearningPointRecord]) -> LearningCurveSummaryRecord:
    maes = np.asarray([p.mae_kg for p in group], dtype=np.float64)
    biases = np.asarray([p.bias_kg for p in group], dtype=np.float64)
    ntrains = np.asarray([p.n_train for p in group], dtype=np.float64)
    reused_cnt = sum(1 for p in group if p.artifact_action == "reused")

    return LearningCurveSummaryRecord(
        configuration=config,
        fraction=frac,
        mean_n_train=float(np.mean(ntrains)),
        mean_mae_kg=float(np.mean(maes)),
        std_mae_kg=float(np.std(maes)),
        mean_bias_kg=float(np.mean(biases)),
        reused_points_count=reused_cnt,
    )
