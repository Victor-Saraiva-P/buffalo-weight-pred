"""Identification of Shared Hard Cases and Divergent Cases Between Approaches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


DEFAULT_BASELINE_NAMES = (
    "random_forest_baseline",
    "dense",
    "compact_cnn",
    "resnet18_pretrained_partial",
)
DECILE_PERCENTILE = 90.0
SHARED_HARD_MINIMUM_BASELINES = 3


@dataclass(frozen=True)
class NotableCaseRecord:
    """Record representing a notable case (Shared Hard Case or Divergent Case)."""

    file_name: str
    case_type: str
    observed_weight_kg: float
    weight_category: str
    farm: str
    resolution: str
    metric_value: float
    predictions: dict[str, float]


def identify_notable_cases(
    predictions: list[DiagnosticPrediction],
    baseline_names: tuple[str, ...] | list[str] = DEFAULT_BASELINE_NAMES,
) -> tuple[list[NotableCaseRecord], list[NotableCaseRecord]]:
    """Identify Shared Hard Cases and Divergent Cases from OOF predictions.

    Example: ``identify_notable_cases(preds)`` returns (shared_hard_cases, divergent_cases).
    """
    _validate_predictions(predictions, tuple(baseline_names))
    sample_meta = _build_sample_metadata(predictions)
    pred_matrix = _build_prediction_matrix(predictions)
    worst_deciles = _identify_worst_deciles(pred_matrix, tuple(baseline_names))
    shared_hard = _extract_shared_hard_cases(sample_meta, pred_matrix, worst_deciles)
    divergent = _extract_divergent_cases(sample_meta, pred_matrix, tuple(baseline_names))
    return shared_hard, divergent


def _validate_predictions(
    predictions: list[DiagnosticPrediction],
    baselines: tuple[str, ...],
) -> None:
    if not predictions:
        raise ValueError(f"predictions were {predictions!r}; expected non-empty list of predictions")
    found_configs = {p.configuration for p in predictions}
    missing = set(baselines) - found_configs
    if missing:
        raise ValueError(
            f"missing baseline configurations {sorted(missing)}; expected 4 baseline configurations {list(baselines)}"
        )


def _build_sample_metadata(
    predictions: list[DiagnosticPrediction],
) -> dict[str, DiagnosticPrediction]:
    meta: dict[str, DiagnosticPrediction] = {}
    for p in predictions:
        if p.file_name not in meta:
            meta[p.file_name] = p
    return meta


def _build_prediction_matrix(
    predictions: list[DiagnosticPrediction],
) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for p in predictions:
        row = matrix.setdefault(p.file_name, {})
        row[p.configuration] = p.predicted_weight_kg
        row["_obs"] = p.observed_weight_kg
    return matrix


def _identify_worst_deciles(
    pred_matrix: dict[str, dict[str, float]],
    baselines: tuple[str, ...],
) -> dict[str, set[str]]:
    worst: dict[str, set[str]] = {b: set() for b in baselines}
    for b in baselines:
        errors = {f: abs(preds[b] - preds["_obs"]) for f, preds in pred_matrix.items()}
        threshold = float(np.percentile(list(errors.values()), DECILE_PERCENTILE))
        for f, err in errors.items():
            if err >= threshold:
                worst[b].add(f)
    return worst


def _extract_shared_hard_cases(
    sample_meta: dict[str, DiagnosticPrediction],
    pred_matrix: dict[str, dict[str, float]],
    worst_deciles: dict[str, set[str]],
) -> list[NotableCaseRecord]:
    shared: list[NotableCaseRecord] = []
    for f_name, meta in sorted(sample_meta.items()):
        match_count = sum(1 for b_set in worst_deciles.values() if f_name in b_set)
        if match_count >= SHARED_HARD_MINIMUM_BASELINES:
            preds = {k: v for k, v in pred_matrix[f_name].items() if k != "_obs"}
            shared.append(NotableCaseRecord(
                file_name=f_name,
                case_type="shared_hard_case",
                observed_weight_kg=meta.observed_weight_kg,
                weight_category=meta.weight_category,
                farm=meta.farm,
                resolution=meta.resolution,
                metric_value=float(match_count),
                predictions=preds,
            ))
    return shared


def _extract_divergent_cases(
    sample_meta: dict[str, DiagnosticPrediction],
    pred_matrix: dict[str, dict[str, float]],
    baselines: tuple[str, ...],
) -> list[NotableCaseRecord]:
    amplitudes: dict[str, float] = {}
    for f_name, preds in pred_matrix.items():
        b_preds = [preds[b] for b in baselines if b in preds]
        amplitudes[f_name] = float(np.max(b_preds) - np.min(b_preds))
    threshold = float(np.percentile(list(amplitudes.values()), DECILE_PERCENTILE))
    divergent: list[NotableCaseRecord] = []
    for f_name, meta in sorted(sample_meta.items()):
        amp = amplitudes[f_name]
        if amp >= threshold:
            preds = {k: v for k, v in pred_matrix[f_name].items() if k != "_obs"}
            divergent.append(NotableCaseRecord(
                file_name=f_name,
                case_type="divergent_case",
                observed_weight_kg=meta.observed_weight_kg,
                weight_category=meta.weight_category,
                farm=meta.farm,
                resolution=meta.resolution,
                metric_value=amp,
                predictions=preds,
            ))
    return divergent
