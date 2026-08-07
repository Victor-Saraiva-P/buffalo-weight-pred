"""Stratified error metrics across weight categories, farms, and resolutions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


RESOLUTION_MINIMUM_SAMPLE_COUNT = 10


@dataclass(frozen=True)
class DiagnosticPrediction:
    """Prediction sample decorated with domain strata for diagnostic analysis."""

    configuration: str
    evaluation_role: str
    file_name: str
    weight_category: str
    farm: str
    resolution: str
    observed_weight_kg: float
    predicted_weight_kg: float


@dataclass(frozen=True)
class StratifiedMetricRecord:
    """Stratified error summary record for a specific stratum value and configuration."""

    configuration: str
    evaluation_role: str
    stratum_type: str
    stratum_value: str
    sample_count: int
    mae_kg: float
    median_abs_error_kg: float
    bias_kg: float


def compute_stratified_metrics(
    predictions: list[DiagnosticPrediction],
) -> list[StratifiedMetricRecord]:
    """Compute n, MAE, median absolute error, and bias across weight categories, farms, and resolutions.

    Example: ``compute_stratified_metrics(preds)`` returns stratified metric records.
    """
    if not predictions:
        raise ValueError(
            f"predictions were {predictions!r}; expected non-empty list of diagnostic predictions"
        )
    _validate_finite_predictions(predictions)
    configs = sorted({(p.configuration, p.evaluation_role) for p in predictions})
    records: list[StratifiedMetricRecord] = []
    for config_name, role in configs:
        sub_preds = [p for p in predictions if p.configuration == config_name]
        records.extend(_stratify_by_attribute(sub_preds, config_name, role, "weight_category", lambda p: p.weight_category))
        records.extend(_stratify_by_attribute(sub_preds, config_name, role, "farm", lambda p: p.farm))
        records.extend(_stratify_resolution(sub_preds, config_name, role))
    return records


def _validate_finite_predictions(predictions: list[DiagnosticPrediction]) -> None:
    invalid = [
        (p.file_name, p.observed_weight_kg, p.predicted_weight_kg)
        for p in predictions
        if not math.isfinite(p.observed_weight_kg) or not math.isfinite(p.predicted_weight_kg)
    ]
    if invalid:
        raise ValueError(f"predictions contained non-finite values {invalid!r}; expected finite float weights")


from collections.abc import Callable


def _stratify_by_attribute(
    predictions: list[DiagnosticPrediction],
    configuration: str,
    role: str,
    stratum_type: str,
    key_func: Callable[[DiagnosticPrediction], str],
) -> list[StratifiedMetricRecord]:
    groups: dict[str, list[DiagnosticPrediction]] = {}
    for p in predictions:
        val = key_func(p)
        groups.setdefault(val, []).append(p)
    results: list[StratifiedMetricRecord] = []
    for val in sorted(groups):
        results.append(_summarize_group(groups[val], configuration, role, stratum_type, val))
    return results


def _stratify_resolution(
    predictions: list[DiagnosticPrediction],
    configuration: str,
    role: str,
) -> list[StratifiedMetricRecord]:
    groups: dict[str, list[DiagnosticPrediction]] = {}
    for p in predictions:
        groups.setdefault(p.resolution, []).append(p)
    results: list[StratifiedMetricRecord] = []
    for res_val in sorted(groups):
        group_preds = groups[res_val]
        if len(group_preds) >= RESOLUTION_MINIMUM_SAMPLE_COUNT:
            results.append(_summarize_group(group_preds, configuration, role, "resolution", res_val))
    return results


def _summarize_group(
    predictions: list[DiagnosticPrediction],
    configuration: str,
    role: str,
    stratum_type: str,
    stratum_value: str,
) -> StratifiedMetricRecord:
    observed = np.asarray([p.observed_weight_kg for p in predictions], dtype=np.float64)
    predicted = np.asarray([p.predicted_weight_kg for p in predictions], dtype=np.float64)
    residuals = predicted - observed
    abs_errors = np.abs(residuals)
    return StratifiedMetricRecord(
        configuration=configuration,
        evaluation_role=role,
        stratum_type=stratum_type,
        stratum_value=stratum_value,
        sample_count=len(predictions),
        mae_kg=float(np.mean(abs_errors)),
        median_abs_error_kg=float(np.median(abs_errors)),
        bias_kg=float(np.mean(residuals)),
    )
