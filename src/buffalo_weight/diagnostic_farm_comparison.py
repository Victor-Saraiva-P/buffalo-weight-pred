"""Farm performance comparison across full sample and shared weight range."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


SHARED_WEIGHT_RANGE_MIN_KG = 92.0
SHARED_WEIGHT_RANGE_MAX_KG = 265.0
REMAINING_CONFOUNDING_NOTE = (
    "Fazenda, faixa de peso extremo e aquisição permanecem confundidas na amostra atual."
)


@dataclass(frozen=True)
class FarmComparisonRecord:
    """Farm evaluation record under full sample or shared weight range."""

    configuration: str
    evaluation_role: str
    sample_scope: str
    farm: str
    sample_count: int
    mae_kg: float
    median_abs_error_kg: float
    bias_kg: float
    confounding_note: str


def compare_farms_under_approved_subsets(
    predictions: list[DiagnosticPrediction],
) -> list[FarmComparisonRecord]:
    """Compare farms in full sample and shared 92–265 kg weight range.

    Example: ``compare_farms_under_approved_subsets(preds)`` returns comparison records.
    """
    if not predictions:
        raise ValueError(
            f"predictions were {predictions!r}; expected non-empty list of diagnostic predictions"
        )
    configs = sorted({(p.configuration, p.evaluation_role) for p in predictions})
    records: list[FarmComparisonRecord] = []
    for config_name, role in configs:
        sub_preds = [p for p in predictions if p.configuration == config_name]
        records.extend(_evaluate_scope(sub_preds, config_name, role, "full_sample", sub_preds))
        shared = [
            p for p in sub_preds
            if SHARED_WEIGHT_RANGE_MIN_KG <= p.observed_weight_kg <= SHARED_WEIGHT_RANGE_MAX_KG
        ]
        records.extend(_evaluate_scope(sub_preds, config_name, role, "shared_range_92_265", shared))
    return records


def _evaluate_scope(
    all_preds: list[DiagnosticPrediction],
    configuration: str,
    role: str,
    sample_scope: str,
    scope_preds: list[DiagnosticPrediction],
) -> list[FarmComparisonRecord]:
    farms = sorted({p.farm for p in all_preds})
    results: list[FarmComparisonRecord] = []
    for farm_name in farms:
        farm_preds = [p for p in scope_preds if p.farm == farm_name]
        if farm_preds:
            results.append(_summarize_farm_group(farm_preds, configuration, role, sample_scope, farm_name))
    return results


def _summarize_farm_group(
    predictions: list[DiagnosticPrediction],
    configuration: str,
    role: str,
    sample_scope: str,
    farm: str,
) -> FarmComparisonRecord:
    observed = np.asarray([p.observed_weight_kg for p in predictions], dtype=np.float64)
    predicted = np.asarray([p.predicted_weight_kg for p in predictions], dtype=np.float64)
    residuals = predicted - observed
    abs_errors = np.abs(residuals)
    return FarmComparisonRecord(
        configuration=configuration,
        evaluation_role=role,
        sample_scope=sample_scope,
        farm=farm,
        sample_count=len(predictions),
        mae_kg=float(np.mean(abs_errors)),
        median_abs_error_kg=float(np.median(abs_errors)),
        bias_kg=float(np.mean(residuals)),
        confounding_note=REMAINING_CONFOUNDING_NOTE,
    )
