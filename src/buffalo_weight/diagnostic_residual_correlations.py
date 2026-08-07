"""Pearson correlation measurement over signed residuals paired by file_name."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from buffalo_weight.diagnostic_stratified import DiagnosticPrediction


@dataclass(frozen=True)
class ResidualCorrelationRecord:
    """Pairwise Pearson correlation record between signed residual vectors of model pairs."""

    configuration_1: str
    configuration_2: str
    evaluation_role_1: str
    evaluation_role_2: str
    pearson_r: float


def compute_residual_correlations(
    predictions: list[DiagnosticPrediction],
) -> list[ResidualCorrelationRecord]:
    """Compute pairwise Pearson correlations of signed residuals paired by file_name.

    Example: ``compute_residual_correlations(preds)`` returns residual correlation records.
    """
    if not predictions:
        raise ValueError(
            f"predictions were {predictions!r}; expected non-empty list of diagnostic predictions"
        )
    config_roles = _extract_config_roles(predictions)
    residuals_by_config = _build_residual_vectors(predictions, config_roles)
    records: list[ResidualCorrelationRecord] = []
    configs = list(config_roles.keys())
    for i, c1 in enumerate(configs):
        for c2 in configs[i:]:
            r_val = _pearson_correlation(residuals_by_config[c1], residuals_by_config[c2])
            records.append(ResidualCorrelationRecord(
                configuration_1=c1,
                configuration_2=c2,
                evaluation_role_1=config_roles[c1],
                evaluation_role_2=config_roles[c2],
                pearson_r=r_val,
            ))
    return records


def _extract_config_roles(predictions: list[DiagnosticPrediction]) -> dict[str, str]:
    config_roles: dict[str, str] = {}
    for p in predictions:
        if p.configuration not in config_roles:
            config_roles[p.configuration] = p.evaluation_role
    return config_roles


def _build_residual_vectors(
    predictions: list[DiagnosticPrediction],
    config_roles: dict[str, str],
) -> dict[str, dict[str, float]]:
    residuals: dict[str, dict[str, float]] = {c: {} for c in config_roles}
    for p in predictions:
        residuals[p.configuration][p.file_name] = p.predicted_weight_kg - p.observed_weight_kg
    return residuals


def _pearson_correlation(
    vec1_dict: dict[str, float],
    vec2_dict: dict[str, float],
) -> float:
    common_files = sorted(set(vec1_dict.keys()) & set(vec2_dict.keys()))
    if len(common_files) < 2:
        raise ValueError(
            f"common sample count was {len(common_files)}; expected at least 2 paired samples"
        )
    v1 = np.asarray([vec1_dict[f] for f in common_files], dtype=np.float64)
    v2 = np.asarray([vec2_dict[f] for f in common_files], dtype=np.float64)
    std1, std2 = float(np.std(v1)), float(np.std(v2))
    if std1 == 0.0 or std2 == 0.0:
        return 1.0 if np.array_equal(v1, v2) else 0.0
    cov = float(np.mean((v1 - np.mean(v1)) * (v2 - np.mean(v2))))
    return float(cov / (std1 * std2))
