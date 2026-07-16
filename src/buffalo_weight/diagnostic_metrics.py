from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from buffalo_weight.train import format_metric


METRIC_FIELDS = ["group_type", "group", "n", "mae", "median_ae", "rmse", "bias", "p90_ae", "r2"]


def metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, str]:
    """Summarize regression errors; for example, ``metric_summary(actual, predicted)``."""
    errors = predicted - actual
    absolute = np.abs(errors)
    r2 = r2_score(actual, predicted) if len(actual) > 1 else math.nan
    return {
        "n": str(len(actual)),
        "mae": format_metric(mean_absolute_error(actual, predicted)),
        "median_ae": format_metric(float(np.median(absolute))),
        "rmse": format_metric(mean_squared_error(actual, predicted) ** 0.5),
        "bias": format_metric(float(errors.mean())),
        "p90_ae": format_metric(float(np.quantile(absolute, 0.9))),
        "r2": format_metric(r2),
    }


def grouped_metrics(
    actual: np.ndarray, predicted: np.ndarray, groups: list[str], group_type: str
) -> list[dict[str, str]]:
    """Calculate metrics by group; for example, ``grouped_metrics(y, p, farms, "farm")``."""
    group_array = np.asarray(groups)
    rows = []
    for group in sorted(set(groups)):
        selected = group_array == group
        rows.append({"group_type": group_type, "group": group, **metric_summary(actual[selected], predicted[selected])})
    return rows


def bootstrap_mae(actual: np.ndarray, predicted: np.ndarray, repeats: int = 10000) -> dict[str, str]:
    """Bootstrap MAE by animal; for example, ``bootstrap_mae(actual, predicted)``."""
    rng = np.random.default_rng(42)
    errors = np.abs(predicted - actual)
    samples = errors[rng.integers(0, len(errors), size=(repeats, len(errors)))].mean(axis=1)
    return {
        "mae": format_metric(float(errors.mean())),
        "ci_low": format_metric(float(np.quantile(samples, 0.025))),
        "ci_high": format_metric(float(np.quantile(samples, 0.975))),
    }


def paired_bootstrap(
    actual: np.ndarray, candidate: np.ndarray, reference: np.ndarray, repeats: int = 10000
) -> dict[str, str]:
    """Bootstrap paired MAE differences; for example, ``paired_bootstrap(y, new, old)``."""
    rng = np.random.default_rng(42)
    differences = np.abs(candidate - actual) - np.abs(reference - actual)
    samples = differences[rng.integers(0, len(differences), size=(repeats, len(differences)))].mean(axis=1)
    return {
        "mae_delta": format_metric(float(differences.mean())),
        "ci_low": format_metric(float(np.quantile(samples, 0.025))),
        "ci_high": format_metric(float(np.quantile(samples, 0.975))),
        "probability_candidate_better": format_metric(float((samples < 0).mean())),
        "candidate_win_fraction": format_metric(float((differences < 0).mean())),
    }


def oracle_metrics(actual: np.ndarray, prediction_matrix: np.ndarray) -> dict[str, str]:
    """Measure a per-animal model-selection oracle; for example, ``oracle_metrics(y, predictions)``."""
    minimum_errors = np.min(np.abs(prediction_matrix - actual[:, None]), axis=1)
    return {
        "oracle_mae": format_metric(float(minimum_errors.mean())),
        "oracle_median_ae": format_metric(float(np.median(minimum_errors))),
        "oracle_p90_ae": format_metric(float(np.quantile(minimum_errors, 0.9))),
    }
