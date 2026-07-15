from __future__ import annotations

import statistics

import numpy as np

from buffalo_weight.feature_analysis_core import feature_matrix
from buffalo_weight.train import format_metric


FEATURE_STAT_FIELDS = ["feature", "missing_count", "unique_count", "mean", "std", "min", "max", "negative_count"]
REDUNDANT_FIELDS = ["feature_a", "feature_b", "spearman", "pearson"]
SUMMARY_BY_MODEL_FIELDS = [
    "model_config", "model", "feature", "all_features_mae_mean", "all_features_mae_std",
    "single_feature_mae_mean", "single_feature_mae_std", "without_feature_mae_mean",
    "without_feature_mae_std", "permuted_feature_mae_mean", "permuted_feature_mae_std",
    "removal_impact_mae", "permutation_impact_mae", "mean_baseline_mae_mean", "area_baseline_mae_mean",
]


def feature_stats(rows: list[dict[str, str]], features: list[str]) -> list[dict[str, str]]:
    stats = []
    for feature in features:
        values = feature_matrix(rows, [feature]).ravel()
        stats.append(feature_stat_row(feature, values))
    return stats


def feature_stat_row(feature: str, values: np.ndarray) -> dict[str, str]:
    return {
        "feature": feature, "missing_count": "0", "unique_count": str(len(set(float(value) for value in values))),
        "mean": format_metric(float(np.mean(values))), "std": format_metric(float(np.std(values))),
        "min": format_metric(float(np.min(values))), "max": format_metric(float(np.max(values))),
        "negative_count": str(int(np.sum(values < 0))),
    }


def scenario_maes(rows: list[dict[str, str]], model_config: str, scenario: str, feature: str) -> list[float]:
    return [float(row["mae"]) for row in rows if row["model_config"] == model_config and row["scenario"] == scenario and row["feature"] == feature]


def mean_or_empty(values: list[float]) -> str:
    return format_metric(statistics.mean(values)) if values else ""


def std_or_empty(values: list[float]) -> str:
    return format_metric(statistics.pstdev(values)) if values else ""


def impact(candidate_value: str, base_value: str) -> str:
    return format_metric(float(candidate_value) - float(base_value)) if candidate_value and base_value else ""


def summarize_by_model(rows: list[dict[str, str]], features: list[str]) -> list[dict[str, str]]:
    summaries = []
    for model_config in sorted({row["model_config"] for row in rows}):
        model = next(row["model"] for row in rows if row["model_config"] == model_config)
        base = scenario_maes(rows, model_config, "all_features", "")
        mean = scenario_maes(rows, model_config, "mean_baseline", "")
        area = scenario_maes(rows, model_config, "area_baseline", "area")
        summaries.extend(summary_row(rows, model_config, model, feature, base, mean, area) for feature in features)
    return summaries


def summary_row(
    rows: list[dict[str, str]], model_config: str, model: str, feature: str,
    base: list[float], mean: list[float], area: list[float],
) -> dict[str, str]:
    single = scenario_maes(rows, model_config, "single_feature", feature)
    without = scenario_maes(rows, model_config, "without_feature", feature)
    permuted = scenario_maes(rows, model_config, "permuted_feature", feature)
    base_mean = mean_or_empty(base)
    without_mean = mean_or_empty(without)
    permuted_mean = mean_or_empty(permuted)
    return base_summary_fields(model_config, model, feature, base, single, without, permuted) | {
        "removal_impact_mae": impact(without_mean, base_mean), "permutation_impact_mae": impact(permuted_mean, base_mean),
        "mean_baseline_mae_mean": mean_or_empty(mean), "area_baseline_mae_mean": mean_or_empty(area),
    }


def base_summary_fields(
    model_config: str, model: str, feature: str, base: list[float], single: list[float],
    without: list[float], permuted: list[float],
) -> dict[str, str]:
    return {
        "model_config": model_config, "model": model, "feature": feature,
        "all_features_mae_mean": mean_or_empty(base), "all_features_mae_std": std_or_empty(base),
        "single_feature_mae_mean": mean_or_empty(single), "single_feature_mae_std": std_or_empty(single),
        "without_feature_mae_mean": mean_or_empty(without), "without_feature_mae_std": std_or_empty(without),
        "permuted_feature_mae_mean": mean_or_empty(permuted), "permuted_feature_mae_std": std_or_empty(permuted),
    }


def summarize_features(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    models = sorted({row["model_config"] for row in rows})
    fields = ["feature", *[field for model in models for field in summary_model_fields(model)]]
    summaries = [combined_feature_row(rows, models, feature) for feature in sorted({row["feature"] for row in rows})]
    return sorted(summaries, key=summary_sort_value), fields


def summary_model_fields(model: str) -> list[str]:
    return [f"{model}_removal_impact_mae", f"{model}_permutation_impact_mae", f"{model}_single_feature_mae_mean"]


def combined_feature_row(rows: list[dict[str, str]], models: list[str], feature: str) -> dict[str, str]:
    output = {"feature": feature}
    for model in models:
        matching = [row for row in rows if row["model_config"] == model and row["feature"] == feature]
        if matching:
            output |= combined_model_fields(model, matching[0])
    return output


def combined_model_fields(model: str, row: dict[str, str]) -> dict[str, str]:
    return {
        f"{model}_removal_impact_mae": row["removal_impact_mae"],
        f"{model}_permutation_impact_mae": row["permutation_impact_mae"],
        f"{model}_single_feature_mae_mean": row["single_feature_mae_mean"],
    }


def summary_sort_value(row: dict[str, str]) -> float:
    keys = [key for key in row if key.startswith("random_forest_baseline") and key.endswith("removal_impact_mae")]
    keys = keys or [key for key in row if key.endswith("removal_impact_mae") and row.get(key)]
    return -float(row[keys[0]]) if keys and row.get(keys[0]) else 0.0


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def correlation_matrix(rows: list[dict[str, str]], features: list[str], method: str) -> list[dict[str, str]]:
    matrix = feature_matrix(rows, features)
    if method == "spearman":
        matrix = np.asarray([rank_values(matrix[:, index]) for index in range(matrix.shape[1])]).T
    corr = np.corrcoef(matrix, rowvar=False)
    return [{"feature": feature, **{features[index]: format_metric(float(value)) for index, value in enumerate(corr[row_index])}} for row_index, feature in enumerate(features)]


def redundant_pairs(rows: list[dict[str, str]], features: list[str], threshold: float = 0.95) -> list[dict[str, str]]:
    pearson = {row["feature"]: row for row in correlation_matrix(rows, features, "pearson")}
    spearman = {row["feature"]: row for row in correlation_matrix(rows, features, "spearman")}
    return [redundant_pair(left, right, spearman, pearson) for left_index, left in enumerate(features) for right in features[left_index + 1:] if abs(float(spearman[left][right])) >= threshold]


def redundant_pair(left: str, right: str, spearman: dict[str, dict[str, str]], pearson: dict[str, dict[str, str]]) -> dict[str, str]:
    return {"feature_a": left, "feature_b": right, "spearman": spearman[left][right], "pearson": pearson[left][right]}
