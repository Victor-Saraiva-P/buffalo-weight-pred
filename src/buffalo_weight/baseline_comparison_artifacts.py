"""Deterministic artifacts for controlled baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path

from buffalo_weight.baseline_comparison_metrics import comparison_metric_rows
from buffalo_weight.baseline_comparison_plots import save_baseline_comparison_figures
from buffalo_weight.baseline_comparison_report import approach_selection_report
from buffalo_weight.baseline_comparison_types import ComparisonMetric, ComparisonPrediction
from buffalo_weight.csv_io import (
    format_csv_number,
    format_optional_csv_number,
    write_csv_rows,
)
from buffalo_weight.hashing import sha256_file


METRIC_COLUMNS = [
    "configuration", "approach", "evaluation_role", "scope", "fold", "population",
    "n", "mae_kg", "rmse_kg", "bias_kg", "r2",
]
CANDIDATE_CONFIGURATIONS = [
    {"approach": "random_forest", "baseline_configuration": "random_forest_baseline"},
    {"approach": "dense_feature_network", "baseline_configuration": "dense"},
    {"approach": "compact_cnn", "baseline_configuration": "compact_cnn"},
    {"approach": "resnet18", "baseline_configuration": "resnet18_pretrained_partial"},
]


def write_baseline_comparison_artifacts(
    output_dir: Path, predictions: list[ComparisonPrediction],
) -> list[ComparisonMetric]:
    """Write every non-manifest output; for example, the stage publishes atomically."""
    metrics = _all_metric_rows(predictions)
    write_csv_rows([_metric_record(row) for row in metrics],
                   output_dir / "baseline_metrics.csv", METRIC_COLUMNS)
    report_path = output_dir / "approach_selection_report.md"
    report_path.write_text(approach_selection_report(metrics))
    _write_selected_approach(output_dir / "selected_approach.json", report_path)
    save_baseline_comparison_figures(output_dir, predictions, metrics)
    return metrics


def _all_metric_rows(predictions: list[ComparisonPrediction]) -> list[ComparisonMetric]:
    configurations = []
    for row in predictions:
        if row.configuration not in configurations:
            configurations.append(row.configuration)
    return [metric for configuration in configurations
            for metric in comparison_metric_rows(
                [row for row in predictions if row.configuration == configuration]
            )]


def _metric_record(metric: ComparisonMetric) -> dict[str, str]:
    return {
        "configuration": metric.configuration, "approach": metric.approach,
        "evaluation_role": metric.evaluation_role, "scope": metric.scope,
        "fold": "" if metric.fold is None else str(metric.fold),
        "population": metric.population, "n": str(metric.n),
        "mae_kg": format_csv_number(metric.mae_kg),
        "rmse_kg": format_optional_csv_number(metric.rmse_kg),
        "bias_kg": format_csv_number(metric.bias_kg),
        "r2": format_optional_csv_number(metric.r2),
    }


def _write_selected_approach(path: Path, report_path: Path) -> None:
    contract = {
        "schema_version": 1, "status": "provisional",
        "eligible_approaches": CANDIDATE_CONFIGURATIONS,
        "maximum_tuning_variations": 3, "source_report_sha256": sha256_file(report_path),
        "human_decision": None,
    }
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
