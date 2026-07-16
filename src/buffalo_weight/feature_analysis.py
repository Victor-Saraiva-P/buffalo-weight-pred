from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.feature_analysis_core import evaluate_feature_analysis, validate_feature_values
from buffalo_weight.feature_analysis_plots import save_diagnostic_plots, save_summary_plots
from buffalo_weight.feature_analysis_reports import (
    FEATURE_STAT_FIELDS,
    REDUNDANT_FIELDS,
    SUMMARY_BY_MODEL_FIELDS,
    correlation_matrix,
    feature_stats,
    redundant_pairs,
    summarize_by_model,
    summarize_features,
)
from buffalo_weight.models import FEATURE_FUSION_MODELS, MASK_PREDICTION_MODELS, ModelConfig, parse_model_configs
from buffalo_weight.split import parse_int, read_rows
from buffalo_weight.stability import split_random_states


RAW_METRIC_FIELDS = [
    "model_config", "model", "scenario", "feature", "split_random_state", "fold",
    "mae", "rmse", "r2", "n_train", "n_validation",
]


def write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    write_csv_rows(rows, path, fieldnames)


def classical_model_configs(models_config: dict[object, object]) -> list[ModelConfig]:
    configs = parse_model_configs(models_config)
    mask_configs = [config.name for config in configs if config.model in MASK_PREDICTION_MODELS]
    if mask_configs:
        raise ValueError(f"feature analysis only supports classical models; found: {', '.join(mask_configs)}")
    return [config for config in configs if config.model not in FEATURE_FUSION_MODELS]


def feature_columns(models_config: dict[object, object]) -> list[str]:
    raw_columns = models_config["feature_columns"]
    if isinstance(raw_columns, list):
        return [str(feature) for feature in raw_columns]
    raise ValueError("classical models config feature_columns must be a list")


def analyze_features(
    shared_config_path: Path, models_config_path: Path, start_seed: int, seed_count: int, output_dir: Path
) -> None:
    shared_config = load_config(shared_config_path)
    models_config = load_config(models_config_path)
    features_config = config_section(shared_config, "features")
    split_config = config_section(shared_config, "split")
    features = feature_columns(models_config)
    feature_rows = read_rows(Path(str(features_config["features_index_path"])))
    validate_feature_values(feature_rows, features)
    metric_rows = evaluate_feature_analysis(
        feature_rows, features, classical_model_configs(models_config), parse_int(split_config["k"], "split.k"),
        parse_int(split_config.get("weight_category_count", 4), "split.weight_category_count"),
        split_random_states(start_seed, seed_count),
    )
    write_outputs(feature_rows, features, metric_rows, output_dir)


def config_section(config: dict[object, object], name: str) -> dict[object, object]:
    section = config[name]
    if isinstance(section, dict):
        return section
    raise ValueError(f"shared config {name} section must be a map")


def write_outputs(
    feature_rows: list[dict[str, str]], features: list[str], metric_rows: list[dict[str, str]], output_dir: Path
) -> None:
    summary_by_model = summarize_by_model(metric_rows, features)
    summary, summary_fields = summarize_features(summary_by_model)
    write_csv(metric_rows, output_dir / "raw" / "fold_metrics.csv", RAW_METRIC_FIELDS)
    write_csv(summary_by_model, output_dir / "feature_summary_by_model.csv", SUMMARY_BY_MODEL_FIELDS)
    write_csv(summary, output_dir / "feature_summary.csv", summary_fields)
    write_diagnostics(feature_rows, features, output_dir)
    save_summary_plots(summary_by_model, output_dir)
    save_diagnostic_plots(feature_rows, features, output_dir)


def write_diagnostics(feature_rows: list[dict[str, str]], features: list[str], output_dir: Path) -> None:
    write_csv(feature_stats(feature_rows, features), output_dir / "diagnostics" / "feature_stats.csv", FEATURE_STAT_FIELDS)
    write_csv(correlation_matrix(feature_rows, features, "pearson"), output_dir / "diagnostics" / "pearson_correlation.csv", ["feature", *features])
    write_csv(correlation_matrix(feature_rows, features, "spearman"), output_dir / "diagnostics" / "spearman_correlation.csv", ["feature", *features])
    write_csv(redundant_pairs(feature_rows, features), output_dir / "diagnostics" / "redundant_pairs.csv", REDUNDANT_FIELDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-config", required=True)
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--output-dir", default="generated/feature_analysis")
    args = parser.parse_args(argv)
    try:
        analyze_features(Path(args.shared_config), Path(args.models_config), args.start_seed, args.seed_count, Path(args.output_dir))
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
