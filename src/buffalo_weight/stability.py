from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from buffalo_weight.config import load_config
from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.environment_contract import RuntimeProbe
from buffalo_weight.models import ModelConfig, parse_model_configs
from buffalo_weight.neural_environment import OFFICIAL_NEURAL_DEVICE
from buffalo_weight.neural_preflight import require_model_configs_cuda
from buffalo_weight.split import assign_folds, assign_weight_categories, parse_int, read_rows
from buffalo_weight.train import evaluate_models, format_metric


FOLD_METRIC_FIELDS = ["split_random_state", "fold", "mae", "rmse", "r2", "n_train", "n_validation"]
SEED_SUMMARY_FIELDS = [
    "split_random_state",
    "mae_mean",
    "mae_std",
    "mae_min",
    "mae_max",
    "rmse_mean",
    "r2_mean",
    "n_folds",
]
OVERALL_FIELDS = [
    "split_random_states",
    "mae_mean",
    "mae_std_between_seeds",
    "mae_min_seed",
    "mae_max_seed",
    "mae_range_between_seeds",
]
HARD_EXAMPLE_FIELDS = [
    "file_name",
    "weight",
    "weight_category",
    "weight_category_label",
    "validation_count",
    "abs_error_mean",
    "abs_error_max",
]
COMPARISON_FIELDS = ["model_config", "model", "mae_mean", "mae_std_between_seeds", "mae_min_seed", "mae_max_seed"]
ConfigLoader = Callable[[Path], dict[str, object]]
StabilityOutputs = tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]
]


@dataclass(frozen=True)
class _StabilityInputs:
    rows: list[dict[str, str]]
    feature_columns: list[str]
    k: int
    weight_category_count: int
    model_configs: list[ModelConfig]
    masks_dir: Path | None


def write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    write_csv_rows(rows, path, fieldnames)


def save_seed_mae_plot(seed_summaries: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_config in sorted({row["model_config"] for row in seed_summaries}):
        rows = sorted(
            [row for row in seed_summaries if row["model_config"] == model_config],
            key=lambda row: int(row["split_random_state"]),
        )
        seeds = [int(row["split_random_state"]) for row in rows]
        mae_means = [float(row["mae_mean"]) for row in rows]
        mae_mins = [float(row["mae_min"]) for row in rows]
        mae_maxs = [float(row["mae_max"]) for row in rows]
        ax.plot(seeds, mae_means, marker="o", linewidth=1.5, label=f"{model_config} MAE medio")
        ax.fill_between(seeds, mae_mins, mae_maxs, alpha=0.15)
    ax.set_xlabel("split.random_state")
    ax.set_ylabel("MAE (kg)")
    ax.set_title("Estabilidade do MAE medio entre seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_fold_mae_plot(fold_metrics: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    folds = sorted({int(row["fold"]) for row in fold_metrics})
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_config in sorted({row["model_config"] for row in fold_metrics}):
        for fold in folds:
            rows = sorted(
                [
                    row
                    for row in fold_metrics
                    if int(row["fold"]) == fold and row["model_config"] == model_config
                ],
                key=lambda row: int(row["split_random_state"]),
            )
            if not rows:
                continue
            ax.plot(
                [int(row["split_random_state"]) for row in rows],
                [float(row["mae"]) for row in rows],
                marker="o",
                linewidth=1,
                label=f"{model_config} Fold {fold}",
            )
    ax.set_xlabel("split.random_state")
    ax.set_ylabel("MAE (kg)")
    ax.set_title("MAE por fold em cada seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Fold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_hard_examples_plot(hard_examples: list[dict[str, str]], path: Path, limit: int = 20) -> None:
    import matplotlib.pyplot as plt

    rows = hard_examples[:limit]
    labels = [row["file_name"] for row in rows]
    values = [float(row["abs_error_mean"]) for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(rows) * 0.35)))
    ax.barh(labels, values, color="#d95f02")
    ax.invert_yaxis()
    ax.set_xlabel("Erro absoluto medio (kg)")
    ax.set_ylabel("Mascara")
    ax.set_title(f"Top {len(rows)} mascaras mais dificeis")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_model_comparison_plot(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: float(row["mae_mean"]))
    labels = [row["model_config"] for row in ordered]
    means = [float(row["mae_mean"]) for row in ordered]
    stds = [float(row["mae_std_between_seeds"]) for row in ordered]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(rows) * 0.45)))
    ax.barh(labels, means, xerr=stds, capsize=4)
    ax.invert_yaxis()
    ax.set_xlabel("MAE medio (kg)")
    ax.set_ylabel("Configuração de Modelo")
    ax.set_title("Comparação de estabilidade")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def without_identity(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{key: value for key, value in row.items() if key not in {"model_config", "model"}} for row in rows]


def metric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row[column] and not math.isnan(float(row[column]))]


def summarize_seed(seed: int, model_config: str, model: str, metrics: list[dict[str, str]]) -> dict[str, str]:
    maes = metric_values(metrics, "mae")
    rmses = metric_values(metrics, "rmse")
    r2s = metric_values(metrics, "r2")
    return {
        "split_random_state": str(seed),
        "model_config": model_config,
        "model": model,
        "mae_mean": format_metric(statistics.mean(maes)),
        "mae_std": format_metric(statistics.pstdev(maes) if len(maes) > 1 else 0.0),
        "mae_min": format_metric(min(maes)),
        "mae_max": format_metric(max(maes)),
        "rmse_mean": format_metric(statistics.mean(rmses)),
        "r2_mean": format_metric(statistics.mean(r2s)) if r2s else "",
        "n_folds": str(len(metrics)),
    }


def summarize_overall(seed_summaries: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries = []
    for model_config in sorted({row["model_config"] for row in seed_summaries}):
        rows = [row for row in seed_summaries if row["model_config"] == model_config]
        maes = metric_values(rows, "mae_mean")
        summaries.append(
            {
                "model_config": model_config,
                "model": rows[0]["model"],
                "split_random_states": str(len(rows)),
                "mae_mean": format_metric(statistics.mean(maes)),
                "mae_std_between_seeds": format_metric(statistics.pstdev(maes) if len(maes) > 1 else 0.0),
                "mae_min_seed": format_metric(min(maes)),
                "mae_max_seed": format_metric(max(maes)),
                "mae_range_between_seeds": format_metric(max(maes) - min(maes)),
            }
        )
    return summaries


def summarize_predictions(prediction_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in prediction_rows:
        grouped.setdefault(f"{row['model_config']}\0{row['file_name']}", []).append(row)

    summaries = []
    for _, rows in grouped.items():
        abs_errors = metric_values(rows, "abs_error")
        summaries.append(
            {
                "model_config": rows[0]["model_config"],
                "model": rows[0]["model"],
                "file_name": rows[0]["file_name"],
                "weight": rows[0]["weight"],
                "weight_category": rows[0]["weight_category"],
                "weight_category_label": rows[0]["weight_category_label"],
                "validation_count": str(len(rows)),
                "abs_error_mean": format_metric(statistics.mean(abs_errors)),
                "abs_error_max": format_metric(max(abs_errors)),
            }
        )
    return sorted(summaries, key=lambda row: float(row["abs_error_mean"]), reverse=True)


def evaluate_split_stability(
    rows: list[dict[str, str]], feature_columns: list[str], k: int,
    weight_category_count: int, split_random_states: list[int], model_configs: list[ModelConfig],
    masks_dir: Path | None = None, device: str = OFFICIAL_NEURAL_DEVICE,
    runtime_probe: RuntimeProbe | None = None,
) -> StabilityOutputs:
    """Evaluate split seeds; for example, neural configurations preflight CUDA first."""
    require_model_configs_cuda(model_configs, runtime_probe)
    return _evaluate_cuda_validated_stability(
        rows, feature_columns, k, weight_category_count, split_random_states,
        model_configs, masks_dir, device,
    )


def _evaluate_cuda_validated_stability(
    rows: list[dict[str, str]], feature_columns: list[str], k: int,
    weight_category_count: int, split_random_states: list[int], model_configs: list[ModelConfig],
    masks_dir: Path | None, device: str,
) -> StabilityOutputs:
    fold_metrics = []
    seed_summaries = []
    predictions = []
    for seed in split_random_states:
        metrics, seed_predictions = _evaluate_stability_seed(
            rows, feature_columns, k, weight_category_count, seed, model_configs, masks_dir, device
        )
        _record_seed_results(seed, metrics, seed_predictions, fold_metrics, seed_summaries, predictions)
    return fold_metrics, seed_summaries, summarize_overall(seed_summaries), summarize_predictions(predictions)


def _evaluate_stability_seed(
    rows: list[dict[str, str]], feature_columns: list[str], k: int, weight_category_count: int,
    seed: int, model_configs: list[ModelConfig], masks_dir: Path | None, device: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    seed_rows = [row.copy() for row in rows]
    assign_weight_categories(seed_rows, weight_category_count)
    assign_folds(seed_rows, k, seed)
    return evaluate_models(seed_rows, feature_columns, model_configs, masks_dir, device)


def _record_seed_results(
    seed: int, metrics: list[dict[str, str]], seed_predictions: list[dict[str, str]],
    fold_metrics: list[dict[str, str]], seed_summaries: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> None:
    fold_metrics.extend({"split_random_state": str(seed), **row} for row in metrics)
    predictions.extend({"split_random_state": str(seed), **row} for row in seed_predictions)
    for model_config in sorted({row["model_config"] for row in metrics}):
        model_rows = [row for row in metrics if row["model_config"] == model_config]
        seed_summaries.append(summarize_seed(seed, model_config, model_rows[0]["model"], model_rows))


def split_random_states(start_seed: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("--seed-count must be at least 1")
    return list(range(start_seed, start_seed + count))


def run_stability(
    config_path: Path, start_seed: int, seed_count: int, output_dir: Path,
    runtime_probe: RuntimeProbe | None = None, config_loader: ConfigLoader = load_config,
) -> None:
    """Run legacy-config stability; for example, ``run_stability(path, 0, 30, output)``."""
    inputs = _legacy_stability_inputs(config_loader(config_path), runtime_probe)
    _execute_stability(inputs, start_seed, seed_count, output_dir)


def run_stability_with_configs(
    shared_config_path: Path,
    models_config_path: Path,
    start_seed: int,
    seed_count: int,
    output_dir: Path,
    runtime_probe: RuntimeProbe | None = None,
    config_loader: ConfigLoader = load_config,
) -> None:
    """Run split stability; for example, pass shared and model config paths separately."""
    inputs = _shared_stability_inputs(
        config_loader(shared_config_path), config_loader(models_config_path), runtime_probe
    )
    _execute_stability(inputs, start_seed, seed_count, output_dir)


def _legacy_stability_inputs(
    config: dict[str, object], runtime_probe: RuntimeProbe | None
) -> _StabilityInputs:
    output = _config_map(config, "output", "config")
    split = _config_map(config, "split", "config")
    training = _config_map(config, "training", "config")
    model_configs = parse_model_configs(training)
    require_model_configs_cuda(model_configs, runtime_probe)
    rows = read_rows(Path(str(output["features_index_path"])))
    return _stability_inputs(rows, training, split, config.get("data"), model_configs)


def _shared_stability_inputs(
    shared: dict[str, object], models: dict[str, object], runtime_probe: RuntimeProbe | None
) -> _StabilityInputs:
    features = _config_map(shared, "features", "shared config")
    split = _config_map(shared, "split", "shared config")
    model_configs = parse_model_configs(models)
    require_model_configs_cuda(model_configs, runtime_probe)
    rows = read_rows(Path(str(features["features_index_path"])))
    return _stability_inputs(rows, models, split, shared.get("data"), model_configs)


def _stability_inputs(
    rows: list[dict[str, str]], models: dict[object, object], split: dict[object, object],
    shared_data_section: object, model_configs: list[ModelConfig],
) -> _StabilityInputs:
    columns = models.get("feature_columns")
    if not isinstance(columns, list):
        raise ValueError(f"config feature_columns was {columns!r}; expected a list")
    masks_dir = (
        Path(str(shared_data_section["masks_dir"]))
        if isinstance(shared_data_section, dict) and "masks_dir" in shared_data_section
        else None
    )
    return _StabilityInputs(
        rows, [str(column) for column in columns], parse_int(split["k"], "split.k"),
        parse_int(split.get("weight_category_count", 4), "split.weight_category_count"),
        model_configs, masks_dir,
    )


def _config_map(
    config: dict[str, object], key: str, source_name: str
) -> dict[object, object]:
    section = config.get(key)
    if isinstance(section, dict):
        return section
    raise ValueError(f"{source_name} {key} section was {section!r}; expected a map")


def _execute_stability(
    inputs: _StabilityInputs, start_seed: int, seed_count: int, output_dir: Path
) -> None:
    outputs = _evaluate_cuda_validated_stability(
        inputs.rows, inputs.feature_columns, inputs.k, inputs.weight_category_count,
        split_random_states(start_seed, seed_count), inputs.model_configs,
        inputs.masks_dir, OFFICIAL_NEURAL_DEVICE,
    )
    _write_stability_outputs(outputs, output_dir)


def _write_stability_outputs(outputs: StabilityOutputs, output_dir: Path) -> None:
    fold_metrics, seed_summaries, overall, hard_examples = outputs
    for model_config in sorted({row["model_config"] for row in overall}):
        _write_model_stability_outputs(
            model_config, fold_metrics, seed_summaries, overall, hard_examples, output_dir
        )
    comparison = _comparison_rows(overall)
    write_csv(comparison, output_dir / "model_comparison.csv", COMPARISON_FIELDS)
    save_model_comparison_plot(comparison, output_dir / "model_comparison.png")


def _write_model_stability_outputs(
    name: str, fold_metrics: list[dict[str, str]], seed_summaries: list[dict[str, str]],
    overall: list[dict[str, str]], hard_examples: list[dict[str, str]], output_dir: Path,
) -> None:
    config_dir = output_dir / name
    selected_folds = [row for row in fold_metrics if row["model_config"] == name]
    selected_seeds = [row for row in seed_summaries if row["model_config"] == name]
    selected_overall = [row for row in overall if row["model_config"] == name]
    selected_hard = [row for row in hard_examples if row["model_config"] == name]
    write_csv(without_identity(selected_folds), config_dir / "fold_metrics.csv", FOLD_METRIC_FIELDS)
    write_csv(without_identity(selected_seeds), config_dir / "seed_summary.csv", SEED_SUMMARY_FIELDS)
    write_csv(without_identity(selected_overall), config_dir / "overall.csv", OVERALL_FIELDS)
    write_csv(without_identity(selected_hard), config_dir / "hard_examples.csv", HARD_EXAMPLE_FIELDS)
    save_seed_mae_plot(selected_seeds, config_dir / "seed_mae.png")
    save_fold_mae_plot(selected_folds, config_dir / "fold_mae.png")
    save_hard_examples_plot(selected_hard, config_dir / "hard_examples.png")


def _comparison_rows(overall: list[dict[str, str]]) -> list[dict[str, str]]:
    fields = (
        "model_config", "model", "mae_mean", "mae_std_between_seeds",
        "mae_min_seed", "mae_max_seed",
    )
    return [{field: row[field] for field in fields} for row in overall]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--shared-config")
    parser.add_argument("--models-config")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--output-dir", default="generated/stability")
    return parser


def _run_stability_command(
    args: argparse.Namespace, runtime_probe: RuntimeProbe | None, config_loader: ConfigLoader
) -> None:
    if args.config:
        run_stability(
            Path(args.config), args.start_seed, args.seed_count, Path(args.output_dir),
            runtime_probe, config_loader,
        )
        return
    if args.shared_config and args.models_config:
        run_stability_with_configs(
            Path(args.shared_config), Path(args.models_config), args.start_seed,
            args.seed_count, Path(args.output_dir), runtime_probe, config_loader,
        )
        return
    received = (args.config, args.shared_config, args.models_config)
    raise ValueError(
        f"stability config arguments were {received!r}; "
        "expected --config or both --shared-config and --models-config"
    )


def main(
    argv: list[str] | None = None,
    runtime_probe: RuntimeProbe | None = None,
    config_loader: ConfigLoader = load_config,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run stability analysis; for example, neural configs require a CUDA probe."""
    try:
        _run_stability_command(_build_parser().parse_args(argv), runtime_probe, config_loader)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
