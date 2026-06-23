from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.split import assign_folds, assign_weight_categories, parse_int, read_rows
from buffalo_weight.train import evaluate_random_forest, format_metric


def write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def save_seed_mae_plot(seed_summaries: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(seed_summaries, key=lambda row: int(row["split_random_state"]))
    seeds = [int(row["split_random_state"]) for row in rows]
    mae_means = [float(row["mae_mean"]) for row in rows]
    mae_mins = [float(row["mae_min"]) for row in rows]
    mae_maxs = [float(row["mae_max"]) for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seeds, mae_means, marker="o", linewidth=1.5, label="MAE medio")
    ax.fill_between(seeds, mae_mins, mae_maxs, alpha=0.2, label="Min-max entre folds")
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
    for fold in folds:
        rows = sorted(
            [row for row in fold_metrics if int(row["fold"]) == fold],
            key=lambda row: int(row["split_random_state"]),
        )
        ax.plot(
            [int(row["split_random_state"]) for row in rows],
            [float(row["mae"]) for row in rows],
            marker="o",
            linewidth=1,
            label=f"Fold {fold}",
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


def metric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows if row[column] and not math.isnan(float(row[column]))]


def summarize_seed(seed: int, metrics: list[dict[str, str]]) -> dict[str, str]:
    maes = metric_values(metrics, "mae")
    rmses = metric_values(metrics, "rmse")
    r2s = metric_values(metrics, "r2")
    return {
        "split_random_state": str(seed),
        "mae_mean": format_metric(statistics.mean(maes)),
        "mae_std": format_metric(statistics.pstdev(maes) if len(maes) > 1 else 0.0),
        "mae_min": format_metric(min(maes)),
        "mae_max": format_metric(max(maes)),
        "rmse_mean": format_metric(statistics.mean(rmses)),
        "r2_mean": format_metric(statistics.mean(r2s)) if r2s else "",
        "n_folds": str(len(metrics)),
    }


def summarize_overall(seed_summaries: list[dict[str, str]]) -> list[dict[str, str]]:
    maes = metric_values(seed_summaries, "mae_mean")
    return [
        {
            "split_random_states": str(len(seed_summaries)),
            "mae_mean": format_metric(statistics.mean(maes)),
            "mae_std_between_seeds": format_metric(statistics.pstdev(maes) if len(maes) > 1 else 0.0),
            "mae_min_seed": format_metric(min(maes)),
            "mae_max_seed": format_metric(max(maes)),
            "mae_range_between_seeds": format_metric(max(maes) - min(maes)),
        }
    ]


def summarize_predictions(prediction_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in prediction_rows:
        grouped.setdefault(row["file_name"], []).append(row)

    summaries = []
    for file_name, rows in grouped.items():
        abs_errors = metric_values(rows, "abs_error")
        summaries.append(
            {
                "file_name": file_name,
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
    rows: list[dict[str, str]],
    feature_columns: list[str],
    k: int,
    split_random_states: list[int],
    n_estimators: int,
    training_random_state: int,
) -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]
]:
    fold_metrics = []
    seed_summaries = []
    predictions = []

    for seed in split_random_states:
        seed_rows = [row.copy() for row in rows]
        assign_weight_categories(seed_rows)
        assign_folds(seed_rows, k, seed)
        metrics, seed_predictions = evaluate_random_forest(
            seed_rows, feature_columns, n_estimators, training_random_state
        )
        for row in metrics:
            fold_metrics.append({"split_random_state": str(seed), **row})
        for row in seed_predictions:
            predictions.append({"split_random_state": str(seed), **row})
        seed_summaries.append(summarize_seed(seed, metrics))

    return fold_metrics, seed_summaries, summarize_overall(seed_summaries), summarize_predictions(predictions)


def split_random_states(start_seed: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("--seed-count must be at least 1")
    return list(range(start_seed, start_seed + count))


def run_stability(config_path: Path, start_seed: int, seed_count: int, output_dir: Path) -> None:
    config = load_config(config_path)
    output = config["output"]
    split = config["split"]
    training = config["training"]
    if not isinstance(output, dict):
        raise ValueError("config output section must be a map")
    if not isinstance(split, dict):
        raise ValueError("config split section must be a map")
    if not isinstance(training, dict):
        raise ValueError("config training section must be a map")

    feature_columns = training["feature_columns"]
    if not isinstance(feature_columns, list):
        raise ValueError("config training.feature_columns must be a list")

    feature_rows = read_rows(Path(str(output["features_index_path"])))
    fold_metrics, seed_summaries, overall, hard_examples = evaluate_split_stability(
        feature_rows,
        [str(column) for column in feature_columns],
        parse_int(split["k"], "split.k"),
        split_random_states(start_seed, seed_count),
        parse_int(training["n_estimators"], "training.n_estimators"),
        parse_int(training["random_state"], "training.random_state"),
    )

    write_csv(
        fold_metrics,
        output_dir / "split_stability_fold_metrics.csv",
        [
            "split_random_state",
            "model",
            "fold",
            "mae",
            "rmse",
            "r2",
            "n_train",
            "n_validation",
        ],
    )
    write_csv(
        seed_summaries,
        output_dir / "split_stability_seed_summary.csv",
        [
            "split_random_state",
            "mae_mean",
            "mae_std",
            "mae_min",
            "mae_max",
            "rmse_mean",
            "r2_mean",
            "n_folds",
        ],
    )
    write_csv(
        overall,
        output_dir / "split_stability_overall.csv",
        [
            "split_random_states",
            "mae_mean",
            "mae_std_between_seeds",
            "mae_min_seed",
            "mae_max_seed",
            "mae_range_between_seeds",
        ],
    )
    write_csv(
        hard_examples,
        output_dir / "split_stability_hard_examples.csv",
        [
            "file_name",
            "weight",
            "weight_category",
            "weight_category_label",
            "validation_count",
            "abs_error_mean",
            "abs_error_max",
        ],
    )
    save_seed_mae_plot(seed_summaries, output_dir / "split_stability_seed_mae.png")
    save_fold_mae_plot(fold_metrics, output_dir / "split_stability_fold_mae.png")
    save_hard_examples_plot(hard_examples, output_dir / "split_stability_hard_examples.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--output-dir", default="generated/diagnostics")
    args = parser.parse_args(argv)

    try:
        run_stability(
            Path(args.config), args.start_seed, args.seed_count, Path(args.output_dir)
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
