from __future__ import annotations

import argparse
import csv
import sys
import statistics
from pathlib import Path

from buffalo_weight.config import load_config
from buffalo_weight.split import assign_folds, assign_weight_categories, parse_int, parse_weight, read_rows
from buffalo_weight.stability import evaluate_split_stability, split_random_states
from buffalo_weight.train import parse_model_names


def write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def save_mae_plot(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted({row["model"] for row in rows}):
        ordered = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: int(row["weight_category_count"]),
        )
        counts = [int(row["weight_category_count"]) for row in ordered]
        means = [float(row["mae_mean"]) for row in ordered]
        stds = [float(row["mae_std_between_seeds"]) for row in ordered]
        ax.errorbar(counts, means, yerr=stds, marker="o", capsize=5, label=model)
    ax.set_xlabel("Quantidade de Categorias de Peso")
    ax.set_ylabel("MAE medio (kg)")
    ax.set_title("MAE por granularidade da Categoria de Peso")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Modelo")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_seed_variation_plot(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted({row["model"] for row in rows}):
        ordered = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: int(row["weight_category_count"]),
        )
        counts = [int(row["weight_category_count"]) for row in ordered]
        stds = [float(row["mae_std_between_seeds"]) for row in ordered]
        ranges = [float(row["mae_range_between_seeds"]) for row in ordered]
        ax.plot(counts, stds, marker="o", label=f"{model} desvio")
        ax.plot(counts, ranges, marker="o", label=f"{model} range")
    ax.set_xlabel("Quantidade de Categorias de Peso")
    ax.set_ylabel("MAE (kg)")
    ax.set_title("Variação do MAE entre seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_weight_scatter_plot(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = sorted({row["weight_category"] for row in rows})
    for category in categories:
        category_rows = [row for row in rows if row["weight_category"] == category]
        ax.scatter(
            [int(row["fold"]) for row in category_rows],
            [parse_weight(row["weight"], row.get("file_name", "")) for row in category_rows],
            alpha=0.75,
            label=category,
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Distribuição de pesos por fold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Categoria de Peso", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_weight_heatmap_plot(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    folds = sorted({int(row["fold"]) for row in rows})
    categories = sorted({row["weight_category"] for row in rows})
    counts = []
    for category in categories:
        counts.append(
            [
                len(
                    [
                        row
                        for row in rows
                        if int(row["fold"]) == fold and row["weight_category"] == category
                    ]
                )
                for fold in folds
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(len(folds)), [str(fold) for fold in folds])
    ax.set_yticks(range(len(categories)), categories)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Categoria de Peso")
    ax.set_title("Contagem por fold e Categoria de Peso")
    for y, row in enumerate(counts):
        for x, count in enumerate(row):
            ax.text(x, y, str(count), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="n")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def assign_split(rows: list[dict[str, str]], category_count: int, seed: int, k: int) -> list[dict[str, str]]:
    split_rows = [row.copy() for row in rows]
    assign_weight_categories(split_rows, category_count)
    assign_folds(split_rows, k, seed)
    return split_rows


def summarize_split_balance(
    rows: list[dict[str, str]], category_count: int, seed: int, k: int
) -> list[dict[str, str]]:
    split_rows = assign_split(rows, category_count, seed, k)
    summaries = []
    for fold in sorted({int(row["fold"]) for row in split_rows}):
        fold_rows = [row for row in split_rows if int(row["fold"]) == fold]
        for category in sorted({row["weight_category"] for row in fold_rows}):
            category_rows = [row for row in fold_rows if row["weight_category"] == category]
            weights = [parse_weight(row["weight"], row.get("file_name", "")) for row in category_rows]
            summaries.append(
                {
                    "weight_category_count": str(category_count),
                    "split_random_state": str(seed),
                    "fold": str(fold),
                    "weight_category": category,
                    "n_validation": str(len(category_rows)),
                    "weight_min": str(min(weights)),
                    "weight_median": str(statistics.median(weights)),
                    "weight_max": str(max(weights)),
                }
            )
    return summaries


def run_category_comparison(
    config_path: Path,
    category_counts: list[int],
    start_seed: int,
    seed_count: int,
    output_dir: Path,
) -> None:
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
    overall_rows = []
    fold_metric_rows = []
    split_balance_rows = []
    seeds = split_random_states(start_seed, seed_count)
    for category_count in category_counts:
        fold_metrics, _, overall, _ = evaluate_split_stability(
            feature_rows,
            [str(column) for column in feature_columns],
            parse_int(split["k"], "split.k"),
            category_count,
            seeds,
            parse_int(training["n_estimators"], "training.n_estimators"),
            parse_int(training["random_state"], "training.random_state"),
            parse_model_names(training),
        )
        for row in fold_metrics:
            fold_metric_rows.append({"weight_category_count": str(category_count), **row})
        for row in overall:
            overall_rows.append({"weight_category_count": str(category_count), **row})
        for seed in seeds:
            split_balance_rows.extend(
                summarize_split_balance(
                    feature_rows, category_count, seed, parse_int(split["k"], "split.k")
                )
            )

    write_csv(
        overall_rows,
        output_dir / "category_comparison_overall.csv",
        [
            "weight_category_count",
            "model",
            "split_random_states",
            "mae_mean",
            "mae_std_between_seeds",
            "mae_min_seed",
            "mae_max_seed",
            "mae_range_between_seeds",
        ],
    )
    write_csv(
        fold_metric_rows,
        output_dir / "category_comparison_fold_metrics.csv",
        [
            "weight_category_count",
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
        split_balance_rows,
        output_dir / "category_comparison_split_balance.csv",
        [
            "weight_category_count",
            "split_random_state",
            "fold",
            "weight_category",
            "n_validation",
            "weight_min",
            "weight_median",
            "weight_max",
        ],
    )
    save_mae_plot(overall_rows, output_dir / "category_comparison_mae.png")
    save_seed_variation_plot(
        overall_rows, output_dir / "category_comparison_seed_variation.png"
    )
    for category_count in category_counts:
        split_rows = assign_split(
            feature_rows, category_count, start_seed, parse_int(split["k"], "split.k")
        )
        save_weight_scatter_plot(
            split_rows,
            output_dir / f"category_comparison_weight_scatter_c{category_count}.png",
        )
        save_weight_heatmap_plot(
            split_rows,
            output_dir / f"category_comparison_weight_heatmap_c{category_count}.png",
        )


def parse_category_counts(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category-counts", default="4,6,8")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--output-dir", default="generated/diagnostics")
    args = parser.parse_args(argv)

    try:
        run_category_comparison(
            Path(args.config),
            parse_category_counts(args.category_counts),
            args.start_seed,
            args.seed_count,
            Path(args.output_dir),
        )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
