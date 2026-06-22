from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from buffalo_weight.config import load_config
from buffalo_weight.split import parse_int, parse_weight, read_rows


MODEL_NAME = "random_forest"


def parse_float(value: str, column: str, file_name: str) -> float:
    try:
        return float(value.replace(",", "."))
    except ValueError as error:
        raise ValueError(f"Invalid {column} for {file_name}: {value}") from error


def join_rows(
    feature_rows: list[dict[str, str]], split_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    split_by_file = {row["file_name"]: row for row in split_rows}
    joined = []
    for feature_row in feature_rows:
        file_name = feature_row["file_name"]
        if file_name not in split_by_file:
            continue
        joined.append({**feature_row, **split_by_file[file_name]})
    if len(joined) != len(split_rows):
        raise ValueError("feature index and split rows do not match")
    return joined


def rows_to_arrays(
    rows: list[dict[str, str]], feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    x = [
        [parse_float(row[column], column, row.get("file_name", "")) for column in feature_columns]
        for row in rows
    ]
    y = [parse_weight(row["weight"], row.get("file_name", "")) for row in rows]
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def format_metric(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.12g}"


def evaluate_random_forest(
    rows: list[dict[str, str]], feature_columns: list[str], n_estimators: int, random_state: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metrics = []
    predictions = []
    folds = sorted({int(row["fold"]) for row in rows})

    for fold in folds:
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        validation_rows = [row for row in rows if int(row["fold"]) == fold]
        x_train, y_train = rows_to_arrays(train_rows, feature_columns)
        x_validation, y_validation = rows_to_arrays(validation_rows, feature_columns)

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_validation)

        mae = mean_absolute_error(y_validation, y_pred)
        rmse = mean_squared_error(y_validation, y_pred) ** 0.5
        r2 = r2_score(y_validation, y_pred) if len(validation_rows) > 1 else math.nan
        metrics.append(
            {
                "model": MODEL_NAME,
                "fold": str(fold),
                "mae": format_metric(mae),
                "rmse": format_metric(rmse),
                "r2": format_metric(r2),
                "n_train": str(len(train_rows)),
                "n_validation": str(len(validation_rows)),
            }
        )

        for row, predicted, actual in zip(validation_rows, y_pred, y_validation, strict=True):
            error = float(predicted - actual)
            predictions.append(
                {
                    "model": MODEL_NAME,
                    "fold": str(fold),
                    "file_name": row["file_name"],
                    "weight": format_metric(float(actual)),
                    "y_pred": format_metric(float(predicted)),
                    "error": format_metric(error),
                    "abs_error": format_metric(abs(error)),
                    "weight_category": row["weight_category"],
                    "weight_category_label": row["weight_category_label"],
                }
            )
    return metrics, predictions


def write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def plot_metrics(metrics: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    models = sorted({row["model"] for row in metrics})
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        model_rows = sorted(
            [row for row in metrics if row["model"] == model], key=lambda row: int(row["fold"])
        )
        ax.plot(
            [int(row["fold"]) for row in model_rows],
            [float(row["mae"]) for row in model_rows],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (kg)")
    ax.set_title("Erro medio absoluto por fold")
    ax.set_xticks(sorted({int(row["fold"]) for row in metrics}))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Modelo")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(config_path: Path) -> None:
    config = load_config(config_path)
    output = config["output"]
    training = config["training"]
    if not isinstance(output, dict):
        raise ValueError("config output section must be a map")
    if not isinstance(training, dict):
        raise ValueError("config training section must be a map")
    feature_columns = training["feature_columns"]
    if not isinstance(feature_columns, list):
        raise ValueError("config training.feature_columns must be a list")

    feature_rows = read_rows(Path(str(output["features_index_path"])))
    split_rows = read_rows(Path(str(training["split_path"])))
    rows = join_rows(feature_rows, split_rows)
    metrics, predictions = evaluate_random_forest(
        rows,
        [str(column) for column in feature_columns],
        parse_int(training["n_estimators"], "training.n_estimators"),
        parse_int(training["random_state"], "training.random_state"),
    )

    write_csv(
        metrics,
        Path(str(training["metrics_path"])),
        ["model", "fold", "mae", "rmse", "r2", "n_train", "n_validation"],
    )
    write_csv(
        predictions,
        Path(str(training["predictions_path"])),
        [
            "model",
            "fold",
            "file_name",
            "weight",
            "y_pred",
            "error",
            "abs_error",
            "weight_category",
            "weight_category_label",
        ],
    )
    plot_metrics(metrics, Path(str(training["metrics_plot_path"])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    try:
        train(Path(args.config))
    except (KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
