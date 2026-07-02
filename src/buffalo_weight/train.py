from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from buffalo_weight.cnn_mask import CnnMaskRegressor
from buffalo_weight.models import CNN_MASK_MODEL, ModelConfig, build_model, parse_model_configs
from buffalo_weight.split import parse_int, parse_weight, read_rows


METRIC_FIELDS = ["fold", "mae", "rmse", "r2", "n_train", "n_validation"]
PREDICTION_FIELDS = [
    "fold",
    "file_name",
    "weight",
    "y_pred",
    "error",
    "abs_error",
    "weight_category",
    "weight_category_label",
]
COMPARISON_FIELDS = ["model_config", "model", "mae_mean", "mae_min", "mae_max", "rmse_mean", "r2_mean"]


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


def evaluate_model(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    model_config: ModelConfig,
    masks_dir: Path | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metrics = []
    predictions = []
    folds = sorted({int(row["fold"]) for row in rows})

    for fold in folds:
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        validation_rows = [row for row in rows if int(row["fold"]) == fold]
        y_validation = np.asarray(
            [parse_weight(row["weight"], row.get("file_name", "")) for row in validation_rows],
            dtype=float,
        )

        if model_config.model == CNN_MASK_MODEL:
            if masks_dir is None:
                raise ValueError("cnn_mask requires data.masks_dir")
            model = CnnMaskRegressor(masks_dir, model_config.params)
            model.fit(train_rows, validation_rows)
            y_pred = model.predict(validation_rows)
        else:
            x_train, y_train = rows_to_arrays(train_rows, feature_columns)
            x_validation, _ = rows_to_arrays(validation_rows, feature_columns)
            model = build_model(model_config)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_validation)

        mae = mean_absolute_error(y_validation, y_pred)
        rmse = mean_squared_error(y_validation, y_pred) ** 0.5
        r2 = r2_score(y_validation, y_pred) if len(validation_rows) > 1 else math.nan
        metrics.append(
            {
                "model_config": model_config.name,
                "model": model_config.model,
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
                    "model_config": model_config.name,
                    "model": model_config.model,
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


def evaluate_models(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    model_configs: list[ModelConfig],
    masks_dir: Path | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metrics = []
    predictions = []
    for model_config in model_configs:
        model_metrics, model_predictions = evaluate_model(rows, feature_columns, model_config, masks_dir)
        metrics.extend(model_metrics)
        predictions.extend(model_predictions)
    return metrics, predictions


def evaluate_random_forest(
    rows: list[dict[str, str]], feature_columns: list[str], n_estimators: int, random_state: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    config = ModelConfig(
        "random_forest",
        "random_forest",
        {"n_estimators": n_estimators, "random_state": random_state},
    )
    return evaluate_model(rows, feature_columns, config)


def write_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def plot_fold_metrics(metrics: list[dict[str, str]], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = sorted(metrics, key=lambda row: int(row["fold"]))
    ax.plot(
        [int(row["fold"]) for row in rows],
        [float(row["mae"]) for row in rows],
        marker="o",
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (kg)")
    ax.set_title(title)
    ax.set_xticks(sorted({int(row["fold"]) for row in metrics}))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_model_comparison(metrics: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries = []
    for model_config in sorted({row["model_config"] for row in metrics}):
        rows = [row for row in metrics if row["model_config"] == model_config]
        maes = [float(row["mae"]) for row in rows]
        rmses = [float(row["rmse"]) for row in rows]
        r2s = [float(row["r2"]) for row in rows if row["r2"]]
        summaries.append(
            {
                "model_config": model_config,
                "model": rows[0]["model"],
                "mae_mean": format_metric(float(np.mean(maes))),
                "mae_min": format_metric(min(maes)),
                "mae_max": format_metric(max(maes)),
                "rmse_mean": format_metric(float(np.mean(rmses))),
                "r2_mean": format_metric(float(np.mean(r2s))) if r2s else "",
            }
        )
    return summaries


def plot_model_comparison(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: float(row["mae_mean"]))
    labels = [row["model_config"] for row in ordered]
    means = [float(row["mae_mean"]) for row in ordered]
    lower = [mean - float(row["mae_min"]) for mean, row in zip(means, ordered, strict=True)]
    upper = [float(row["mae_max"]) - mean for mean, row in zip(means, ordered, strict=True)]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(rows) * 0.45)))
    ax.barh(labels, means, xerr=[lower, upper], capsize=4)
    ax.invert_yaxis()
    ax.set_xlabel("MAE medio (kg)")
    ax.set_ylabel("Configuração de Modelo")
    ax.set_title("Comparação de Configurações de Modelo")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_predicted_vs_actual(
    predictions: list[dict[str, str]], path: Path, title: str, hard_limit: int = 10
) -> None:
    import matplotlib.pyplot as plt

    weights = [float(row["weight"]) for row in predictions]
    predicted = [float(row["y_pred"]) for row in predictions]
    hard_rows = sorted(predictions, key=lambda row: float(row["abs_error"]), reverse=True)[:hard_limit]
    min_value = min(weights + predicted)
    max_value = max(weights + predicted)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(weights, predicted, alpha=0.65, label="Predições")
    ax.scatter(
        [float(row["weight"]) for row in hard_rows],
        [float(row["y_pred"]) for row in hard_rows],
        color="#d95f02",
        edgecolor="black",
        linewidth=0.6,
        label=f"Top {len(hard_rows)} erros",
    )
    for row in hard_rows:
        ax.annotate(
            row["file_name"][:8],
            (float(row["weight"]), float(row["y_pred"])),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Peso real (kg)")
    ax.set_ylabel("Peso predito (kg)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def without_identity(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{key: value for key, value in row.items() if key not in {"model_config", "model"}} for row in rows]


def write_training_outputs(
    output_dir: Path,
    model_configs: list[ModelConfig],
    metrics: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> None:
    for model_config in model_configs:
        config_dir = output_dir / model_config.name
        config_metrics = [row for row in metrics if row["model_config"] == model_config.name]
        config_predictions = [row for row in predictions if row["model_config"] == model_config.name]
        write_csv(without_identity(config_metrics), config_dir / "fold_metrics.csv", METRIC_FIELDS)
        write_csv(without_identity(config_predictions), config_dir / "predictions.csv", PREDICTION_FIELDS)
        plot_fold_metrics(
            config_metrics,
            config_dir / "fold_mae.png",
            f"{model_config.name} ({model_config.model}) - MAE por fold",
        )
        plot_predicted_vs_actual(
            config_predictions,
            config_dir / "predicted_vs_actual.png",
            f"{model_config.name} ({model_config.model}) - Peso real vs predito",
        )


def write_model_comparison_from_outputs(output_dir: Path, model_configs: list[ModelConfig]) -> None:
    metrics = []
    by_name = {model_config.name: model_config for model_config in model_configs}
    for model_config in model_configs:
        path = output_dir / model_config.name / "fold_metrics.csv"
        for row in read_rows(path):
            metrics.append({"model_config": model_config.name, "model": by_name[model_config.name].model, **row})
    comparison = summarize_model_comparison(metrics)
    write_csv(comparison, output_dir / "model_comparison.csv", COMPARISON_FIELDS)
    plot_model_comparison(comparison, output_dir / "model_comparison.png")


def main() -> int:
    print(
        "buffalo_weight.train is a shared training library; use buffalo_weight.train_pipeline",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
