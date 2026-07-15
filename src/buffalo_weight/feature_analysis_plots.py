from __future__ import annotations

from pathlib import Path

import numpy as np

from buffalo_weight.feature_analysis_core import parse_feature_value
from buffalo_weight.feature_analysis_reports import correlation_matrix
from buffalo_weight.split import parse_weight


def save_summary_plots(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    save_impact_plot(summary_rows, output_dir / "summary", "removal_impact_mae", "Impacto de remocao no MAE")
    save_impact_plot(summary_rows, output_dir / "summary", "permutation_impact_mae", "Impacto de permutacao no MAE")
    save_single_feature_plot(summary_rows, output_dir / "summary")


def save_impact_plot(rows: list[dict[str, str]], directory: Path, column: str, title: str) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for model_config in sorted({row["model_config"] for row in rows}):
        model_rows = sorted([row for row in rows if row["model_config"] == model_config], key=lambda row: float(row[column] or 0), reverse=True)
        fig, ax = plt.subplots(figsize=(9, max(5, len(model_rows) * 0.35)))
        ax.barh([row["feature"] for row in model_rows], [float(row[column] or 0) for row in model_rows])
        ax.invert_yaxis(); ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Delta MAE (kg)"); ax.set_title(f"{title} - {model_config}")
        ax.grid(axis="x", alpha=0.25); fig.tight_layout()
        fig.savefig(directory / f"{column}_{model_config}.png", dpi=160)
        plt.close(fig)


def save_single_feature_plot(rows: list[dict[str, str]], directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for model_config in sorted({row["model_config"] for row in rows}):
        model_rows = sorted([row for row in rows if row["model_config"] == model_config], key=lambda row: float(row["single_feature_mae_mean"]))
        fig, ax = plt.subplots(figsize=(9, max(5, len(model_rows) * 0.35)))
        ax.barh([row["feature"] for row in model_rows], [float(row["single_feature_mae_mean"]) for row in model_rows])
        ax.axvline(float(model_rows[0]["mean_baseline_mae_mean"]), color="black", linestyle="--", label="mean_baseline")
        ax.axvline(float(model_rows[0]["area_baseline_mae_mean"]), color="#d95f02", linestyle="--", label="area_baseline")
        ax.invert_yaxis(); ax.set_xlabel("MAE medio (kg)")
        ax.set_title(f"Feature isolada vs baselines - {model_config}")
        ax.grid(axis="x", alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(directory / f"single_feature_mae_{model_config}.png", dpi=160)
        plt.close(fig)


def save_diagnostic_plots(rows: list[dict[str, str]], features: list[str], output_dir: Path) -> None:
    save_correlation_plot(rows, features, output_dir / "diagnostics" / "spearman_correlation.png", "spearman")
    for feature in features:
        save_feature_scatter(rows, feature, output_dir / "diagnostics" / f"{feature}_vs_weight.png")


def save_correlation_plot(rows: list[dict[str, str]], features: list[str], path: Path, method: str) -> None:
    import matplotlib.pyplot as plt

    matrix_rows = correlation_matrix(rows, features, method)
    matrix = np.asarray([[float(row[feature]) for feature in features] for row in matrix_rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(features) * 0.45), max(6, len(features) * 0.45)))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=90); ax.set_yticks(range(len(features)), features)
    ax.set_title(f"Correlacao {method}"); fig.colorbar(image, ax=ax, label="correlacao")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_feature_scatter(rows: list[dict[str, str]], feature: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    values = [parse_feature_value(row, feature) for row in rows]
    weights = [parse_weight(row["weight"], row.get("file_name", "")) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(values, weights, alpha=0.7)
    if len(values) > 1:
        slope, intercept = np.polyfit(values, weights, 1)
        x_values = np.asarray([min(values), max(values)])
        ax.plot(x_values, slope * x_values + intercept, color="#d95f02", linewidth=1)
    ax.set_xlabel(feature); ax.set_ylabel("Peso (kg)"); ax.set_title(f"{feature} vs peso")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
