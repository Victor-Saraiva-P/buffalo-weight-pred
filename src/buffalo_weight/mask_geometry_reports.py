from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from buffalo_weight.artifact_provenance import TrainingEvidence
from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.models import ModelConfig
from buffalo_weight.split import read_rows
from buffalo_weight.train import write_training_outputs


COMPARISON_FIELDS = ["model", "input", "mae_kg", "r2"]


def write_mask_geometry_reports(
    metrics: list[dict[str, str]],
    predictions: list[dict[str, str]],
    model_configs: list[ModelConfig],
    evidence: TrainingEvidence,
    output_dir: Path,
) -> list[dict[str, str]]:
    """Write comparable OOF evidence; for example, ``write_mask_geometry_reports(..., output)``."""
    completed = _completed_configs(model_configs, predictions)
    if completed:
        write_training_outputs(output_dir, completed, metrics, predictions, evidence)
    stored_predictions = _stored_predictions(output_dir, model_configs)
    comparison = _comparison_rows(stored_predictions, model_configs)
    comparison.extend(_pure_geometry_rows(output_dir.parent / "pure_geometry" / "model_comparison.csv"))
    comparison.sort(key=lambda row: float(row["mae_kg"]))
    write_csv_rows(comparison, output_dir / "model_comparison.csv", COMPARISON_FIELDS)
    _plot_comparison(comparison, output_dir / "model_comparison.png")
    _write_report(comparison, output_dir / "report.md")
    return comparison


def _completed_configs(
    model_configs: list[ModelConfig], predictions: list[dict[str, str]]
) -> list[ModelConfig]:
    predicted_names = {row["model_config"] for row in predictions}
    return [config for config in model_configs if config.name in predicted_names]


def _stored_predictions(
    output_dir: Path, model_configs: list[ModelConfig]
) -> list[dict[str, str]]:
    rows = []
    for config in model_configs:
        path = output_dir / config.name / "predictions.csv"
        rows.extend(
            {"model_config": config.name, "model": config.model, **row}
            for row in read_rows(path)
        )
    return rows


def _comparison_rows(
    predictions: list[dict[str, str]], model_configs: list[ModelConfig]
) -> list[dict[str, str]]:
    rows = []
    for config in model_configs:
        selected = [row for row in predictions if row["model_config"] == config.name]
        actual = np.asarray([float(row["weight"]) for row in selected])
        predicted = np.asarray([float(row["y_pred"]) for row in selected])
        rows.append(_comparison_row(config, actual, predicted))
    return sorted(rows, key=lambda row: float(row["mae_kg"]))


def _comparison_row(
    config: ModelConfig, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, str]:
    input_name = "máscara + geometria pura" if config.model == "cnn_mask_geometry" else "máscara"
    return {
        "model": config.name,
        "input": input_name,
        "mae_kg": f"{mean_absolute_error(actual, predicted):.12g}",
        "r2": f"{r2_score(actual, predicted):.12g}",
    }


def _pure_geometry_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return [
        {
            "model": row["model"],
            "input": "geometria pura",
            "mae_kg": row["mae_kg"],
            "r2": row["r2"],
        }
        for row in read_rows(path)
    ]


def _plot_comparison(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh([row["model"] for row in rows], [float(row["mae_kg"]) for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("MAE OOF (kg)")
    axis.set_title("CNN: máscara versus fusão com geometria pura")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_report(rows: list[dict[str, str]], path: Path) -> None:
    best = rows[0]
    cnn_rows = [row for row in rows if row["input"] != "geometria pura"]
    best_cnn = min(cnn_rows, key=lambda row: float(row["mae_kg"]))
    lines = [
        "# Comparação CNN com geometria pura",
        "",
        "As CNNs usam os mesmos cinco folds e early stopping interno ao treino.",
        "A fusão recebe somente as dez features permitidas pela avaliação geometry-only.",
        "",
        f"A melhor CNN foi `{best_cnn['model']}`: MAE OOF {float(best_cnn['mae_kg']):.2f} kg e R² {float(best_cnn['r2']):.3f}.",
        f"O melhor resultado da tabela foi `{best['model']}`: MAE OOF {float(best['mae_kg']):.2f} kg.",
        "",
        "A escolha das arquiteturas foi informada por explorações anteriores nestes mesmos animais;",
        "portanto, o resultado continua exploratório até validação prospectiva externa.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
