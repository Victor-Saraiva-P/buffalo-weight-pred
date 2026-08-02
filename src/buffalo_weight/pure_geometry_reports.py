from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.pure_geometry_evaluation import PURE_GEOMETRY_FEATURES, NestedEvaluation
from buffalo_weight.split import parse_weight
from buffalo_weight.train import format_metric


def write_pure_geometry_reports(
    evaluation: NestedEvaluation, source_rows: list[dict[str, str]], output_dir: Path
) -> list[dict[str, str]]:
    """Persist metrics and scientific diagnostics for geometry-only models.

    Example: ``write_pure_geometry_reports(evaluation, rows, Path("generated/pure_geometry"))``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = summarize_oof_predictions(evaluation.predictions)
    _write_evaluation_csvs(evaluation, comparison, output_dir)
    correlation_rows = feature_correlation_rows(source_rows)
    write_csv_rows(correlation_rows, output_dir / "correlation_matrix.csv", ["method", "feature", *PURE_GEOMETRY_FEATURES, "weight"])
    _write_plots(evaluation, source_rows, comparison, output_dir)
    write_scientific_report(evaluation, comparison, correlation_rows, output_dir / "report.md")
    return comparison


def summarize_oof_predictions(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    """Calculate pooled OOF MAE and R² for an unbiased model comparison.

    Example: ``summarize_oof_predictions(prediction_rows)``.
    """
    summaries = []
    for model_name in sorted({row["model"] for row in predictions}):
        model_rows = [row for row in predictions if row["model"] == model_name]
        actual = np.asarray([float(row["weight"]) for row in model_rows])
        predicted = np.asarray([float(row["prediction"]) for row in model_rows])
        summaries.append(_comparison_row(model_name, actual, predicted))
    return sorted(summaries, key=lambda row: float(row["mae_kg"]))


def _comparison_row(model_name: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, str]:
    residual = predicted - actual
    heavy_cutoff = float(np.quantile(actual, 0.8))
    heavy = actual >= heavy_cutoff
    return {
        "model": model_name,
        "mae_kg": format_metric(mean_absolute_error(actual, predicted)),
        "r2": format_metric(r2_score(actual, predicted)),
        "bias_kg": format_metric(float(np.mean(residual))),
        "heavy_20pct_mae_kg": format_metric(mean_absolute_error(actual[heavy], predicted[heavy])),
        "heavy_20pct_bias_kg": format_metric(float(np.mean(residual[heavy]))),
    }


def feature_correlation_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build Pearson and Spearman matrices including the target weight.

    Example: ``feature_correlation_rows(rows)[0]["method"] == "pearson"``.
    """
    names = [*PURE_GEOMETRY_FEATURES, "weight"]
    matrix = _geometry_target_matrix(source_rows)
    pearson = np.corrcoef(matrix, rowvar=False)
    spearman = np.corrcoef(np.apply_along_axis(rankdata, 0, matrix), rowvar=False)
    return _matrix_rows("pearson", names, pearson) + _matrix_rows("spearman", names, spearman)


def _geometry_target_matrix(source_rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    for row in source_rows:
        features = [float(row[feature].replace(",", ".")) for feature in PURE_GEOMETRY_FEATURES]
        values.append([*features, parse_weight(row["weight"], row.get("file_name", ""))])
    return np.asarray(values, dtype=float)


def _matrix_rows(method: str, names: list[str], matrix: np.ndarray) -> list[dict[str, str]]:
    return [
        {"method": method, "feature": name, **{other: format_metric(float(matrix[index, column])) for column, other in enumerate(names)}}
        for index, name in enumerate(names)
    ]


def _write_evaluation_csvs(
    evaluation: NestedEvaluation, comparison: list[dict[str, str]], output_dir: Path
) -> None:
    write_csv_rows(evaluation.fold_metrics, output_dir / "fold_metrics.csv", list(evaluation.fold_metrics[0]))
    write_csv_rows(evaluation.predictions, output_dir / "predictions.csv", list(evaluation.predictions[0]))
    write_csv_rows(evaluation.tuning_results, output_dir / "tuning_results.csv", list(evaluation.tuning_results[0]))
    write_csv_rows(evaluation.importance_rows, output_dir / "feature_importance.csv", list(evaluation.importance_rows[0]))
    write_csv_rows(comparison, output_dir / "model_comparison.csv", list(comparison[0]))


def _write_plots(
    evaluation: NestedEvaluation, source_rows: list[dict[str, str]],
    comparison: list[dict[str, str]], output_dir: Path,
) -> None:
    plot_feature_importance(evaluation.importance_rows, output_dir / "feature_importance.png")
    best_model = comparison[0]["model"]
    best_predictions = [row for row in evaluation.predictions if row["model"] == best_model]
    plot_residuals_vs_prediction(best_predictions, output_dir / "residuals_vs_prediction.png")
    plot_correlation_matrix(source_rows, output_dir / "correlation_matrix.png")


def plot_feature_importance(importance_rows: list[dict[str, str]], path: Path) -> None:
    """Plot validation permutation importance as MAE increase.

    Example: ``plot_feature_importance(rows, Path("importance.png"))``.
    """
    import matplotlib.pyplot as plt

    models = sorted({row["model"] for row in importance_rows})
    figure, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for axis, model_name in zip(axes[0], models, strict=True):
        values = _mean_importance_by_feature(importance_rows, model_name)
        axis.barh(list(values), list(values.values()))
        axis.set(title=model_name, xlabel="Aumento do MAE após permutação (kg)")
    figure.suptitle("Importância OOF das features de geometria pura")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _mean_importance_by_feature(rows: list[dict[str, str]], model_name: str) -> dict[str, float]:
    values = {
        feature: float(np.mean([float(row["mae_increase_mean"]) for row in rows if row["model"] == model_name and row["feature"] == feature]))
        for feature in PURE_GEOMETRY_FEATURES
    }
    return dict(sorted(values.items(), key=lambda item: item[1]))


def plot_residuals_vs_prediction(predictions: list[dict[str, str]], path: Path) -> None:
    """Plot OOF residuals against predictions to reveal systematic bias.

    Example: ``plot_residuals_vs_prediction(rows, Path("residuals.png"))``.
    """
    import matplotlib.pyplot as plt

    predicted = np.asarray([float(row["prediction"]) for row in predictions])
    residuals = np.asarray([float(row["residual"]) for row in predictions])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(predicted, residuals, alpha=0.75)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Predição OOF (kg)", ylabel="Resíduo: predito - real (kg)", title="Resíduos vs. predição")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_correlation_matrix(source_rows: list[dict[str, str]], path: Path) -> None:
    """Plot Spearman correlations among allowed features and weight.

    Example: ``plot_correlation_matrix(rows, Path("correlation.png"))``.
    """
    import matplotlib.pyplot as plt

    names = [*PURE_GEOMETRY_FEATURES, "weight"]
    matrix = _geometry_target_matrix(source_rows)
    correlation = np.corrcoef(np.apply_along_axis(rankdata, 0, matrix), rowvar=False)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(names)), names, rotation=75, ha="right")
    axis.set_yticks(range(len(names)), names)
    figure.colorbar(image, ax=axis, label="Correlação de Spearman")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_scientific_report(
    evaluation: NestedEvaluation, comparison: list[dict[str, str]],
    correlations: list[dict[str, str]], path: Path,
) -> None:
    """Write an evidence-only interpretation of the nested evaluation.

    Example: ``write_scientific_report(evaluation, comparison, correlations, path)``.
    """
    best = comparison[0]
    ridge = next(row for row in comparison if row["model"] == "ridge")
    diagnostics = _residual_diagnostics(evaluation.predictions, best["model"])
    strongest = _strongest_weight_correlation(correlations)
    redundancy = _strongest_feature_pair(correlations)
    train_mae = _mean_training_mae(evaluation.fold_metrics, best["model"])
    text = _report_text(best, ridge, diagnostics, strongest, redundancy, train_mae)
    path.write_text(text)


def _residual_diagnostics(predictions: list[dict[str, str]], model_name: str) -> dict[str, float]:
    rows = [row for row in predictions if row["model"] == model_name]
    predicted = np.asarray([float(row["prediction"]) for row in rows])
    actual = np.asarray([float(row["weight"]) for row in rows])
    residuals = np.asarray([float(row["residual"]) for row in rows])
    pred_correlation, pred_p_value = spearmanr(predicted, residuals)
    actual_correlation, actual_p_value = spearmanr(actual, residuals)
    return {
        "pred_spearman": float(pred_correlation), "pred_p_value": float(pred_p_value),
        "actual_spearman": float(actual_correlation), "actual_p_value": float(actual_p_value),
    }


def _strongest_weight_correlation(correlations: list[dict[str, str]]) -> tuple[str, float]:
    rows = [row for row in correlations if row["method"] == "spearman" and row["feature"] != "weight"]
    strongest = max(rows, key=lambda row: abs(float(row["weight"])))
    return strongest["feature"], float(strongest["weight"])


def _strongest_feature_pair(correlations: list[dict[str, str]]) -> tuple[str, str, float]:
    rows = [row for row in correlations if row["method"] == "spearman" and row["feature"] != "weight"]
    pairs = [(row["feature"], feature, float(row[feature])) for row in rows for feature in PURE_GEOMETRY_FEATURES if row["feature"] < feature]
    return max(pairs, key=lambda pair: abs(pair[2]))


def _mean_training_mae(fold_metrics: list[dict[str, str]], model_name: str) -> float:
    values = [float(row["train_mae"]) for row in fold_metrics if row["model"] == model_name]
    if not values:
        raise ValueError(f"fold metrics had model {model_name!r} 0 times; expected at least one training MAE")
    return float(np.mean(values))


def _report_text(
    best: dict[str, str], ridge: dict[str, str], diagnostics: dict[str, float],
    strongest: tuple[str, float], redundancy: tuple[str, str, float], train_mae: float,
) -> str:
    delta = float(ridge["mae_kg"]) - float(best["mae_kg"])
    sections = [
        "# Avaliação de geometria pura",
        _protocol_section(),
        _result_section(best, ridge, diagnostics, strongest, redundancy, train_mae, delta),
        _evidence_section(),
    ]
    return "\n\n".join(sections) + "\n"


def _protocol_section() -> str:
    return f"""## Protocolo

Foram usados exclusivamente `{', '.join(PURE_GEOMETRY_FEATURES)}`. Cada predição é OOF de um Stratified K-Fold externo com 5 folds e 10 faixas de peso. A seleção de hiperparâmetros ocorreu em folds internos criados apenas a partir das amostras de treino do fold externo."""


def _result_section(
    best: dict[str, str], ridge: dict[str, str], diagnostics: dict[str, float],
    strongest: tuple[str, float], redundancy: tuple[str, str, float], train_mae: float, delta: float,
) -> str:
    return f"""## Resultado

O melhor modelo foi `{best['model']}`: MAE OOF {float(best['mae_kg']):.2f} kg e R² {float(best['r2']):.3f}. A baseline Ridge obteve MAE {float(ridge['mae_kg']):.2f} kg e R² {float(ridge['r2']):.3f}; a diferença de MAE foi {delta:.2f} kg.

Nos 20% mais pesados, o melhor modelo teve MAE {float(best['heavy_20pct_mae_kg']):.2f} kg e viés {float(best['heavy_20pct_bias_kg']):.2f} kg. O MAE médio no treino foi {train_mae:.2f} kg, contra {float(best['mae_kg']):.2f} kg OOF, evidenciando um gap de generalização de {float(best['mae_kg']) - train_mae:.2f} kg.

A correlação de Spearman entre peso real e resíduo foi {diagnostics['actual_spearman']:.3f} (p={diagnostics['actual_p_value']:.3g}), quantificando regressão sistemática em direção à média. A associação monotônica mais forte entre feature e peso foi `{strongest[0]}` (Spearman {strongest[1]:.3f}). O par `{redundancy[0]}`/`{redundancy[1]}` teve Spearman {redundancy[2]:.3f}, evidenciando redundância entre descritores. Juntos, o gap treino-validação, a redundância e o viés extremo sustentam limitação de generalização com estas entradas; não provam que a curva de aprendizado tenha atingido um plateau."""


def _evidence_section() -> str:
    return """## Evidências geradas

- `feature_importance.png`: importância por permutação medida nos folds externos.
- `residuals_vs_prediction.png`: dispersão dos resíduos OOF contra a predição.
- `correlation_matrix.png`: matriz de Spearman das features permitidas e peso.
- `fold_metrics.csv`, `predictions.csv` e `tuning_results.csv`: evidência numérica reproduzível.

Esses resultados medem generalização dentro da amostra disponível; não demonstram generalização prospectiva para novas fazendas, câmeras ou protocolos de aquisição."""
