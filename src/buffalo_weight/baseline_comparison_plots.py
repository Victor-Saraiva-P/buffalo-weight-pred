"""Canonical figures for controlled baseline comparison."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from buffalo_weight.baseline_comparison_types import ComparisonMetric, ComparisonPrediction

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


FIGURE_SIZE = (8.0, 6.0)
FIGURE_DPI = 300
APPROACH_ORDER = ("random_forest", "dense_feature_network", "compact_cnn", "resnet18")
APPROACH_LABELS = {
    "random_forest": "Random Forest",
    "dense_feature_network": "Rede Densa por Feições",
    "compact_cnn": "CNN compacta",
    "resnet18": "ResNet-18 pré-treinada",
}
APPROACH_COLORS = {
    "random_forest": "#0072B2", "dense_feature_network": "#D55E00",
    "compact_cnn": "#009E73", "resnet18": "#CC79A7",
}


def save_baseline_comparison_figures(
    output_dir: Path, predictions: list[ComparisonPrediction], metrics: list[ComparisonMetric],
) -> None:
    """Save three candidate-only figures; for example, the mean reference is excluded."""
    candidates = [row for row in predictions if row.evaluation_role == "candidate"]
    _save_global_mae(output_dir / "global_mae.png", metrics)
    _save_prediction_panels(output_dir / "predicted_vs_observed.png", candidates)
    _save_residual_panels(output_dir / "residuals_vs_observed.png", candidates)


def _save_global_mae(path: Path, metrics: list[ComparisonMetric]) -> None:
    indexed = {row.approach: row.mae_kg for row in metrics if row.evaluation_role == "candidate"
               and row.scope == "oof" and row.population == "all"}
    values = [indexed[approach] for approach in APPROACH_ORDER]
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.bar([APPROACH_LABELS[name] for name in APPROACH_ORDER], values,
             color=[APPROACH_COLORS[name] for name in APPROACH_ORDER])
    axis.set_ylabel("MAE OOF Pós-Seleção (kg)")
    axis.set_title("Comparação controlada das quatro candidatas")
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, path)


def _save_prediction_panels(path: Path, predictions: list[ComparisonPrediction]) -> None:
    bounds = _observed_prediction_bounds(predictions)
    _save_scatter_panels(
        path, predictions, lambda row: row.predicted_weight_kg,
        "Predição OOF (kg)", bounds,
    )


def _save_residual_panels(path: Path, predictions: list[ComparisonPrediction]) -> None:
    _save_scatter_panels(
        path, predictions, lambda row: row.residual_kg, "Resíduo (kg)", None,
    )


def _save_scatter_panels(
    path: Path, predictions: list[ComparisonPrediction],
    y_value: Callable[[ComparisonPrediction], float], y_label: str,
    identity_bounds: tuple[float, float] | None,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex=True, sharey=True)
    for approach, axis in zip(APPROACH_ORDER, axes.flat, strict=True):
        rows = [row for row in predictions if row.approach == approach]
        axis.scatter([row.observed_weight_kg for row in rows],
                     [y_value(row) for row in rows], s=10, alpha=0.7,
                     color=APPROACH_COLORS[approach])
        _draw_panel_reference(axis, identity_bounds)
        _label_panel(axis, approach, "Peso observado (kg)", y_label)
    _save_figure(figure, path)


def _draw_panel_reference(axis: Axes, identity_bounds: tuple[float, float] | None) -> None:
    if identity_bounds is None:
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        return
    axis.plot(identity_bounds, identity_bounds, color="black", linewidth=0.8, linestyle="--")


def _observed_prediction_bounds(predictions: list[ComparisonPrediction]) -> tuple[float, float]:
    values = [value for row in predictions
              for value in (row.observed_weight_kg, row.predicted_weight_kg)]
    return float(np.min(values)), float(np.max(values))


def _label_panel(axis: Axes, approach: str, x_label: str, y_label: str) -> None:
    axis.set_title(APPROACH_LABELS[approach])
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


def _save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, metadata={"Software": "buffalo-weight-pred"})
    plt.close(figure)
