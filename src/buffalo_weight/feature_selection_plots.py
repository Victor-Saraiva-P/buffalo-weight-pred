"""Canonical 300 DPI figures for feature-selection evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from buffalo_weight.feature_evaluation import FeatureEvidence
from buffalo_weight.feature_recommendations import RemovalRecommendation
from buffalo_weight.feature_redundancy import FeatureRedundancy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FIGURE_SIZE = (8.0, 6.0)
FIGURE_DPI = 300


def save_feature_selection_figures(
    output_dir: Path, feature_names: tuple[str, ...], redundancy: list[FeatureRedundancy],
    evidence: list[FeatureEvidence], recommendations: list[RemovalRecommendation],
) -> None:
    """Save three canonical figures; for example, every PNG is fixed at 2400×1800."""
    _save_redundancy_heatmap(output_dir / "redundancy_heatmap.png", feature_names, redundancy)
    _save_removal_heatmap(output_dir / "removal_heatmap.png", recommendations)
    _save_permutation_effects(output_dir / "permutation_effects.png", feature_names, evidence)


def _save_redundancy_heatmap(
    path: Path, features: tuple[str, ...], rows: list[FeatureRedundancy]
) -> None:
    matrix = np.eye(len(features))
    index = {name: position for position, name in enumerate(features)}
    for row in rows:
        value = row.pearson if row.pearson is not None else np.nan
        first, second = index[row.feature_a], index[row.feature_b]
        matrix[first, second] = matrix[second, first] = value
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    _feature_ticks(axis, features, features)
    axis.set_title("Redundância observada entre features (Pearson)")
    figure.colorbar(image, ax=axis, label="Correlação")
    _save_figure(figure, path)


def _save_removal_heatmap(
    path: Path, recommendations: list[RemovalRecommendation]
) -> None:
    matrix = np.asarray([[item.random_forest_delta_mae_kg, item.dense_delta_mae_kg]
                         for item in recommendations])
    limit = max(float(np.max(np.abs(matrix))), 1.0)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    _feature_ticks(axis, ("Random Forest", "Rede Densa"),
                   tuple(item.target for item in recommendations))
    axis.set_title("Efeito OOF da retirada (retirada − completo)")
    figure.colorbar(image, ax=axis, label="Δ MAE (kg)")
    _save_figure(figure, path)


def _save_permutation_effects(
    path: Path, features: tuple[str, ...], evidence: list[FeatureEvidence]
) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    positions = np.arange(len(features))
    for offset, baseline, label, color in ((-0.12, "random_forest", "Random Forest", "#0072B2"),
                                            (0.12, "dense", "Rede Densa", "#D55E00")):
        means, lower, upper = _permutation_intervals(features, evidence, baseline)
        axis.errorbar(positions + offset, means, yerr=[lower, upper], fmt="o", markersize=3,
                      capsize=2, label=label, color=color)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, features, rotation=90, fontsize=5)
    axis.set_ylabel("Δ MAE após permutação (kg)")
    axis.set_title("Dependência fora da amostra por feature")
    axis.legend()
    _save_figure(figure, path)


def _permutation_intervals(
    features: tuple[str, ...], evidence: list[FeatureEvidence], baseline: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deltas = [[row.delta_mae_kg for row in evidence if row.scope == "oof"
               and row.experiment == "permutation" and row.baseline == baseline
               and row.target == feature and row.delta_mae_kg is not None] for feature in features]
    means = np.asarray([np.mean(values) for values in deltas])
    lower = np.asarray([mean - min(values) for mean, values in zip(means, deltas)])
    upper = np.asarray([max(values) - mean for mean, values in zip(means, deltas)])
    return means, lower, upper


def _feature_ticks(axis: Axes, x_labels: tuple[str, ...], y_labels: tuple[str, ...]) -> None:
    axis.set_xticks(range(len(x_labels)), x_labels, rotation=90, fontsize=5)
    axis.set_yticks(range(len(y_labels)), y_labels, fontsize=5)
    axis.tick_params(axis="both", pad=1)


def _save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, metadata={"Software": "buffalo-weight-pred"})
    plt.close(figure)
