"""Canonical plot generation for controlled learning curves diagnostic slice.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from buffalo_weight.diagnostic_learning_types import LearningPointRecord


CONFIGURATION_LABELS = {
    "random_forest_baseline": "Random Forest (Feições)",
    "dense_baseline": "Rede Densa por Feições",
    "compact_cnn_baseline": "CNN Compacta",
    "resnet18_pretrained_partial": "ResNet-18 Pré-treinada",
}

CONFIGURATION_COLORS = {
    "random_forest_baseline": "#2878a5",
    "dense_baseline": "#b44d3a",
    "compact_cnn_baseline": "#5d6b45",
    "resnet18_pretrained_partial": "#7b5294",
}

CONFIGURATION_MARKERS = {
    "random_forest_baseline": "o",
    "dense_baseline": "s",
    "compact_cnn_baseline": "^",
    "resnet18_pretrained_partial": "D",
}


def plot_canonical_learning_curves(
    points: tuple[LearningPointRecord, ...],
    path: Path,
) -> None:
    """Plot controlled learning curves comparing 50%, 75%, and 100% fractions for four baselines.

    Example: ``plot_canonical_learning_curves(points, output_dir / "learning_curves_canonical.png")``.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))

    fractions = (0.50, 0.75, 1.00)
    frac_percents = [50, 75, 100]

    for config, label in CONFIGURATION_LABELS.items():
        _plot_single_configuration(axis, points, config, label, fractions, frac_percents)

    _format_learning_curves_axis(axis, frac_percents)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_single_configuration(
    axis: object,
    points: tuple[LearningPointRecord, ...],
    config: str,
    label: str,
    fractions: tuple[float, ...],
    frac_percents: list[int],
) -> None:
    config_points = [p for p in points if p.configuration == config]
    if not config_points:
        return

    mean_maes = _calculate_mean_maes(config_points, fractions)
    color = CONFIGURATION_COLORS.get(config, "#333333")
    marker = CONFIGURATION_MARKERS.get(config, "o")

    axis.plot(  # type: ignore
        frac_percents, mean_maes, marker=marker, linewidth=2.0,
        markersize=7, label=label, color=color,
    )


def _calculate_mean_maes(
    config_points: list[LearningPointRecord], fractions: tuple[float, ...]
) -> list[float]:
    mean_maes: list[float] = []
    for frac in fractions:
        frac_pts = [p for p in config_points if abs(p.fraction - frac) < 1e-4]
        mae_val = float(np.mean([p.mae_kg for p in frac_pts])) if frac_pts else 0.0
        mean_maes.append(mae_val)
    return mean_maes


def _format_learning_curves_axis(axis: object, frac_percents: list[int]) -> None:
    axis.set_xlabel("Fração da Partição de Treino Externo (%)")  # type: ignore
    axis.set_ylabel("MAE OOF (kg)")  # type: ignore
    axis.set_title("Curvas de Aprendizado Controladas dos Baselines")  # type: ignore
    axis.set_xticks(frac_percents)  # type: ignore
    axis.set_xticklabels(["50%", "75%", "100%"])  # type: ignore
    axis.grid(True, linestyle="--", alpha=0.5)  # type: ignore
    axis.legend(loc="upper right")  # type: ignore
