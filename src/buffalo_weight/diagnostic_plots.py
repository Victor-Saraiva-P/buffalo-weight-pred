from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_residuals(actual: np.ndarray, predicted: np.ndarray, farms: list[str], path: Path) -> None:
    """Plot residuals against weight; for example, ``plot_residuals(y, p, farms, path)``."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    colors = {"Faco": "#b44d3a", "Manezinho": "#2878a5"}
    for farm in sorted(set(farms)):
        selected = np.asarray(farms) == farm
        axis.scatter(actual[selected], predicted[selected] - actual[selected], label=farm, alpha=0.8, color=colors.get(farm))
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Peso real (kg)", ylabel="Erro predito - real (kg)", title="Resíduos OOF por peso e fazenda")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_group_mae(metric_rows: list[dict[str, str]], group_type: str, path: Path) -> None:
    """Plot MAE by a conditional group; for example, ``plot_group_mae(rows, "weight_category", path)``."""
    import matplotlib.pyplot as plt

    selected = [row for row in metric_rows if row["group_type"] == group_type]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar([row["group"] for row in selected], [float(row["mae"]) for row in selected], color="#5d6b45")
    axis.set(xlabel=group_type, ylabel="MAE (kg)", title=f"Erro por {group_type}")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_curve(summary_rows: list[dict[str, str]], path: Path) -> None:
    """Plot train and validation learning curves; for example, ``plot_learning_curve(rows, path)``."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = [float(row["n_train_mean"]) for row in summary_rows]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(sizes, [float(row["train_mae_mean"]) for row in summary_rows], yerr=[float(row["train_mae_std"]) for row in summary_rows], marker="o", label="Treino")
    axis.errorbar(sizes, [float(row["validation_mae_mean"]) for row in summary_rows], yerr=[float(row["validation_mae_std"]) for row in summary_rows], marker="o", label="Validação")
    axis.errorbar(sizes, [float(row["heavy_mae_mean"]) for row in summary_rows], yerr=[float(row["heavy_mae_std"]) for row in summary_rows], marker="o", label="Validação, 20% mais pesados")
    axis.set(xlabel="Amostras de treino", ylabel="MAE (kg)", title="Learning curve repetida")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_correlation(names: list[str], correlation: np.ndarray, path: Path) -> None:
    """Plot model error correlation; for example, ``plot_error_correlation(names, matrix, path)``."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(names)), labels=names, rotation=75, ha="right")
    axis.set_yticks(range(len(names)), labels=names)
    fig.colorbar(image, ax=axis, label="Correlação de Pearson")
    axis.set_title("Correlação dos erros OOF")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_quality_error(audit_rows: list[dict[str, str]], errors: dict[str, float], path: Path) -> None:
    """Plot mask morphology against error; for example, ``plot_quality_error(rows, errors, path)``."""
    import matplotlib.pyplot as plt

    metrics = ("foreground_ratio", "component_count", "hole_ratio", "largest_component_fraction")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    error_values = [errors[row["file_name"]] for row in audit_rows]
    for axis, metric in zip(axes.reshape(-1), metrics, strict=True):
        axis.scatter([float(row[metric]) for row in audit_rows], error_values, alpha=0.65)
        axis.set(xlabel=metric, ylabel="Erro absoluto OOF (kg)")
    fig.suptitle("Qualidade morfológica da máscara versus erro")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_nearest_pairs(
    pair_rows: list[dict[str, str]], masks: np.ndarray, index_by_name: dict[str, int], path: Path
) -> None:
    """Plot contradictory visual neighbors; for example, ``plot_nearest_pairs(rows, masks, index, path)``."""
    import matplotlib.pyplot as plt

    shown = pair_rows[:6]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(shown), 2, figsize=(6, 2.6 * len(shown)))
    for row_index, row in enumerate(shown):
        for column, suffix in enumerate(("a", "b")):
            file_name = row[f"file_name_{suffix}"]
            axes[row_index, column].imshow(masks[index_by_name[file_name]], cmap="gray")
            axes[row_index, column].set_title(f"{row[f'weight_{suffix}']} kg | {file_name[:8]}")
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(f"IoU {float(row['canonical_iou']):.2f}\nΔ {float(row['weight_difference']):.0f} kg")
    fig.suptitle("Máscaras canônicas semelhantes com pesos diferentes")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
