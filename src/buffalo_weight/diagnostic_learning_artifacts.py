"""Artifact writing for controlled learning curves diagnostic slice.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

from pathlib import Path

from buffalo_weight.csv_io import format_csv_number, write_csv_rows
from buffalo_weight.diagnostic_learning_plots import plot_canonical_learning_curves
from buffalo_weight.diagnostic_learning_types import LearningCurvesSlice


POINT_COLUMNS = [
    "configuration",
    "fold",
    "fraction",
    "n_train",
    "evaluated_population",
    "n_eval",
    "mae_kg",
    "bias_kg",
    "artifact_action",
]

SUMMARY_COLUMNS = [
    "configuration",
    "fraction",
    "mean_n_train",
    "mean_mae_kg",
    "std_mae_kg",
    "mean_bias_kg",
    "reused_points_count",
]


def write_learning_curves_artifacts(
    output_dir: Path,
    slice_data: LearningCurvesSlice,
) -> None:
    """Write tidy CSV files, canonical plot, and markdown report for controlled learning curves.

    Example: ``write_learning_curves_artifacts(output_dir, slice_data)`` saves all artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_points_csv(output_dir / "learning_curves_points.csv", slice_data)
    _write_summary_csv(output_dir / "learning_curves_summary.csv", slice_data)
    plot_canonical_learning_curves(slice_data.point_records, output_dir / "learning_curves_canonical.png")
    _write_report_markdown(output_dir / "learning_curves_report.md", slice_data)


def _write_points_csv(path: Path, data: LearningCurvesSlice) -> None:
    rows = [
        {
            "configuration": p.configuration,
            "fold": str(p.fold),
            "fraction": f"{p.fraction:.2f}",
            "n_train": str(p.n_train),
            "evaluated_population": p.evaluated_population,
            "n_eval": str(p.n_eval),
            "mae_kg": format_csv_number(p.mae_kg),
            "bias_kg": format_csv_number(p.bias_kg),
            "artifact_action": p.artifact_action,
        }
        for p in data.point_records
    ]
    write_csv_rows(rows, path, POINT_COLUMNS)


def _write_summary_csv(path: Path, data: LearningCurvesSlice) -> None:
    rows = [
        {
            "configuration": s.configuration,
            "fraction": f"{s.fraction:.2f}",
            "mean_n_train": format_csv_number(s.mean_n_train),
            "mean_mae_kg": format_csv_number(s.mean_mae_kg),
            "std_mae_kg": format_csv_number(s.std_mae_kg),
            "mean_bias_kg": format_csv_number(s.mean_bias_kg),
            "reused_points_count": str(s.reused_points_count),
        }
        for s in data.summary_records
    ]
    write_csv_rows(rows, path, SUMMARY_COLUMNS)


def _write_report_markdown(path: Path, data: LearningCurvesSlice) -> None:
    lines = [
        "# Relatório de Diagnóstico: Curvas de Aprendizado Controladas",
        "",
        f"Total de pontos de avaliação processados: {len(data.point_records)}",
        f"Total de configurações avaliadas: {len(set(p.configuration for p in data.point_records))}",
        "",
        "## Resumo das Curvas de Aprendizado (MAE OOF Médio em kg)",
        "",
        "| Configuração | 50% Treino | 75% Treino | 100% Treino |",
        "| :--- | :---: | :---: | :---: |",
    ]

    configs = sorted({s.configuration for s in data.summary_records})
    for config in configs:
        res: dict[float, float] = {}
        for s in data.summary_records:
            if s.configuration == config:
                res[s.fraction] = s.mean_mae_kg
        f50 = f"{res.get(0.50, 0.0):.2f}"
        f75 = f"{res.get(0.75, 0.0):.2f}"
        f100 = f"{res.get(1.00, 0.0):.2f}"
        lines.append(f"| {config} | {f50} | {f75} | {f100} |")

    lines.extend([
        "",
        "Nota: Subconjuntos de 50% e 75% foram gerados de forma aninhada, estratificada e determinística com seed 45.",
        "Os pontos de 100% reutilizam os artefatos existentes apenas quando a proveniência é válida e atual.",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
