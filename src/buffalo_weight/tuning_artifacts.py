"""Output artifact construction for configuration tuning evidence."""

from __future__ import annotations

from pathlib import Path

from buffalo_weight.baseline_comparison_artifacts import METRIC_COLUMNS, _metric_record
from buffalo_weight.baseline_comparison_types import ComparisonMetric, ComparisonPrediction
from buffalo_weight.csv_io import write_csv_rows
from buffalo_weight.tuning_types import TuningVariation


def write_tuning_artifacts(
    output_dir: Path, predictions: list[ComparisonPrediction],
    metrics: list[ComparisonMetric], selected_approach: str,
    baseline_config: str, variations: tuple[TuningVariation, ...],
) -> None:
    """Write metrics and markdown report for tuning stage.

    Example: ``write_tuning_artifacts(output_dir, preds, metrics, ...)`` writes files to disk.
    """
    write_csv_rows([_metric_record(row) for row in metrics],
                   output_dir / "tuning_metrics.csv", METRIC_COLUMNS)
    report_text = generate_tuning_report(metrics, selected_approach, baseline_config, variations)
    (output_dir / "tuning_report.md").write_text(report_text)


def generate_tuning_report(
    metrics: list[ComparisonMetric], selected_approach: str,
    baseline_config: str, variations: tuple[TuningVariation, ...],
) -> str:
    """Render markdown report for tuning evidence.

    Example: ``generate_tuning_report(metrics, "random_forest", ...)`` returns markdown string.
    """
    lines = _report_header(selected_approach, baseline_config)
    lines.extend(_variations_summary_table(metrics, variations))
    lines.extend(_conclusions_section(metrics, baseline_config))
    return "\n".join(lines) + "\n"


def _report_header(selected_approach: str, baseline_config: str) -> list[str]:
    return [
        "# Relatório de Ajuste Fino de Configuração", "",
        f"- Abordagem Selecionada: `{selected_approach}`",
        f"- Configuração Baseline: `{baseline_config}`", "",
        "Este relatório compara o desempenho OOF das variações ajustadas pré-registradas "
        "com a configuração baseline confirmada, sem reabrir a seleção de features.",
    ]


def _variations_summary_table(
    metrics: list[ComparisonMetric], variations: tuple[TuningVariation, ...],
) -> list[str]:
    lines = ["", "## Desempenho das Variações Pré-registradas", "",
             "| Variação | Função | MAE (kg) | RMSE (kg) | Viés (kg) | R² |",
             "|---|---|---:|---:|---:|---:|"]
    for row in metrics:
        if row.scope == "oof" and row.population == "all":
            rmse = f"{row.rmse_kg:.2f}" if row.rmse_kg is not None else ""
            r2 = f"{row.r2:.3f}" if row.r2 is not None else ""
            lines.append(
                f"| `{row.configuration}` | {row.evaluation_role} | {row.mae_kg:.2f} | "
                f"{rmse} | {row.bias_kg:.2f} | {r2} |"
            )
    return lines


def _conclusions_section(metrics: list[ComparisonMetric], baseline_config: str) -> list[str]:
    baseline_rows = [m for m in metrics if m.configuration == baseline_config and m.scope == "oof" and m.population == "all"]
    baseline_mae = baseline_rows[0].mae_kg if baseline_rows else None
    lines = ["", "## Conclusões", ""]
    if baseline_mae is not None:
        lines.append(f"O MAE OOF do baseline `{baseline_config}` é {baseline_mae:.2f} kg.")
    lines.extend([
        "",
        "As variações avaliadas foram pré-registradas conforme o orçamento de no máximo 3 variações "
        "e mantiveram exatamente o mesmo protocolo e conjunto de features.",
    ])
    return lines
