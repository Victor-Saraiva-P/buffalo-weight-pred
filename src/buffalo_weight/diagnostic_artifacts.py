"""Output artifact writing for descriptive diagnostic slice."""

from __future__ import annotations

from pathlib import Path

from buffalo_weight.csv_io import format_csv_number, write_csv_rows
from buffalo_weight.diagnostic_descriptive_slice import DescriptiveDiagnosticSlice


COVERAGE_COLUMNS = ["stratum_type", "stratum_value", "sample_count"]
STRATIFIED_COLUMNS = [
    "configuration", "evaluation_role", "stratum_type", "stratum_value",
    "sample_count", "mae_kg", "median_abs_error_kg", "bias_kg",
]
FARM_COMPARISON_COLUMNS = [
    "configuration", "evaluation_role", "sample_scope", "farm",
    "sample_count", "mae_kg", "median_abs_error_kg", "bias_kg", "confounding_note",
]
RESIDUAL_CORRELATION_COLUMNS = [
    "configuration_1", "configuration_2", "evaluation_role_1", "evaluation_role_2", "pearson_r",
]
NOTABLE_CASE_COLUMNS = [
    "file_name", "case_type", "observed_weight_kg", "weight_category",
    "farm", "resolution", "metric_value",
]


def write_descriptive_diagnostic_artifacts(
    output_dir: Path,
    slice_data: DescriptiveDiagnosticSlice,
) -> None:
    """Write tidy CSV files and report for descriptive diagnostic slice.

    Example: ``write_descriptive_diagnostic_artifacts(output_dir, slice_data)`` saves artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_coverage_csv(output_dir / "sample_coverage.csv", slice_data)
    _write_stratified_csv(output_dir / "stratified_metrics.csv", slice_data)
    _write_farm_comparison_csv(output_dir / "farm_comparison.csv", slice_data)
    _write_correlations_csv(output_dir / "residual_correlations.csv", slice_data)
    _write_notable_cases_csv(output_dir / "notable_cases.csv", slice_data)
    _write_report_markdown(output_dir / "descriptive_diagnostics_report.md", slice_data)


def _write_coverage_csv(path: Path, data: DescriptiveDiagnosticSlice) -> None:
    summary = data.coverage_summary
    rows: list[dict[str, str]] = []
    for cat, count in sorted(summary.category_counts.items()):
        rows.append({"stratum_type": "weight_category", "stratum_value": cat, "sample_count": str(count)})
    for farm, count in sorted(summary.farm_counts.items()):
        rows.append({"stratum_type": "farm", "stratum_value": farm, "sample_count": str(count)})
    for res, count in sorted(summary.resolution_counts.items()):
        rows.append({"stratum_type": "resolution", "stratum_value": res, "sample_count": str(count)})
    write_csv_rows(rows, path, COVERAGE_COLUMNS)


def _write_stratified_csv(path: Path, data: DescriptiveDiagnosticSlice) -> None:
    rows = [
        {
            "configuration": r.configuration,
            "evaluation_role": r.evaluation_role,
            "stratum_type": r.stratum_type,
            "stratum_value": r.stratum_value,
            "sample_count": str(r.sample_count),
            "mae_kg": format_csv_number(r.mae_kg),
            "median_abs_error_kg": format_csv_number(r.median_abs_error_kg),
            "bias_kg": format_csv_number(r.bias_kg),
        }
        for r in data.stratified_metrics
    ]
    write_csv_rows(rows, path, STRATIFIED_COLUMNS)


def _write_farm_comparison_csv(path: Path, data: DescriptiveDiagnosticSlice) -> None:
    rows = [
        {
            "configuration": r.configuration,
            "evaluation_role": r.evaluation_role,
            "sample_scope": r.sample_scope,
            "farm": r.farm,
            "sample_count": str(r.sample_count),
            "mae_kg": format_csv_number(r.mae_kg),
            "median_abs_error_kg": format_csv_number(r.median_abs_error_kg),
            "bias_kg": format_csv_number(r.bias_kg),
            "confounding_note": r.confounding_note,
        }
        for r in data.farm_comparisons
    ]
    write_csv_rows(rows, path, FARM_COMPARISON_COLUMNS)


def _write_correlations_csv(path: Path, data: DescriptiveDiagnosticSlice) -> None:
    rows = [
        {
            "configuration_1": r.configuration_1,
            "configuration_2": r.configuration_2,
            "evaluation_role_1": r.evaluation_role_1,
            "evaluation_role_2": r.evaluation_role_2,
            "pearson_r": format_csv_number(r.pearson_r),
        }
        for r in data.residual_correlations
    ]
    write_csv_rows(rows, path, RESIDUAL_CORRELATION_COLUMNS)


def _write_notable_cases_csv(path: Path, data: DescriptiveDiagnosticSlice) -> None:
    all_cases = [*data.shared_hard_cases, *data.divergent_cases]
    configs = sorted({cfg for c in all_cases for cfg in c.predictions})
    headers = [*NOTABLE_CASE_COLUMNS, *[f"{cfg}_pred_kg" for cfg in configs]]
    rows = [_notable_case_row(c, configs) for c in all_cases]
    write_csv_rows(rows, path, headers)


def _notable_case_row(c: NotableCaseRecord, configs: list[str]) -> dict[str, str]:
    row = {
        "file_name": c.file_name,
        "case_type": c.case_type,
        "observed_weight_kg": format_csv_number(c.observed_weight_kg),
        "weight_category": c.weight_category,
        "farm": c.farm,
        "resolution": c.resolution,
        "metric_value": format_csv_number(c.metric_value),
    }
    for cfg in configs:
        val = c.predictions.get(cfg)
        row[f"{cfg}_pred_kg"] = format_csv_number(val) if val is not None else ""
    return row


def _write_report_markdown(path: Path, data: DescriptiveDiagnosticSlice) -> None:
    lines = [
        "# Relatório de Diagnóstico Expandido: Caracterização de Cobertura e Erros",
        "",
        f"Amostra total analisada: {data.coverage_summary.sample_count} Máscaras Válidas.",
        "",
        "## Resumo de Cobertura da Amostra",
        f"- Categorias de peso: {len(data.coverage_summary.category_counts)} faixas (B1–B10)",
        f"- Fazendas: {len(data.coverage_summary.farm_counts)}",
        f"- Resoluções: {len(data.coverage_summary.resolution_counts)}",
        "",
        "## Casos Notáveis",
        f"- Casos Difíceis Compartilhados: {len(data.shared_hard_cases)}",
        f"- Casos Divergentes Entre Abordagens: {len(data.divergent_cases)}",
        "",
        "Nota: Fazenda, faixa de peso extremo e aquisição permanecem confundidas na amostra atual.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
